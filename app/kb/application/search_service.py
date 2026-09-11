"""
Search workflow for the KB domain — 6-step retrieval pipeline.

Pipeline stages:
1. (Optional) HyDE query expansion
2. Embed query → dense + sparse vectors
3. Hybrid search (top_k=50) → SearchResult[]
4. Fetch child chunks → get child text + breadcrumbs
5. Cross-encoder rerank chunks → top-8
6. Hydrate parents + siblings + cross-refs → merge + dedupe
"""

import math
import re
from datetime import datetime
from typing import Dict, List, Optional, Set
import structlog

from app.kb.domain.interfaces import ITextEmbedder, IVectorStore, IKBRepository, IReranker, IQueryExpander, SearchResult
from app.kb.domain.models import PDFDocument, RetrievedContext, RetrievedDocument
from app.kb.application.retrieval_strategies import BaselineStrategy, RetrievalStrategy

logger = structlog.get_logger(__name__)

# Pipeline defaults
INITIAL_SEARCH_TOP_K = 50
RERANK_TOP_K = 8
HYDE_FUSION_RRF_K = 60

# Hydration budget.
#
# Siblings and cross-refs carry score 0.0 — they were never scored against the
# query, they are adjacency guesses. Uncapped they crowd the result: reranked
# candidates dedupe by parent down to 4-8 unique chunks, so unranked hydration
# was taking 7-11 of 15 slots and up to 68% of the context characters the LLM
# sees. These caps keep hydration a supplement to the ranked results instead of
# a replacement for them.
MAX_SIBLING_CONTEXTS = 3
MAX_CROSS_REF_CONTEXTS = 3

# A single chunk can mention dozens of "Pasal N", and each prefix is a separate
# path-prefix query returning every matching chunk in the KB — 1366 fetched
# chunks for one observed query, essentially all of them discarded at top_k.
# Bound both the fan-out and the per-prefix result set.
MAX_CROSS_REF_PREFIXES = 8
MAX_CHUNKS_PER_CROSS_REF = 2

# Shortest hydrated chunk worth a context slot. 1070 parent chunks in this KB
# are heading-only skeletons ("MEMUTUSKAN:\n\nPasal 1\n\nPasal 2\n\nPasal 3") —
# a "Pasal 3" path-prefix lookup matches them across unrelated documents, and
# they were arriving as evidence that says nothing. Primaries are exempt: those
# were scored against the query, so a short one earned its place.
MIN_HYDRATED_CHARS = 200


class SearchService:
    """Orchestrates document search combining Qdrant vectors and Postgres full-text chunks."""

    def __init__(
        self,
        text_embedder: ITextEmbedder,
        vector_store: IVectorStore,
        kb_repo: IKBRepository,
        reranker: Optional[IReranker] = None,
        query_expander: Optional[IQueryExpander] = None,
        retrieval_strategy: Optional[RetrievalStrategy] = None,
        rerank_probe: str = "hyde",
    ):
        """Wire the collaborators used by the 6-step pipeline (see module
        docstring): ``text_embedder`` (step 2), ``vector_store`` (step 3),
        ``kb_repo`` (steps 4, 6), ``reranker`` (step 5, optional — skipped
        if None), and ``query_expander`` (step 1 HyDE, optional — injected
        across the chat→kb boundary when enabled, see
        ``app/kb/domain/interfaces.py::IQueryExpander``). ``retrieval_strategy``
        is the final document-ranking rule applied by ``search_documents``;
        defaults to the identity baseline.

        ``rerank_probe`` selects what the cross-encoder scores against when
        expansion ran: ``"hyde"`` (the generated passage) or ``"query"`` (the
        raw question). See ``RetrievalSettings.rerank_probe``."""
        self.text_embedder = text_embedder
        self.vector_store = vector_store
        self.kb_repo = kb_repo
        self.reranker = reranker
        self.query_expander = query_expander
        self.retrieval_strategy = retrieval_strategy or BaselineStrategy()
        self.rerank_probe = rerank_probe

    async def search(
        self,
        query: str,
        top_k: int = 15,
        session_id: Optional[str] = None,
        mode: str = "hybrid",
        rerank: bool = True,
        use_expansion: bool = True,
        hydrate: bool = True,
    ) -> List[RetrievedContext]:
        """Search the Knowledge Base using the 6-step retrieval pipeline.

        When an :class:`IQueryExpander` is configured, retrieval fuses the
        raw-query ranking with a dense-only ranking from the normalized mean of
        several HyDE passage embeddings. Sparse features always remain those of
        the raw query. Child fetch, reranking, and parent hydration occur after
        the rank-fusion point.

        ``rerank=False`` skips the cross-encoder step and keeps the fusion
        ranking — for ablations comparing the fusion strategies themselves
        rather than three reranked variants of them. Production callers leave it
        True.

        ``use_expansion=False`` skips HyDE for this call even when an expander
        is wired. Query expansion costs one LLM round-trip per passage, so a
        caller that only needs a coarse signal — the relevance pre-check, which
        looks at top-k scores rather than reading the chunks — should not pay
        for it. ``hydrate=False`` likewise skips sibling/cross-reference
        expansion, which only appends unranked adjacency (score 0.0) and costs
        a burst of DB queries.
        """
        if not query.strip():
            return []

        logger.info("kb.search.started", query_length=len(query), top_k=top_k, mode=mode)

        # --- Step 1: raw embedding, then optional three-passage HyDE ---
        # The raw query is always retained.  A mean HyDE dense vector supplies
        # a second ranking, rather than replacing raw dense or sparse evidence.
        raw_embeddings = await self.text_embedder.embed_texts([query], is_query=True)
        if not raw_embeddings:
            return []
        raw_query_emb = raw_embeddings[0]
        hyde_dense_vector: Optional[List[float]] = None
        # Kept for step 5: the hypothetical answer is a sharper cross-encoder
        # probe than the question, because it is written in the register of the
        # passage that should match (see RetrievalSettings.rerank_probe).
        hyde_probe: Optional[str] = None
        if self.query_expander is not None and use_expansion:
            try:
                hyde_docs = await self.query_expander.expand_many(query)
                if hyde_docs:
                    hyde_probe = "\n\n".join(hyde_docs)
                    # Deliberately NOT is_query: HyDE passages are hypothetical
                    # documents, so for an asymmetric encoder they belong in
                    # document space — that identity is the whole point of HyDE.
                    hyde_embeddings = await self.text_embedder.embed_texts(hyde_docs)
                    hyde_dense_vector = self._mean_normalized_dense(
                        [embedding.dense for embedding in hyde_embeddings]
                    )
                    if hyde_dense_vector is not None:
                        logger.info(
                            "kb.search.hyde_ensemble_generated",
                            query_len=len(query),
                            passages=len(hyde_embeddings),
                            strategy="raw_rank_plus_mean_hyde_dense_rrf",
                        )
                else:
                    logger.warning("kb.search.hyde_empty", query_len=len(query))
            except Exception as exc:
                logger.warning("kb.search.hyde_failed", error=str(exc), query_len=len(query))

        # --- Steps 2–3: raw retrieval, optional HyDE retrieval, rank fusion ---
        raw_results = await self.vector_store.hybrid_search(
            dense_vector=raw_query_emb.dense,
            sparse_indices=raw_query_emb.sparse_indices,
            sparse_values=raw_query_emb.sparse_values,
            top_k=INITIAL_SEARCH_TOP_K,
            session_id=session_id,
            mode=mode,
        )
        search_results = raw_results
        # HyDE expands a dense representation.  Sparse-only ablations therefore
        # deliberately remain raw-query only.
        if hyde_dense_vector is not None and mode != "sparse":
            hyde_results = await self.vector_store.hybrid_search(
                dense_vector=hyde_dense_vector,
                sparse_indices=raw_query_emb.sparse_indices,
                sparse_values=raw_query_emb.sparse_values,
                top_k=INITIAL_SEARCH_TOP_K,
                session_id=session_id,
                mode="dense",
            )
            search_results = self._rrf_fuse([raw_results, hyde_results])
            logger.info(
                "kb.search.hyde_rank_fused",
                raw_candidates=len(raw_results),
                hyde_candidates=len(hyde_results),
                fused_candidates=len(search_results),
            )

        if not search_results:
            return []

        logger.info("kb.search.hybrid_done", candidates=len(search_results))

        # --- Step 4: Fetch child chunks ---
        chunk_ids = [r.chunk_id for r in search_results]
        child_chunks = await self.kb_repo.get_child_chunks_by_ids(chunk_ids)
        child_map = {c.id: c for c in child_chunks}

        # Build (search_result, child_chunk) pairs for children that exist
        candidates = []
        for sr in search_results:
            child = child_map.get(sr.chunk_id)
            if child:
                candidates.append((sr, child))

        if not candidates:
            # Fallback: no child chunks persisted — use parent text directly
            return await self._fallback_parent_search(
                search_results, query, top_k
            )

        # --- Step 5: Cross-encoder rerank chunks → top-8 ---
        if rerank and self.reranker is not None and candidates:
            try:
                rerank_texts = await self._rerank_texts(candidates)
                # Scoring against the raw question lets a chunk that merely
                # echoes its wording outrank the one that answers it — "berapa
                # UKT 2026" pulls decrees that happen to say "UKT" and "2026".
                # Concatenating query + passage is worse than either alone: it
                # re-injects the wording the passage was meant to get past.
                probe = query
                if self.rerank_probe == "hyde" and hyde_probe:
                    probe = hyde_probe
                rerank_results = await self.reranker.rerank(
                    query=probe,
                    documents=rerank_texts,
                    top_k=RERANK_TOP_K,
                )
                candidates = [
                    candidates[r.index] for r in rerank_results
                    if 0 <= r.index < len(candidates)
                ][:RERANK_TOP_K]
                logger.info("kb.search.rerank_done", kept=len(candidates))
            except Exception as exc:
                logger.warning("kb.search.rerank_failed", error=str(exc))
                # Truncate to RERANK_TOP_K by original search score
                candidates = candidates[:RERANK_TOP_K]
        else:
            candidates = candidates[:RERANK_TOP_K]

        # --- Step 6: Hydrate parents + siblings + cross-refs ---
        parent_ids = list(set(c.parent_chunk_id for _, c in candidates))
        parent_chunks = await self.kb_repo.get_parent_chunks_by_ids(parent_ids)
        parent_map = {pc.id: pc for pc in parent_chunks}

        # Fetch doc titles
        doc_ids = list(set(r.doc_id for r, _ in candidates))
        pdf_docs: List[PDFDocument] = await self.kb_repo.get_pdfs_by_ids(doc_ids)
        doc_title_map: Dict[str, str] = {doc.id: doc.title or doc.id for doc in pdf_docs}
        doc_released_map: Dict[str, Optional[datetime]] = {doc.id: doc.released_date for doc in pdf_docs}

        # Build primary contexts. The hierarchical breadcrumb is appended to the
        # parent text here (post-retrieval) rather than embedded into the child
        # vectors at ingestion time — "post-generation" hierarchical chunking.
        contexts: List[RetrievedContext] = []
        for sr, child in candidates:
            parent = parent_map.get(child.parent_chunk_id)
            if parent:
                breadcrumbs = parent.breadcrumbs or []
                text = (
                    (" > ".join(breadcrumbs) + "\n\n" + parent.text)
                    if breadcrumbs
                    else parent.text
                )
                contexts.append(
                    RetrievedContext(
                        chunk_id=sr.chunk_id,
                        parent_chunk_id=child.parent_chunk_id,
                        doc_id=sr.doc_id,
                        text=text,
                        score=sr.score,
                        source_title=doc_title_map.get(sr.doc_id, sr.doc_id),
                        page=parent.page,
                        breadcrumbs=breadcrumbs,
                        content_type=getattr(parent, "content_type", "text") or "text",
                        child_text=child.text,
                        path=getattr(parent, "path", "") or "",
                        depth=getattr(parent, "depth", 0) or 0,
                        released_date=doc_released_map.get(sr.doc_id),
                    )
                )

        # --- Step 6b/6c: Sibling + cross-reference hydration (bounded) ---
        # Both are pure DB fan-out and only add unranked adjacency, so a caller
        # that just needs retrieval scores can turn them off.
        sibling_contexts: List[RetrievedContext] = []
        cross_ref_contexts: List[RetrievedContext] = []
        if hydrate:
            sibling_contexts = await self._hydrate_siblings(
                contexts, parent_map, doc_title_map
            )
            cross_ref_contexts = await self._detect_and_fetch_cross_refs(
                contexts, parent_map, doc_title_map
            )

        # --- Step 7: Merge + dedupe ---
        all_contexts = self._merge_and_dedupe(contexts, sibling_contexts, cross_ref_contexts)

        # Truncate to final top_k
        result = all_contexts[:top_k]

        logger.info(
            "kb.search.completed",
            results_count=len(result),
            primary=len(contexts),
            siblings=len(sibling_contexts),
            cross_refs=len(cross_ref_contexts),
        )
        return result

    async def search_documents(
        self,
        query: str,
        top_k: int = 15,
        session_id: Optional[str] = None,
        mode: str = "hybrid",
        rerank: bool = True,
    ) -> List[RetrievedDocument]:
        """Run the chunk-level pipeline and aggregate to document-level output."""
        contexts = await self.search(
            query=query,
            top_k=top_k,
            session_id=session_id,
            mode=mode,
            rerank=rerank,
        )
        return await self.aggregate_documents(contexts)

    async def aggregate_documents(
        self, contexts: List[RetrievedContext]
    ) -> List[RetrievedDocument]:
        """Aggregate chunk-level contexts into document-level output.

        Returns one :class:`RetrievedDocument` per unique source document, with
        the document's original title, release date, and the concatenated
        retrieved parent sections (in retrieval order, deduplicated by parent
        chunk). The document score is the best chunk score; the configured
        retrieval strategy then produces the final ordering (e.g. date priority).
        """
        if not contexts:
            return []

        # Group ranked contexts by document, preserving retrieval order.
        grouped: Dict[str, List[RetrievedContext]] = {}
        for ctx in contexts:
            grouped.setdefault(ctx.doc_id, []).append(ctx)

        # Derive title/released_date from the contexts themselves (populated by
        # ``search()``), avoiding a redundant ``get_pdfs_by_ids`` on the hot
        # path. Only docs whose contexts all lack ``released_date`` (e.g. a
        # cross-reference introducing a doc outside the primary candidate set)
        # are refetched — a rare fallback.
        title_by_doc: Dict[str, str] = {}
        released_by_doc: Dict[str, Optional[datetime]] = {}
        missing: List[str] = []
        for doc_id, ctxs in grouped.items():
            title_by_doc[doc_id] = ctxs[0].source_title or doc_id
            released = next(
                (c.released_date for c in ctxs if c.released_date is not None),
                None,
            )
            released_by_doc[doc_id] = released
            if released is None:
                missing.append(doc_id)

        if missing:
            pdf_docs = await self.kb_repo.get_pdfs_by_ids(missing)
            for pdf in pdf_docs:
                if pdf.title:
                    title_by_doc[pdf.id] = pdf.title
                released_by_doc[pdf.id] = pdf.released_date

        documents: List[RetrievedDocument] = []
        for doc_id, ctxs in grouped.items():
            # Sibling/cross-ref hydration can surface the same parent more than
            # once — deduplicate so document content isn't repeated.
            seen: Set[str] = set()
            parts: List[str] = []
            for ctx in ctxs:
                key = ctx.parent_chunk_id or ctx.chunk_id
                if key in seen:
                    continue
                seen.add(key)
                parts.append(ctx.text)

            documents.append(
                RetrievedDocument(
                    doc_id=doc_id,
                    title=title_by_doc.get(doc_id, doc_id),
                    released_date=released_by_doc.get(doc_id),
                    content="\n\n".join(parts),
                    score=max(ctx.score for ctx in ctxs),
                )
            )

        return self.retrieval_strategy.rank_documents(documents)

    @staticmethod
    def _mean_normalized_dense(vectors: List[List[float]]) -> Optional[List[float]]:
        """Return the L2-normalized centroid of compatible dense vectors."""
        if not vectors or not vectors[0]:
            return None
        width = len(vectors[0])
        compatible = [vector for vector in vectors if len(vector) == width]
        if not compatible:
            return None
        mean = [sum(vector[index] for vector in compatible) / len(compatible) for index in range(width)]
        norm = math.sqrt(sum(value * value for value in mean))
        return [value / norm for value in mean] if norm else None

    @staticmethod
    def _rrf_fuse(rankings: List[List[SearchResult]], k: int = HYDE_FUSION_RRF_K) -> List[SearchResult]:
        """Fuse rankings by reciprocal rank, keyed by child chunk id."""
        fused: Dict[str, tuple[SearchResult, float]] = {}
        for ranking in rankings:
            for rank, result in enumerate(ranking, start=1):
                existing, score = fused.get(result.chunk_id, (result, 0.0))
                fused[result.chunk_id] = (existing, score + 1.0 / (k + rank))
        return [
            SearchResult(
                chunk_id=result.chunk_id,
                parent_chunk_id=result.parent_chunk_id,
                doc_id=result.doc_id,
                score=score,
            )
            for result, score in sorted(fused.values(), key=lambda item: item[1], reverse=True)
        ]

    async def _rerank_texts(self, candidates: List) -> List[str]:
        """Return the text each candidate should be scored on.

        Children are ~512 chars (``CHUNKING_CHILD_MAX_CHARS``), which is too
        little for a cross-encoder to judge relevance: most children of the
        *correct* document are boilerplate ("Segala biaya yang timbul…") that
        genuinely does not answer the question in isolation. A reranker asked
        about them correctly answers "no" — for the right document and the wrong
        one alike — so their scores collapse into an undifferentiated floor and
        sorting by score shuffles right-document chunks against wrong-document
        ones. Measured on Qwen3-Reranker-0.6B: 0.000294 vs 0.000175 for
        boilerplate children (noise), against 0.98 vs 0.0007 for the same two
        documents at parent size — a 1470x separation instead of 1.7x.

        The missing context is not more surrounding prose — it is the
        document's *identity*, and that lives outside the chunk text entirely.
        A child reading "membayar biaya UKT 500.000" is unjudgeable on its own:
        which programme, which year? Only the document title says
        "1313-UN40-KM.02.02-2026 - Peserta Program Outbound Student Mobility".
        Parent text does not contain it either — parents are sections, while
        the title is a column on ``PDFDocument`` — so scoring parents fixes the
        collapse without answering "which document is this".

        So each candidate is scored on its own text prefixed with the title,
        release year, and heading breadcrumbs. That is ~700 chars against
        ~4096 for a parent, and it is the only variant that carries the decree
        number the identifier-style questions actually ask for.

        Degrades cleanly: a document whose title is missing is scored on its
        chunk text alone, exactly as before.
        """
        doc_ids = list({sr.doc_id for sr, _ in candidates})
        titles: Dict[str, str] = {}
        codes: Dict[str, str] = {}
        released: Dict[str, Optional[datetime]] = {}
        crumbs: Dict[str, List[str]] = {}
        try:
            docs = await self.kb_repo.get_pdfs_by_ids(doc_ids)
            titles = {d.id: (d.title or "") for d in docs}
            codes = {d.id: (getattr(d, "code", None) or "") for d in docs}
            released = {d.id: d.released_date for d in docs}
            # Breadcrumbs live on ParentChunk, not ChildChunk — reading
            # child.breadcrumbs silently yields nothing.
            parents = await self.kb_repo.get_parent_chunks_by_ids(
                list({c.parent_chunk_id for _, c in candidates})
            )
            crumbs = {p.id: (p.breadcrumbs or []) for p in parents}
        except Exception as exc:
            # Metadata is an enrichment, not a requirement — fall back to the
            # bare chunk rather than failing the whole rerank step.
            logger.warning("kb.search.rerank_meta_fetch_failed", error=str(exc))

        texts: List[str] = []
        for sr, child in candidates:
            header: List[str] = []
            code = codes.get(sr.doc_id, "").strip()
            if code:
                # Its own labelled line, not buried in the title: the
                # identifier-style questions ask for exactly this string.
                header.append(f"Nomor: {code}")
            title = titles.get(sr.doc_id, "").strip()
            if title:
                header.append(f"Dokumen: {title}")
            year = released.get(sr.doc_id)
            if year is not None:
                header.append(f"Tahun: {year.year}")
            path = [c for c in crumbs.get(child.parent_chunk_id, []) if c]
            if path:
                header.append("Bagian: " + " > ".join(path))
            texts.append(
                ("\n".join(header) + "\n\n" + child.text) if header else child.text
            )
        return texts

    async def _fallback_parent_search(
        self,
        search_results: List,
        query: str,
        top_k: int,
    ) -> List[RetrievedContext]:
        """Fallback when child chunks are not persisted — use parent text directly.

        Args:
            search_results: Raw search results from Qdrant.
            query: Original query (for reranking).
            top_k: Maximum results to return.

        Returns:
            List of RetrievedContext.
        """
        parent_chunk_ids = list(set(r.parent_chunk_id for r in search_results))
        parent_chunks = await self.kb_repo.get_parent_chunks_by_ids(parent_chunk_ids)
        parent_chunk_map = {pc.id: pc for pc in parent_chunks}

        doc_ids = list(set(r.doc_id for r in search_results))
        pdf_docs = await self.kb_repo.get_pdfs_by_ids(doc_ids)
        doc_title_map: Dict[str, str] = {doc.id: doc.title or doc.id for doc in pdf_docs}

        contexts: List[RetrievedContext] = []
        for result in search_results:
            parent = parent_chunk_map.get(result.parent_chunk_id)
            if parent:
                contexts.append(
                    RetrievedContext(
                        chunk_id=result.chunk_id,
                        parent_chunk_id=result.parent_chunk_id,
                        doc_id=result.doc_id,
                        text=parent.text,
                        score=result.score,
                        source_title=doc_title_map.get(result.doc_id, result.doc_id),
                        page=parent.page,
                        breadcrumbs=parent.breadcrumbs or [],
                        content_type=getattr(parent, "content_type", "text") or "text",
                    )
                )

        if self.reranker is not None and contexts:
            try:
                rerank_results = await self.reranker.rerank(
                    query=query,
                    documents=[c.text for c in contexts],
                )
                contexts = [contexts[r.index] for r in rerank_results if 0 <= r.index < len(contexts)]
            except Exception as exc:
                logger.warning("kb.rerank.skipped", error=str(exc))

        return contexts[:top_k]

    async def _hydrate_siblings(
        self,
        primary_contexts: List[RetrievedContext],
        parent_map: Dict,
        doc_title_map: Dict[str, str],
    ) -> List[RetrievedContext]:
        """Fetch sibling parent chunks for each primary context.

        Siblings are parent chunks sharing the same parent_id (i.e., sections
        under the same parent section). This provides adjacent context that
        may be relevant but wasn't directly matched by vector search.

        Args:
            primary_contexts: The primary retrieved contexts.
            parent_map: Map of parent_chunk_id → ParentChunk.
            doc_title_map: Map of doc_id → title.

        Returns:
            List of sibling RetrievedContext (deduped, excludes primary contexts).
        """
        if not primary_contexts:
            return []

        sibling_contexts: List[RetrievedContext] = []
        seen_parent_ids: Set[str] = {
            c.parent_chunk_id for c in primary_contexts
        }

        for ctx in primary_contexts:
            if len(sibling_contexts) >= MAX_SIBLING_CONTEXTS:
                break
            parent = parent_map.get(ctx.parent_chunk_id)
            if not parent or not getattr(parent, "parent_id", None):
                continue

            try:
                siblings = await self.kb_repo.get_sibling_chunks(parent.parent_id)
            except Exception as exc:
                logger.warning("kb.search.sibling_fetch_failed", error=str(exc))
                continue

            for sib in siblings:
                if len(sibling_contexts) >= MAX_SIBLING_CONTEXTS:
                    break
                if sib.id in seen_parent_ids:
                    continue
                if len(sib.text or "") < MIN_HYDRATED_CHARS:
                    continue
                seen_parent_ids.add(sib.id)
                sibling_contexts.append(
                    RetrievedContext(
                        chunk_id=sib.id,
                        parent_chunk_id=sib.id,
                        doc_id=sib.doc_id,
                        text=sib.text,
                        score=0.0,  # Siblings have no direct search score
                        source_title=doc_title_map.get(sib.doc_id, sib.doc_id),
                        page=sib.page,
                        breadcrumbs=sib.breadcrumbs or [],
                        content_type=getattr(sib, "content_type", "text") or "text",
                        path=getattr(sib, "path", "") or "",
                        depth=getattr(sib, "depth", 0) or 0,
                    )
                )

        logger.info("kb.search.siblings_hydrated", count=len(sibling_contexts))
        return sibling_contexts

    async def _detect_and_fetch_cross_refs(
        self,
        primary_contexts: List[RetrievedContext],
        parent_map: Dict,
        doc_title_map: Dict[str, str],
    ) -> List[RetrievedContext]:
        """Detect cross-references in retrieved text and fetch referenced chunks.

        Scans child text and parent text for references like "Pasal N",
        "Ayat N", "BAB N" and fetches the referenced parent chunks by path
        prefix lookup.

        Args:
            primary_contexts: The primary retrieved contexts.
            parent_map: Map of parent_chunk_id → ParentChunk.
            doc_title_map: Map of doc_id → title.

        Returns:
            List of cross-referenced RetrievedContext (deduped).
        """
        if not primary_contexts:
            return []

        # Collect cross-reference prefixes in *first-appearance* order, walking
        # primaries best-first. A set would iterate in hash order, which varies
        # with PYTHONHASHSEED — the surviving cross-refs would then differ
        # between runs of the same query.
        path_prefixes: List[str] = []
        seen_prefixes: Set[str] = set()
        for ctx in primary_contexts:
            texts_to_scan = []
            if ctx.child_text:
                texts_to_scan.append(ctx.child_text)
            texts_to_scan.append(ctx.text)

            for text in texts_to_scan:
                for prefix in self._extract_cross_references(text):
                    if prefix not in seen_prefixes:
                        seen_prefixes.add(prefix)
                        path_prefixes.append(prefix)

        if not path_prefixes:
            return []

        # References from the highest-ranked primaries are the ones worth
        # spending queries on.
        path_prefixes = path_prefixes[:MAX_CROSS_REF_PREFIXES]

        cross_ref_contexts: List[RetrievedContext] = []
        seen_ids: Set[str] = {c.parent_chunk_id for c in primary_contexts}

        for prefix in path_prefixes:
            if len(cross_ref_contexts) >= MAX_CROSS_REF_CONTEXTS:
                break
            try:
                referenced = await self.kb_repo.get_chunks_by_path_prefix(prefix)
            except Exception as exc:
                logger.warning("kb.search.crossref_fetch_failed", prefix=prefix, error=str(exc))
                continue

            for ref in referenced[:MAX_CHUNKS_PER_CROSS_REF]:
                if len(cross_ref_contexts) >= MAX_CROSS_REF_CONTEXTS:
                    break
                if ref.id in seen_ids:
                    continue
                if len(ref.text or "") < MIN_HYDRATED_CHARS:
                    continue
                seen_ids.add(ref.id)
                cross_ref_contexts.append(
                    RetrievedContext(
                        chunk_id=ref.id,
                        parent_chunk_id=ref.id,
                        doc_id=ref.doc_id,
                        text=ref.text,
                        score=0.0,  # Cross-refs have no direct search score
                        source_title=doc_title_map.get(ref.doc_id, ref.doc_id),
                        page=ref.page,
                        breadcrumbs=ref.breadcrumbs or [],
                        content_type=getattr(ref, "content_type", "text") or "text",
                        path=getattr(ref, "path", "") or "",
                        depth=getattr(ref, "depth", 0) or 0,
                    )
                )

        logger.info("kb.search.crossrefs_detected", count=len(cross_ref_contexts))
        return cross_ref_contexts

    def _extract_cross_references(self, text: str) -> List[str]:
        """Extract cross-reference path prefixes from text.

        Detects Indonesian legal cross-references like:
        - ``Pasal 5`` → path prefix ``pasal_5``
        - ``BAB II`` → path prefix ``bab_ii``
        - ``Ayat 3`` → path prefix ``ayat_3``

        Args:
            text: The text to scan for cross-references.

        Returns:
            List of path prefix strings to search for in the KB.
        """
        if not text:
            return []

        prefixes: List[str] = []

        # Pasal N → pasal_n
        for m in re.finditer(r"Pasal\s+(\d+)", text, re.IGNORECASE):
            prefixes.append(f"pasal_{m.group(1)}")

        # BAB N (Roman or Arabic) → bab_n
        for m in re.finditer(r"BAB\s+([IVXLC]+|\d+)", text, re.IGNORECASE):
            prefixes.append(f"bab_{m.group(1).lower()}")

        # Ayat N → ayat_n
        for m in re.finditer(r"Ayat\s+\(?(\d+)\)?", text, re.IGNORECASE):
            prefixes.append(f"ayat_{m.group(1)}")

        return prefixes

    def _merge_and_dedupe(
        self,
        primary: List[RetrievedContext],
        siblings: List[RetrievedContext],
        cross_refs: List[RetrievedContext],
    ) -> List[RetrievedContext]:
        """Merge primary, sibling, and cross-ref contexts with deduplication.

        Preserves original ranking order: primary contexts first (by score),
        then siblings, then cross-refs. Deduplicates by parent_chunk_id.

        Args:
            primary: Primary retrieved contexts (ordered by relevance).
            siblings: Sibling contexts (adjacent sections).
            cross_refs: Cross-referenced contexts.

        Returns:
            Merged and deduplicated list of RetrievedContext.
        """
        seen: Set[str] = set()
        result: List[RetrievedContext] = []

        for ctx in primary + siblings + cross_refs:
            if ctx.parent_chunk_id not in seen:
                seen.add(ctx.parent_chunk_id)
                result.append(ctx)

        return result
