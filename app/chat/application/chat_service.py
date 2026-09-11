"""
Orchestrator for the Chat Module.

Coordinates the Knowledge Base, IVM (Safety/Relevance), RAM (Response Assessment),
and the LLM inference engine.
"""

import json
import uuid
import asyncio
import structlog
from typing import AsyncGenerator, List, Dict, Any, Optional, Tuple

from app.chat.application.history import (
    approx_token_count,
    build_history,
    format_history_for_condenser,
)
from app.chat.application.query_condenser import QueryCondenser
from app.chat.domain.interfaces import IChatRepository, ILLMConnection
from app.chat.domain.usage import collect_usage
from app.kb.application.search_service import SearchService
from app.guardrails.ivm.service import IVMService, MaliciousPromptException
from app.guardrails.ivm.relevance_service import RelevanceService, IrrelevantQueryException
from app.rag.prompts import build_prompt
from app.guardrails.ram.service import LABEL_CONTRADICTION, RAMService
from app.guardrails.ram.interfaces import ClaimUnit, RetrievedContext as RAMRetrievedContext
from app.guardrails.ram.claim_parser import (
    extract_citations,
    parse_table_block,
    split_cells,
    split_claims,
)
from app.guardrails.ram.clause_splitter import ClauseSplitter
from app.guardrails.ram.text_utils import is_unciteable_statement

logger = structlog.get_logger(__name__)

# Minimum claim length (chars) worth an NLI call / an "Unverified" flag.
MIN_ASSESSABLE_LENGTH = 8

class ChatService:
    """Orchestrates the chat request pipeline (application-layer service).

    ``process_chat_message`` runs the streaming RAG pipeline for one user turn:
    persist the user message → IVM safety check → IVM relevance pre-check →
    deep KB retrieval (top_k=15) → prompt build (with anti-injection
    delimiter) → stream the LLM completion → per-proposition RAM entailment
    assessment with citation markers → persist the assistant message with its
    RAG context.

    Collaborators (ports from ``app/chat/domain/interfaces.py`` or services
    from other bounded contexts, wired in
    ``app/chat/dependency.py::get_chat_service``): ``chat_repo`` (session/
    message persistence), ``llm_conn`` (LLM calls), ``search_service`` (hybrid
    KB retrieval for both the pre-check and deep fetch), ``ivm_service``
    (prompt-injection safety gate), ``relevance_service`` (topical/OOD gate),
    and ``ram_service`` (per-sentence NLI citation markers).

    Each defense is independently disableable for ablation — ``skip_ivm``,
    ``skip_ram``, ``skip_nonce`` — with ``skip_guardrails`` as the shorthand
    for the first two. Retrieval always runs so the LLM has context.
    """

    def __init__(
        self,
        chat_repo: IChatRepository,
        llm_conn: ILLMConnection,
        search_service: SearchService,
        ivm_service: IVMService,
        relevance_service: RelevanceService,
        ram_service: RAMService,
        model_name: str,
        system_prompt: str,
        clause_splitter: Optional[ClauseSplitter] = None,
        temperature: float = 0.0,
        attachment_search_excerpt_chars: int = 4000,
        history_max_tokens: int = 3000,
        context_max_tokens: int = 6000,
        query_condenser: Optional[QueryCondenser] = None,
        refusal_message: str = "Maaf, pertanyaan ini tidak tercakup dalam dokumen yang tersedia.",
        safety_block_message: str = "Maaf, permintaan ini diblokir oleh filter keamanan.",
        internal_error_message: str = "Maaf, terjadi kesalahan saat memproses permintaan Anda.",
    ):
        """Wire the pipeline's collaborators and generation settings.

        ``attachment_search_excerpt_chars`` caps how much of an uploaded
        attachment's text is folded into the KB search query (the full text
        still goes to the LLM prompt). ``history_max_tokens`` budgets the
        replayed conversation history. ``context_max_tokens`` budgets the
        retrieved context block (see ``_clamp_contexts``). ``query_condenser``
        rewrites elliptical follow-ups for retrieval; None disables
        condensation. The three message strings are the user-facing
        refusal/error texts.
        """
        self.chat_repo = chat_repo
        self.llm_conn = llm_conn
        self.search_service = search_service
        self.ivm_service = ivm_service
        self.relevance_service = relevance_service
        self.ram_service = ram_service
        self.clause_splitter = clause_splitter
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.attachment_search_excerpt_chars = attachment_search_excerpt_chars
        self.history_max_tokens = history_max_tokens
        self.context_max_tokens = context_max_tokens
        self.query_condenser = query_condenser
        self.refusal_message = refusal_message
        self.safety_block_message = safety_block_message
        self.internal_error_message = internal_error_message

    async def create_session(self) -> Dict[str, Any]:
        """Create a new chat session.

        Commits before returning: the client POSTs to ``/stream`` with this ID
        as soon as it has the response, and FastAPI only unwinds the DB
        dependency (and its commit) *after* the response is sent — so without
        an explicit commit the stream request can fail to see the session it
        was just handed.
        """
        session_id = str(uuid.uuid4())
        session = await self.chat_repo.create_session(session_id, "New Chat")
        await self.chat_repo.commit()
        return {"id": session.id, "title": session.title}

    async def list_sessions(self) -> List[Dict[str, Any]]:
        """List all chat sessions."""
        sessions = await self.chat_repo.get_all_sessions()
        return [{"id": s.id, "title": s.title} for s in sessions]

    async def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session details."""
        session = await self.chat_repo.get_session_by_id(session_id, load_messages=True)
        if not session:
            return None
        return {
            "id": session.id,
            "title": session.title,
            "messages": [
                {
                    "role": m.role,
                    "content": m.content,
                    "context": m.context,
                    "sources": m.sources,
                    "attachment_filename": m.attachment_filename,
                }
                for m in session.messages
            ],
        }

    async def delete_session(self, session_id: str) -> bool:
        """Delete a chat session."""
        return await self.chat_repo.delete_session(session_id)

    @staticmethod
    def _format_citation(result: Any) -> str:
        """Format an NLI result into the citation badge grammar.

        Always includes all three per-class scores so the frontend can show
        Supported / Neutral / Contradicted together; source/page/doc_id/
        evidence appear only when present. Returns "" for None.
        """
        if result is None:
            return ""
        parts = [
            f"Supported:{result.entailment_score:.2f}",
            f"Neutral:{result.neutral_score:.2f}",
            f"Contradicted:{result.contradiction_score:.2f}",
        ]
        if result.source_title:
            parts.append(result.source_title)
        if result.page is not None:
            parts.append(f"Page {result.page}")
        if result.doc_id:
            parts.append(f"DocID:{result.doc_id}")
        if result.evidence_snippet:
            parts.append(f'Evidence:"{result.evidence_snippet}"')
        return f" *({'; '.join(parts)})*"

    @staticmethod
    def _format_unverified() -> str:
        """Badge for a factual-looking claim the LLM left uncited."""
        return " *(Unverified)*"

    @staticmethod
    def _worst_result(results: List[Any]) -> Any:
        """Collapse a sentence's per-clause NLI results into the one to show.

        A sentence is only as trustworthy as its weakest clause, so a
        contradiction anywhere wins; otherwise the lowest entailment score
        does. Mirrors ``RAMService._pick_best`` inverted — that picks the most
        supporting evidence *for one claim*, this picks the least supported
        *claim in one sentence*. Returns None when nothing was assessed.
        """
        assessed = [r for r in results if r is not None]
        if not assessed:
            return None
        contradictions = [r for r in assessed if r.label == LABEL_CONTRADICTION]
        if contradictions:
            return max(contradictions, key=lambda r: r.contradiction_score)
        return min(assessed, key=lambda r: r.entailment_score)

    def _split_claims(self, buffer: str) -> Tuple[List[ClaimUnit], str]:
        """Split ``buffer`` into complete claim units + an incomplete remainder.

        Synchronous on purpose: the caller offloads it to a worker thread via
        ``asyncio.to_thread`` so the Stanza dependency parser (CPU-bound,
        50-200ms/sentence) never blocks the event loop.
        """
        return split_claims(buffer, self.clause_splitter)

    @staticmethod
    def _clamp_contexts(
        contexts: List[RAMRetrievedContext],
        max_tokens: int,
    ) -> List[RAMRetrievedContext]:
        """Truncate retrieved contexts to ``max_tokens`` (approximate), keeping
        retrieval order (highest-ranked first). A non-positive budget disables
        clamping. The same (clamped) list feeds the prompt and RAM's citation
        lookup, so the ``[CIT:N]`` numbering stays consistent.
        """
        if max_tokens <= 0:
            return contexts
        kept: List[RAMRetrievedContext] = []
        used = 0
        for ctx in contexts:
            cost = (
                approx_token_count(ctx.text)
                + approx_token_count(ctx.source_title or "")
                + 4
            )
            if kept and used + cost > max_tokens:
                break
            kept.append(ctx)
            used += cost
        return kept

    async def _assess_table(
        self,
        table_text: str,
        ram_contexts: List[RAMRetrievedContext],
        skip_ram: bool,
    ) -> AsyncGenerator[str, None]:
        """Assess a complete Markdown table and re-emit it with in-cell badges."""
        table = parse_table_block(table_text)
        if table is None:
            clean, ids = extract_citations(table_text)
            async for out in self._handle_claim(
                ClaimUnit(text=clean, citation_ids=ids, kind="prose", separator="\n"),
                ram_contexts=ram_contexts, skip_ram=skip_ram, table_rows=[],
            ):
                yield out
            return

        badges: Dict[Tuple[int, int], str] = {}
        if not skip_ram:
            for claim in table.claims:
                result = await self.ram_service.assess_claim(
                    claim.cell_text, ram_contexts, claim.citation_ids
                )
                badges[(claim.row_index, claim.col_index)] = self._format_citation(result)

        header = extract_citations(table.header)[0].strip()
        lines = [header, table.separator]
        for row_index, line in enumerate(table.data_lines):
            cells = [extract_citations(c)[0].strip() for c in split_cells(line)]
            for col_index in range(len(cells)):
                badge = badges.get((row_index, col_index), "")
                if badge:
                    cells[col_index] = cells[col_index] + badge
            lines.append("| " + " | ".join(cells) + " |")
        yield "\n".join(lines) + "\n\n"

    async def _handle_claim(
        self,
        unit: ClaimUnit,
        *,
        ram_contexts: List[RAMRetrievedContext],
        skip_ram: bool,
        table_rows: List[Tuple[str, str]],
        pending_clauses: Optional[List[Any]] = None,
    ) -> AsyncGenerator[str, None]:
        """Process one complete claim unit, yielding output chunks in stream order.

        Table rows are buffered (not emitted) until the block ends, because the
        in-cell badges can only be rendered once the whole table is parsed.

        ``pending_clauses`` accumulates the NLI results of the non-final
        clauses of a multi-clause sentence (see ``ClaimUnit.is_sentence_end``).
        Every clause is still assessed individually, but the badge is held back
        until the sentence ends and then reflects the weakest clause — a badge
        in the middle of a sentence reads as a stray "." to the user.
        """
        if pending_clauses is None:
            pending_clauses = []

        if unit.kind == "table_row":
            table_rows.append((unit.text, unit.separator))
            return

        if table_rows:
            table_text = "\n".join(row for row, _ in table_rows)
            table_rows.clear()
            # A table can only start at a sentence boundary, so any clause
            # results still pending belong to a sentence that never completed.
            pending_clauses.clear()
            async for out in self._assess_table(table_text, ram_contexts, skip_ram):
                yield out

        text = unit.text.strip()
        sep = unit.separator
        if unit.kind not in ("prose", "list_item") or not text:
            pending_clauses.clear()
            yield text + sep
            return

        if unit.is_sentence_end and not text.endswith((".", "?", "!")):
            text += "."

        if skip_ram:
            yield text + sep
            return

        result = None
        if unit.citation_ids:
            result = await self.ram_service.assess_claim(text, ram_contexts, unit.citation_ids)

        # Mid-sentence clause: keep its verdict, emit the text bare.
        if not unit.is_sentence_end:
            pending_clauses.append((text, result))
            yield text + sep
            return

        sentence = pending_clauses + [(text, result)]
        pending_clauses.clear()
        results = [r for _, r in sentence]

        if any(r is not None for r in results):
            badge = self._format_citation(self._worst_result(results))
        else:
            # Uncited: judge "worth flagging" on the whole sentence, not just
            # its trailing clause. A refusal or a piece of advice has no citable
            # source by construction, so flagging it would put a warning on an
            # honest "I don't have that" — see is_unciteable_statement.
            sentence_text = " ".join(t for t, _ in sentence)
            length = sum(len(t) for t, _ in sentence)
            worth_flagging = (
                length > MIN_ASSESSABLE_LENGTH
                and not is_unciteable_statement(sentence_text)
            )
            badge = self._format_unverified() if worth_flagging else ""

        yield text + badge + sep

    async def process_chat_message(
        self,
        session_id: str,
        message_text: str,
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        """Run one chat turn, accounting for what it cost.

        A thin wrapper over :meth:`_process_chat_message` that installs a
        turn-scoped usage collector. The turn fans out into LLM calls owned by
        three different objects — HyDE inside ``SearchService``, the condenser,
        and generation here — so collecting by async context attributes them
        together without threading a callback through every signature.

        Emits one ``chat.turn.cost`` event per turn, which is what makes spend
        answerable in production ("what does a turn actually cost?") rather than
        only inside a benchmark.
        """
        with collect_usage() as usage:
            async for event in self._process_chat_message(
                session_id, message_text, **kwargs
            ):
                yield event

        if usage.call_count:
            logger.info(
                "chat.turn.cost",
                session_id=session_id,
                llm_calls=usage.call_count,
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                cached_prompt_tokens=usage.cached_prompt_tokens,
                cost_usd=usage.cost_usd,
            )

    async def _process_chat_message(
        self,
        session_id: str,
        message_text: str,
        skip_guardrails: bool = False,
        attachment_text: Optional[str] = None,
        attachment_filename: Optional[str] = None,
        skip_ivm: Optional[bool] = None,
        skip_ram: Optional[bool] = None,
        skip_nonce: bool = False,
    ) -> AsyncGenerator[str, None]:
        """The main generation pipeline: Safety → Pre-check → Context → Generate → Assess.

        The three defenses disable independently so an experiment can attribute
        an effect to one of them: ``skip_ivm`` (safety + relevance),
        ``skip_ram`` (per-sentence assessment), ``skip_nonce`` (the
        anti-injection delimiter, a structural defense separate from the IVM
        classifier). ``skip_guardrails`` is the shorthand for ``skip_ivm`` +
        ``skip_ram`` unless either is passed explicitly; it does not imply
        ``skip_nonce``. Retrieval always runs so the LLM has context.

        ``attachment_text`` (extracted text of a chat-uploaded PDF, if any) is
        treated as part of this turn's prompt only: folded into the same IVM
        check, KB search queries, and delimiter as the typed message, but never
        persisted or replayed on later turns.
        """

        # Resolve the individual switches from the shorthand.
        skip_ivm = skip_guardrails if skip_ivm is None else skip_ivm
        skip_ram = skip_guardrails if skip_ram is None else skip_ram

        # The attachment is data the user is asking about, so it shares the
        # typed message's trust boundary rather than being system context.
        if attachment_text:
            combined_text = (
                f"{message_text}\n\n[Dokumen terlampir: {attachment_filename}]\n{attachment_text}"
            )
            # Capped excerpt for KB search only (full text still goes to the
            # IVM check and LLM prompt): a short question alone often won't
            # retrieve the right chunks since the topic lives in the
            # attachment, but a full document would degrade embedding/HyDE.
            search_query_text = (
                f"{message_text}\n\n{attachment_text[: self.attachment_search_excerpt_chars]}"
            )
        else:
            combined_text = message_text
            search_query_text = message_text

        # Everything from here runs inside the try so that any failure — DB or
        # otherwise — becomes a handled NDJSON error event rather than tearing
        # the generator down mid-stream and rolling the turn back silently.
        try:
            # 1. Initialize or get session. create_session is idempotent, so a
            # racing creator can't fail this turn (see PostgresChatRepository).
            session = await self.chat_repo.get_session_by_id(session_id, load_messages=True)
            if not session:
                session = await self.chat_repo.create_session(
                    session_id, message_text[:20] + "..."
                )
            elif session.title == "New Chat":
                new_title = message_text[:30] + ("..." if len(message_text) > 30 else "")
                await self.chat_repo.update_session_title(session, new_title)

            # Capture history before any DB writes to avoid async lazy-load
            # errors (greenlet_spawn after flush/commit). Trimmed to a token
            # budget and replayed from raw_content — see application/history.py.
            history = build_history(
                list(session.messages) if session.messages else [],
                max_tokens=self.history_max_tokens,
            )

            # Record the user message. Only the typed text and the attachment's
            # filename are persisted; the extracted text is single-turn only.
            await self.chat_repo.create_message(
                session_id, "user", message_text, raw_content=message_text,
                attachment_filename=attachment_filename,
            )
            # Commit the user turn immediately: it is durable regardless of what
            # the rest of the pipeline does, and the next request can see it.
            await self.chat_repo.commit()

            # 2. Safety check (IVM) over the combined message + attachment,
            # since an attached PDF can carry a prompt injection too. Always the
            # raw text — never a condensed rewrite.
            if not skip_ivm:
                await self.ivm_service.check_malicious(combined_text)

            # 2b. Resolve an elliptical follow-up ("kenapa begitu?") against the
            # conversation before it reaches retrieval or the relevance gate,
            # which otherwise see a query with no antecedent and abstain on it.
            # Retrieval-side only: combined_text (the LLM prompt) is untouched.
            if self.query_condenser and history:
                search_query_text = await self.query_condenser.condense(
                    format_history_for_condenser(history), search_query_text
                )

            # 3. Relevance pre-check (IVM + KB). No session_id is passed: the
            # chat session ID is unrelated to the KB chunk session_id payload,
            # and filtering on it would return zero results. HyDE is skipped
            # here (``use_expansion=False``): the gate reads retrieval scores,
            # not the chunks, so it does not justify an LLM round-trip per
            # passage — the deep fetch below still expands.
            precheck_contexts = await self.search_service.search(
                search_query_text, top_k=3, use_expansion=False, hydrate=False
            )
            if not skip_ivm:
                if not precheck_contexts:
                    raise IrrelevantQueryException("No relevant contexts found in the knowledge base.")

                context_chunks = [ctx.text for ctx in precheck_contexts]
                context_scores = [ctx.score for ctx in precheck_contexts]
                await self.relevance_service.check_relevance(search_query_text, context_chunks, context_scores)

            # 4. Deep Context Retrieval (KB). Chunk-level contexts are kept for
            # RAM's per-sentence evidence lookup; document-level aggregation
            # feeds the LLM prompt and the emitted "view RAG context" panel.
            full_contexts = await self.search_service.search(search_query_text, top_k=15)
            documents = await self.search_service.aggregate_documents(full_contexts)

            # Chunk-level contexts for both the LLM prompt (numbered Sumber N →
            # [CIT:N]) and RAM's citation-local evidence lookup. Preserve
            # breadcrumbs/hierarchy and the matching child text.
            doc_map = {doc.doc_id: doc for doc in documents}
            ram_contexts = [
                RAMRetrievedContext(
                    text=ctx.text,
                    source_title=ctx.source_title,
                    page=ctx.page,
                    breadcrumbs=ctx.breadcrumbs,
                    content_type=ctx.content_type,
                    chunk_id=ctx.chunk_id,
                    path=ctx.path,
                    doc_id=ctx.doc_id,
                    child_text=ctx.child_text or "",
                    parent_chunk_id=ctx.parent_chunk_id,
                    released_date=(
                        doc_map[ctx.doc_id].released_date.isoformat()
                        if ctx.doc_id in doc_map and doc_map[ctx.doc_id].released_date
                        else None
                    ),
                )
                for ctx in full_contexts
            ]

            # Clamp the retrieved context to the token budget BEFORE building
            # the prompt and before RAM's citation lookup, so the [CIT:N]
            # numbering the model sees matches the evidence RAM verifies.
            ram_contexts = self._clamp_contexts(ram_contexts, self.context_max_tokens)

            # 5. Prompt build: Indonesian system/context prompt plus a random
            # per-request delimiter wrapping the raw user message (injection
            # defense on top of the IVM check).
            bundle = build_prompt(
                combined_text, ram_contexts, self.system_prompt, use_nonce=not skip_nonce
            )

            # History is already trimmed to the token budget and normalized to
            # {"role", "content"} dicts by build_history().
            messages = [{"role": "system", "content": bundle.system_prompt}]
            messages.extend(history)
            messages.append({"role": "user", "content": bundle.user_turn})

            # Emit the retrieved context as one NDJSON event before streaming,
            # for the frontend's collapsible "view RAG context" panel and for
            # downstream consumers; the same payload is persisted with the
            # assistant message so it survives a refresh.
            #
            # "content" is a flat join of document texts; "chunks" is now the
            # document-level structure the chat UI renders (one object per
            # unique source document).
            context_payload = {
                "content": "\n\n".join(doc.content for doc in documents),
                "chunks": [
                    {
                        "doc_id": doc.doc_id,
                        "title": doc.title,
                        "released_date": doc.released_date.isoformat() if doc.released_date else None,
                        "content": doc.content,
                    }
                    for doc in documents
                ],
            }
            yield json.dumps({"type": "context", **context_payload}) + "\n"

            # Buffer the stream by claim unit; assess each complete unit.
            buffer = ""
            raw_output = ""
            final_output = ""
            # Accumulates contiguous table-row units until a non-row unit ends
            # the block (see _handle_claim). Local to this call — never
            # instance state, since ChatService may be reused across requests.
            table_rows: List[Tuple[str, str]] = []
            # Per-clause NLI results of the sentence currently being emitted,
            # held until its final clause carries the badge (see _handle_claim).
            pending_clauses: List[Any] = []

            # 6. Stream and Assess
            stream = self.llm_conn.stream_chat(
                model=self.model_name,
                messages=messages,
                max_tokens=1024,
                temperature=self.temperature,
            )

            async for chunk in stream:
                raw_output += chunk
                buffer += chunk
                units, remainder = await asyncio.to_thread(self._split_claims, buffer)
                for unit in units:
                    async for out in self._handle_claim(
                        unit, ram_contexts=ram_contexts,
                        skip_ram=skip_ram, table_rows=table_rows,
                        pending_clauses=pending_clauses,
                    ):
                        final_output += out
                        yield json.dumps({"type": "chunk", "content": out}) + "\n"
                buffer = remainder

            # Flush any remaining buffer through the same dispatch.
            if buffer.strip():
                units, remainder = await asyncio.to_thread(self._split_claims, buffer)
                for unit in units:
                    async for out in self._handle_claim(
                        unit, ram_contexts=ram_contexts,
                        skip_ram=skip_ram, table_rows=table_rows,
                        pending_clauses=pending_clauses,
                    ):
                        final_output += out
                        yield json.dumps({"type": "chunk", "content": out}) + "\n"
                if remainder.strip():
                    clean, ids = extract_citations(remainder)
                    async for out in self._handle_claim(
                        ClaimUnit(text=clean, citation_ids=ids, kind="prose", separator=""),
                        ram_contexts=ram_contexts, skip_ram=skip_ram, table_rows=table_rows,
                        pending_clauses=pending_clauses,
                    ):
                        final_output += out
                        yield json.dumps({"type": "chunk", "content": out}) + "\n"

            # If the answer ended inside a table (last content was rows, so no
            # trailing non-row unit triggered the flush), assess and flush the
            # accumulated block now.
            if table_rows:
                table_text = "\n".join(row for row, _ in table_rows)
                table_rows.clear()
                async for out in self._assess_table(table_text, ram_contexts, skip_ram):
                    final_output += out
                    yield json.dumps({"type": "chunk", "content": out}) + "\n"

            # Persist the assistant message with its RAG context/sources so the
            # frontend can restore the "view RAG context" panel after a refresh.
            await self.chat_repo.create_message(
                session_id, "assistant", final_output, raw_content=raw_output,
                context=context_payload["content"], sources=context_payload["chunks"],
            )
            # Commit before signalling "done": the client may start the next
            # turn the moment it sees this, and FastAPI's own commit doesn't run
            # until after the response is fully sent.
            await self.chat_repo.commit()
            yield json.dumps({"type": "done"}) + "\n"

        except MaliciousPromptException:
            async for evt in self._emit_refusal(session_id, self.safety_block_message, "unsafe"):
                yield evt
        except IrrelevantQueryException:
            async for evt in self._emit_refusal(session_id, self.refusal_message, "irrelevant"):
                yield evt
        except Exception as e:
            logger.error("chat.pipeline.failed", error=str(e), exc_info=True)
            # The failure may have poisoned the transaction, so roll back before
            # writing. The user message is already committed above, so this only
            # discards partial work from this turn.
            try:
                await self.chat_repo.rollback()
            except Exception:  # pragma: no cover - rollback is best-effort
                logger.warning("chat.pipeline.rollback_failed", exc_info=True)
            async for evt in self._emit_refusal(
                session_id, self.internal_error_message, "internal"
            ):
                yield evt

    async def _emit_refusal(
        self, session_id: str, message: str, reason: str
    ) -> AsyncGenerator[str, None]:
        """Persist a refusal/error reply and emit its NDJSON events.

        ``reason`` distinguishes an intentional abstention ("irrelevant",
        "unsafe") from a pipeline failure ("internal"). The event type stays
        "error" for backward compatibility — the eval harness
        (``exp4_end_to_end/run.py``) and ``tools/visualize/chat_viz.py`` key on
        it — but the frontend uses ``reason`` to render a deliberate abstention
        differently from a crash, and Exp 4's abstention metric can use it to
        stop counting infrastructure failures as correct refusals.

        Persisting is best-effort: if the DB is the thing that broke, the user
        must still receive the message and a terminating "done".
        """
        try:
            await self.chat_repo.create_message(
                session_id, "assistant", message, raw_content=message
            )
            await self.chat_repo.commit()
        except Exception:
            logger.warning("chat.refusal.persist_failed", reason=reason, exc_info=True)

        yield json.dumps({"type": "error", "content": message, "reason": reason}) + "\n"
        yield json.dumps({"type": "done"}) + "\n"
