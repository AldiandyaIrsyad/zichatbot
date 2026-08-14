"""Build Subset D — RAM Ground Truth.

Generates sentence-level NLI annotations by running questions through the
full RAG pipeline, decomposing responses into sentences, and having the
panel label each sentence.

Pipeline:
    1. Load Subset A questions (stratified sample)
    2. Run each question through the full chat pipeline via API
    3. Decompose the response into sentences, dropping fragments too short
       to be a real proposition (MIN_SENTENCE_TOKENS)
    4. Panel labels each sentence directly as entailment, neutral,
       contradiction, or no_source_needed using the most relevant chunks.
    5. Keep a label only when at least 4/5 successful panel votes agree; store
       3/5 disagreements outside the principal inference set.
    6. Assemble exactly 150 balanced diagnostic rows (50 per NLI class) and
       60 natural rows, then lock 30 balanced diagnostic rows for calibration
       and the remaining 180 rows for evaluation.
    7. Output CSV with columns: question_id, question, full_response,
       sentence_id, sentence_text, retrieved_context, label, verifier_note,
       plus slice metadata (construction, perturbation_family, intended_label,
       perturbation_of, difficulty_band, split, edit_note)
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
import structlog

from evals._dataset_gen.config import DatasetGenSettings, get_dataset_gen_settings
from evals._dataset_gen.concordance import BlindInjectionTracker
from evals._dataset_gen.generator import DatasetGenerator
from evals._dataset_gen.checkpoint import IncrementalCSVWriter, resume_rows
from evals._dataset_gen.panel import EvaluatorPanel
from evals._dataset_gen.perturbations import (
    FAMILIES,
    LABEL_TO_NLI,
    MIN_CLASS_ROWS,
    SUPPORTED,
    assert_class_balance,
    assert_no_shortcut_features,
    band_by_nli,
    generate_perturbation,
    label_distribution,
    nli_class_counts,
    shortcut_report,
)
from evals._dataset_gen.provenance import write_provenance
from evals._shared.clients import EvalNLIClient
from app.guardrails.ram.text_utils import split_sentences as _split_sentences_production

logger = structlog.get_logger(__name__)

# Inline citation marker format emitted by ChatService._format_citation(),
# e.g. *(Supported: 0.66; <doc title>; Page 3; DocID:<uuid>; Evidence:"...")*
# Must be stripped BEFORE sentence splitting: abbreviated titles/names inside
# the marker (e.g. "M.Ag.", "Dr.", "S.Pd.") contain periods that the naive
# sentence splitter otherwise mistakes for sentence boundaries, corrupting
# sentence_text with citation-marker fragments instead of real sentences.
CITATION_MARKER_PATTERN = re.compile(
    r"\*?\s*\("
    r"(?:Supported|Contradiction|Neutral)"
    r":\s*[\d.]+"
    r";\s*[^;]+"
    r"(?:;\s*Page\s+\d+)?"
    r"(?:;\s*DocID:[\w-]+)?"
    r'(?:;\s*Evidence:"[^"]*")?'
    r"\)\s*\*?"
)

LABELS = ["entailment", "neutral", "contradiction", "no_source_needed"]

# Question and Full Response are identical across every sentence of one
# question; Retrieved Context (narrowed per sentence by
# select_relevant_chunks, below) is usually identical too since most
# sentences of one response draw on the same chunk(s). Sentence is the one
# field that varies per call. Ordering the prompt so the varying field comes
# LAST maximizes the shared prefix across a question's per-sentence panel
# calls, which is what OpenRouter's provider-side prompt caching actually
# caches against.
VALIDATION_PROMPT = """\
You are evaluating a sentence from an LLM response for a hallucination \
detection benchmark.

Question: {question}
Full Response: {full_response}
Retrieved Context: {retrieved_context}
Sentence (ID {sentence_id}): {sentence_text}

Label the sentence as one of:
- "entailment": The complete proposition is directly supported by the retrieved context
- "neutral": The context does not establish the complete proposition and does not conflict with it
- "contradiction": The retrieved context explicitly conflicts with the proposition
- "no_source_needed": The sentence doesn't need verification (greetings, transitions, etc.)

Decision rules:
- entailment requires support for every material clause, quantifier, entity, and relation.
- neutral includes absent details and compound propositions that are only partly supported.
- contradiction requires explicit incompatible evidence, not merely missing support.
- no_source_needed is only for non-factual discourse; a factual claim without evidence is neutral.
Return exactly one label and no explanation.
"""

# Minimum token count for a sentence to be considered a real proposition
# rather than a fragment (e.g. a bare name split out of a markdown list:
# "Arief Johari, S.T., M.Ds."). Fragments have no defensible entailment
# label and dilute label quality. 5 is a pragmatic floor, not a linguistic
# parse — a real Indonesian clause is rarely shorter than this once citation
# markers are already stripped.
MIN_SENTENCE_TOKENS = 5

# How many of the response's retrieved chunks to keep per sentence when
# building the panel prompt (see select_relevant_chunks). Narrowing from
# "all ~15 chunks" (15-22k chars observed) to the few chunks actually
# relevant to this sentence is both a quality fix (the panel isn't drowned in
# irrelevant text) and the single largest cost lever available — larger than
# any model swap.
MAX_CONTEXT_CHUNKS_FOR_PANEL = 3


def split_sentences(text: str) -> List[str]:
    """Split text into sentences.

    Strips inline citation markers first (see ``CITATION_MARKER_PATTERN``),
    since their embedded periods (e.g. in abbreviated titles) otherwise get
    mistaken for sentence boundaries, corrupting the result with citation
    fragments instead of real sentences. Delegates the actual splitting to
    the production splitter (``app.guardrails.ram.text_utils.split_sentences``),
    which already handles markdown list markers ("1.", "2.") correctly —
    production only ever splits pre-citation text, so this eval-only
    citation-stripping step has no equivalent there.
    """
    if not text or not text.strip():
        return []
    cleaned = CITATION_MARKER_PATTERN.sub("", text)
    return _split_sentences_production(cleaned)


async def run_pipeline(
    api_url: str,
    session_id: str,
    question: str,
) -> Optional[Dict[str, Any]]:
    """Run a question through the full chat pipeline.

    Calls the streaming chat endpoint (``/api/chat/sessions/{id}/stream``)
    and consumes the NDJSON stream, accumulating ``chunk`` events into the
    response text and capturing the ``context`` event emitted before
    streaming begins.

    Returns a dict with response text, flat retrieved context, and the
    structured per-chunk list (``chunks``, each a dict with ``title``/``page``/
    ``breadcrumbs``/``text`` — see chat_service.py's context payload), or None
    on error. ``chunks`` lets callers narrow context per sentence
    (select_relevant_chunks) instead of guessing chunk boundaries out of the
    flat joined string.
    """
    async with httpx.AsyncClient(base_url=api_url, timeout=180.0) as client:
        # Send message and collect NDJSON stream.
        # The chat API expects {"message": ...} (see app/chat/api.py ChatRequest).
        response = await client.post(
            f"/api/chat/sessions/{session_id}/stream",
            json={"message": question},
            headers={"Accept": "application/x-ndjson"},
        )
        if response.status_code != 200:
            return None

        full_response = ""
        retrieved_context = ""
        retrieved_chunks: List[Dict[str, Any]] = []

        for line in response.text.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            try:
                chunk = json.loads(line)
                chunk_type = chunk.get("type", "")
                if chunk_type == "chunk":
                    full_response += chunk.get("content", "")
                elif chunk_type == "context":
                    retrieved_context = chunk.get("content", "")
                    retrieved_chunks = chunk.get("chunks", []) or []
                elif chunk_type == "error":
                    # Pipeline rejected the query (e.g. IVM block); stop early
                    return {
                        "response": chunk.get("content", ""),
                        "context": retrieved_context,
                        "chunks": retrieved_chunks,
                    }
                elif chunk_type == "done":
                    break
            except Exception:
                continue

        return {
            "response": full_response,
            "context": retrieved_context,
            "chunks": retrieved_chunks,
        }


def select_relevant_chunks(
    sentence: str,
    chunks: List[Dict[str, Any]],
    max_chunks: int = MAX_CONTEXT_CHUNKS_FOR_PANEL,
) -> str:
    """Narrow the retrieved chunks to the few most relevant to ``sentence``.

    Ranks chunks by token containment of ``sentence`` (``|sent ∩ chunk| /
    |sent|``) and returns the top ``max_chunks``, joined in their original
    retrieval order (not by score) so the excerpt still reads coherently.
    Falls back to joining everything if there are no real chunk boundaries
    to work with (e.g. an API response without the ``chunks`` field).

    This exists for two reasons at once: it keeps the panel from having to
    read all ~15 retrieved chunks (15-22k chars observed) to label a single
    20-token sentence — the single largest cost lever available for this
    pipeline — and it improves label quality, since a human annotator would
    naturally focus on the passage that actually supports the sentence rather
    than being handed the whole context indiscriminately.

    Returns the narrowed context text (chunk texts joined with blank lines).
    """
    if not chunks:
        return ""
    sent_tokens = set(re.findall(r"\w+", sentence.lower()))
    if not sent_tokens:
        return "\n\n".join(c.get("text", "") for c in chunks[:max_chunks])

    def _containment(chunk: Dict[str, Any]) -> float:
        chunk_tokens = set(re.findall(r"\w+", chunk.get("text", "").lower()))
        if not chunk_tokens:
            return 0.0
        return len(sent_tokens & chunk_tokens) / len(sent_tokens)

    ranked = sorted(range(len(chunks)), key=lambda i: _containment(chunks[i]), reverse=True)
    keep = sorted(ranked[:max_chunks])  # restore original retrieval order
    return "\n\n".join(chunks[i].get("text", "") for i in keep)


def select_relevant_passage(sentence: str, context: str, radius: int = 1) -> str:
    """Select a compact evidence passage when structured chunks are unavailable.

    Resume checkpoints retain the flat retrieved context but not the API's chunk
    objects. Rank non-empty paragraphs by sentence-token containment and return
    the best paragraph with its immediate neighbours. This preserves local legal
    context while avoiding a full-document panel prompt.
    """
    paragraphs = [part.strip() for part in context.split("\n\n") if part.strip()]
    if not paragraphs:
        return context
    sent_tokens = set(re.findall(r"\w+", sentence.lower()))
    if not sent_tokens:
        return "\n\n".join(paragraphs[: 2 * radius + 1])

    def score(paragraph: str) -> float:
        tokens = set(re.findall(r"\w+", paragraph.lower()))
        return len(sent_tokens & tokens) / len(sent_tokens) if tokens else 0.0

    best = max(range(len(paragraphs)), key=lambda i: score(paragraphs[i]))
    lo = max(0, best - radius)
    hi = min(len(paragraphs), best + radius + 1)
    return "\n\n".join(paragraphs[lo:hi])


def is_fragment_sentence(sentence: str, min_tokens: int = MIN_SENTENCE_TOKENS) -> bool:
    """Check whether ``sentence`` is too short to be a real proposition.

    Fragments like bare names have no defensible entailment label and are
    skipped.
    """
    return len(sentence.split()) < min_tokens


async def label_sentence(
    panel: EvaluatorPanel,
    question: str,
    question_id: str,
    full_response: str,
    sentence: str,
    sent_idx: int,
    retrieved_context: str,
    chunks: List[Dict[str, Any]],
    session_id: Optional[str] = None,
) -> Tuple[Optional[Dict[str, Any]], str, bool]:
    """Label one sentence via the panel, with plurality fallback.

    Shared by build_subset_d.py and build_subset_d_hard.py so both scripts
    get the same fragment filter, chunk narrowing, and accept/dispute/drop
    logic without duplicating it.

    The stored ``retrieved_context`` in the returned row is always the
    FULL flat context (not the panel's narrowed view) — the narrowing in
    select_relevant_chunks is a labeling aid to keep the panel's prompt
    small and focused, not a claim about what "the" retrieved context was;
    downstream consumers (Exp3, Exp4) should still see everything the
    system actually retrieved.

    Args:
        chunks: Structured per-chunk list for this question (may be empty
            for responses captured before this field existed).
        session_id: Optional OpenRouter sticky-routing session id — pass
            the same id for every sentence of one question.

    Returns:
        Tuple of (row dict or None, status, unanimous). status is one of
        "accepted" (clean majority), "disputed" (plurality fallback),
        "fragment" (skipped before the panel call), "no_signal" (panel ran
        but no label reached even a 2-vote plurality), or "error" (the panel
        call itself failed). ``unanimous`` is True only when every panel
        member independently agreed — the criterion the blind-injection
        tracker uses for candidate selection.
    """
    if is_fragment_sentence(sentence):
        return None, "fragment", False

    narrowed_context = select_relevant_chunks(sentence, chunks) or retrieved_context
    validation_context = VALIDATION_PROMPT.format(
        question=question,
        full_response=full_response,
        sentence_id=sent_idx,
        sentence_text=sentence,
        retrieved_context=narrowed_context,
    )

    try:
        verdict = await panel.evaluate_label(
            prompt="Assign the correct label to this sentence.",
            context=validation_context,
            valid_labels=LABELS,
            session_id=session_id,
        )
    except Exception as e:
        logger.error(
            "datagen.subset_d.panel_error",
            question_id=question_id,
            sentence_id=sent_idx,
            error=str(e),
            exc_info=True,
        )
        return None, "error", False

    if not verdict.accepted or not verdict.accepted_label:
        return None, "contested" if verdict.contested else "no_signal", False
    label = verdict.accepted_label
    status = "accepted"
    top_count = verdict.label_counts[label]

    vote_summary = ", ".join(f"{v.model}@{v.provider or 'unknown'}:{v.label or 'none'}" for v in verdict.votes)
    note_prefix = "Panel majority" if status == "accepted" else "Panel plurality, below acceptance threshold"
    row: Dict[str, Any] = {
        "question_id": question_id,
        "question": question,
        "full_response": full_response,
        "sentence_id": sent_idx,
        "sentence_text": sentence,
        "retrieved_context": retrieved_context,
        "label": label,
        "verifier_note": f"{note_prefix} ({verdict.label_counts}; votes: {vote_summary})",
    }
    unanimous = top_count == len(verdict.votes)
    return row, status, unanimous


async def create_session(api_url: str) -> str:
    """Create a new chat session and return its ID."""
    async with httpx.AsyncClient(base_url=api_url, timeout=30.0) as client:
        response = await client.post("/api/chat/sessions")
        response.raise_for_status()
        return response.json()["id"]


def load_subset_a(path: str, count: int) -> List[Dict[str, str]]:
    """Load a stratified sample of questions from the Subset A CSV."""
    input_path = Path(path)
    if not input_path.exists():
        logger.error("datagen.subset_d.subset_a_not_found", path=path)
        sys.exit(1)

    with input_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        questions = list(reader)

    # Stratified sample: try to get diverse categories
    by_category: Dict[str, List[Dict[str, str]]] = {}
    for q in questions:
        cat = q.get("category", "unknown")
        by_category.setdefault(cat, []).append(q)

    # Round-robin sample across categories
    sampled: List[Dict[str, str]] = []
    idx = 0
    while len(sampled) < count:
        added = False
        for cat in by_category:
            if idx < len(by_category[cat]):
                sampled.append(by_category[cat][idx])
                added = True
                if len(sampled) >= count:
                    break
        if not added:
            break
        idx += 1

    return sampled


# Full CSV schema for the rebuilt Subset D: the base sentence-label columns
# plus the slice metadata the hard rebuild adds (see
# _shared/dataset.SubsetDRow). The blind-injection sidecar keeps only the base
# columns.
BASE_FIELDS = [
    "question_id", "question", "full_response", "sentence_id",
    "sentence_text", "retrieved_context", "label", "verifier_note",
]
SLICE_FIELDS = [
    "construction", "perturbation_family", "intended_label",
    "perturbation_of", "difficulty_band", "split", "evaluation_split",
    "edit_note",
]
FIELDNAMES = BASE_FIELDS + SLICE_FIELDS


def core_resume_state(loaded_core: List[Dict[str, Any]]) -> Tuple[Dict[str, int], set, int]:
    """Reconstruct phase-2 resume bookkeeping from checkpointed core rows.

    Everything phase 2 needs to continue is derivable from the rows already on
    disk, so an interrupted build resumes without re-paying for verified edits.

    Returns ``(kept_by_family, done_parent_family_pairs, max_counter)`` — the
    per-family kept counts to seed the quota check, the ``(parent_ref,
    family)`` pairs to skip re-perturbing, and the highest ``qp-…-NNNN``
    index seen so new ids don't collide.
    """
    kept = dict(Counter(r["perturbation_family"] for r in loaded_core if r.get("perturbation_family")))
    done_pairs = {(r.get("perturbation_of", ""), r.get("perturbation_family", "")) for r in loaded_core}
    counter = 0
    for r in loaded_core:
        try:
            counter = max(counter, int(str(r.get("question_id", "")).rsplit("-", 1)[-1]))
        except (ValueError, IndexError):
            pass
    return kept, done_pairs, counter


def _tag(row: Dict[str, Any], **overrides: str) -> Dict[str, Any]:
    """Return a copy of ``row`` with every slice column present (defaulting empty)."""
    out = dict(row)
    for key in SLICE_FIELDS:
        out.setdefault(key, "")
    out.update(overrides)
    return out


async def assemble_questions(
    generator: DatasetGenerator,
    api_url: str,
    subset_a_path: str,
    subset_c_path: Optional[str],
    counts: Dict[str, int],
    seed: int,
) -> List[Tuple[str, str]]:
    """Assemble the hard question mix for the natural backbone (phase 1).

    Reuses the question generators from ``build_subset_d_hard`` (imported lazily
    to avoid an import cycle, since that module imports this one's primitives):
    reweighted Subset A + precise-detail + cross-document + evaluative questions
    generated from real KB text, plus a few Subset C boundary queries.

    Returns a list of (question, source) pairs.
    """
    from evals._dataset_gen.build_subset_a import fetch_kb_documents
    from evals._dataset_gen.build_subset_d_hard import (
        generate_crossdoc_questions,
        generate_detail_questions,
        generate_evaluative_questions,
        sample_boundary_queries,
        sample_reweighted_subset_a,
    )
    from evals._shared.dataset import load_subset_a as load_subset_a_rows
    from evals._shared.dataset import load_subset_c

    random.seed(seed)
    subset_a_rows = load_subset_a_rows(subset_a_path)
    questions: List[Tuple[str, str]] = [
        (q, "reweighted") for q in sample_reweighted_subset_a(subset_a_rows, counts["reweighted"])
    ]

    if counts["detail"] or counts["crossdoc"] or counts["evaluative"]:
        docs = await fetch_kb_documents(api_url) or []
        if docs and counts["detail"]:
            questions += [(q, "detail") for q in await generate_detail_questions(generator, api_url, docs, counts["detail"])]
        if docs and counts["crossdoc"]:
            questions += [(q, "crossdoc") for q in await generate_crossdoc_questions(generator, api_url, docs, counts["crossdoc"])]
        if docs and counts["evaluative"]:
            questions += [(q, "evaluative") for q in await generate_evaluative_questions(generator, api_url, docs, counts["evaluative"])]

    if counts["boundary"] and subset_c_path:
        c_rows = load_subset_c(subset_c_path)
        questions += [(q, "boundary") for q in sample_boundary_queries(c_rows, counts["boundary"])]

    return questions


async def phase1_natural(
    panel: EvaluatorPanel,
    api_url: str,
    questions_by_source: List[Tuple[str, str]],
    blind_tracker: BlindInjectionTracker,
    status_counts: Dict[str, int],
    writer: Optional["IncrementalCSVWriter"] = None,
    done_qids: Optional[set] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
    """Run each question through the live pipeline and panel-label its sentences.

    This is the unchanged natural methodology (no fabrication) — it supplies the
    realistic base-rate slice and the ``supported`` parents phase 2 perturbs.

    Returns (natural_rows, chunks_by_question_id). Rows carry
    ``construction=natural`` and ``split=natural``; the chunk map lets phase 2
    re-narrow grounding for a chosen parent sentence.
    """
    natural_rows: List[Dict[str, Any]] = []
    chunks_by_qid: Dict[str, List[Dict[str, Any]]] = {}
    source_counters: Dict[str, int] = {}
    done_qids = done_qids or set()

    for question, source in questions_by_source:
        source_counters[source] = source_counters.get(source, 0) + 1
        question_id = f"q-{source}-{source_counters[source]:03d}"
        if question_id in done_qids:
            # Already panel-labeled and checkpointed by an earlier run; the
            # counter still advanced above so IDs stay aligned (--resume).
            continue
        logger.info("datagen.subset_d.processing", question_id=question_id, source=source, question=question[:60])
        try:
            session_id = await create_session(api_url)
            result = await run_pipeline(api_url, session_id, question)
        except Exception as e:
            logger.error("datagen.subset_d.pipeline_error", question_id=question_id, error=str(e), exc_info=True)
            continue
        if not result or not result.get("response"):
            logger.warning("datagen.subset_d.no_response", question_id=question_id)
            continue

        full_response = result["response"]
        retrieved_context = result.get("context", "")
        chunks = result.get("chunks", [])
        chunks_by_qid[question_id] = chunks
        sentences = split_sentences(full_response)
        logger.info("datagen.subset_d.decomposed", question_id=question_id, sentence_count=len(sentences))

        panel_session_id = f"subset-d-{question_id}"
        for sent_idx, sentence in enumerate(sentences):
            row, status, unanimous = await label_sentence(
                panel=panel, question=question, question_id=question_id,
                full_response=full_response, sentence=sentence, sent_idx=sent_idx,
                retrieved_context=retrieved_context, chunks=chunks, session_id=panel_session_id,
            )
            status_counts[status] = status_counts.get(status, 0) + 1
            if row is None:
                continue
            tagged = _tag(row, construction="natural", split="natural")
            natural_rows.append(tagged)
            if writer is not None:
                writer.append(tagged)  # flush per row so an abort keeps the panel spend
            if unanimous:
                blind_tracker.add_candidate({**row, "_panel_label": row["label"]})

    return natural_rows, chunks_by_qid


def _family_quotas(core_floor: int) -> Dict[str, int]:
    """Per-family kept-row targets so each NLI class can clear ``core_floor``.

    ``contradiction`` is split across factual and scope reversals; ``neutral``
    is split across absent-detail and evaluative additions. ``far_paraphrase``
    supplies half the entailment target and natural entailments fill the rest.
    """
    per_class_family = -(-core_floor // 2)
    return {
        "factual_flip": per_class_family,
        "scope_negation_flip": per_class_family,
        "plausible_absent_detail": per_class_family,
        "evaluative_graft": per_class_family,
        "far_paraphrase": max(1, core_floor // 2),
    }


async def phase2_perturb(
    panel: EvaluatorPanel,
    generator: DatasetGenerator,
    natural_rows: List[Dict[str, Any]],
    chunks_by_qid: Dict[str, List[Dict[str, Any]]],
    core_floor: int,
    seed: int,
    writer: Optional["IncrementalCSVWriter"] = None,
    resume_kept: Optional[Dict[str, int]] = None,
    resume_counter: int = 0,
    done_pairs: Optional[set] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, int], Dict[str, int]]:
    """Generate and panel-verify counterfactual edits to fill the balanced core.

    For each ``supported`` parent sentence (with its per-sentence grounding), each
    still-short family produces one minimal edit; the panel then re-labels the
    edit against the same grounding and the row is kept only if the panel's
    majority label equals the family's intended label (the verification gate).

    Args:
        panel: The evaluator panel (verifier).
        generator: DeepSeek generator (perturber).
        natural_rows: Phase-1 rows; ``supported`` ones are the parents.
        chunks_by_qid: Per-question chunk map for grounding.
        core_floor: Per-class floor driving the family quotas.
        seed: RNG seed for parent ordering.

    Returns:
        Tuple of (core_rows, kept_per_family, gate_stats).
    """
    quotas = _family_quotas(core_floor)
    kept: Counter = Counter(resume_kept or {})  # seed from checkpointed core rows (--resume)
    done_pairs = done_pairs or set()
    gate_stats = {"generated": 0, "gen_failed": 0, "verified": 0, "rejected": 0}
    parents = [
        r for r in natural_rows
        if r["label"] == SUPPORTED and len(r["sentence_text"].split()) >= 5
    ]
    random.Random(seed).shuffle(parents)

    core_rows: List[Dict[str, Any]] = []
    counter = resume_counter
    for parent in parents:
        if all(kept[fam.name] >= quotas[fam.name] for fam in FAMILIES):
            break
        chunks = chunks_by_qid.get(parent["question_id"], [])
        # A perturbation changes one proposition derived from one parent
        # sentence. Validate it against that sentence's single best evidence
        # chunk: adding unrelated chunks increases cost and can create accidental
        # support for an otherwise invalid edit. Natural-row labeling still uses
        # the broader MAX_CONTEXT_CHUNKS_FOR_PANEL setting.
        grounding = (
            select_relevant_chunks(parent["sentence_text"], chunks, max_chunks=1)
            if chunks
            else select_relevant_passage(
                parent["sentence_text"], parent["retrieved_context"]
            )
        )
        parent_ref = f"{parent['question_id']}:{parent['sentence_id']}"
        for family in FAMILIES:
            if kept[family.name] >= quotas[family.name]:
                continue
            if (parent_ref, family.name) in done_pairs:
                continue  # this (parent, family) edit was already verified + checkpointed
            out = await generate_perturbation(generator, parent["sentence_text"], grounding, family)
            gate_stats["generated"] += 1
            if out is None:
                gate_stats["gen_failed"] += 1
                continue
            perturbed, note = out
            counter += 1
            new_qid = f"qp-{family.name}-{counter:04d}"
            # Verification gate: the panel independently labels the edited
            # sentence against the same grounding; keep only if it confirms the
            # intended label. A one-sentence "full response" is passed since the
            # verifier scores this sentence in isolation.
            row, status, unanimous = await label_sentence(
                panel=panel, question=parent["question"], question_id=new_qid,
                full_response=perturbed, sentence=perturbed, sent_idx=0,
                retrieved_context=grounding, chunks=[], session_id=f"verify-{new_qid}",
            )
            if row is None or row["label"] != family.intended_label:
                gate_stats["rejected"] += 1
                continue
            gate_stats["verified"] += 1
            kept[family.name] += 1
            tagged = _tag(
                row,
                construction="perturbed",
                perturbation_family=family.name,
                intended_label=family.intended_label,
                perturbation_of=parent_ref,
                split="core",
                edit_note=note,
            )
            core_rows.append(tagged)
            if writer is not None:
                writer.append(tagged)  # flush per verified edit (--resume)
            logger.info("datagen.subset_d.perturb_kept", family=family.name, kept=kept[family.name], target=quotas[family.name])

    return core_rows, dict(kept), gate_stats


def promote_supported_to_core(
    natural_rows: List[Dict[str, Any]],
    core_rows: List[Dict[str, Any]],
    core_floor: int,
    seed: int,
) -> int:
    """Move natural ``supported`` rows into the core until entailment is balanced.

    The core's entailment target is the larger of its contradiction/neutral counts
    (so the three classes match), floored at ``core_floor``. Promotion flips a
    natural row's ``split`` to ``core`` in place; the far_paraphrase positives
    already in ``core_rows`` count toward the target first.

    Returns the number of natural rows promoted into the core.
    """
    counts = nli_class_counts(core_rows)
    target = max(core_floor, counts.get("contradiction", 0), counts.get("neutral", 0))
    need = max(0, target - counts.get("entailment", 0))
    pool = [r for r in natural_rows if r["label"] == SUPPORTED]
    random.Random(seed + 1).shuffle(pool)
    promoted = 0
    for row in pool:
        if promoted >= need:
            break
        row["split"] = "core"
        promoted += 1
    return promoted


async def phase3_band(nli_client: EvalNLIClient, rows: List[Dict[str, Any]]) -> None:
    """Tag each row's difficulty from how the production NLI verifier fares on it.

    A reported slice only, never an inclusion gate — so the dataset's composition
    stays independent of any one model's blind spots. ``no_source_needed`` rows
    (excluded from Exp3 scoring) get an empty band.
    """
    for row in rows:
        truth_nli = LABEL_TO_NLI.get(row["label"])
        if truth_nli is None:
            row["difficulty_band"] = ""
            continue
        try:
            result = await nli_client.check(premise=row["retrieved_context"], hypothesis=row["sentence_text"])
            score = result.entailment_score if result.label == "entailment" else None
            row["difficulty_band"] = band_by_nli(result.label, truth_nli, score)
        except Exception as e:
            logger.warning("datagen.subset_d.banding_error", error=str(e))
            row["difficulty_band"] = ""


def finalize_principal_rows(
    natural_rows: List[Dict[str, Any]],
    core_rows: List[Dict[str, Any]],
    seed: int,
    natural_target: int = 60,
    calibration_per_class: int = 10,
) -> List[Dict[str, Any]]:
    """Create the fixed 30-calibration/180-locked principal Subset D.

    ``split`` retains the analytical slice (core versus natural), while
    ``evaluation_split`` records threshold-calibration isolation. Calibration
    contains exactly ``calibration_per_class`` rows from each NLI class; the
    natural base-rate slice is locked evaluation only.
    """
    diagnostic = [r for r in natural_rows + core_rows if r.get("split") == "core"]
    expected = {"entailment": 50, "neutral": 50, "contradiction": 50}
    counts = nli_class_counts(diagnostic)
    if counts != expected:
        raise AssertionError(f"diagnostic core must be exactly 50/class, got {counts}")

    natural_pool = [r for r in natural_rows if r.get("split") == "natural"]
    if len(natural_pool) < natural_target:
        raise AssertionError(
            f"need {natural_target} natural rows after core promotion; got {len(natural_pool)}"
        )
    rng = random.Random(seed + 2)
    rng.shuffle(natural_pool)
    selected_natural = natural_pool[:natural_target]

    for row in diagnostic + selected_natural:
        row["evaluation_split"] = "locked_test"
    for offset, label in enumerate(("entailment", "neutral", "contradiction")):
        candidates = [r for r in diagnostic if r["label"] == label]
        random.Random(seed + 100 + offset).shuffle(candidates)
        for row in candidates[:calibration_per_class]:
            row["evaluation_split"] = "calibration"

    principal = diagnostic + selected_natural
    split_counts = Counter(r["evaluation_split"] for r in principal)
    expected_calibration = 3 * calibration_per_class
    if split_counts != {
        "calibration": expected_calibration,
        "locked_test": len(principal) - expected_calibration,
    }:
        raise AssertionError(f"invalid evaluation split: {dict(split_counts)}")
    return principal


async def build_subset_d(
    settings: DatasetGenSettings,
    api_url: str,
    subset_a_path: str,
    subset_c_path: Optional[str],
    output_path: str,
    question_counts: Dict[str, int],
    core_floor: int,
    class_floor: int,
    infinity_url: str,
    nli_model: str,
    seed: int,
    natural_target: int = 60,
    calibration_per_class: int = 10,
    resume: bool = False,
    finalize_only: bool = False,
) -> None:
    """Build the single, hardened Subset D and save to CSV.

    Four phases: a natural backbone, a panel-verified
    counterfactual-perturbation core, adversarial difficulty banding, then
    shortcut/balance assertions before writing one file.
    """
    if not settings.openrouter_api_key:
        logger.error("datagen.subset_d.missing_api_key")
        sys.exit(1)

    generator = DatasetGenerator(settings)
    panel = EvaluatorPanel(settings)
    blind_tracker = BlindInjectionTracker()
    status_counts: Dict[str, int] = {"accepted": 0, "contested": 0, "fragment": 0, "no_signal": 0, "error": 0}

    # Fail-safe checkpoints: the panel spend (5 models × ~hundreds of sentences)
    # is written per-row as it is accepted, so an abort/crash/restart never
    # re-pays for it — --resume skips whatever is already on disk.
    natural_part = output_path.replace(".csv", ".natural.part.csv")
    core_part = output_path.replace(".csv", ".core.part.csv")
    loaded_natural = resume_rows(natural_part, FIELDNAMES) if resume else []
    loaded_core = resume_rows(core_part, FIELDNAMES) if resume else []
    done_qids = {r["question_id"] for r in loaded_natural}
    resume_kept, done_pairs, resume_counter = core_resume_state(loaded_core)
    if resume and (loaded_natural or loaded_core):
        logger.info("datagen.subset_d.resume", natural=len(loaded_natural), core=len(loaded_core))

    questions_by_source: List[Tuple[str, str]] = []
    try:
        if finalize_only:
            if not loaded_natural or not loaded_core:
                raise ValueError("--finalize-only requires non-empty resumed part files")
            natural_rows = loaded_natural
            core_rows = loaded_core
            kept_per_family = resume_kept
            gate_stats = {"finalize_only": 1}
        else:
            questions_by_source = await assemble_questions(
                generator, api_url, subset_a_path, subset_c_path, question_counts, seed
            )
            if not questions_by_source:
                logger.error("datagen.subset_d.no_questions")
                sys.exit(1)
            logger.info("datagen.subset_d.questions_assembled", count=len(questions_by_source))

            with IncrementalCSVWriter(natural_part, FIELDNAMES, resume=resume) as nat_writer:
                new_natural, chunks_by_qid = await phase1_natural(
                    panel, api_url, questions_by_source, blind_tracker, status_counts,
                    writer=nat_writer, done_qids=done_qids,
                )
            natural_rows = loaded_natural + new_natural

            with IncrementalCSVWriter(core_part, FIELDNAMES, resume=resume) as core_writer:
                new_core, kept_per_family, gate_stats = await phase2_perturb(
                    panel, generator, natural_rows, chunks_by_qid, core_floor, seed,
                    writer=core_writer, resume_kept=resume_kept, resume_counter=resume_counter,
                    done_pairs=done_pairs,
                )
            core_rows = loaded_core + new_core
        promoted = promote_supported_to_core(natural_rows, core_rows, core_floor, seed)
    finally:
        await generator.aclose()
        await panel.aclose()

    all_rows = finalize_principal_rows(
        natural_rows, core_rows, seed, natural_target, calibration_per_class
    )

    # Difficulty is reported by construction family. It is deliberately not
    # defined by the NLI model evaluated in Experiment 3.
    for row in all_rows:
        row["difficulty_band"] = ""

    # Phase 4: assertions BEFORE writing — a shortcut or a degenerate core must
    # fail the build loudly rather than ship a quietly-broken dataset. Both guards
    # target the balanced ``core`` slice: that is where the counterfactual edits are
    # minimal (so negatives share the positives' overlap band) and where macro-F1 is
    # meaningful. The ``natural`` backbone is by design the realistic, supported-heavy
    # base-rate slice and would trip a surface-feature check for that very
    # reason, so its cross-tab is reported for transparency but not asserted on.
    core_slice = [r for r in all_rows if r.get("split") == "core"]
    shortcut = assert_no_shortcut_features(core_slice)
    shortcut_all = shortcut_report(all_rows)
    core_class_counts = assert_class_balance(core_slice, floor=class_floor)

    blind_tracker.write_sidecar(
        output_path.replace(".csv", "_blind_injection.csv"),
        fieldnames=BASE_FIELDS,
    )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)

    write_provenance(
        output_path,
        subset="d",
        settings=settings,
        row_count=len(all_rows),
        extra={
            "source_subset_a": subset_a_path,
            "source_subset_c": subset_c_path,
            "questions_processed": len(questions_by_source),
            "processing_status_counts": dict(status_counts),
            # The headline sanity check: this must show all three NLI classes
            # populated on the core (a supported-heavy build with zero
            # partially_supported is the defect the perturbation core exists to
            # fix).
            "label_distribution": label_distribution(all_rows),
            "by_construction": dict(Counter(r.get("construction", "") for r in all_rows)),
            "by_split": dict(Counter(r.get("split", "") for r in all_rows)),
            "by_evaluation_split": dict(
                Counter(r.get("evaluation_split", "") for r in all_rows)
            ),
            "by_difficulty_band": dict(Counter(r.get("difficulty_band", "") for r in all_rows)),
            "perturbation_kept_by_family": kept_per_family,
            "perturbation_gate_stats": gate_stats,
            "naturals_promoted_to_core": promoted,
            "core_nli_class_counts": core_class_counts,
            # Asserted on the balanced core; the full-set cross-tab (natural backbone
            # included) is reported unasserted so the base-rate slice's surface skew
            # stays visible.
            "shortcut_report": shortcut,
            "shortcut_report_all_rows": shortcut_all,
        },
    )

    logger.info(
        "datagen.subset_d.complete",
        sentences_written=len(all_rows),
        core=len(core_slice),
        perturbed=len(core_rows),
        promoted=promoted,
        core_nli=core_class_counts,
        output=output_path,
    )


def main() -> None:
    """Entry point for the hardened Subset D generation."""
    parser = argparse.ArgumentParser(
        description="Build the single, hardened Subset D (RAM ground truth): natural "
        "backbone + panel-verified counterfactual-perturbation core + difficulty banding."
    )
    parser.add_argument("--api-url", default="http://localhost:8000", help="Base URL of the running application")
    parser.add_argument("--subset-a", default="evals/data/subset_a.csv", help="Path to Subset A CSV")
    parser.add_argument("--subset-c", default="evals/data/subset_c.csv", help="Path to Subset C CSV (boundary queries)")
    parser.add_argument("--output", default="evals/data/subset_d.csv", help="Output CSV path")
    parser.add_argument("--count-reweighted", type=int, default=20, help="Reweighted Subset A questions (phase 1)")
    parser.add_argument("--count-detail", type=int, default=10, help="Precise-detail questions (phase 1)")
    parser.add_argument("--count-crossdoc", type=int, default=8, help="Cross-document questions (phase 1)")
    parser.add_argument("--count-evaluative", type=int, default=12, help="Evaluative/rationale questions (phase 1)")
    parser.add_argument("--count-boundary", type=int, default=6, help="Boundary queries injected from Subset C (phase 1)")
    parser.add_argument(
        "--core-floor", type=int, default=MIN_CLASS_ROWS,
        help="Per-NLI-class target driving perturbation quotas + entailment balance",
    )
    parser.add_argument(
        "--natural-rows", type=int, default=60,
        help="Natural base-rate rows retained in locked evaluation",
    )
    parser.add_argument(
        "--calibration-per-class", type=int, default=10,
        help="Diagnostic rows per NLI class reserved for calibration",
    )
    parser.add_argument(
        "--class-floor", type=int, default=MIN_CLASS_ROWS,
        help="Floor the balance assertion enforces on the core slice (build fails below it)",
    )
    parser.add_argument("--infinity-url", default="http://localhost:7997", help="Infinity base URL (NLI banding)")
    parser.add_argument(
        "--nli-model", default="StevenLimcorn/indo-roberta-indonli",
        help="NLI model identifier for difficulty banding (must be loaded on Infinity)",
    )
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for reproducibility")
    parser.add_argument(
        "--finalize-only",
        action="store_true",
        help="Assemble and validate existing resumed part files without network calls",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an interrupted build: reuse the panel-labeled rows already written to "
        "<output>.natural.part.csv / <output>.core.part.csv (skipping questions/perturbations "
        "already paid for) instead of restarting from zero. Safe to pass on the first run too "
        "(no checkpoint yet = fresh build).",
    )
    args = parser.parse_args()

    question_counts = {
        "reweighted": args.count_reweighted,
        "detail": args.count_detail,
        "crossdoc": args.count_crossdoc,
        "evaluative": args.count_evaluative,
        "boundary": args.count_boundary,
    }

    settings = get_dataset_gen_settings()
    asyncio.run(
        build_subset_d(
            settings=settings,
            api_url=args.api_url,
            subset_a_path=args.subset_a,
            subset_c_path=args.subset_c,
            output_path=args.output,
            question_counts=question_counts,
            core_floor=args.core_floor,
            class_floor=args.class_floor,
            infinity_url=args.infinity_url,
            nli_model=args.nli_model,
            seed=args.seed,
            natural_target=args.natural_rows,
            calibration_per_class=args.calibration_per_class,
            resume=args.resume,
            finalize_only=args.finalize_only,
        )
    )


if __name__ == "__main__":
    main()
