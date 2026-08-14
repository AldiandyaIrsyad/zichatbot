"""
Orchestrator for the Chat Module.

Coordinates the Knowledge Base, IVM (Safety/Relevance), RAM (Response Assessment),
and the LLM inference engine.
"""

import json
import re
import uuid
import structlog
from typing import AsyncGenerator, List, Dict, Any, Optional, Tuple

from app.chat.application.history import build_history, format_history_for_condenser
from app.chat.application.query_condenser import QueryCondenser
from app.chat.domain.interfaces import IChatRepository, ILLMConnection
from app.kb.application.search_service import SearchService
from app.guardrails.ivm.service import IVMService, MaliciousPromptException
from app.guardrails.ivm.relevance_service import RelevanceService, IrrelevantQueryException
from app.rag.prompts import build_prompt
from app.guardrails.ram.service import RAMService
from app.guardrails.ram.interfaces import RetrievedContext as RAMRetrievedContext
from app.guardrails.ram.text_utils import split_sentences_with_seps

logger = structlog.get_logger(__name__)

# A Markdown table row (header, separator, or data row) starts and ends with
# "|". Keeps citation markers off table rows — appending text after a row's
# closing "|" would corrupt GFM table syntax.
_TABLE_ROW_RE = re.compile(r'^\s*\|.*\|\s*$')

# Minimum proposition length (chars) worth an NLI call.
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
        temperature: float = 0.0,
        attachment_search_excerpt_chars: int = 4000,
        history_max_tokens: int = 3000,
        query_condenser: Optional[QueryCondenser] = None,
        refusal_message: str = "Maaf, pertanyaan ini tidak tercakup dalam dokumen yang tersedia.",
        safety_block_message: str = "Maaf, permintaan ini diblokir oleh filter keamanan.",
        internal_error_message: str = "Maaf, terjadi kesalahan saat memproses permintaan Anda.",
    ):
        """Wire the pipeline's collaborators and generation settings.

        ``attachment_search_excerpt_chars`` caps how much of an uploaded
        attachment's text is folded into the KB search query (the full text
        still goes to the LLM prompt). ``history_max_tokens`` budgets the
        replayed conversation history. ``query_condenser`` rewrites elliptical
        follow-ups for retrieval; None disables condensation. The three message
        strings are the user-facing refusal/error texts.
        """
        self.chat_repo = chat_repo
        self.llm_conn = llm_conn
        self.search_service = search_service
        self.ivm_service = ivm_service
        self.relevance_service = relevance_service
        self.ram_service = ram_service
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.attachment_search_excerpt_chars = attachment_search_excerpt_chars
        self.history_max_tokens = history_max_tokens
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
        """Format an NLI result into a canonical citation marker.

        Produces ``*(STATUS: SCORE; SOURCE; Page N; DocID:ID; Evidence:"snippet")*``
        where STATUS is Supported/Contradiction and SCORE the dominant NLI
        confidence; SOURCE, Page, DocID, and Evidence appear only when present
        (DocID enables a frontend "Download PDF" link). Parsed by the frontend
        ``renderMessage`` to render citation badges with tooltips.

        Returns "" for None or non-Supported/Contradiction labels (neutral and
        error-fallback results carry nothing useful to show).
        """
        if result is None:
            return ""

        label_map = {
            "entailment": "Supported",
            "contradiction": "Contradiction",
        }
        status = label_map.get(result.label)
        if status is None:
            return ""

        # Pick the dominant score for the predicted label
        score_map = {
            "entailment": result.entailment_score,
            "contradiction": result.contradiction_score,
        }
        score = score_map[result.label]

        parts = [f"{status}: {score:.2f}"]
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
    def _split_propositions(text: str) -> List[Tuple[str, str]]:
        """Heuristically split a buffer into (proposition, trailing_separator) pairs.

        Splits on standard sentence boundaries and major Indonesian conjunctions
        to evaluate smaller facts independently. The trailing separator is the
        whitespace that followed the proposition in the source text (e.g.
        "\\n\\n" for a paragraph break), so callers can preserve the original
        formatting instead of always rejoining with a single space.
        """
        # Sentence boundaries (., ?, !) and newlines, via the shared splitter
        # (guards against markdown list markers like "1." being read as
        # sentence ends, and reports the separator at each boundary).
        sentence_pairs = split_sentences_with_seps(text)

        propositions: List[Tuple[str, str]] = []
        for part, sep in sentence_pairs:
            # Further split on Indonesian conjunctions that introduce new
            # claims, keeping the delimiter attached to the following part.
            sub_parts = re.split(r'(?i)(,\s*yang\s+|,\s*dan\s+|,\s*karena\s+|,\s*sehingga\s+)', part)

            subs = []
            current_prop = ""
            for sp in sub_parts:
                if re.match(r'(?i)(,\s*yang\s+|,\s*dan\s+|,\s*karena\s+|,\s*sehingga\s+)', sp):
                    # Delimiter: start a new proposition with it.
                    if current_prop.strip():
                        subs.append(current_prop.strip())
                    current_prop = sp.lstrip(", ")  # drop leading comma
                else:
                    current_prop += sp

            if current_prop.strip():
                subs.append(current_prop.strip())

            # Conjunction splits are always mid-line, so only the last
            # sub-proposition of a sentence carries the sentence's real
            # trailing separator (e.g. a paragraph break); earlier ones
            # just get a plain space.
            for i, sub in enumerate(subs):
                propositions.append((sub, sep if i == len(subs) - 1 else " "))

        return propositions

    @staticmethod
    def _is_table_row(prop_text: str) -> bool:
        """True if a proposition is a single Markdown table line (starts/ends
        with '|'). Keeps citation markers off row lines, which would corrupt
        GFM table syntax. A bare-pipe sentence like "|x| = 5" false-positives,
        acceptable in this domain.
        """
        return bool(_TABLE_ROW_RE.match(prop_text.strip()))

    async def _assess_and_format(
        self,
        text: str,
        premise: str,
        ram_contexts: List[RAMRetrievedContext],
        skip_ram: bool,
    ) -> str:
        """Run RAM assessment (unless baseline) and format a citation marker
        (possibly ""). Shared by the prose and table-block paths.
        """
        if skip_ram:
            return ""
        result = await self.ram_service.assess_sentence(text, premise, ram_contexts)
        return self._format_citation(result)

    async def _handle_complete_proposition(
        self,
        prop_text: str,
        sep: str,
        *,
        ram_contexts: List[RAMRetrievedContext],
        premise: str,
        skip_ram: bool,
        table_rows: List[Tuple[str, str]],
    ) -> AsyncGenerator[str, None]:
        """Process one complete (proposition, sep) pair, yielding output chunks
        in stream order.

        Table rows are accumulated in the caller-owned ``table_rows`` list
        rather than assessed individually — a lone row has no headers, and a
        marker on a row line would corrupt GFM syntax. When a non-row
        proposition ends the block, the rows are assessed as one unit and the
        citation is emitted as its own paragraph.
        """
        if self._is_table_row(prop_text):
            table_rows.append((prop_text, sep))
            yield prop_text + sep
            return

        if table_rows:
            table_text = "\n".join(row for row, _ in table_rows)
            table_rows.clear()
            citation = await self._assess_and_format(table_text, premise, ram_contexts, skip_ram)
            if citation:
                yield "\n" + citation.strip() + "\n\n"

        if not prop_text.endswith((".", "?", "!")):
            prop_text += "."

        if len(prop_text) > MIN_ASSESSABLE_LENGTH:
            citation = await self._assess_and_format(prop_text, premise, ram_contexts, skip_ram)
            prop_text += citation

        yield prop_text + sep

    async def process_chat_message(
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
            # and filtering on it would return zero results.
            precheck_contexts = await self.search_service.search(search_query_text, top_k=3)
            if not skip_ivm:
                if not precheck_contexts:
                    raise IrrelevantQueryException("No relevant contexts found in the knowledge base.")

                context_chunks = [ctx.text for ctx in precheck_contexts]
                context_scores = [ctx.score for ctx in precheck_contexts]
                await self.relevance_service.check_relevance(search_query_text, context_chunks, context_scores)

            # 4. Deep Context Retrieval (KB)
            full_contexts = await self.search_service.search(search_query_text, top_k=15)
            
            # Map KB contexts to RAM contexts (preserve breadcrumbs + hierarchy)
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
                )
                for ctx in full_contexts
            ]

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

            premise = self.ram_service.build_premise(ram_contexts)

            # Emit the retrieved context as one NDJSON event before streaming,
            # for the frontend's collapsible "view RAG context" panel and for
            # downstream consumers; the same payload is persisted with the
            # assistant message so it survives a refresh.
            #
            # "content" is a flat join of chunk texts and MUST keep this shape
            # (build_subset_d.py reads it verbatim as RAM ground truth);
            # "chunks" is the per-source structure the chat UI renders.
            context_payload = {
                "content": "\n\n".join(ctx.text for ctx in ram_contexts),
                "chunks": [
                    {
                        "title": ctx.source_title,
                        "page": ctx.page,
                        "breadcrumbs": ctx.breadcrumbs,
                        "text": ctx.text,
                    }
                    for ctx in ram_contexts
                ],
            }
            yield json.dumps({"type": "context", **context_payload}) + "\n"

            # Buffer the stream by sentence to assess each complete proposition.
            buffer = ""
            final_output = ""
            # Accumulates contiguous table-row propositions until a non-row
            # proposition ends the block (see _handle_complete_proposition).
            # Local to this call — never instance state, since ChatService may
            # be reused across concurrent requests.
            table_rows: List[Tuple[str, str]] = []

            # 6. Stream and Assess
            stream = self.llm_conn.stream_chat(
                model=self.model_name,
                messages=messages,
                max_tokens=1024,
                temperature=self.temperature,
            )

            async for chunk in stream:
                buffer += chunk
                # On a likely proposition boundary, flush all but the trailing
                # incomplete fragment.
                if any(punct in chunk for punct in [". ", "? ", "! ", "\n", ", yang ", ", dan ", ", karena ", ", sehingga "]):
                    propositions = self._split_propositions(buffer)
                    if len(propositions) > 1:
                        for prop, sep in propositions[:-1]:
                            async for out in self._handle_complete_proposition(
                                prop, sep, ram_contexts=ram_contexts, premise=premise,
                                skip_ram=skip_ram, table_rows=table_rows,
                            ):
                                final_output += out
                                yield json.dumps({"type": "chunk", "content": out}) + "\n"

                        buffer = propositions[-1][0]

            # Flush any remaining buffer through the same row/prose dispatch.
            if buffer.strip():
                for prop, sep in self._split_propositions(buffer.strip()):
                    async for out in self._handle_complete_proposition(
                        prop, sep, ram_contexts=ram_contexts, premise=premise,
                        skip_ram=skip_ram, table_rows=table_rows,
                    ):
                        final_output += out
                        yield json.dumps({"type": "chunk", "content": out}) + "\n"

            # If the answer ended inside a table (last content was rows, so no
            # trailing prose proposition triggered the flush), assess and flush
            # the accumulated block now.
            if table_rows:
                table_text = "\n".join(row for row, _ in table_rows)
                table_rows.clear()
                citation = await self._assess_and_format(table_text, premise, ram_contexts, skip_ram)
                if citation:
                    out = "\n" + citation.strip() + "\n\n"
                    final_output += out
                    yield json.dumps({"type": "chunk", "content": out}) + "\n"

            # Persist the assistant message with its RAG context/sources so the
            # frontend can restore the "view RAG context" panel after a refresh.
            await self.chat_repo.create_message(
                session_id, "assistant", final_output, raw_content=final_output,
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
