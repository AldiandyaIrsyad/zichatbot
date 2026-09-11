"""History-aware query condensation for retrieval.

A follow-up turn is usually elliptical — "kenapa begitu?", "kalau untuk S2?",
"pasal berapa?" — and carries almost no retrievable signal on its own. The chat
pipeline previously sent that raw text to both the KB search and the IVM
relevance gate, so a perfectly reasonable follow-up would retrieve poorly and
then get abstained on as out-of-domain.

This module rewrites such a turn into a standalone question using the recent
conversation, before it reaches retrieval. It sits *before* HyDE: condensation
resolves what the user is asking, HyDE (inside ``SearchService``) then imagines
a document answering it.

Mirrors the ``app/thesis/ivm/judge.py::LLMJudge`` shape — a narrow
``stream_chat``-only dependency plus a configurable prompt/template — so it can
be swapped or disabled from config without touching the pipeline.
"""

from typing import Optional

import structlog

from app.chat.domain.interfaces import ILLMConnection

logger = structlog.get_logger(__name__)

DEFAULT_CONDENSER_PROMPT = (
    "Anda adalah penulis ulang pertanyaan untuk sistem pencarian dokumen. "
    "Diberikan riwayat percakapan dan pertanyaan lanjutan, tulis ulang "
    "pertanyaan lanjutan tersebut menjadi satu pertanyaan yang berdiri "
    "sendiri dalam Bahasa Indonesia, dengan mengganti kata ganti dan rujukan "
    "implisit ('itu', 'tersebut', 'begitu') dengan entitas yang dimaksud dari "
    "riwayat.\n"
    "Aturan:\n"
    "- Jika pertanyaan sudah berdiri sendiri, salin persis tanpa perubahan.\n"
    "- Jangan menjawab pertanyaan.\n"
    "- Jangan menambahkan informasi yang tidak ada di riwayat.\n"
    "- Keluarkan HANYA pertanyaan hasil penulisan ulang, tanpa penjelasan."
)

DEFAULT_CONDENSER_USER_TEMPLATE = (
    "Riwayat percakapan:\n{history}\n\n"
    "Pertanyaan lanjutan: {question}\n\n"
    "Pertanyaan yang berdiri sendiri:"
)

# Generous enough to absorb a reasoning model's preamble before the rewritten
# question, for the same reason LLMJudge uses 400 (see its max_tokens comment).
_MAX_TOKENS = 400


class QueryCondenser:
    """Rewrites a follow-up question into a standalone retrieval query."""

    def __init__(
        self,
        llm_connection: ILLMConnection,
        model: str,
        system_prompt: Optional[str] = None,
        user_template: Optional[str] = None,
    ) -> None:
        """``user_template`` must contain ``{history}`` and ``{question}``."""
        self._llm = llm_connection
        self._model = model
        self._system_prompt = system_prompt or DEFAULT_CONDENSER_PROMPT
        self._user_template = user_template or DEFAULT_CONDENSER_USER_TEMPLATE

    async def condense(self, history_text: str, question: str) -> str:
        """Return a standalone version of ``question``, or ``question`` itself.

        Fails **open**: on any error, an empty result, or an implausible rewrite
        the raw question is returned unchanged. Condensation is a retrieval aid,
        so it must never become a new source of refusals or dropped turns.
        """
        if not history_text.strip() or not question.strip():
            return question

        user_content = self._user_template.replace("{history}", history_text).replace(
            "{question}", question
        )
        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": user_content},
        ]

        try:
            chunks = []
            async for chunk in self._llm.stream_chat(
                model=self._model,
                messages=messages,
                max_tokens=_MAX_TOKENS,
                temperature=0.0,
                call="condenser",
            ):
                chunks.append(chunk)
            rewritten = "".join(chunks).strip()
        except Exception as exc:
            logger.warning("chat.condense.failed", error=str(exc), exc_info=True)
            return question

        # A reasoning model may emit a preamble before the question; keep the
        # last non-empty line, which is where the answer lands.
        lines = [ln.strip() for ln in rewritten.splitlines() if ln.strip()]
        rewritten = lines[-1] if lines else ""
        rewritten = rewritten.strip().strip('"').strip()

        # Guard against degenerate rewrites (empty, or a runaway explanation
        # that would poison retrieval more than the original ever could).
        if not rewritten or len(rewritten) > max(400, len(question) * 8):
            logger.info("chat.condense.rejected", original=question[:100])
            return question

        if rewritten != question:
            logger.info(
                "chat.condense.rewritten",
                original=question[:100],
                rewritten=rewritten[:100],
            )
        return rewritten
