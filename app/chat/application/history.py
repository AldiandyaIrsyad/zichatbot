"""Conversation-history assembly for the chat pipeline.

Turns persisted :class:`app.chat.domain.models.Message` rows into the
``{"role", "content"}`` dicts ``ChatService`` splices between the system prompt
and the current user turn.

Replaces a hardcoded ``session.messages[-10:]`` slice. A message-count slice
has three problems this module fixes:

* it is not a context budget — ten short turns and ten long ones cost wildly
  different numbers of tokens, and overflow was silent;
* it can start the replayed history on an assistant turn, leaving the model
  with a dangling reply whose question it cannot see;
* it replayed ``Message.content``, which for assistant rows carries the inline
  RAM citation markers (``*(Supported: 0.92; …)*``). Those markers are a
  rendering concern, they cost tokens, and feeding them back teaches the model
  to imitate a format the system prompt forbids. ``raw_content`` holds the
  clean pre-citation text for exactly this purpose.

Trimming itself is delegated to ``langchain_core.messages.trim_messages``
rather than hand-rolled: it already implements the budget walk plus the
``start_on``/``include_system`` invariants above. ``langchain_core`` is already
in the environment (it arrives with ``langchain-text-splitters``, used by the
KB context), so this adds no new dependency. The heavier ``langchain`` /
``langgraph`` packages are deliberately not used.
"""

from typing import Any, Dict, Iterable, List, Sequence

import structlog
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, trim_messages

logger = structlog.get_logger(__name__)

# Average characters per token for Indonesian text under a Qwen-family
# tokenizer. The chat model is served remotely (OpenRouter), so there is no
# local tokenizer to consult, and tiktoken's BPE is both wrong for Qwen and a
# network fetch on first use. A budget only needs to be approximately right,
# and this errs low (i.e. over-counts tokens) so the estimate stays safe.
_CHARS_PER_TOKEN = 3.0

# Rough per-message cost of the chat template's role/delimiter scaffolding.
_PER_MESSAGE_OVERHEAD_TOKENS = 4


def approx_token_count(text: str) -> int:
    """Estimate the token cost of ``text``. See ``_CHARS_PER_TOKEN``."""
    return int(len(text) / _CHARS_PER_TOKEN) + 1


def approx_token_counter(messages: Iterable[BaseMessage]) -> int:
    """``trim_messages``-compatible counter over LangChain message objects."""
    total = 0
    for msg in messages:
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        total += approx_token_count(content) + _PER_MESSAGE_OVERHEAD_TOKENS
    return total


def to_lc_messages(messages: Sequence[Any]) -> List[BaseMessage]:
    """Convert persisted ``Message`` rows to LangChain message objects.

    Prefers ``raw_content`` (clean text) and falls back to ``content``: the
    column is nullable and rows written before it existed have it NULL.
    Rows with neither are skipped, as are roles other than user/assistant —
    the system prompt is supplied fresh each turn, never replayed from the DB.
    """
    converted: List[BaseMessage] = []
    for msg in messages:
        text = getattr(msg, "raw_content", None) or getattr(msg, "content", None)
        if not text:
            continue
        if msg.role == "user":
            converted.append(HumanMessage(content=text))
        elif msg.role == "assistant":
            converted.append(AIMessage(content=text))
    return converted


def build_history(messages: Sequence[Any], max_tokens: int) -> List[Dict[str, str]]:
    """Build the trimmed history to replay, as ``{"role", "content"}`` dicts.

    Keeps the most recent turns that fit in ``max_tokens`` and guarantees the
    result begins on a user turn. ``max_tokens`` covers the replayed history
    only — the system prompt, retrieved context, and current user turn are
    budgeted by the caller.

    Never raises: history is an enhancement, so a trimming failure degrades to
    "no history" rather than failing the turn.
    """
    if not messages or max_tokens <= 0:
        return []

    lc_messages = to_lc_messages(messages)
    if not lc_messages:
        return []

    try:
        trimmed = trim_messages(
            lc_messages,
            max_tokens=max_tokens,
            strategy="last",
            token_counter=approx_token_counter,
            # The replayed block must open on a user turn; an assistant message
            # whose question was trimmed away is confusing context.
            start_on="human",
            # The system prompt is prepended separately by the caller.
            include_system=False,
            # Keep whole messages: a truncated turn is worse than a dropped one.
            allow_partial=False,
        )
    except Exception as exc:
        logger.warning("chat.history.trim_failed", error=str(exc), exc_info=True)
        return []

    return [
        {
            "role": "user" if isinstance(m, HumanMessage) else "assistant",
            "content": m.content if isinstance(m.content, str) else str(m.content),
        }
        for m in trimmed
    ]


def format_history_for_condenser(history: Sequence[Dict[str, str]], max_turns: int = 6) -> str:
    """Render recent history as a plain transcript for the query condenser.

    Only the last ``max_turns`` messages are included — rewriting a follow-up
    needs the immediate antecedent, not the whole conversation.
    """
    recent = list(history)[-max_turns:]
    label = {"user": "Pengguna", "assistant": "Asisten"}
    return "\n".join(f"{label.get(m['role'], m['role'])}: {m['content']}" for m in recent)
