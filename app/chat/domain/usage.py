"""Token/cost accounting for LLM calls, and a context-scoped collector.

One chat turn fans out into several LLM calls owned by different objects —
HyDE inside ``SearchService``, the condenser inside ``QueryCondenser``,
generation inside ``ChatService``. Threading a callback through all three
would put cost plumbing in every signature between them.

Instead the adapter reports each call into a :class:`UsageCollector` bound to
the current async context, so a caller can wrap a turn and get the total
without any intermediate layer knowing that cost is being measured.
``contextvars`` propagate into tasks spawned by ``asyncio.gather`` (as HyDE
does for its passage ensemble), and because the collector is mutable, records
appended inside those child tasks are visible to the parent.
"""

from __future__ import annotations

import contextvars
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable, Iterator, List, Optional


@dataclass(frozen=True)
class LLMUsage:
    """Token accounting for one LLM call.

    ``cost_usd`` and ``provider`` are only populated by gateways that report
    them (OpenRouter does, when asked); a plain OpenAI-compatible backend
    leaves them None and the caller prices the tokens itself.

    ``cached_prompt_tokens`` is the slice of the prompt served from the
    provider's prefix cache — the only reliable way to tell whether prompt
    caching is actually firing, as opposed to being merely available.
    """

    model: str
    prompt_tokens: int
    completion_tokens: int
    cached_prompt_tokens: int = 0
    cost_usd: Optional[float] = None
    provider: Optional[str] = None
    generation_id: Optional[str] = None
    call: str = "generation"

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


# Invoked once per LLM call, after the response completes. Kept a plain
# callback rather than a return value so ``stream_chat`` stays an
# ``AsyncIterator[str]`` for every existing caller.
UsageCallback = Callable[[LLMUsage], None]


@dataclass
class UsageCollector:
    """Accumulates the LLM usage of one logical unit of work (a chat turn)."""

    records: List[LLMUsage] = field(default_factory=list)

    def add(self, usage: LLMUsage) -> None:
        self.records.append(usage)

    @property
    def prompt_tokens(self) -> int:
        return sum(r.prompt_tokens for r in self.records)

    @property
    def completion_tokens(self) -> int:
        return sum(r.completion_tokens for r in self.records)

    @property
    def cached_prompt_tokens(self) -> int:
        return sum(r.cached_prompt_tokens for r in self.records)

    @property
    def cost_usd(self) -> Optional[float]:
        """Total reported cost, or None when no call reported one."""
        costs = [r.cost_usd for r in self.records if r.cost_usd is not None]
        return sum(costs) if costs else None

    @property
    def call_count(self) -> int:
        return len(self.records)


_active: contextvars.ContextVar[Optional[UsageCollector]] = contextvars.ContextVar(
    "chat_usage_collector", default=None
)


@contextmanager
def collect_usage() -> Iterator[UsageCollector]:
    """Collect the usage of every LLM call made inside this block."""
    collector = UsageCollector()
    token = _active.set(collector)
    try:
        yield collector
    finally:
        _active.reset(token)


def record_usage(usage: LLMUsage) -> None:
    """Report one call to the active collector, if a caller installed one."""
    collector = _active.get()
    if collector is not None:
        collector.add(usage)
