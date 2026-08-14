"""Repeated-run self-agreement for non-deterministic evaluators.

A hosted LLM is not bit-reproducible even at temperature 0, so a single pass
gives a number with no stated stability. Repeating an identical run converts
that from an unknown into a measured quantity: if the label is stable the point
estimate is quotable, and if it is not, the instability is itself the finding.

Shared by Exp1a's prompting baseline and Exp1b's LLM-judge, which both classify
with a hosted model and both need the same agreement reporting.
"""

from __future__ import annotations

from typing import Awaitable, Callable, List, Sequence, Tuple, TypeVar

T = TypeVar("T")


async def repeat_passes(
    run_once: Callable[[], Awaitable[List[T]]],
    repeats: int,
    label: str = "pass",
) -> List[List[T]]:
    """Run one classification pass ``repeats`` times over identical input.

    ``run_once`` is an async callable returning one pass's per-row results;
    ``label`` is the human-readable name printed with each pass number.
    Returns one result list per pass, in order.
    """
    runs: List[List[T]] = []
    for attempt in range(repeats):
        print(f"\n  {label} {attempt + 1}/{repeats}...")
        runs.append(await run_once())
    return runs


def self_agreement(runs: Sequence[Sequence[T]]) -> Tuple[float, List[int]]:
    """Measure how often the passes agree row by row.

    ``runs`` is one result list per pass, all the same length. Returns (share
    of rows labelled identically in every pass, per-row count of distinct
    labels — 1 means stable, >1 means the row flipped).
    """
    if not runs:
        return 0.0, []
    n = len(runs[0])
    distinct_per_row = [len({run[i] for run in runs}) for i in range(n)]
    unanimous = sum(1 for count in distinct_per_row if count == 1)
    agreement = unanimous / n if n else 0.0
    return agreement, distinct_per_row
