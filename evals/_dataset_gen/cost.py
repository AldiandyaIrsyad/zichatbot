"""OpenRouter usage accounting with a hard, checkpoint-friendly budget."""

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


class BudgetExceededError(RuntimeError):
    """Raised when recorded OpenRouter spend reaches the configured ceiling."""


@dataclass(frozen=True)
class UsageRecord:
    timestamp_utc: str
    phase: str
    model: str
    provider: str
    generation_id: str
    prompt_tokens: int
    completion_tokens: int
    reasoning_tokens: int
    cost_usd: Optional[float]


def usage_from_response(data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract normalized usage fields from an OpenRouter response."""
    usage = data.get("usage") or {}
    completion_details = usage.get("completion_tokens_details") or {}
    cost = usage.get("cost")
    try:
        normalized_cost: Optional[float] = float(cost) if cost is not None else None
    except (TypeError, ValueError):
        normalized_cost = None
    return {
        "provider": str(data.get("provider") or ""),
        "generation_id": str(data.get("id") or ""),
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "completion_tokens": int(usage.get("completion_tokens") or 0),
        "reasoning_tokens": int(completion_details.get("reasoning_tokens") or 0),
        "cost_usd": normalized_cost,
    }


class CostLedger:
    """Append-only billed-cost ledger shared by generator and panel clients."""

    def __init__(self, path: str, limit_usd: float) -> None:
        self.path = Path(path)
        self.limit_usd = limit_usd
        self._lock = asyncio.Lock()
        self._known_cost_usd = self._load_known_cost()

    def _load_known_cost(self) -> float:
        if not self.path.exists():
            return 0.0
        total = 0.0
        for line in self.path.read_text(encoding="utf-8").splitlines():
            try:
                value = json.loads(line).get("cost_usd")
                if value is not None:
                    total += float(value)
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
        return total

    @property
    def known_cost_usd(self) -> float:
        return self._known_cost_usd

    async def assert_available(self) -> None:
        if self.limit_usd > 0 and self._known_cost_usd >= self.limit_usd:
            raise BudgetExceededError(
                f"OpenRouter budget exhausted: ${self._known_cost_usd:.4f} "
                f"recorded against ${self.limit_usd:.2f} limit"
            )

    async def record(
        self,
        *,
        phase: str,
        model: str,
        response_data: Dict[str, Any],
    ) -> UsageRecord:
        normalized = usage_from_response(response_data)
        record = UsageRecord(
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            phase=phase,
            model=model,
            **normalized,
        )
        async with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
            if record.cost_usd is not None:
                self._known_cost_usd += record.cost_usd
            if self.limit_usd > 0 and self._known_cost_usd >= self.limit_usd:
                raise BudgetExceededError(
                    "OpenRouter budget reached after response: "
                    f"${self._known_cost_usd:.4f} recorded against "
                    f"${self.limit_usd:.2f} limit; checkpoint before resuming"
                )
        return record

_LEDGERS: Dict[str, CostLedger] = {}

def get_cost_ledger(path: str, limit_usd: float) -> Optional[CostLedger]:
    """Return one in-process ledger per path; blank paths disable accounting."""
    if not path:
        return None
    ledger = _LEDGERS.get(path)
    if ledger is None:
        ledger = CostLedger(path, limit_usd)
        _LEDGERS[path] = ledger
    elif limit_usd > 0:
        ledger.limit_usd = min(ledger.limit_usd, limit_usd) if ledger.limit_usd > 0 else limit_usd
    return ledger
