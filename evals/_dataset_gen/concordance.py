"""Human-Machine Concordance Rate helper.

    - Inject 5/5-consensus items blindly into the researcher's verification queue
    - Compute concordance rate (target ≥95%)
    - If concordance < 95%, regenerate the affected subset
"""

from __future__ import annotations

import csv
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

import structlog

logger = structlog.get_logger(__name__)

# Fixed seed for reproducibility (matches _shared/metrics.py bootstrap seed)
BLIND_INJECTION_SEED = 42


@dataclass
class BlindInjectionTracker:
    """Tracks 5/5-consensus items for blind injection into human verification.

    ~20% of the items shown to the human verifier are "blind injections" —
    items where all 5 panel models unanimously agreed. The human verifies
    these without knowing their status. If the human agrees with the panel on
    >=95% of these, the pipeline is deemed reliable.
    """

    candidates: List[Dict[str, Any]] = field(default_factory=list)
    selected: List[Dict[str, Any]] = field(default_factory=list)
    injection_ratio: float = 0.20
    min_count: int | None = None
    max_count: int | None = None

    def add_candidate(self, item: Dict[str, Any]) -> None:
        """Record a 5/5-unanimous panel item as a blind-injection candidate."""
        self.candidates.append(item)

    def select_for_injection(self) -> List[Dict[str, Any]]:
        """Randomly select items for blind injection (fixed seed, clamped to bounds)."""
        if not self.candidates:
            self.selected = []
            return []
        rng = random.Random(BLIND_INJECTION_SEED)
        n_select = max(1, int(len(self.candidates) * self.injection_ratio))
        if self.min_count is not None:
            n_select = max(n_select, self.min_count)
        if self.max_count is not None:
            n_select = min(n_select, self.max_count)
        n_select = min(n_select, len(self.candidates))
        self.selected = rng.sample(self.candidates, n_select)
        return self.selected

    def write_sidecar(self, path: str, fieldnames: List[str]) -> None:
        """Write the blind-injection items to a sidecar CSV."""
        if not self.selected:
            self.select_for_injection()
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for item in self.selected:
                # Strip non-CSV fields (like verdict objects)
                row = {k: v for k, v in item.items() if k in fieldnames}
                writer.writerow(row)
        logger.info(
            "concordance.blind_injection_written",
            path=str(output),
            count=len(self.selected),
        )


@dataclass
class ConcordanceTracker:
    """Tracks human-machine agreement for dataset quality control."""

    pairs: List[Tuple[str, str]] = field(default_factory=list)
    target_rate: float = 0.95

    def record(self, machine_label: str, human_label: str) -> None:
        """Record a single machine-human label pair."""
        self.pairs.append((machine_label.lower().strip(), human_label.lower().strip()))

    def concordance_rate(self) -> float:
        """Compute the proportion of matching labels."""
        if not self.pairs:
            return 0.0
        matches = sum(1 for m, h in self.pairs if m == h)
        return matches / len(self.pairs)

    def is_acceptable(self) -> bool:
        """Check if concordance meets the target rate."""
        rate = self.concordance_rate()
        acceptable = rate >= self.target_rate
        if not acceptable:
            logger.warning(
                "concordance.below_target",
                rate=rate,
                target=self.target_rate,
                total=len(self.pairs),
            )
        return acceptable

    def disagreement_examples(self) -> List[Tuple[str, str, int]]:
        """Get (machine_label, human_label, count) tuples for disagreements."""
        from collections import Counter
        disagreements = Counter(
            (m, h) for m, h in self.pairs if m != h
        )
        return [(m, h, c) for (m, h), c in disagreements.most_common()]

    def summary(self) -> str:
        """Generate a human-readable summary."""
        rate = self.concordance_rate()
        total = len(self.pairs)
        matches = sum(1 for m, h in self.pairs if m == h)
        status = "ACCEPTABLE" if self.is_acceptable() else "BELOW TARGET"

        lines = [
            f"Concordance Rate: {rate:.4f} ({matches}/{total}) — {status}",
            f"Target: {self.target_rate:.2%}",
        ]

        if not self.is_acceptable():
            lines.append("Disagreement examples:")
            for m, h, c in self.disagreement_examples()[:5]:
                lines.append(f"  Machine: {m} → Human: {h} (×{c})")

        return "\n".join(lines)
