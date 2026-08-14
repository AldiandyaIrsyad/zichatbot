"""Controlled counterfactual perturbations for the hard Subset D rebuild.

Moves the difficulty lever from the question to the **response**: a known-good
(``supported``) sentence plus its grounding is edited *minimally* into a
counterfactual whose label is known **by construction** — then the same 5-model
panel *verifies* the label before it is kept (``build_subset_d`` phase 2). The
negatives are subtle: a single flipped number keeps almost all of the sentence's
lexical overlap with the context, which is exactly the case a lexical or
weak-NLI verifier gets wrong.

Nothing here calls the network except ``generate_perturbation`` (one generator
call) — the banding and assertion helpers are pure so they run in the builder
and in tests without a stack.

Design notes:
    - Every family carries the NLI-mapped ``intended_label`` (Exp3's LABEL_MAP),
      so the verification gate and the balance assertion speak the same language
      the RAM experiment scores in.
    - The shortcut assertion mirrors ``build_subset_b_train.assert_no_shortcut_features``:
      a surface feature (sentence length, or lexical overlap with the context)
      must not be able to stand in for the label. Because the perturbations are
      minimal edits, the ``not_supported`` rows sit in the *same* high-overlap
      band as ``supported`` ones — the assertion proves that on the built set.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from evals._dataset_gen.generator import parse_json_object
from evals._shared.metrics import token_containment_similarity

# --- direct three-class NLI ground truth --------------------
SUPPORTED = "entailment"
PARTIALLY_SUPPORTED = "neutral"
NOT_SUPPORTED = "contradiction"
NO_SOURCE_NEEDED = "no_source_needed"

# Identity mapping retained for shared balance helpers; no unsupported-to-contradiction conversion.
# balance floor and banding are expressed in the classes Exp3 scores.
LABEL_TO_NLI: Dict[str, str] = {
    SUPPORTED: "entailment",
    NOT_SUPPORTED: "contradiction",
    PARTIALLY_SUPPORTED: "neutral",
    # no_source_needed is excluded from scoring.
}


@dataclass(frozen=True)
class PerturbationFamily:
    """One counterfactual edit strategy.

    ``system_prompt`` must instruct a *minimal* edit and a JSON reply with
    ``perturbed`` and ``edit_note`` fields.
    """

    name: str
    intended_label: str
    system_prompt: str


# The response is always a single JSON object so the generator's output parses
# the same way for every family; the builder reads .perturbed / .edit_note.
_JSON_TAIL = (
    ' Output a single JSON object with exactly two fields: "perturbed" (the edited '
    'sentence, in Indonesian) and "edit_note" (a short English description of the '
    "one change you made). Do not include markdown code fences."
)

FAMILIES: Tuple[PerturbationFamily, ...] = (
    PerturbationFamily(
        name="factual_flip",
        intended_label=NOT_SUPPORTED,
        system_prompt=(
            "You edit a sentence for a hallucination-detection benchmark about UPI "
            "internal legal/regulatory documents (JDIH). You are given a sentence that "
            "IS supported by the context and the context itself. Change EXACTLY ONE "
            "concrete detail (a number, percentage, date, article/pasal number, or a "
            "person's name/title) to a plausible but WRONG value that the context does "
            "NOT support. Keep every other word identical, so the sentence still reads "
            "naturally and still overlaps heavily with the context." + _JSON_TAIL
        ),
    ),
    PerturbationFamily(
        name="scope_negation_flip",
        intended_label=NOT_SUPPORTED,
        system_prompt=(
            "You edit a sentence for a hallucination-detection benchmark about UPI "
            "internal legal/regulatory documents (JDIH). You are given a sentence that "
            "IS supported by the context and the context itself. Make the SMALLEST "
            "possible edit that reverses its meaning relative to the context: flip a "
            "quantifier or modality (semua<->sebagian, wajib<->dapat, harus<->boleh, "
            "selalu<->kadang) or insert/remove a negation (tidak/bukan/tanpa). Change "
            "nothing else." + _JSON_TAIL
        ),
    ),
    PerturbationFamily(
        name="plausible_absent_detail",
        intended_label=PARTIALLY_SUPPORTED,
        system_prompt=(
            "You edit a sentence for a hallucination-detection benchmark about UPI "
            "internal legal/regulatory documents (JDIH). You are given a sentence that "
            "IS supported by the context and the context itself. Append or insert ONE "
            "specific-sounding claim (an extra article number, a date, a named body, a "
            "figure) that is NOT present in the context. Keep the original, supported "
            "part unchanged so only the added detail is ungrounded." + _JSON_TAIL
        ),
    ),
    PerturbationFamily(
        name="evaluative_graft",
        intended_label=PARTIALLY_SUPPORTED,
        system_prompt=(
            "You edit a sentence for a hallucination-detection benchmark about UPI "
            "internal legal/regulatory documents (JDIH). You are given a sentence that "
            "IS supported by the context and the context itself. Keep the grounded fact "
            "exactly as is, and graft on an evaluative or causal clause the context does "
            "NOT state (e.g. '...yang sangat penting', '...dan merupakan yang pertama di "
            "Indonesia', '...sehingga berdampak besar bagi mahasiswa'). The result must "
            "be half-grounded: the fact holds, the added judgement does not." + _JSON_TAIL
        ),
    ),
    PerturbationFamily(
        name="far_paraphrase",
        intended_label=SUPPORTED,
        system_prompt=(
            "You edit a sentence for a hallucination-detection benchmark about UPI "
            "internal legal/regulatory documents (JDIH). You are given a sentence that "
            "IS supported by the context and the context itself. Rewrite it so it keeps "
            "the SAME meaning and stays fully supported by the context, but shares as "
            "few surface words with the context as possible (synonyms, reordering, voice "
            "change). Do not add or drop any factual claim — this is a hard *positive*, "
            "still entailed by the context." + _JSON_TAIL
        ),
    ),
)

FAMILY_BY_NAME: Dict[str, PerturbationFamily] = {f.name: f for f in FAMILIES}


def build_perturbation_prompt(sentence: str, grounding: str) -> str:
    """Assemble the user-message body for a perturbation call."""
    return (
        f"Context:\n{grounding}\n\n"
        f"Supported sentence:\n{sentence}\n\n"
        "Produce the edited sentence now."
    )


async def generate_perturbation(
    generator: Any,
    sentence: str,
    grounding: str,
    family: PerturbationFamily,
) -> Optional[Tuple[str, str]]:
    """Generate one counterfactual edit of ``sentence`` for ``family``.

    Returns None when the generator errors, returns unparseable output, or
    returns an edit identical to the input (a no-op edit carries no label
    signal and must not be kept).
    """
    prompt = build_perturbation_prompt(sentence, grounding)
    try:
        raw = await generator.generate_single(prompt=prompt, system_prompt=family.system_prompt)
    except Exception:
        return None
    parsed = _parse_perturbation(raw)
    if parsed is None:
        return None
    perturbed, note = parsed
    if not perturbed or perturbed.strip() == sentence.strip():
        return None  # no-op edit: no label signal
    return perturbed.strip(), note.strip()


def _parse_perturbation(raw: str) -> Optional[Tuple[str, str]]:
    """Extract (perturbed, edit_note) from a generator reply via parse_json_object.

    Returns None if no object with a non-empty ``perturbed`` field is recovered.
    """
    obj = parse_json_object(raw)
    if obj and str(obj.get("perturbed", "")).strip():
        return str(obj["perturbed"]), str(obj.get("edit_note", ""))
    return None


# --- Difficulty banding (a TAG, never a gate) --------------------------------
BAND_HARD = "hard"
BAND_MEDIUM = "medium"
BAND_EASY = "easy"

# When the production verifier is *correct*, confidence splits easy from medium.
# When it is *wrong*, the row is hard regardless of confidence.
_BAND_CONFIDENT = 0.80


def band_by_nli(
    predicted_nli: str,
    truth_nli: str,
    score: Optional[float] = None,
) -> str:
    """Rate one row's difficulty from how the production NLI verifier fared on it.

    Used only to *label* a slice, never to decide inclusion — so the dataset's
    composition stays independent of any one model's blind spots.

    Returns ``hard`` if the verifier is wrong, else ``easy`` (correct and
    confident) or ``medium`` (correct but not confident, or confidence unknown).
    """
    if predicted_nli != truth_nli:
        return BAND_HARD
    if score is not None and score >= _BAND_CONFIDENT:
        return BAND_EASY
    return BAND_MEDIUM


# --- Shortcut + balance assertions (mirrors build_subset_b_train) -------------
# A cell smaller than this says nothing about balance.
MIN_CELL_ROWS = 15
# The rarer of {supported, not_supported} must hold at least this share of a
# cell: the surface feature must not be *decisive*, not that the cell be balanced.
MIN_MINORITY_SHARE = 0.25
# Per-class floor for the balanced core (NLI classes).
MIN_CLASS_ROWS = 50

_LENGTH_BANDS: Tuple[Tuple[str, int, int], ...] = (
    ("<=8", 0, 8),
    ("9-16", 9, 16),
    ("17-32", 17, 32),
    (">32", 33, 10**9),
)
_OVERLAP_BANDS: Tuple[Tuple[str, float, float], ...] = (
    ("0.0-0.5", 0.0, 0.5),
    ("0.5-0.7", 0.5, 0.7),
    ("0.7-0.85", 0.7, 0.85),
    ("0.85-1.0", 0.85, 1.0001),
)


def _length_band(word_count: int) -> str:
    for name, low, high in _LENGTH_BANDS:
        if low <= word_count <= high:
            return name
    return _LENGTH_BANDS[-1][0]


def _overlap_band(overlap: float) -> str:
    for name, low, high in _OVERLAP_BANDS:
        if low <= overlap < high:
            return name
    return _OVERLAP_BANDS[-1][0]


def shortcut_report(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Cross-tabulate the supported/not_supported split against surface features.

    Two features are checked: sentence length, and lexical overlap between the
    sentence and its retrieved context (token containment). Only the two
    decisive classes are counted — ``partially_supported`` and
    ``no_source_needed`` are not part of a lexical-overlap shortcut.
    """
    cells: Dict[Tuple[str, str], Counter] = {}
    for row in rows:
        label = row.get("label", "?")
        if label not in (SUPPORTED, NOT_SUPPORTED):
            continue
        sentence = row.get("sentence_text", "")
        context = row.get("retrieved_context", "")
        length_key = ("length", _length_band(len(str(sentence).split())))
        overlap = token_containment_similarity(sentence, context)
        overlap_key = ("overlap", _overlap_band(overlap))
        cells.setdefault(length_key, Counter())[label] += 1
        cells.setdefault(overlap_key, Counter())[label] += 1

    report: List[Dict[str, Any]] = []
    for (feature, band), counts in sorted(cells.items()):
        total = sum(counts.values())
        minority = min(counts.get(SUPPORTED, 0), counts.get(NOT_SUPPORTED, 0))
        report.append(
            {
                "feature": feature,
                "band": band,
                "supported": counts.get(SUPPORTED, 0),
                "not_supported": counts.get(NOT_SUPPORTED, 0),
                "total": total,
                "minority_share": minority / total if total else 0.0,
            }
        )
    return report


def assert_no_shortcut_features(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Fail the build if length or lexical overlap can stand in for the label.

    The counterfactual perturbations are minimal edits precisely so the
    negatives land in the same high-overlap band as the positives; this
    assertion proves that held on the assembled set.

    Raises:
        AssertionError: If a sufficiently large cell is dominated by one label.
    """
    report = shortcut_report(rows)
    offenders = [
        cell
        for cell in report
        if cell["total"] >= MIN_CELL_ROWS and cell["minority_share"] < MIN_MINORITY_SHARE
    ]
    if offenders:
        detail = "; ".join(
            f"{c['feature']}={c['band']} supported={c['supported']} "
            f"not_supported={c['not_supported']} (minority {c['minority_share']:.1%})"
            for c in offenders
        )
        raise AssertionError(
            f"a surface feature predicts the faithfulness label in {len(offenders)} "
            f"cell(s): {detail}. A verifier can satisfy its objective on this split "
            "without judging entailment."
        )
    return report


def label_distribution(rows: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    """Count rows per direct NLI class."""
    return dict(Counter(row.get("label", "?") for row in rows))


def nli_class_counts(rows: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    """Count rows per NLI class (no_source_needed excluded, as Exp3 does)."""
    counts: Counter = Counter()
    for row in rows:
        nli = LABEL_TO_NLI.get(row.get("label", ""))
        if nli is not None:
            counts[nli] += 1
    return dict(counts)


def assert_class_balance(
    rows: Sequence[Dict[str, Any]],
    floor: int = MIN_CLASS_ROWS,
) -> Dict[str, int]:
    """Fail the build unless every NLI class clears ``floor`` on the core.

    The whole point of the rebuild is a non-degenerate distribution: macro-F1 is
    meaningless if ``contradiction`` or ``neutral`` has a handful of rows. Applied
    to the balanced *core* slice, not the natural slice (which is allowed to keep
    its realistic ~all-supported base rate).

    Args:
        rows: The core-slice rows.
        floor: Minimum rows required in each of the three NLI classes.

    Returns:
        The per-NLI-class counts.

    Raises:
        AssertionError: If any NLI class is below ``floor``.
    """
    counts = nli_class_counts(rows)
    short = {cls: counts.get(cls, 0) for cls in ("entailment", "neutral", "contradiction") if counts.get(cls, 0) < floor}
    if short:
        raise AssertionError(
            f"class balance floor {floor} not met: {short}. The core needs enough of "
            "every NLI class for macro-F1 to be meaningful — raise the perturbation "
            "quota for the short class(es)."
        )
    return counts
