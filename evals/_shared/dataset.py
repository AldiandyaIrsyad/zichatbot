"""Dataset loaders for the thesis evaluation subsets.

Subset A — RAG QA triplets:
    question, category, ground_truth_answer, source_doc_id, source_context

Subset B — Adversarial inputs:
    query, label, attack_type

Subset C — Boundary relevance:
    query, label, subtype

Subset D — RAM ground truth:
    question_id, question, full_response, sentence_id, sentence_text,
    retrieved_context, label, verifier_note

IndoNLI — the external Indonesian NLI benchmark (EMNLP 2021):
    train / val / test / test_lay / test_expert JSONL from
    ``github.com/ir-nlp-csui/indonli``. Loaded via :func:`load_indonli`.
"""

from __future__ import annotations

import csv
import json
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass(frozen=True)
class SubsetARow:
    """A single row from Subset A (RAG QA triplets).

    ``source_doc_id`` may hold several pipe-separated IDs — see
    ``gold_doc_ids``. ``source_context`` is NONE for out-of-domain rows.
    """

    question: str
    category: str
    ground_truth_answer: str
    source_doc_id: str
    source_context: str

    @property
    def gold_doc_ids(self) -> List[str]:
        """Return the relevant document IDs for retrieval scoring.

        A multi-hop question can legitimately have more than one correct source
        document, but the column holds a single ID, so such a question would be
        scored as a miss whenever retrieval surfaces one of its *other* valid
        sources. Pipe-separated IDs are therefore accepted here; single-ID rows
        are unaffected. Empty for out-of-domain rows, which carry the sentinel
        "NONE".
        """
        raw = (self.source_doc_id or "").strip()
        if not raw or raw.upper() == "NONE":
            return []
        return [part.strip() for part in raw.split("|") if part.strip()]


@dataclass(frozen=True)
class SubsetBRow:
    """A single row from Subset B (adversarial inputs).

    ``label`` is 'safe' or 'malicious'; ``attack_type`` is the subtype
    (jailbreak, dan_attempt, hidden_instruction, safe_normal, safe_complex).
    """

    query: str
    label: str
    attack_type: str


@dataclass(frozen=True)
class SubsetCRow:
    """A single row from Subset C (boundary relevance).

    ``label`` is 'in_domain' or 'out_of_domain'. ``panel_yes``/``panel_size``
    record the panel vote at generation time; both default to 0 when unrecorded.
    ``split`` is blank for legacy files.
    """

    query: str
    label: str
    subtype: str
    panel_yes: int = 0
    panel_size: int = 0
    split: str = ""

    def is_contested(self, strict_threshold: int = 4) -> bool:
        """Whether the panel fell short of the strict rule on this row.

        Subset C's boundary subtypes — ``near_miss_government`` above all —
        sit near the domain edge, so a split panel is information about the
        boundary rather than evidence of a bad item. Rows admitted one vote
        below the strict threshold are kept and marked here, so the strict and
        contested slices can be reported separately instead of pooled or
        silently discarded.

        Args:
            strict_threshold: The strict acceptance threshold to compare
                against (the generation-time ``acceptance_threshold``). Rows
                with no recorded vote count are treated as not contested, since
                they could only have been accepted strictly.
        """
        return 0 < self.panel_yes < strict_threshold


@dataclass(frozen=True)
class SubsetDRow:
    """A single row from Subset D (RAM ground truth).

    ``label`` is one of supported, partially_supported, not_supported,
    no_source_needed. The optional slice-metadata fields default to empty:
    ``construction`` is ``natural`` (real pipeline output) or ``perturbed`` (a
    verified counterfactual edit); ``perturbation_family`` is the edit strategy
    for a perturbed row (e.g. ``factual_flip``); ``intended_label`` is the label
    a perturbed row was designed to carry (panel-confirmed equal to ``label``);
    ``perturbation_of`` is the parent ``question_id:sentence_id``;
    ``difficulty_band`` is ``easy``/``medium``/``hard`` from how the production
    NLI verifier fared (a reported slice, not an inclusion gate); ``split`` is
    ``core`` (balanced diagnostic) or ``natural`` (realistic base rate);
    ``edit_note`` describes the perturbation for auditing.
    """

    question_id: str
    question: str
    full_response: str
    sentence_id: int
    sentence_text: str
    retrieved_context: str
    label: str
    verifier_note: str
    # Optional slice metadata. All default to empty and load via ``.get()``, so
    # files lacking these columns (older schema) still read unchanged.
    construction: str = ""
    perturbation_family: str = ""
    intended_label: str = ""
    perturbation_of: str = ""
    difficulty_band: str = ""
    split: str = ""
    evaluation_split: str = ""
    edit_note: str = ""


def _int_or_zero(value: Optional[str]) -> int:
    """Parse an optional CSV cell as an int, defaulting to 0.

    Absent or blank cells (a column added after a dataset was generated, or a
    hand-edited blank) must not crash a load.
    """
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return 0


def _open_csv(path: str) -> csv.DictReader:
    """Open a CSV file and return a DictReader.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    csv_path = Path(path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")
    fh = csv_path.open(newline="", encoding="utf-8")
    return csv.DictReader(fh)


def load_subset_a(path: str) -> List[SubsetARow]:
    """Load Subset A (RAG QA triplets) from CSV."""
    reader = _open_csv(path)
    rows: List[SubsetARow] = []
    for row in reader:
        rows.append(
            SubsetARow(
                question=row["question"].strip(),
                category=row["category"].strip().lower(),
                ground_truth_answer=row.get("ground_truth_answer", "").strip(),
                source_doc_id=row.get("source_doc_id", "").strip(),
                source_context=row.get("source_context", "").strip(),
            )
        )
    return rows


def load_subset_b(path: str) -> List[SubsetBRow]:
    """Load Subset B (adversarial inputs) from CSV."""
    reader = _open_csv(path)
    rows: List[SubsetBRow] = []
    for row in reader:
        rows.append(
            SubsetBRow(
                query=row["query"].strip(),
                label=row["label"].strip().lower(),
                attack_type=row.get("attack_type", "").strip().lower(),
            )
        )
    return rows


def load_subset_c(path: str) -> List[SubsetCRow]:
    """Load Subset C (boundary relevance) from CSV."""
    reader = _open_csv(path)
    rows: List[SubsetCRow] = []
    for row in reader:
        rows.append(
            SubsetCRow(
                query=row["query"].strip(),
                label=row["label"].strip().lower(),
                subtype=row.get("subtype", "").strip().lower(),
                panel_yes=_int_or_zero(row.get("panel_yes")),
                panel_size=_int_or_zero(row.get("panel_size")),
                split=row.get("split", "").strip().lower(),
            )
        )
    return rows


def load_subset_d(path: str) -> List[SubsetDRow]:
    """Load Subset D (RAM ground truth) from CSV."""
    reader = _open_csv(path)
    rows: List[SubsetDRow] = []
    for row in reader:
        rows.append(
            SubsetDRow(
                question_id=row.get("question_id", "").strip(),
                question=row.get("question", "").strip(),
                full_response=row.get("full_response", "").strip(),
                sentence_id=int(row.get("sentence_id", 0) or 0),
                sentence_text=row["sentence_text"].strip(),
                retrieved_context=row.get("retrieved_context", "").strip(),
                label=row["label"].strip().lower(),
                verifier_note=row.get("verifier_note", "").strip(),
                construction=row.get("construction", "").strip().lower(),
                perturbation_family=row.get("perturbation_family", "").strip().lower(),
                intended_label=row.get("intended_label", "").strip().lower(),
                perturbation_of=row.get("perturbation_of", "").strip(),
                difficulty_band=row.get("difficulty_band", "").strip().lower(),
                split=row.get("split", "").strip().lower(),
                evaluation_split=row.get("evaluation_split", "").strip().lower(),
                edit_note=row.get("edit_note", "").strip(),
            )
        )
    return rows


# ---------------------------------------------------------------------------
# IndoNLI (external Indonesian NLI benchmark)
# ---------------------------------------------------------------------------

INDONLI_RAW_BASE = (
    "https://raw.githubusercontent.com/ir-nlp-csui/indonli/main/data/indonli"
)

# IndoNLI's single-letter labels → canonical 3-way NLI labels.
INDONLI_LABEL_MAP = {"e": "entailment", "n": "neutral", "c": "contradiction"}

INDONLI_SPLITS = ("train", "val", "test", "test_lay", "test_expert", "diagnostic")


@dataclass(frozen=True)
class IndoNLIRow:
    """A single IndoNLI pair.

    ``label`` is canonical (entailment/neutral/contradiction); ``annotator_type``
    is ``lay`` or ``expert``; ``sentence_size`` is ``single`` or ``multi``.
    """

    pair_id: int
    premise: str
    hypothesis: str
    label: str
    annotator_type: str = ""
    sentence_size: str = ""


def _ensure_indonli_file(split: str, data_dir: str) -> Path:
    """Return the local path for an IndoNLI split, downloading it from GitHub on
    first use. A missing download raises ``FileNotFoundError``."""
    target = Path(data_dir) / "indonli" / f"{split}.jsonl"
    if target.is_file():
        return target

    url = f"{INDONLI_RAW_BASE}/{split}.jsonl"
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".jsonl.tmp")
    try:
        urllib.request.urlretrieve(url, tmp)
        os.replace(tmp, target)
    except Exception as exc:  # noqa: BLE001 - surface as a clear load error
        if tmp.exists():
            tmp.unlink()
        raise FileNotFoundError(f"IndoNLI split '{split}' could not be downloaded: {exc}") from exc
    return target


def load_indonli(split: str, data_dir: str = "evals/data") -> List[IndoNLIRow]:
    """Load an IndoNLI split, mapping single-letter labels to canonical names.

    ``split`` is one of ``train``/``val``/``test``/``test_lay``/``test_expert``.
    Rows are downloaded from the upstream GitHub repo on first use and cached
    under ``{data_dir}/indonli/{split}.jsonl``.
    """
    if split not in INDONLI_SPLITS:
        raise ValueError(f"Unknown IndoNLI split: {split} (expected one of {INDONLI_SPLITS})")

    path = _ensure_indonli_file(split, data_dir)
    rows: List[IndoNLIRow] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            raw_label = str(obj.get("label", "")).strip().lower()
            label = INDONLI_LABEL_MAP.get(raw_label, raw_label)
            rows.append(
                IndoNLIRow(
                    pair_id=int(obj.get("pair_id", 0)),
                    premise=str(obj.get("premise", "")).strip(),
                    hypothesis=str(obj.get("hypothesis", "")).strip(),
                    label=label,
                    annotator_type=str(obj.get("annotator_type", "")).strip(),
                    sentence_size=str(obj.get("sentence_size", "")).strip(),
                )
            )
    return rows
