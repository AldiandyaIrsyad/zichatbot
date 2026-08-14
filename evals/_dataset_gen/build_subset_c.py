"""Build Subset C — Boundary Relevance.

Generates boundary relevance queries (in-domain and out-of-domain) using
the Generator-Evaluator architecture.

Pipeline:
    1. Generator produces draft boundary queries per subtype
    2. Panel labels each query as in_domain or out_of_domain
    3. Accept if ≥4/5 panel members agree on the label
    4. Write strict rows to the principal CSV and 3/5 disagreements to a
       separate contested sidecar.
    5. Assign 60 principal rows to calibration and 140 to locked evaluation.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List

import structlog

from evals._dataset_gen.config import DatasetGenSettings, get_dataset_gen_settings
from evals._dataset_gen.checkpoint import IncrementalCSVWriter, resume_rows
from evals._dataset_gen.concordance import BlindInjectionTracker
from evals._dataset_gen.generator import DatasetGenerator
from evals._dataset_gen.panel import EvaluatorPanel, PanelUnavailableError
from evals._dataset_gen.provenance import append_panel_vote_record, write_provenance

logger = structlog.get_logger(__name__)

SUBTYPES = [
    # (subtype, expected_label, count) — total 200, balanced 100 in / 100 out.
    #
    # Every subtype clears roughly ±14pp (the in-domain pair clears ±11pp), so
    # per-subtype conclusions are possible. The in/out split is held at 100/100
    # because Exp1b reports Accuracy over the whole set: an uneven split would
    # let the majority class drive the headline number without that being
    # visible in the table.
    ("direct_upi", "in_domain", 50),
    ("indirect_upi", "in_domain", 50),
    ("near_miss_government", "out_of_domain", 34),
    ("adjacent_legal", "out_of_domain", 33),
    ("off_topic", "out_of_domain", 33),
]

CALIBRATION_TARGETS = {
    "direct_upi": 15,
    "indirect_upi": 15,
    "near_miss_government": 10,
    "adjacent_legal": 10,
    "off_topic": 10,
}

FIELDNAMES = ["query", "label", "subtype", "split", "panel_yes", "panel_size"]

# In-domain = UPI internal legal/regulatory documents published via JDIH.
DOMAIN_DESC = (
    "the internal legal/regulatory documents of Universitas Pendidikan "
    "Indonesia (UPI), published via its JDIH portal (Peraturan Rektor, SK "
    "Rektor, Statuta UPI, keputusan Senat Akademik / MWA, internal pedoman)"
)

SUBTYPE_SPECS = {
    "direct_upi": "Explicitly asks about a UPI internal legal document, office, rule, or regulated procedure.",
    "indirect_upi": "Still asks about a UPI internal rule or procedure, but uses colloquial wording, anaphora, abbreviation, or omits the document title; it must remain answerable from JDIH UPI.",
    "near_miss_government": "The benchmark's broad legacy identifier for near misses that exclusively request a document or rule from another university, ministry, regional government, or other non-UPI institution. It must not ask to compare with UPI or ask how an external rule affects UPI; therefore it is unambiguously out of domain.",
    "adjacent_legal": "Concerns Indonesian law or education regulation generally, without asking about an internal UPI instrument or implementation; therefore out of domain.",
    "off_topic": "Unrelated to internal legal/regulatory documents even if written in formal bureaucratic language.",
}


GENERATOR_SYSTEM_PROMPT = f"""\
You are a dataset generator for a boundary relevance benchmark about \
{DOMAIN_DESC}. Generate queries in Indonesian. Output each item as a JSON \
object on its own line (JSONL) with:
- "query": The query text
- "label": Either "in_domain" or "out_of_domain"
- "subtype": The boundary subtype

Items must be unique and self-contained. Do not decide the label from keywords alone:
UPI-specific institutional scope controls the boundary. Near-miss and adjacent-law rows
must include enough information to establish that they are not requests about UPI internal rules.

Do not include markdown code fences. Output one JSON object per line.
"""

VALIDATION_PROMPT = f"""\
You are verifying a proposed annotation for a boundary-relevance benchmark.

CANDIDATE QUERY: {{query}}
CLAIMED LABEL: {{label}}
CLAIMED SUBTYPE: {{subtype}}
OPERATIONAL SUBTYPE DEFINITION: {{subtype_definition}}

Your task is NOT to answer whether the query itself is in-domain. Your task is
to judge whether the proposed label and subtype are correct. YES means the
annotation is fully correct; it does not mean "in domain."

Vote YES only if all are true:
1. The label follows this boundary: in_domain asks about {DOMAIN_DESC};
   out_of_domain does not.
2. The claimed subtype matches its operational definition. The identifier
   near_miss_government is a broad legacy name that also includes other
   universities and non-UPI institutions.
3. The query is self-contained and its institutional scope is explicit enough
   to classify without guessing.

Return exactly YES or NO.
"""


async def build_subset_c(
    settings: DatasetGenSettings,
    output_path: str,
    count: int,
    resume: bool = False,
) -> None:
    """Build Subset C (boundary relevance queries) and save to CSV.

    Args:
        resume: Continue an interrupted run, keeping rows already written to
            ``output_path`` and rebuilding the per-subtype counters from them.
    """
    if not settings.openrouter_api_key:
        logger.error("datagen.subset_c.missing_api_key")
        sys.exit(1)

    generator = DatasetGenerator(settings)
    panel = EvaluatorPanel(settings)
    blind_tracker = BlindInjectionTracker(min_count=20)

    accepted_items: List[Dict[str, str]] = list(
        resume_rows(output_path, FIELDNAMES) if resume else []
    )
    contested_path = output_path.replace(".csv", "_contested.csv")
    contested_items: List[Dict[str, str]] = list(
        resume_rows(contested_path, FIELDNAMES) if resume else []
    )
    contested_queries = {row["query"] for row in contested_items}
    total_generated = 0
    total_rejected = 0

    # Safety bound on retry batches per subtype, in case acceptance rate for a
    # hard boundary subtype (e.g. near_miss_government) is persistently low —
    # mirrors build_subset_a's max_cycles retry design.
    max_batches_per_subtype = 25

    # v2 principal rows always require the strict 4/5 threshold. Boundary
    # disagreement is informative, but it is not silently promoted to truth.

    # Rows are flushed as they are accepted so an interrupted run leaves a
    # valid partial dataset that --resume can continue.
    writer_ctx = IncrementalCSVWriter(output_path, FIELDNAMES, resume=resume)
    contested_writer_ctx = IncrementalCSVWriter(
        contested_path, FIELDNAMES, resume=resume
    )
    try:
        with writer_ctx as row_writer, contested_writer_ctx as contested_writer:
            for subtype, expected_label, per_subtype_count in SUBTYPES:
                if len(accepted_items) >= count:
                    break

                accepted_for_subtype = sum(
                    1 for r in accepted_items if r.get("subtype") == subtype
                )
                subtype_target = min(
                    per_subtype_count, count - len(accepted_items) + accepted_for_subtype
                )
                batch_num = 0
                seed_prompt_base = SUBTYPE_SPECS[subtype]
                while (
                    accepted_for_subtype < subtype_target
                    and len(accepted_items) < count
                    and batch_num < max_batches_per_subtype
                ):
                    effective_threshold = settings.acceptance_threshold
                    n = max(5, subtype_target - accepted_for_subtype)
                    logger.info(
                        "datagen.subset_c.generating",
                        subtype=subtype,
                        batch_size=n,
                        batch_num=batch_num + 1,
                        threshold=effective_threshold,
                        expected_label=expected_label,
                        have=accepted_for_subtype,
                        target=subtype_target,
                    )

                    seed_prompt = f"Generate {n} UNIQUE queries for subtype {subtype!r}.\nRequired label: {expected_label}.\nOperational definition: {seed_prompt_base}\nVary syntax, topic, and named entities; do not make template paraphrases."

                    try:
                        drafts = await generator.generate(
                            seed_prompt=seed_prompt,
                            count=n,
                            system_prompt=GENERATOR_SYSTEM_PROMPT,
                        )
                    except Exception as e:
                        logger.error("datagen.subset_c.generator_error", error=str(e), exc_info=True)
                        batch_num += 1
                        continue

                    total_generated += len(drafts)

                    for draft in drafts:
                        if accepted_for_subtype >= subtype_target or len(accepted_items) >= count:
                            break

                        if not isinstance(draft.parsed, dict):
                            continue

                        item = draft.parsed
                        query = item.get("query", "").strip()
                        if not query:
                            continue

                        label = item.get("label", expected_label).lower().strip()
                        stype = item.get("subtype", subtype).lower().strip()

                        validation_context = VALIDATION_PROMPT.format(
                            query=query,
                            label=label,
                            subtype=stype,
                            subtype_definition=SUBTYPE_SPECS.get(stype, "unknown subtype"),
                        )

                        try:
                            verdict = await panel.evaluate(
                                prompt=(
                                    "Verify the candidate annotation. YES means the claimed "
                                    "label and subtype are fully correct; NO means they are not."
                                ),
                                context=validation_context,
                            )
                        except PanelUnavailableError:
                            # The API is down, not this candidate. Propagating stops the
                            # run with its output intact instead of burning the batch
                            # budget marking every item rejected; --resume continues it.
                            raise
                        except Exception as e:
                            logger.error("datagen.subset_c.panel_error", error=str(e), exc_info=True)
                            continue

                        append_panel_vote_record(
                            output_path, "c",
                            {"query": query, "label": label, "subtype": stype},
                            verdict, verdict.yes_count >= effective_threshold,
                        )
                        if verdict.yes_count >= effective_threshold:
                            calibration_count = sum(
                                1 for existing in accepted_items
                                if existing.get("subtype") == subtype
                                and existing.get("split") == "calibration"
                            )
                            row = {
                                "query": query,
                                "label": label,
                                "subtype": stype,
                                "split": (
                                    "calibration"
                                    if calibration_count < CALIBRATION_TARGETS.get(subtype, 0)
                                    else "locked_test"
                                ),
                                # Recorded per row so any threshold can be applied
                                # after the fact: "contested" stays a derived
                                # predicate rather than a decision baked into the
                                # data at generation time.
                                "panel_yes": verdict.yes_count,
                                "panel_size": len(verdict.votes),
                            }
                            accepted_items.append(row)
                            row_writer.append(row)
                            accepted_for_subtype += 1
                            # Track 5/5-unanimous items for blind injection
                            if verdict.yes_count == len(verdict.votes):
                                blind_tracker.add_candidate({**row, "_panel_yes": verdict.yes_count})
                            logger.info(
                                "datagen.subset_c.accepted",
                                subtype=subtype,
                                accepted=len(accepted_items),
                                panel_yes=verdict.yes_count,
                                threshold=effective_threshold,
                                contested=verdict.yes_count < settings.acceptance_threshold,
                                target=count,
                            )
                        elif verdict.yes_count == 3:
                            if query not in contested_queries:
                                contested_row = {
                                    "query": query,
                                    "label": label,
                                    "subtype": stype,
                                    "split": "contested",
                                    "panel_yes": verdict.yes_count,
                                    "panel_size": len(verdict.votes),
                                }
                                contested_writer.append(contested_row)
                                contested_items.append(contested_row)
                                contested_queries.add(query)
                            total_rejected += 1
                        else:
                            total_rejected += 1
                            logger.info("datagen.subset_c.rejected", yes=verdict.yes_count, total=verdict.no_count + verdict.yes_count)

                    batch_num += 1

                if accepted_for_subtype < subtype_target:
                    logger.warning(
                        "datagen.subset_c.subtype_underfilled",
                        subtype=subtype,
                        accepted=accepted_for_subtype,
                        target=subtype_target,
                    )

    finally:
        await generator.aclose()
        await panel.aclose()

    # Write blind-injection sidecar
    blind_tracker.write_sidecar(
        output_path.replace(".csv", "_blind_injection.csv"),
        fieldnames=FIELDNAMES,
    )

    # Write CSV
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    # CSV already complete — rows were flushed as accepted (IncrementalCSVWriter).

    strict_rows = sum(
        1 for r in accepted_items if int(r["panel_yes"]) >= settings.acceptance_threshold
    )
    write_provenance(
        output_path,
        subset="c",
        settings=settings,
        row_count=len(accepted_items),
        extra={
            "generated": total_generated,
            "rejected": total_rejected,
            "targets": {s: c for s, _, c in SUBTYPES},
            "by_subtype": Counter(r["subtype"] for r in accepted_items),
            "by_label": Counter(r["label"] for r in accepted_items),
            "by_split": Counter(r["split"] for r in accepted_items),
            # Every principal row meets the configured strict threshold.
            # Disagreements are excluded from scientific evaluation.
            "acceptance": {
                "strict_threshold": settings.acceptance_threshold,
                "strict_rows": strict_rows,
                "contested_rows": len(accepted_items) - strict_rows,
                "contested_sidecar_rows": len(contested_items),
            },
            "panel_yes_distribution": Counter(int(r["panel_yes"]) for r in accepted_items),
        },
    )

    logger.info(
        "datagen.subset_c.complete",
        generated=total_generated,
        accepted=len(accepted_items),
        rejected=total_rejected,
        output=output_path,
    )


def main() -> None:
    """Entry point for Subset C generation."""
    parser = argparse.ArgumentParser(
        description="Build Subset C (Boundary Relevance) using Generator-Evaluator architecture."
    )
    parser.add_argument(
        "--output",
        default="evals/data/subset_c.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=200,
        help="Target number of accepted items",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue an interrupted run: keep the rows already in --output and "
        "generate only the shortfall. Rows are written as they are accepted, so a "
        "run stopped by an outage or Ctrl-C can always be continued this way.",
    )
    args = parser.parse_args()

    settings = get_dataset_gen_settings()
    asyncio.run(build_subset_c(settings, args.output, args.count, resume=args.resume))


if __name__ == "__main__":
    main()
