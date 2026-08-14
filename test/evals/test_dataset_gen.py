"""Tests for the dataset generation pipeline.

Verifies:
    - EvaluatorPanel majority voting (≥4/5 accept, <4/5 reject)
    - EvaluatorPanel fail-closed on errors (defaults to NO)
    - _parse_yes_no() parsing logic
    - DatasetGenerator JSONL parsing
    - DatasetGenerator handles markdown fences
"""

from __future__ import annotations

import json
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from evals._dataset_gen.config import DatasetGenSettings
from evals._dataset_gen.generator import DatasetGenerator, GeneratedItem
from evals._dataset_gen.panel import EvaluatorPanel, PanelUnavailableError, PanelVerdict


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_settings(
    api_key: str = "test-key",
    panel_models: str = "model-a,model-b,model-c,model-d,model-e",
    threshold: int = 4,
) -> DatasetGenSettings:
    """Create DatasetGenSettings for testing."""
    return DatasetGenSettings(
        openrouter_api_key=api_key,
        openrouter_base_url="https://openrouter.ai/api/v1",
        generator_model="deepseek/deepseek-chat",
        generator_temperature=0.0,
        panel_models=panel_models,
        panel_temperature=0.0,
        acceptance_threshold=threshold,
        budget_usd=0,
        cost_ledger_path="",
    )


def _mock_chat_response(content: str) -> MagicMock:
    """Create a mock OpenRouter chat completion response."""
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {
        "choices": [{"message": {"content": content}}]
    }
    return resp


# ---------------------------------------------------------------------------
# EvaluatorPanel tests
# ---------------------------------------------------------------------------


class TestEvaluatorPanel:
    """Tests for the EvaluatorPanel majority voting."""

    @pytest.mark.asyncio
    async def test_accept_when_four_yes(self) -> None:
        """4/5 YES → accepted=True."""
        settings = _make_settings()
        panel = EvaluatorPanel(settings)
        panel._client = MagicMock(spec=httpx.AsyncClient)

        # 4 YES, 1 NO
        responses = [
            _mock_chat_response("YES"),
            _mock_chat_response("YES"),
            _mock_chat_response("YES"),
            _mock_chat_response("YES"),
            _mock_chat_response("NO"),
        ]
        panel._client.post = AsyncMock(side_effect=responses)

        verdict = await panel.evaluate("Is this valid?", "context here")

        assert verdict.accepted is True
        assert verdict.yes_count == 4
        assert verdict.no_count == 1
        assert verdict.acceptance_threshold == 4

    @pytest.mark.asyncio
    async def test_reject_when_three_yes(self) -> None:
        """3/5 YES → accepted=False (below threshold)."""
        settings = _make_settings()
        panel = EvaluatorPanel(settings)
        panel._client = MagicMock(spec=httpx.AsyncClient)

        responses = [
            _mock_chat_response("YES"),
            _mock_chat_response("YES"),
            _mock_chat_response("YES"),
            _mock_chat_response("NO"),
            _mock_chat_response("NO"),
        ]
        panel._client.post = AsyncMock(side_effect=responses)

        verdict = await panel.evaluate("Is this valid?", "context here")

        assert verdict.accepted is False
        assert verdict.yes_count == 3
        assert verdict.no_count == 2

    @pytest.mark.asyncio
    async def test_unanimous_yes(self) -> None:
        """5/5 YES → accepted=True."""
        settings = _make_settings()
        panel = EvaluatorPanel(settings)
        panel._client = MagicMock(spec=httpx.AsyncClient)

        responses = [_mock_chat_response("YES") for _ in range(5)]
        panel._client.post = AsyncMock(side_effect=responses)

        verdict = await panel.evaluate("Is this valid?", "context")
        assert verdict.accepted is True
        assert verdict.yes_count == 5

    @pytest.mark.asyncio
    async def test_unanimous_no(self) -> None:
        """0/5 YES → accepted=False."""
        settings = _make_settings()
        panel = EvaluatorPanel(settings)
        panel._client = MagicMock(spec=httpx.AsyncClient)

        responses = [_mock_chat_response("NO") for _ in range(5)]
        panel._client.post = AsyncMock(side_effect=responses)

        verdict = await panel.evaluate("Is this valid?", "context")
        assert verdict.accepted is False
        assert verdict.yes_count == 0

    @pytest.mark.asyncio
    async def test_error_makes_round_unavailable(self) -> None:
        """Infrastructure errors must not become semantic NO votes."""
        settings = _make_settings()
        panel = EvaluatorPanel(settings)
        panel._client = MagicMock(spec=httpx.AsyncClient)
        responses = [
            _mock_chat_response("YES"), _mock_chat_response("YES"),
            _mock_chat_response("YES"), httpx.ConnectError("Connection refused"),
            httpx.ConnectError("Timeout"),
        ]
        panel._client.post = AsyncMock(side_effect=responses)
        with pytest.raises(PanelUnavailableError, match="incomplete panel round"):
            await panel.evaluate("Is this valid?", "context")

    @pytest.mark.asyncio
    async def test_votes_recorded(self) -> None:
        """Individual votes should be recorded in the verdict."""
        settings = _make_settings()
        panel = EvaluatorPanel(settings)
        panel._client = MagicMock(spec=httpx.AsyncClient)

        responses = [_mock_chat_response("YES") for _ in range(5)]
        panel._client.post = AsyncMock(side_effect=responses)

        verdict = await panel.evaluate("prompt", "context")

        assert len(verdict.votes) == 5
        assert all(v.parsed for v in verdict.votes)
        assert all(v.model in ["model-a", "model-b", "model-c", "model-d", "model-e"]
                    for v in verdict.votes)


class TestParseYesNo:
    """Tests for EvaluatorPanel._parse_yes_no()."""

    def test_explicit_yes(self) -> None:
        """Explicit YES → True."""
        assert EvaluatorPanel._parse_yes_no("YES") is True
        assert EvaluatorPanel._parse_yes_no("yes") is True
        assert EvaluatorPanel._parse_yes_no("Yes") is True

    def test_explicit_no(self) -> None:
        """Explicit NO → False."""
        assert EvaluatorPanel._parse_yes_no("NO") is False
        assert EvaluatorPanel._parse_yes_no("no") is False
        assert EvaluatorPanel._parse_yes_no("No") is False

    def test_yes_in_sentence(self) -> None:
        """YES embedded in text → True."""
        assert EvaluatorPanel._parse_yes_no("Yes, this is correct.") is True

    def test_no_in_sentence(self) -> None:
        """NO embedded in text → False."""
        assert EvaluatorPanel._parse_yes_no("No, this is wrong.") is False

    def test_ambiguous(self) -> None:
        """Ambiguous text is unavailable, not a semantic NO."""
        assert EvaluatorPanel._parse_yes_no("Maybe") is None
        assert EvaluatorPanel._parse_yes_no("") is None
        assert EvaluatorPanel._parse_yes_no("I.m not sure") is None


# ---------------------------------------------------------------------------
# DatasetGenerator tests
# ---------------------------------------------------------------------------


class TestDatasetGenerator:
    """Tests for the DatasetGenerator JSONL parsing."""

    def test_parse_jsonl_simple(self) -> None:
        """Parse simple JSONL output."""
        text = '{"question": "Apa itu Statuta UPI?", "answer": "Peraturan dasar UPI"}\n{"question": "Apa itu MWA?", "answer": "Majelis Wali Amanat"}'
        items = DatasetGenerator._parse_jsonl(text)

        assert len(items) == 2
        assert isinstance(items[0].parsed, dict)
        assert items[0].parsed["question"] == "Apa itu Statuta UPI?"
        assert items[1].parsed["answer"] == "Majelis Wali Amanat"

    def test_parse_jsonl_with_markdown_fences(self) -> None:
        """Parse JSONL wrapped in markdown code fences."""
        text = '```json\n{"question": "Apa itu Statuta UPI?"}\n{"question": "Apa itu MWA?"}\n```'
        items = DatasetGenerator._parse_jsonl(text)

        assert len(items) == 2
        assert items[0].parsed["question"] == "Apa itu Statuta UPI?"

    def test_parse_jsonl_empty_lines(self) -> None:
        """Empty lines should be skipped."""
        text = '{"q": "1"}\n\n{"q": "2"}\n'
        items = DatasetGenerator._parse_jsonl(text)

        assert len(items) == 2

    def test_parse_jsonl_invalid_json_fallback(self) -> None:
        """Invalid JSON lines should be kept as raw strings."""
        text = '{"q": "1"}\nnot json at all\n{"q": "3"}'
        items = DatasetGenerator._parse_jsonl(text)

        assert len(items) == 3
        assert items[0].parsed == {"q": "1"}
        assert items[1].parsed == "not json at all"
        assert items[2].parsed == {"q": "3"}

    def test_parse_jsonl_empty(self) -> None:
        """Empty text → empty list."""
        items = DatasetGenerator._parse_jsonl("")
        assert items == []

    def test_parse_jsonl_extract_json_from_line(self) -> None:
        """Extract JSON object from a line with surrounding text."""
        text = 'The answer is: {"q": "test"} done'
        items = DatasetGenerator._parse_jsonl(text)

        assert len(items) == 1
        assert items[0].parsed == {"q": "test"}

    @pytest.mark.asyncio
    async def test_generate_calls_api(self) -> None:
        """generate() should call the OpenRouter API and parse results."""
        settings = _make_settings()
        gen = DatasetGenerator(settings)
        gen._client = MagicMock(spec=httpx.AsyncClient)

        jsonl_output = '{"question": "Q1"}\n{"question": "Q2"}'
        gen._client.post = AsyncMock(return_value=_mock_chat_response(jsonl_output))

        items = await gen.generate("Generate questions", count=2)

        assert len(items) == 2
        assert items[0].parsed["question"] == "Q1"
        gen._client.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_single(self) -> None:
        """generate_single() should return raw text."""
        settings = _make_settings()
        gen = DatasetGenerator(settings)
        gen._client = MagicMock(spec=httpx.AsyncClient)
        gen._client.post = AsyncMock(return_value=_mock_chat_response("raw response"))

        result = await gen.generate_single("Tell me about Statuta UPI")
        assert result == "raw response"


# ---------------------------------------------------------------------------
# Subset A document-coverage tests
# ---------------------------------------------------------------------------


class TestSubsetADocumentCoverage:
    """Subset A must spread its questions across the corpus.

    The committed data/subset_a.csv drew 98 questions from 11 of 309 documents
    because the builder walked the KB API's document order and stopped as soon
    as the per-category targets filled. These tests pin the seeded shuffle and
    the per-document cap that replaced it.
    """

    @staticmethod
    async def _run(
        tmp_path: Any,
        n_docs: int = 60,
        count: int = 30,
        max_items_per_doc: int = 2,
        seed: int = 42,
    ) -> List[Dict[str, str]]:
        """Run build_subset_a against stub KB/generator/panel, return the rows."""
        import csv as _csv

        from evals._dataset_gen import build_subset_a as mod

        docs = [{"id": f"doc-{i:03d}", "title": f"Dokumen {i}"} for i in range(n_docs)]
        counter = {"n": 0}

        async def fake_generate(seed_prompt: str, count: int, system_prompt: str = "") -> List[Any]:
            items = []
            for _ in range(count):
                counter["n"] += 1
                items.append(
                    GeneratedItem(
                        raw="{}",
                        parsed={
                            "question": f"Pertanyaan {counter['n']}?",
                            "ground_truth_answer": "Jawaban",
                            "source_context": "Konteks",
                        },
                    )
                )
            return items

        out = tmp_path / "subset_a.csv"
        with (
            patch.object(mod, "fetch_kb_documents", AsyncMock(return_value=docs)),
            patch.object(mod, "fetch_document_text", AsyncMock(return_value="Isi dokumen.")),
            patch.object(mod.DatasetGenerator, "generate", AsyncMock(side_effect=fake_generate)),
            patch.object(mod.DatasetGenerator, "aclose", AsyncMock()),
            # Every candidate accepted, so coverage is decided purely by the
            # loop's document selection rather than by panel randomness.
            patch.object(
                mod.EvaluatorPanel,
                "evaluate",
                AsyncMock(
                    return_value=PanelVerdict(
                        votes=["YES"] * 5,
                        yes_count=5,
                        no_count=0,
                        accepted=True,
                        acceptance_threshold=4,
                    )
                ),
            ),
            patch.object(mod.EvaluatorPanel, "aclose", AsyncMock()),
        ):
            await mod.build_subset_a(
                _make_settings(),
                "http://localhost:8000",
                str(out),
                count,
                seed=seed,
                max_items_per_doc=max_items_per_doc,
            )

        with out.open(encoding="utf-8") as f:
            return list(_csv.DictReader(f))

    @pytest.mark.asyncio
    async def test_spreads_across_many_documents(self, tmp_path: Any) -> None:
        """Doc-bound rows must respect the per-document cap, not cluster."""
        rows = await self._run(tmp_path, n_docs=60, count=30, max_items_per_doc=2)

        doc_bound = [r for r in rows if r["source_doc_id"] != "NONE"]
        per_doc: Dict[str, int] = {}
        for r in doc_bound:
            per_doc[r["source_doc_id"]] = per_doc.get(r["source_doc_id"], 0) + 1

        assert doc_bound, "expected at least some doc-bound rows"
        assert max(per_doc.values()) <= 2, f"per-doc cap violated: {per_doc}"
        # With a cap of 2, N doc-bound rows need at least ceil(N/2) documents.
        assert len(per_doc) >= (len(doc_bound) + 1) // 2

    @pytest.mark.asyncio
    async def test_seed_changes_document_selection(self, tmp_path: Any) -> None:
        """The shuffle must actually depend on the seed (and be deterministic)."""
        rows_a = await self._run(tmp_path / "a", seed=1)
        rows_b = await self._run(tmp_path / "b", seed=2)
        rows_a2 = await self._run(tmp_path / "a2", seed=1)

        docs_a = {r["source_doc_id"] for r in rows_a if r["source_doc_id"] != "NONE"}
        docs_b = {r["source_doc_id"] for r in rows_b if r["source_doc_id"] != "NONE"}
        docs_a2 = {r["source_doc_id"] for r in rows_a2 if r["source_doc_id"] != "NONE"}

        assert docs_a == docs_a2, "same seed must select the same documents"
        assert docs_a != docs_b, "different seeds must select different documents"

    @pytest.mark.asyncio
    async def test_small_corpus_still_reaches_target(self, tmp_path: Any) -> None:
        """A corpus too small for the cap must loosen it, not stall short.

        Compared against a corpus large enough that the cap never binds, rather
        than against ``count``: the per-category targets are scaled with int()
        truncation, so they sum to slightly under ``count`` (this is why
        ``--count 100`` yields 98 rows).
        """
        small = await self._run(tmp_path / "small", n_docs=3, count=20, max_items_per_doc=2)
        roomy = await self._run(tmp_path / "roomy", n_docs=60, count=20, max_items_per_doc=2)

        assert len(small) == len(roomy)
        # 3 docs x cap 2 = 6 doc-bound slots on the first pass, so reaching the
        # target at all proves the cap loosened on later passes.
        assert sum(1 for r in small if r["source_doc_id"] != "NONE") > 6


# ---------------------------------------------------------------------------
# Provenance sidecar tests
# ---------------------------------------------------------------------------


class TestProvenance:
    """Every generated subset must record how it was generated.

    Without this, which panel validated a given CSV can only be asserted from
    memory — there are no dataset-generation logs and no metadata columns.
    """

    def test_records_panel_and_generator(self, tmp_path: Any) -> None:
        from evals._dataset_gen.provenance import write_provenance

        settings = _make_settings(panel_models="m1,m2,m3", threshold=2)
        csv_path = tmp_path / "subset_x.csv"

        sidecar = write_provenance(str(csv_path), subset="x", settings=settings, row_count=7)

        assert sidecar == tmp_path / "subset_x.meta.json"
        meta = json.loads(sidecar.read_text(encoding="utf-8"))
        assert meta["subset"] == "x"
        assert meta["row_count"] == 7
        assert meta["panel"]["models"] == ["m1", "m2", "m3"]
        assert meta["panel"]["size"] == 3
        assert meta["panel"]["acceptance_threshold"] == 2
        assert meta["generator"]["model"] == "deepseek/deepseek-chat"
        assert meta["generated_at_utc"]

    def test_extra_fields_are_merged(self, tmp_path: Any) -> None:
        from evals._dataset_gen.provenance import write_provenance

        sidecar = write_provenance(
            str(tmp_path / "subset_a.csv"),
            subset="a",
            settings=_make_settings(),
            row_count=98,
            extra={"seed": 7, "distinct_source_documents": 43},
        )
        meta = json.loads(sidecar.read_text(encoding="utf-8"))
        assert meta["seed"] == 7
        assert meta["distinct_source_documents"] == 43


# ---------------------------------------------------------------------------
# Panel payload tests
# ---------------------------------------------------------------------------


class TestPanelPayload:
    """The panel request payload carries the reproducibility guarantees."""

    def test_provider_pinned_by_default(self) -> None:
        """One slug can be served by several providers at different
        quantizations, so temperature=0.0 alone does not pin the rater."""
        panel = EvaluatorPanel(_make_settings())
        payload = panel._build_payload("m", [{"role": "user", "content": "x"}], 10)

        assert payload["provider"] == {"allow_fallbacks": False}

    def test_provider_order_included_when_configured(self) -> None:
        settings = _make_settings()
        settings.panel_provider_order = "DeepInfra, Together"
        panel = EvaluatorPanel(settings)
        payload = panel._build_payload("m", [{"role": "user", "content": "x"}], 10)

        assert payload["provider"]["order"] == ["DeepInfra", "Together"]

    def test_session_id_is_not_sent(self) -> None:
        """Measured as a no-op: caching is automatic and prefix-driven.

        Verified 2026-07-22 with preflight --check-session-id (3/6 cache hits
        with it, 2/6 without, session_id condition run first so it could not
        inherit a warm cache). Sending it implied a mechanism that does not
        exist; the parameter is still accepted at the call sites.
        """
        panel = EvaluatorPanel(_make_settings())
        payload = panel._build_payload(
            "m", [{"role": "user", "content": "x"}], 10, session_id="anything"
        )

        assert "session_id" not in payload

    def test_reasoning_disabled_unless_mandatory(self) -> None:
        panel = EvaluatorPanel(_make_settings())
        assert panel._build_payload("m", [], 10)["reasoning"] == {"enabled": False}

        panel._reasoning_mandatory = {"m"}
        assert "reasoning" not in panel._build_payload("m", [], 10)


# ---------------------------------------------------------------------------
# Subset C two-pass acceptance
# ---------------------------------------------------------------------------


class TestSubsetCTwoPassAcceptance:
    """Panel disagreement on a boundary query is signal, not noise.

    near_miss_government is defined as sitting near the domain edge, so a
    strict >=4/5 rule fights the subtype's own definition — it previously
    delivered 6 rows against a target of 10 by exhausting its retry budget.
    The two-pass rule keeps strict acceptance wherever strict acceptance is
    achievable and only relaxes for a subtype that genuinely cannot fill.
    """

    @staticmethod
    async def _run(tmp_path: Any, yes_counts: List[int], target: int = 2) -> List[Dict[str, str]]:
        """Run build_subset_c with a stubbed panel returning `yes_counts` in order."""
        import csv as _csv

        from evals._dataset_gen import build_subset_c as mod

        counter = {"n": 0}

        async def fake_generate(seed_prompt: str, count: int, system_prompt: str = "") -> List[Any]:
            items = []
            for _ in range(count):
                counter["n"] += 1
                items.append(
                    GeneratedItem(
                        raw="{}",
                        parsed={
                            "query": f"Kueri {counter['n']}?",
                            "label": "out_of_domain",
                            "subtype": "near_miss_government",
                        },
                    )
                )
            return items

        seq = list(yes_counts)

        async def fake_evaluate(*args: Any, **kwargs: Any) -> PanelVerdict:
            # Hold the last scripted value once the script runs out, so a test
            # that scripts "always 2/5" stays "always 2/5" for the whole budget.
            yes = seq.pop(0) if len(seq) > 1 else (seq[0] if seq else 5)
            return PanelVerdict(
                votes=["v"] * 5, yes_count=yes, no_count=5 - yes,
                accepted=yes >= 4, acceptance_threshold=4,
            )

        out = tmp_path / "subset_c.csv"
        single = [("near_miss_government", "out_of_domain", target)]
        with (
            patch.object(mod, "SUBTYPES", single),
            patch.object(mod.DatasetGenerator, "generate", AsyncMock(side_effect=fake_generate)),
            patch.object(mod.DatasetGenerator, "aclose", AsyncMock()),
            patch.object(mod.EvaluatorPanel, "evaluate", AsyncMock(side_effect=fake_evaluate)),
            patch.object(mod.EvaluatorPanel, "aclose", AsyncMock()),
        ):
            await mod.build_subset_c(_make_settings(), str(out), target)

        with out.open(encoding="utf-8") as f:
            return list(_csv.DictReader(f))

    @pytest.mark.asyncio
    async def test_strict_items_accepted_and_marked(self, tmp_path: Any) -> None:
        rows = await self._run(tmp_path, yes_counts=[5, 4], target=2)

        assert len(rows) == 2
        assert [int(r["panel_yes"]) for r in rows] == [5, 4]
        assert all(int(r["panel_size"]) == 5 for r in rows)

    @pytest.mark.asyncio
    async def test_three_of_five_rejected_during_strict_pass(self, tmp_path: Any) -> None:
        """A 3/5 item must NOT be admitted while strict batches remain."""
        # First 12 batches are strict (25 // 2); feed 3/5 throughout those,
        # then 5/5 once the relaxed pass begins.
        rows = await self._run(tmp_path, yes_counts=[3] * 12 + [5, 5], target=2)

        assert all(int(r["panel_yes"]) >= 4 for r in rows), (
            "a 3/5 item was admitted before the strict budget was exhausted"
        )

    @pytest.mark.asyncio
    async def test_three_of_five_never_enters_principal_set(self, tmp_path: Any) -> None:
        """A 3/5 disagreement may be audited but is never principal data."""
        rows = await self._run(tmp_path, yes_counts=[3] * 40, target=2)

        assert rows == []

    @pytest.mark.asyncio
    async def test_two_of_five_never_admitted(self, tmp_path: Any) -> None:
        """Relaxing by one vote must not become 'accept anything'."""
        rows = await self._run(tmp_path, yes_counts=[2] * 40, target=2)

        assert rows == []


class TestSubsetCLoaderAgreement:
    """panel_yes must round-trip, and older CSVs must still load."""

    def test_contested_predicate(self) -> None:
        from evals._shared.dataset import SubsetCRow

        strict = SubsetCRow(query="q", label="in_domain", subtype="s", panel_yes=4, panel_size=5)
        contested = SubsetCRow(query="q", label="in_domain", subtype="s", panel_yes=3, panel_size=5)
        legacy = SubsetCRow(query="q", label="in_domain", subtype="s")

        assert strict.is_contested() is False
        assert contested.is_contested() is True
        # A CSV predating the column can only have held strictly-accepted rows,
        # so absence must not read as "contested".
        assert legacy.is_contested() is False

    def test_loader_reads_and_defaults(self, tmp_path: Any) -> None:
        from evals._shared.dataset import load_subset_c

        new = tmp_path / "new.csv"
        new.write_text(
            "query,label,subtype,panel_yes,panel_size\n"
            "Apa itu Statuta?,in_domain,direct_upi,3,5\n",
            encoding="utf-8",
        )
        old = tmp_path / "old.csv"
        old.write_text("query,label,subtype\nApa itu Statuta?,in_domain,direct_upi\n", encoding="utf-8")

        assert load_subset_c(str(new))[0].panel_yes == 3
        assert load_subset_c(str(old))[0].panel_yes == 0

    def test_compute_by_agreement_splits(self) -> None:
        from evals._shared.dataset import SubsetCRow
        from evals.exp1b_relevance.run import compute_by_agreement

        dataset = [
            SubsetCRow("q1", "in_domain", "direct_upi", 5, 5),
            SubsetCRow("q2", "in_domain", "direct_upi", 4, 5),
            SubsetCRow("q3", "out_of_domain", "near_miss_government", 3, 5),
        ]
        # Correct on both strict rows, wrong on the contested one.
        out = compute_by_agreement([True, True, True], dataset, strict_threshold=4)

        assert out["strict"].total == 2
        assert out["strict"].accuracy == 1.0
        assert out["contested"].total == 1
        assert out["contested"].accuracy == 0.0


@pytest.mark.usefixtures("no_retry_sleep")
class TestPanelTransientFailures:
    """A rate limit must not become a content decision.

    A failed call is counted as NO, so with 5 members and a threshold of 4,
    two failures cap the achievable score at 3/5 and force a rejection
    regardless of the item. That is infrastructure noise entering the dataset
    — and invisible in the CSV afterwards. A live dry run hit exactly this: a
    429 from one panel member turned into a NO vote on an otherwise unanimous
    item.
    """

    @pytest.mark.asyncio
    async def test_429_is_retried_not_counted_as_no(self) -> None:
        settings = _make_settings()
        panel = EvaluatorPanel(settings)
        panel._client = MagicMock(spec=httpx.AsyncClient)

        rate_limited = MagicMock()
        rate_limited.raise_for_status = MagicMock(
            side_effect=httpx.HTTPStatusError(
                "429", request=MagicMock(), response=MagicMock(status_code=429)
            )
        )
        # First model 429s once then succeeds; the rest answer YES directly.
        panel._client.post = AsyncMock(
            side_effect=[rate_limited] + [_mock_chat_response("YES") for _ in range(5)]
        )

        verdict = await panel.evaluate("prompt", "context")

        assert verdict.yes_count == 5, "retry should have recovered the rate-limited vote"
        assert verdict.error_count == 0
        assert verdict.accepted is True

    @pytest.mark.asyncio
    async def test_persistent_failure_invalidates_round(self) -> None:
        """Errors that outlive retries invalidate rather than bias the vote."""
        settings = _make_settings()
        panel = EvaluatorPanel(settings)
        panel._client = MagicMock(spec=httpx.AsyncClient)
        panel._client.post = AsyncMock(side_effect=httpx.ConnectError("down"))

        with pytest.raises(PanelUnavailableError, match="incomplete panel round"):
            await panel.evaluate("prompt", "context")

    @pytest.mark.asyncio
    async def test_non_retryable_4xx_is_not_retried(self) -> None:
        """A 401/422 fails identically every time — retrying just wastes time."""
        settings = _make_settings()
        panel = EvaluatorPanel(settings)
        panel._client = MagicMock(spec=httpx.AsyncClient)

        bad_key = MagicMock()
        bad_key.raise_for_status = MagicMock(
            side_effect=httpx.HTTPStatusError(
                "401", request=MagicMock(), response=MagicMock(status_code=401)
            )
        )
        panel._client.post = AsyncMock(return_value=bad_key)

        with pytest.raises(PanelUnavailableError, match="incomplete panel round"):
            await panel.evaluate("prompt", "context")
        # One attempt per model, no retries.
        assert panel._client.post.await_count == 5

# ---------------------------------------------------------------------------
# Crash safety / resume
# ---------------------------------------------------------------------------


class TestCheckpointing:
    """A multi-hour run must survive an interruption.

    The builders used to hold every accepted row in memory and write the CSV
    once at the end, so any interruption discarded the whole run along with the
    API spend behind it.
    """

    def test_rows_are_on_disk_before_the_writer_closes(self, tmp_path: Any) -> None:
        from evals._dataset_gen.checkpoint import IncrementalCSVWriter

        out = tmp_path / "x.csv"
        with IncrementalCSVWriter(str(out), ["a", "b"]) as w:
            w.append({"a": "1", "b": "2"})
            # Read the file while the writer is still open — this is what a
            # crash would leave behind.
            assert out.read_text(encoding="utf-8").strip().splitlines() == ["a,b", "1,2"]

    def test_resume_appends_without_duplicating_header(self, tmp_path: Any) -> None:
        from evals._dataset_gen.checkpoint import IncrementalCSVWriter, resume_rows

        out = tmp_path / "x.csv"
        with IncrementalCSVWriter(str(out), ["a", "b"]) as w:
            w.append({"a": "1", "b": "2"})

        existing = resume_rows(str(out), ["a", "b"])
        assert existing == [{"a": "1", "b": "2"}]

        with IncrementalCSVWriter(str(out), ["a", "b"], resume=True) as w:
            w.append({"a": "3", "b": "4"})

        assert out.read_text(encoding="utf-8").strip().splitlines() == ["a,b", "1,2", "3,4"]

    def test_without_resume_the_file_is_truncated(self, tmp_path: Any) -> None:
        """A fresh run must not inherit rows made under an older panel."""
        from evals._dataset_gen.checkpoint import IncrementalCSVWriter

        out = tmp_path / "x.csv"
        with IncrementalCSVWriter(str(out), ["a", "b"]) as w:
            w.append({"a": "old", "b": "old"})
        with IncrementalCSVWriter(str(out), ["a", "b"]) as w:
            w.append({"a": "new", "b": "new"})

        assert "old" not in out.read_text(encoding="utf-8")

    def test_schema_mismatch_is_not_silently_merged(self, tmp_path: Any) -> None:
        from evals._dataset_gen.checkpoint import resume_rows

        out = tmp_path / "x.csv"
        out.write_text("different,columns\n1,2\n", encoding="utf-8")
        assert resume_rows(str(out), ["a", "b"]) == []


@pytest.fixture
def no_retry_sleep(monkeypatch: Any) -> None:
    """Collapse the retry backoff so these tests don't sleep for real.

    The production policy waits 4s, 8s, ... between attempts, which is correct
    against a rate-limited API and unbearable in a test suite — the tests below
    deliberately exhaust retries on every panel member.
    """
    import tenacity.asyncio

    async def _instant(_seconds: float) -> None:
        return None

    # AsyncRetrying takes `sleep` as a constructor default, so the decorator
    # captured it at import time — patch the function it defaults to.
    monkeypatch.setattr(tenacity.asyncio, "_portable_async_sleep", _instant)
    for retrying in _panel_retry_objects():
        monkeypatch.setattr(retrying, "sleep", _instant)


def _panel_retry_objects() -> List[Any]:
    """Return the AsyncRetrying instances bound to the panel's retried methods."""
    from evals._dataset_gen.panel import EvaluatorPanel

    found = []
    for name in ("_evaluate_single", "_evaluate_label_single"):
        fn = getattr(EvaluatorPanel, name)
        retry_obj = getattr(fn, "retry", None)
        if retry_obj is not None:
            found.append(retry_obj)
    return found


@pytest.mark.usefixtures("no_retry_sleep")
@pytest.mark.usefixtures("no_retry_sleep")
class TestStrictPanelAvailability:
    """Every principal decision requires five successful semantic votes."""

    @pytest.mark.asyncio
    async def test_total_failure_raises_immediately(self) -> None:
        panel = EvaluatorPanel(_make_settings())
        panel._client = MagicMock(spec=httpx.AsyncClient)
        panel._client.post = AsyncMock(side_effect=httpx.ConnectError("down"))

        with pytest.raises(PanelUnavailableError, match="incomplete panel round"):
            await panel.evaluate("prompt", "context")

    @pytest.mark.asyncio
    async def test_good_round_after_failure_is_accepted(self) -> None:
        """A later retry/checkpoint round can proceed once all models recover."""
        panel = EvaluatorPanel(_make_settings())
        panel._client = MagicMock(spec=httpx.AsyncClient)

        panel._client.post = AsyncMock(side_effect=httpx.ConnectError("down"))
        with pytest.raises(PanelUnavailableError):
            await panel.evaluate("p", "c")

        panel._client.post = AsyncMock(side_effect=[_mock_chat_response("YES") for _ in range(5)])
        verdict = await panel.evaluate("p", "c")
        assert verdict.accepted is True
        assert verdict.successful_votes == 5

    @pytest.mark.asyncio
    async def test_partial_failure_invalidates_round(self) -> None:
        """A partial outage is unavailable, not a semantic disagreement."""
        panel = EvaluatorPanel(_make_settings())
        panel._client = MagicMock(spec=httpx.AsyncClient)

        panel._client.post = AsyncMock(
            side_effect=[_mock_chat_response("YES")] + [httpx.ConnectError("x")] * 4
        )
        with pytest.raises(PanelUnavailableError, match="incomplete panel round"):
            await panel.evaluate("p", "c")
