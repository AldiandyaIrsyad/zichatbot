"""Tests for HyDEExpander (app/chat/infra/hyde_expander.py).

Covers:
    - Baseline {query} substitution (unchanged behavior)
    - {kb_context} grounding filled from a mocked IKBRepository
    - Inactive PDFs excluded from the grounding list
    - TTL cache avoids re-calling get_all_pdfs() within the refresh window
    - kb_repo=None resolves {kb_context} to "" rather than leaking the placeholder
"""
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import app.chat.infra.hyde_expander as hyde_expander_module
from app.chat.infra.hyde_expander import HyDEExpander


SYSTEM_PROMPT = "Domain prompt. Docs:\n{kb_context}\n\nEnd."
PROMPT_TEMPLATE = "Pertanyaan: {query}"


def _make_llm(response: str = "Hypothetical doc.") -> MagicMock:
    llm = MagicMock()
    llm.generate = AsyncMock(return_value=response)
    return llm


def _make_pdf(title: str, description: str = "desc", active: bool = True) -> SimpleNamespace:
    return SimpleNamespace(title=title, description=description, active=active)


def _make_kb_repo(pdfs: list) -> MagicMock:
    repo = MagicMock()
    repo.get_all_pdfs = AsyncMock(return_value=pdfs)
    return repo


@pytest.fixture(autouse=True)
def _reset_kb_context_cache():
    """The grounding cache is module-level (shared across instances/requests)
    so it must be reset between tests to avoid cross-test leakage."""
    hyde_expander_module._kb_context_cache["text"] = ""
    hyde_expander_module._kb_context_cache["expires_at"] = 0.0
    yield
    hyde_expander_module._kb_context_cache["text"] = ""
    hyde_expander_module._kb_context_cache["expires_at"] = 0.0


class TestBaselineExpand:
    @pytest.mark.asyncio
    async def test_query_substitution_without_kb_repo(self) -> None:
        llm = _make_llm()
        expander = HyDEExpander(
            llm=llm,
            model="test-model",
            prompt_template=PROMPT_TEMPLATE,
            system_prompt=SYSTEM_PROMPT,
        )

        result = await expander.expand("sepeda listrik")

        assert result == "Hypothetical doc."
        messages = llm.generate.call_args.kwargs["messages"]
        assert messages[1]["content"] == "Pertanyaan: sepeda listrik"
        # No kb_repo configured — {kb_context} must resolve to "", not leak through.
        assert "{kb_context}" not in messages[0]["content"]

    @pytest.mark.asyncio
    async def test_expand_many_returns_three_variant_passages(self) -> None:
        llm = _make_llm()
        expander = HyDEExpander(
            llm=llm, model="m", prompt_template="Q: {query} / {variant_instruction}",
            system_prompt=SYSTEM_PROMPT, num_passages=3,
        )

        result = await expander.expand_many("cuti pegawai")

        assert result == ["Hypothetical doc."] * 3
        assert llm.generate.await_count == 3
        variant_prompts = [call.kwargs["messages"][1]["content"] for call in llm.generate.await_args_list]
        assert "variant 1 of 3" in variant_prompts[0]
        assert "variant 2 of 3" in variant_prompts[1]
        assert "variant 3 of 3" in variant_prompts[2]

    @pytest.mark.asyncio
    async def test_empty_query_short_circuits(self) -> None:
        llm = _make_llm()
        expander = HyDEExpander(
            llm=llm, model="m", prompt_template=PROMPT_TEMPLATE, system_prompt=SYSTEM_PROMPT
        )

        result = await expander.expand("   ")

        assert result == ""
        llm.generate.assert_not_called()


class TestKBGrounding:
    @pytest.mark.asyncio
    async def test_kb_context_filled_from_active_pdfs(self) -> None:
        repo = _make_kb_repo([
            _make_pdf("SOP Cuti", "Prosedur pengajuan cuti pegawai"),
            _make_pdf("SOP Perjalanan Dinas", "Ketentuan perjalanan dinas"),
        ])
        llm = _make_llm()
        expander = HyDEExpander(
            llm=llm,
            model="m",
            prompt_template=PROMPT_TEMPLATE,
            system_prompt=SYSTEM_PROMPT,
            kb_repo=repo,
        )

        await expander.expand("sepeda listrik")

        messages = llm.generate.call_args.kwargs["messages"]
        system_content = messages[0]["content"]
        assert "SOP Cuti" in system_content
        assert "SOP Perjalanan Dinas" in system_content
        assert "{kb_context}" not in system_content

    @pytest.mark.asyncio
    async def test_inactive_pdfs_excluded(self) -> None:
        repo = _make_kb_repo([
            _make_pdf("Active Doc", "visible", active=True),
            _make_pdf("Inactive Doc", "hidden", active=False),
        ])
        llm = _make_llm()
        expander = HyDEExpander(
            llm=llm, model="m", prompt_template=PROMPT_TEMPLATE,
            system_prompt=SYSTEM_PROMPT, kb_repo=repo,
        )

        await expander.expand("query")

        system_content = llm.generate.call_args.kwargs["messages"][0]["content"]
        assert "Active Doc" in system_content
        assert "Inactive Doc" not in system_content

    @pytest.mark.asyncio
    async def test_context_max_docs_caps_list(self) -> None:
        repo = _make_kb_repo([_make_pdf(f"Doc {i}") for i in range(5)])
        llm = _make_llm()
        expander = HyDEExpander(
            llm=llm, model="m", prompt_template=PROMPT_TEMPLATE,
            system_prompt=SYSTEM_PROMPT, kb_repo=repo, context_max_docs=2,
        )

        await expander.expand("query")

        system_content = llm.generate.call_args.kwargs["messages"][0]["content"]
        assert "Doc 0" in system_content
        assert "Doc 1" in system_content
        assert "Doc 2" not in system_content

    @pytest.mark.asyncio
    async def test_cache_avoids_repeated_db_calls_within_ttl(self) -> None:
        repo = _make_kb_repo([_make_pdf("Doc")])
        llm = _make_llm()
        expander = HyDEExpander(
            llm=llm, model="m", prompt_template=PROMPT_TEMPLATE,
            system_prompt=SYSTEM_PROMPT, kb_repo=repo, context_refresh_seconds=300,
        )

        await expander.expand("first query")
        await expander.expand("second query")

        repo.get_all_pdfs.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_kb_repo_none_resolves_context_to_empty_string(self) -> None:
        llm = _make_llm()
        expander = HyDEExpander(
            llm=llm, model="m", prompt_template=PROMPT_TEMPLATE,
            system_prompt=SYSTEM_PROMPT, kb_repo=None,
        )

        await expander.expand("query")

        system_content = llm.generate.call_args.kwargs["messages"][0]["content"]
        assert system_content == "Domain prompt. Docs:\n\n\nEnd."
