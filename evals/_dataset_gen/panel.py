"""Strict evaluator panel for versioned dataset generation.

Every configured member must return a usable vote after retries. Infrastructure
failure is never converted into a semantic NO or an empty multiclass label.
Principal rows require the configured absolute threshold (4/5 for v2).
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import httpx
import structlog

from app.shared.retry import RetryableResponseError, external_api_retry
from evals._dataset_gen.config import DatasetGenSettings
from evals._dataset_gen.cost import get_cost_ledger, usage_from_response

logger = structlog.get_logger(__name__)


class PanelUnavailableError(RuntimeError):
    """A panel round was incomplete and must be retried from its checkpoint."""


@dataclass(frozen=True)
class PanelVote:
    model: str
    vote: str
    parsed: Optional[bool]
    provider: str = ""
    error: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    cost_usd: Optional[float] = None


@dataclass(frozen=True)
class LabelVote:
    model: str
    vote: str
    label: str
    provider: str = ""
    error: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    cost_usd: Optional[float] = None


@dataclass(frozen=True)
class LabelVerdict:
    votes: List[LabelVote]
    label_counts: Dict[str, int]
    accepted_label: Optional[str]
    accepted: bool
    acceptance_threshold: int
    successful_votes: int
    contested: bool


@dataclass(frozen=True)
class PanelVerdict:
    votes: List[PanelVote]
    yes_count: int
    no_count: int
    accepted: bool
    acceptance_threshold: int
    error_count: int = 0
    successful_votes: int = 0
    contested: bool = False


class EvaluatorPanel:
    """Five independent raters with strict absolute-threshold voting."""

    def __init__(self, settings: DatasetGenSettings) -> None:
        self._settings = settings
        self._client = httpx.AsyncClient(
            base_url=settings.openrouter_base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {settings.openrouter_api_key}"},
            timeout=httpx.Timeout(120.0, connect=10.0),
        )
        self._models = settings.panel_model_list
        self._threshold = settings.acceptance_threshold
        if len(self._models) != 5 or self._threshold != 4:
            raise ValueError(
                "v2 principal datasets require exactly five panel models and a 4/5 threshold"
            )
        self._reasoning_mandatory: set[str] = set()
        self._ledger = get_cost_ledger(settings.cost_ledger_path, settings.budget_usd)

    def _build_payload(
        self,
        model: str,
        messages: List[Dict[str, str]],
        max_tokens: int,
        session_id: Optional[str] = None,
    ) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "model": model,
            "messages": messages,
            "temperature": self._settings.panel_temperature,
            "max_tokens": max_tokens,
            "usage": {"include": True},
        }
        if model not in self._reasoning_mandatory:
            payload["reasoning"] = {"enabled": False}
        provider: Dict[str, object] = {
            "allow_fallbacks": self._settings.panel_allow_fallbacks
        }
        order = [
            p.strip()
            for p in self._settings.panel_provider_order.split(",")
            if p.strip()
        ]
        if order:
            provider["order"] = order
        payload["provider"] = provider
        return payload

    async def evaluate(
        self,
        prompt: str,
        context: str = "",
        session_id: Optional[str] = None,
    ) -> PanelVerdict:
        results = await asyncio.gather(
            *(self._evaluate_single(m, prompt, context, session_id) for m in self._models),
            return_exceptions=True,
        )
        votes: List[PanelVote] = []
        for model, result in zip(self._models, results):
            if isinstance(result, BaseException):
                votes.append(PanelVote(model=model, vote="", parsed=None, error=str(result)))
            else:
                votes.append(result)
        unavailable = [v for v in votes if v.parsed is None]
        if unavailable:
            models = ", ".join(v.model for v in unavailable)
            raise PanelUnavailableError(
                f"incomplete panel round ({len(unavailable)}/5 unavailable: {models}); "
                "checkpoint and retry the item"
            )
        yes_count = sum(v.parsed is True for v in votes)
        no_count = sum(v.parsed is False for v in votes)
        accepted = yes_count >= self._threshold
        return PanelVerdict(
            votes=votes,
            yes_count=yes_count,
            no_count=no_count,
            accepted=accepted,
            acceptance_threshold=self._threshold,
            successful_votes=len(votes),
            contested=(not accepted and yes_count == self._threshold - 1),
        )

    @external_api_retry
    async def _evaluate_single(
        self,
        model: str,
        prompt: str,
        context: str,
        session_id: Optional[str] = None,
    ) -> PanelVote:
        messages = [
            {
                "role": "system",
                "content": "You are an evaluator. Answer with ONLY 'YES' or 'NO'.",
            },
            {"role": "user", "content": f"{prompt}\n\n{context}"},
        ]
        if self._ledger:
            await self._ledger.assert_available()
        response = await self._client.post(
            "/chat/completions",
            json=self._build_payload(model, messages, max_tokens=32, session_id=session_id),
        )
        response.raise_for_status()
        data = response.json()
        if self._ledger:
            await self._ledger.record(phase="panel_binary", model=model, response_data=data)
        vote_text = self._extract_content(data)
        parsed = self._parse_yes_no(vote_text)
        usage = usage_from_response(data)
        if parsed is None:
            raise RetryableResponseError(f"unparseable YES/NO response: {vote_text[:80]!r}")
        return PanelVote(
            model=model,
            vote=vote_text,
            parsed=parsed,
            error="" if parsed is not None else "unparseable YES/NO response",
            **{k: usage[k] for k in (
                "provider", "prompt_tokens", "completion_tokens",
                "reasoning_tokens", "cost_usd"
            )},
        )

    @staticmethod
    def _parse_yes_no(text: str) -> Optional[bool]:
        cleaned = text.upper().strip()
        has_yes = bool(re.search(r"\bYES\b", cleaned))
        has_no = bool(re.search(r"\bNO\b", cleaned))
        if has_yes == has_no:
            return None
        return has_yes

    async def evaluate_label(
        self,
        prompt: str,
        context: str,
        valid_labels: List[str],
        session_id: Optional[str] = None,
    ) -> LabelVerdict:
        results = await asyncio.gather(
            *(
                self._evaluate_label_single(m, prompt, context, valid_labels, session_id)
                for m in self._models
            ),
            return_exceptions=True,
        )
        votes: List[LabelVote] = []
        for model, result in zip(self._models, results):
            if isinstance(result, BaseException):
                votes.append(LabelVote(model=model, vote="", label="", error=str(result)))
            else:
                votes.append(result)
        unavailable = [v for v in votes if not v.label]
        if unavailable:
            models = ", ".join(v.model for v in unavailable)
            raise PanelUnavailableError(
                f"incomplete multiclass panel round ({len(unavailable)}/5 unavailable: "
                f"{models}); checkpoint and retry the item"
            )
        counts: Dict[str, int] = {}
        for vote in votes:
            counts[vote.label] = counts.get(vote.label, 0) + 1
        top_label, top_count = max(counts.items(), key=lambda item: item[1])
        accepted = top_count >= self._threshold
        return LabelVerdict(
            votes=votes,
            label_counts=counts,
            accepted_label=top_label if accepted else None,
            accepted=accepted,
            acceptance_threshold=self._threshold,
            successful_votes=len(votes),
            contested=(not accepted and top_count == self._threshold - 1),
        )

    @external_api_retry
    async def _evaluate_label_single(
        self,
        model: str,
        prompt: str,
        context: str,
        valid_labels: List[str],
        session_id: Optional[str] = None,
    ) -> LabelVote:
        labels = ", ".join(valid_labels)
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an evaluator. Assign exactly one label from: "
                    f"{labels}. Respond with ONLY the label name."
                ),
            },
            {"role": "user", "content": f"{prompt}\n\n{context}"},
        ]
        if self._ledger:
            await self._ledger.assert_available()
        response = await self._client.post(
            "/chat/completions",
            json=self._build_payload(model, messages, max_tokens=32, session_id=session_id),
        )
        response.raise_for_status()
        data = response.json()
        if self._ledger:
            await self._ledger.record(phase="panel_multiclass", model=model, response_data=data)
        vote_text = self._extract_content(data)
        label = self._parse_label(vote_text, valid_labels)
        usage = usage_from_response(data)
        if not label:
            raise RetryableResponseError(f"unparseable label response: {vote_text[:80]!r}")
        return LabelVote(
            model=model,
            vote=vote_text,
            label=label,
            error="" if label else "unparseable label response",
            **{k: usage[k] for k in (
                "provider", "prompt_tokens", "completion_tokens",
                "reasoning_tokens", "cost_usd"
            )},
        )

    @staticmethod
    def _parse_label(text: str, valid_labels: List[str]) -> str:
        cleaned = re.sub(r'["\'.!,;:]', "", text.lower().strip()).strip()
        matches = [
            label.lower().strip()
            for label in valid_labels
            if cleaned == label.lower().strip()
        ]
        return matches[0] if len(matches) == 1 else ""

    @staticmethod
    def _extract_content(data: Dict[str, Any]) -> str:
        message = (data.get("choices") or [{}])[0].get("message") or {}
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content.strip()
        return ""

    async def aclose(self) -> None:
        await self._client.aclose()
