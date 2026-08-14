"""Dataset generator (DeepSeek by default).

Generates draft items (questions, adversarial inputs, boundary queries, RAM
sentences) from seed prompts at temperature 0.0. Drafts are validated by the
EvaluatorPanel.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import httpx
import structlog

from evals._dataset_gen.config import DatasetGenSettings
from evals._dataset_gen.cost import get_cost_ledger

logger = structlog.get_logger(__name__)


def parse_json_object(raw: str) -> Optional[Dict[str, Any]]:
    """Recover a single JSON object from a possibly-messy generator reply.

    Robust to pretty-printed multi-line objects and objects wrapped in prose
    or ```json fences. Extracts the widest brace-balanced span and parses it
    whole. Returns None if no JSON object can be recovered.
    """
    if not raw or not raw.strip():
        return None
    text = re.sub(r"```(?:json)?", "", raw).strip()
    candidates = []
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        candidates.append(match.group())
    candidates.append(text)
    for candidate in candidates:
        try:
            obj = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(obj, dict):
            return obj
    return None


@dataclass(frozen=True)
class GeneratedItem:
    """A single generated draft item (raw output + parsed data)."""

    raw: str
    parsed: Any


class DatasetGenerator:
    """LLM-based dataset generator (DeepSeek by default)."""

    def __init__(self, settings: DatasetGenSettings) -> None:
        self._settings = settings
        self._client = httpx.AsyncClient(
            base_url=settings.openrouter_base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {settings.openrouter_api_key}"},
            timeout=httpx.Timeout(120.0, connect=10.0),
        )
        # Models whose reasoning cannot be disabled (see panel.py). The default
        # generator (deepseek-chat) supports disabling reasoning, so this is
        # empty; add a slug only if OpenRouter rejects reasoning:{enabled:false}.
        self._reasoning_mandatory: set[str] = set()
        self._ledger = get_cost_ledger(settings.cost_ledger_path, settings.budget_usd)

    def _build_payload(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int,
    ) -> Dict[str, object]:
        """Build the OpenRouter request payload for the generator model.

        Conditionally includes ``reasoning: {enabled: false}`` for models
        that support disabling reasoning.
        """
        payload: Dict[str, object] = {
            "model": self._settings.generator_model,
            "messages": messages,
            "temperature": self._settings.generator_temperature,
            "max_tokens": max_tokens,
            "usage": {"include": True},
        }
        if self._settings.generator_model not in self._reasoning_mandatory:
            payload["reasoning"] = {"enabled": False}
        return payload

    async def generate(
        self,
        seed_prompt: str,
        count: int = 10,
        system_prompt: Optional[str] = None,
    ) -> List[GeneratedItem]:
        """Generate draft items from a seed prompt."""
        if system_prompt is None:
            system_prompt = (
                "You are a dataset generator for a RAG evaluation benchmark. "
                "Generate high-quality, diverse items in Indonesian. "
                "Output each item as a JSON object on its own line (JSONL format). "
                "Do not include markdown code fences."
            )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"{seed_prompt}\n\nGenerate exactly {count} items."},
        ]

        if self._ledger:
            await self._ledger.assert_available()
        response = await self._client.post(
            "/chat/completions",
            json=self._build_payload(messages, max_tokens=8192),
        )
        response.raise_for_status()
        data = response.json()
        if self._ledger:
            await self._ledger.record(
                phase="generator", model=self._settings.generator_model, response_data=data
            )
        raw_output = self._extract_content(data)

        items = self._parse_jsonl(raw_output)
        logger.info(
            "dataset_gen.generated",
            requested=count,
            parsed=len(items),
        )
        return items

    @staticmethod
    def _parse_jsonl(text: str) -> List[GeneratedItem]:
        """Parse JSONL output from the generator into GeneratedItem objects."""
        items: List[GeneratedItem] = []
        # Remove markdown code fences if present
        text = re.sub(r'^```(?:json)?\s*', '', text, flags=re.MULTILINE)
        text = re.sub(r'\s*```$', '', text, flags=re.MULTILINE)

        for line in text.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            try:
                parsed = json.loads(line)
                items.append(GeneratedItem(raw=line, parsed=parsed))
            except json.JSONDecodeError:
                # Try to extract JSON from the line
                json_match = re.search(r'\{[^}]+\}', line)
                if json_match:
                    try:
                        parsed = json.loads(json_match.group())
                        items.append(GeneratedItem(raw=line, parsed=parsed))
                    except json.JSONDecodeError:
                        items.append(GeneratedItem(raw=line, parsed=line))
                else:
                    items.append(GeneratedItem(raw=line, parsed=line))
        return items

    async def generate_single(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Generate a single text response (not JSONL)."""
        messages = [
            {"role": "system", "content": system_prompt or "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]

        if self._ledger:
            await self._ledger.assert_available()
        response = await self._client.post(
            "/chat/completions",
            json=self._build_payload(messages, max_tokens=8192),
        )
        response.raise_for_status()
        data = response.json()
        if self._ledger:
            await self._ledger.record(
                phase="generator", model=self._settings.generator_model, response_data=data
            )
        return self._extract_content(data)

    @staticmethod
    def _extract_content(data: Dict[str, Any]) -> str:
        """Extract text content from an OpenRouter chat completion response.

        Handles reasoning models that return ``content: null`` with the text
        in a ``reasoning`` field. Prefers ``content`` when both are present.
        """
        message = data.get("choices", [{}])[0].get("message", {})
        content = message.get("content")
        if content and content.strip():
            return content
        # Fall back to reasoning field (reasoning models with max_tokens too low)
        reasoning = message.get("reasoning")
        if reasoning and reasoning.strip():
            return reasoning
        return "" if content is None else (content or "")

    async def aclose(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()
