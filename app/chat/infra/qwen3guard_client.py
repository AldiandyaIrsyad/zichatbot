"""Hosted generative safety-guard infrastructure adapter.

Calls Qwen3Guard-Gen over an OpenAI-compatible chat-completions API and reduces
its generated verdict to the binary decision the IVM needs. Fulfills
``app/thesis/ivm/interfaces.py::ISafetyModel``; wired in
``app/chat/dependency.py::get_safety_model``.

Unlike ``PromptGuardClient`` (which flags only explicit instruction-override
attempts and was never evaluated on Indonesian), Qwen3Guard was trained on
native Indonesian and classifies against a content policy with a ``Jailbreak``
category — closer to this system's "input berbahaya" scope. It's too large to
deploy on the target GPU, so this adapter calls a hosted endpoint and is meant
for evaluation rather than the per-turn chat path.

Qwen3Guard-Gen emits a fixed two-line verdict::

    Safety: Unsafe
    Categories: Jailbreak

There are three tiers; ``Controversial`` is mapped to safe/unsafe by
configuration (a methodological choice fixed before an experiment runs), not by
a constant here.
"""

from __future__ import annotations

import re
from typing import List, Optional

import httpx
import structlog

from app.guardrails.ivm.interfaces import ISafetyModel, SafetyResult

logger = structlog.get_logger(__name__)

# The verdict is a fixed two-line form the model was trained to emit, so a
# regex parse is appropriate here.
SAFETY_PATTERN = re.compile(r"Safety:\s*(Safe|Unsafe|Controversial)", re.IGNORECASE)
CATEGORY_PATTERN = re.compile(r"Categories?:\s*(.+)", re.IGNORECASE)


class Qwen3GuardClient(ISafetyModel):
    """Hosted generative safety classification."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        controversial_is_unsafe: bool = True,
        timeout: float = 30.0,
    ):
        """Configure the OpenAI-compatible client. ``controversial_is_unsafe``
        sets whether the ``Controversial`` tier counts as unsafe (True matches
        the IVM's fail-closed posture). ``timeout`` is higher than the local
        classifier's since this is a network call to a generative model.
        """
        self.model = model
        self.controversial_is_unsafe = controversial_is_unsafe
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=httpx.Timeout(timeout, connect=10.0),
        )
        logger.info(
            "chat.qwen3guard.initialized",
            model=model,
            base_url=base_url,
            controversial_is_unsafe=controversial_is_unsafe,
        )

    @staticmethod
    def parse_verdict(content: str) -> tuple[Optional[str], List[str]]:
        """Extract the safety tier (None if unparseable) and category names
        from the generated verdict.
        """
        tier_match = SAFETY_PATTERN.search(content or "")
        tier = tier_match.group(1).capitalize() if tier_match else None

        categories: List[str] = []
        category_match = CATEGORY_PATTERN.search(content or "")
        if category_match:
            categories = [
                c.strip()
                for c in category_match.group(1).split(",")
                if c.strip() and c.strip().lower() != "none"
            ]
        return tier, categories

    async def check_prompt(self, text: str) -> SafetyResult:
        """Fulfills ``ISafetyModel.check_prompt``: classify ``text`` by asking a
        hosted Qwen3Guard-Gen model for a safety verdict and reducing its three
        tiers to a binary decision.

        Fails closed: a request error, empty response, or unparseable verdict
        returns ``is_safe=False`` rather than letting unchecked input through.
        An unparseable verdict counts as unsafe because on this gate a wrong
        "safe" is unbounded-cost while a wrong "unsafe" is one rejected turn.
        """
        try:
            response = await self._client.post(
                "/chat/completions",
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": text}],
                    # The verdict is two short lines; anything longer means the
                    # model has drifted off its trained output format.
                    "max_tokens": 64,
                    "temperature": 0.0,
                },
            )
            response.raise_for_status()
            data = response.json()

            choices = data.get("choices", [])
            if not choices:
                logger.warning("chat.qwen3guard.empty_response", model=self.model)
                return SafetyResult(is_safe=False, message="Service unavailable")

            content = (choices[0].get("message") or {}).get("content", "") or ""
            tier, categories = self.parse_verdict(content)

            if tier is None:
                logger.warning(
                    "chat.qwen3guard.unparseable_verdict",
                    model=self.model,
                    raw=content[:200],
                )
                return SafetyResult(is_safe=False, message="Unparseable verdict")

            unsafe = tier == "Unsafe" or (
                tier == "Controversial" and self.controversial_is_unsafe
            )
            logger.debug(
                "chat.qwen3guard.prediction",
                tier=tier,
                categories=categories,
                unsafe=unsafe,
            )

            if unsafe:
                detail = f" ({', '.join(categories)})" if categories else ""
                return SafetyResult(
                    is_safe=False,
                    message=f"Policy violation: {tier}{detail}",
                )
            return SafetyResult(is_safe=True, message=f"Safe ({tier})")

        except Exception as e:
            logger.error("chat.qwen3guard.failed", model=self.model, error=str(e))
            return SafetyResult(is_safe=False, message="Service unavailable")

    async def close(self) -> None:
        """Release the underlying ``httpx.AsyncClient`` connection pool."""
        await self._client.aclose()
