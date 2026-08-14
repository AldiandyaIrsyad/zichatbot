"""Prompt injection detection infrastructure adapter.

Calls an Infinity-hosted Llama-Prompt-Guard-2-86M classifier to detect prompt
injection / jailbreak attempts in user input. Fulfills
``app/thesis/ivm/interfaces.py::ISafetyModel``; wired in
``app/chat/dependency.py::get_prompt_guard_client``.
"""

import httpx
import structlog
from typing import List, Dict, Any

from app.guardrails.ivm.interfaces import ISafetyModel, SafetyResult

logger = structlog.get_logger(__name__)


class PromptGuardClient(ISafetyModel):
    """Prompt injection detection via the Infinity HTTP server."""

    def __init__(self, base_url: str, model: str, security_threshold: float = 0.32):
        """Configure the Infinity client. ``security_threshold`` is the minimum
        malicious-class score (0-1) that flags input as unsafe.
        """
        self.model = model
        self.security_threshold = security_threshold
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=httpx.Timeout(10.0, connect=5.0),
        )
        logger.info(
            "chat.prompt_guard.initialized",
            model=model,
            base_url=base_url,
            threshold=security_threshold,
        )

    async def check_prompt(self, text: str) -> SafetyResult:
        """Classify ``text`` for prompt injection / jailbreak via Infinity's
        ``/classify``, normalizing ``LABEL_0``/``LABEL_1`` to
        ``BENIGN``/``MALICIOUS`` and flagging unsafe only when the malicious
        score meets ``security_threshold``.

        Fails closed: any request error or empty/malformed response returns
        ``is_safe=False`` rather than letting unchecked input through.
        """
        try:
            response = await self._client.post(
                "/classify",
                json={
                    "model": self.model,
                    "input": [text],
                },
            )
            response.raise_for_status()
            data = response.json()

            items = data.get("data", [])
            if not items:
                logger.warning("Empty classify response from Infinity (PromptGuard)")
                return SafetyResult(is_safe=False, message="Service unavailable")

            predictions = items[0]
            if isinstance(predictions, dict) and "results" in predictions:
                predictions = predictions["results"]
            elif isinstance(predictions, dict):
                predictions = [predictions]

            if not predictions:
                logger.warning("No valid predictions found in Infinity response.")
                return SafetyResult(is_safe=False, message="Service unavailable")

            for pred in predictions:
                label = str(pred.get("label", "")).upper()
                score = float(pred.get("score", 0.0))

                if label == "LABEL_0":
                    label = "BENIGN"
                elif label == "LABEL_1":
                    label = "MALICIOUS"

                logger.debug("chat.prompt_guard.prediction", label=label, score=score)
                
                if "BENIGN" not in label:
                    if score >= self.security_threshold:
                        return SafetyResult(
                            is_safe=False,
                            message=f"Policy violation: {label} (Score: {score:.2f} >= {self.security_threshold})",
                        )

            return SafetyResult(is_safe=True, message="Safe")

        except Exception as e:
            logger.error("chat.prompt_guard.failed", error=str(e))
            return SafetyResult(is_safe=False, message="Service unavailable")

    async def close(self) -> None:
        """Release the underlying ``httpx.AsyncClient`` connection pool."""
        await self._client.aclose()
