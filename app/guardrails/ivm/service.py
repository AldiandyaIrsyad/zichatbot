"""Service layer for the Input Validation Module (IVM).

Checks prompts for malicious content (prompt injection / jailbreak detection).
Relevance/OOD checking lives in ``app/thesis/ivm/relevance_service.py`` — the
two concerns have independent dependencies and are composed separately at the
DI layer.

``IVMService`` is a plain application service depending only on the
``ISafetyModel`` Protocol, so it stays in the infra-free ``thesis`` core. Wired
in ``app/chat/dependency.py::get_ivm_service``.
"""
from .interfaces import ISafetyModel

import structlog

logger = structlog.get_logger(__name__)


class IVMException(Exception):
    """Base exception for IVM validation failures."""
    pass


class MaliciousPromptException(IVMException):
    """Raised when a prompt fails the safety check."""
    pass


class IVMService:
    """Input Validation Module (IVM) — prompt safety checks."""

    def __init__(
        self,
        safety_model: ISafetyModel,
    ):
        """``safety_model`` is the prompt-injection classifier backend
        (typically ``app/chat/infra/prompt_guard_client.py::PromptGuardClient``).
        """
        self.safety_model = safety_model

    async def check_malicious(self, query: str) -> None:
        """Validate the query against the safety model using a sliding window.

        Raises:
            MaliciousPromptException: The prompt is malicious or the service fails.
        """
        if not query.strip():
            return

        window_size = 512
        overlap = 50

        # Sliding window; runs at least once even for a short query.
        start = 0
        while start < len(query) or start == 0:
            end = start + window_size
            chunk = query[start:end]
            
            try:
                result = await self.safety_model.check_prompt(chunk)
            except Exception as e:
                logger.error("safety_model.error", error=str(e), exc_info=True)
                raise MaliciousPromptException("Safety check failed due to internal error.") from e
                
            if not result.is_safe:
                logger.warning(
                    "malicious_prompt_detected", 
                    message=result.message,
                    chunk_preview=chunk[:50]
                )
                raise MaliciousPromptException("Malicious prompt detected.")
                
            if end >= len(query):
                break
            start += window_size - overlap
