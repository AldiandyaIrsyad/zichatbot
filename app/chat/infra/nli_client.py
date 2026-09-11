"""NLI infrastructure adapter (backward-compat shim).

The RAM/IVM NLI client now lives in the NLI bounded context
(``app.guardrails.nli.infra.sequence_classify_client.SequenceClassifyNLIClient``).
This module re-exports it under the historical ``NLIClient`` name so existing
imports (``from app.chat.infra import NLIClient`` /
``from app.chat.infra.nli_client import NLIClient``) keep working unchanged.
"""

from app.guardrails.nli.infra.sequence_classify_client import (
    SequenceClassifyNLIClient,
)

NLIClient = SequenceClassifyNLIClient

__all__ = ["NLIClient"]
