"""In-process Qwen3-Reranker adapter.

Fulfills ``app/kb/domain/interfaces.py::IReranker``; wired in
``app/kb/dependency.py::get_reranker``.

Why in-process rather than Infinity (unlike ``infinity_reranker.py``): the
running Infinity image ships ``transformers 4.49.0.dev0`` and Qwen3 needs
>= 4.51.0, so the container rejects the checkpoint outright
(michaelfeil/infinity#611). The app venv is on transformers 5.15.0.

Qwen3-Reranker is not a cross-encoder with a classification head. It is a causal
LM asked a yes/no question, scored from the logits of the "yes" and "no" tokens
at the final position. That difference is why it cannot be dropped into a
sequence-classification serving path.
"""

from __future__ import annotations

import asyncio
from functools import lru_cache
from typing import List, Optional

import structlog
import torch

from app.kb.domain.interfaces import IReranker, RerankResult

logger = structlog.get_logger(__name__)

DEFAULT_RERANK_INSTRUCTION = (
    "Given a question about Indonesian institutional and legal documents, "
    "judge whether the document contains the information needed to answer it"
)

_PREFIX = (
    '<|im_start|>system\nJudge whether the Document meets the requirements '
    'based on the Query and the Instruct provided. Note that the answer can '
    'only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
)
_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


@lru_cache(maxsize=1)
def _load_model(model_name: str, device: str, use_fp16: bool):
    """Load tokenizer + causal LM once as a process-lifetime singleton, and
    resolve the "yes"/"no" token ids the scoring reads.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    dtype = torch.float16 if (use_fp16 and device.startswith("cuda")) else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
    model = model.to(device).eval()

    token_true = tokenizer.convert_tokens_to_ids("yes")
    token_false = tokenizer.convert_tokens_to_ids("no")
    prefix_ids = tokenizer.encode(_PREFIX, add_special_tokens=False)
    suffix_ids = tokenizer.encode(_SUFFIX, add_special_tokens=False)
    return tokenizer, model, token_true, token_false, prefix_ids, suffix_ids


class Qwen3Reranker(IReranker):
    """Causal-LM reranker scoring documents by P(yes) against a query."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Reranker-0.6B",
        device: str = "cuda",
        use_fp16: bool = True,
        batch_size: int = 4,
        max_length: int = 4096,
        instruction: str = DEFAULT_RERANK_INSTRUCTION,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.use_fp16 = use_fp16
        self.batch_size = batch_size
        self.max_length = max_length
        self.instruction = instruction
        # One shared model on an 8 GB GPU: serialise calls so concurrent
        # requests cannot interleave forward passes or stack peak activations.
        self._lock = asyncio.Lock()
        logger.info(
            "Qwen3Reranker initialized", model=model_name, device=device, use_fp16=use_fp16
        )

    async def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None,
    ) -> List[RerankResult]:
        """Score each document against ``query``, returning results in
        descending score order, capped to ``top_k``.
        """
        if not documents:
            return []

        try:
            async with self._lock:
                scores = await asyncio.to_thread(self._score, query, documents)
        except Exception as exc:
            # Fail-closed to the pre-rerank ordering, matching InfinityReranker:
            # a reranker outage should degrade ranking, not fail the search.
            logger.warning("rerank.failed", error=str(exc), doc_count=len(documents))
            return [RerankResult(index=i, score=0.0) for i in range(len(documents))]

        results = [RerankResult(index=i, score=s) for i, s in enumerate(scores)]
        results.sort(key=lambda r: r.score, reverse=True)
        if top_k is not None:
            results = results[:top_k]

        logger.info(
            "kb.rerank.completed",
            query_length=len(query),
            doc_count=len(documents),
            returned_count=len(results),
        )
        return results

    def _score(self, query: str, documents: List[str]) -> List[float]:
        tokenizer, model, token_true, token_false, prefix_ids, suffix_ids = _load_model(
            self.model_name, self.device, self.use_fp16
        )

        pairs = [
            f"<Instruct>: {self.instruction}\n<Query>: {query}\n<Document>: {doc}"
            for doc in documents
        ]
        # Reserve room for the fixed prefix/suffix so truncation eats the
        # document, never the instruction or the assistant turn that the
        # yes/no logits are read from.
        budget = self.max_length - len(prefix_ids) - len(suffix_ids)

        scores: List[float] = []
        for start in range(0, len(pairs), self.batch_size):
            batch = pairs[start : start + self.batch_size]
            encoded = tokenizer(
                batch,
                truncation=True,
                max_length=budget,
                return_attention_mask=False,
                add_special_tokens=False,
            )
            encoded["input_ids"] = [
                prefix_ids + ids + suffix_ids for ids in encoded["input_ids"]
            ]
            # padding=True pads to the longest sequence in the batch; passing
            # max_length here would be ignored (and warned about), since
            # truncation already happened above against `budget`.
            padded = tokenizer.pad(encoded, padding=True, return_tensors="pt")
            padded = {k: v.to(model.device) for k, v in padded.items()}

            with torch.no_grad():
                logits = model(**padded).logits[:, -1, :]
                true_logits = logits[:, token_true]
                false_logits = logits[:, token_false]
                # log_softmax over just the yes/no pair, then exp -> P(yes) in
                # [0, 1]. Calibrated across queries, unlike a raw logit.
                stacked = torch.stack([false_logits, true_logits], dim=1).float()
                probs = torch.nn.functional.log_softmax(stacked, dim=1)[:, 1].exp()

            scores.extend(probs.tolist())

        return scores

    async def close(self) -> None:
        # Process-lifetime singleton shared across instances.
        pass
