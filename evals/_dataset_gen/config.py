"""Configuration for the dataset generation pipeline.

Generator–Evaluator architecture:
    - Generator: DeepSeek — cheap, strong instruction-following
    - Panel: 5 distinct-lab models (generator excluded to avoid self-eval
      bias), all verified live on OpenRouter.
    - Temperature: 0.0 (deterministic), with upstream provider pinned — see
      ``panel_allow_fallbacks``, since temperature alone is not enough for
      reproducibility when a slug is served by several providers.
    - Acceptance threshold: ≥4/5 majority vote

Model slugs must resolve on OpenRouter; an invalid slug fails closed (panel
votes NO) and silently rejects every candidate. Run
``python -m evals._dataset_gen.preflight`` before a full run — it
resolves slugs, checks each model votes correctly in both directions, and
reports which upstream provider served each call.

Avoid the "flash-lite"/smallest tier of a model family for judging — these
can give bare unexplained "NO" votes or exhaust their token budget on hidden
reasoning and return empty content (which fails closed as NO).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from typing import List

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatasetGenSettings(BaseSettings):
    """Settings for the dataset generation pipeline.

    All values can be overridden via environment variables with prefix
    ``DATAGEN_`` (e.g., ``DATAGEN_OPENROUTER_API_KEY``).
    """

    model_config = SettingsConfigDict(env_file=".env", env_prefix="DATAGEN_", extra="ignore")

    # --- API ---
    openrouter_api_key: str = Field(
        default="",
        description="OpenRouter API key for LLM access",
    )
    openrouter_base_url: str = Field(
        default="https://openrouter.ai/api/v1",
        description="OpenRouter API base URL",
    )

    # --- Generator ---
    generator_model: str = Field(
        default="deepseek/deepseek-v4-pro",
        description="Model for draft generation (DeepSeek). Kept out of the panel.",
    )
    generator_temperature: float = Field(
        default=0.0,
        description="Temperature for generation (0.0 = deterministic)",
    )

    # --- Evaluator Panel ---
    # 5 distinct-lab models. Panel size stays 5 with a majority-of-4
    # threshold; cost is negligible at this scale (the whole A-C regeneration
    # is well under $1). Blended input price is ~$0.36/M. Since each panel
    # call's output is a single label word, INPUT price dominates cost.
    #
    # Position carries no meaning: EvaluatorPanel.evaluate gathers all five
    # votes concurrently and counts them equally.
    #
    # Generator (deepseek) is intentionally excluded to avoid self-evaluation
    # bias.
    #
    # Caveats: avoid gemini flash-**lite** for judging (unreliable — bare
    # unexplained "NO" votes / empty responses from an exhausted reasoning
    # budget), and avoid ':free' variants, which are separately rate-limited.
    # Any model that rejects this panel's ``reasoning: {enabled: false}``
    # request errors into a fail-closed NO vote on every call, so run
    # ``python -m evals._dataset_gen.preflight`` before a full
    # generation.
    panel_models: str = Field(
        default="google/gemini-2.5-flash,meta-llama/llama-3.3-70b-instruct,qwen/qwen3.7-plus,z-ai/glm-5.2,xiaomi/mimo-v2.5",
        description=(
            "Comma-separated list of 5 distinct-lab panel models for majority "
            "voting. Generator (deepseek) intentionally excluded."
        ),
    )
    panel_provider_order: str = Field(
        default="",
        description=(
            "Optional comma-separated OpenRouter provider preference order for "
            "panel calls (e.g. 'DeepInfra,Together'). Empty = let OpenRouter "
            "choose, but still pin the choice per run via allow_fallbacks."
        ),
    )
    panel_allow_fallbacks: bool = Field(
        default=False,
        description=(
            "Whether OpenRouter may fall back to another upstream provider. "
            "False by default: one slug can be served by several providers at "
            "different quantizations, so temperature=0.0 alone does NOT make "
            "the panel reproducible — a mid-run reroute silently changes the "
            "rater. Set True only if pinning causes availability failures."
        ),
    )
    panel_temperature: float = Field(
        default=0.0,
        description="Temperature for panel evaluation (0.0 = deterministic)",
    )
    acceptance_threshold: int = Field(
        default=4,
        description=(
            "Minimum votes out of 5 to accept an item (≥4/5 = accept) — "
            "MUST move in lockstep with panel size. A threshold at or above "
            "the panel size means no label can ever be accepted."
        ),
    )

    # --- Output ---
    output_dir: str = Field(
        default="data",
        description="Directory for generated CSV files",
    )

    # --- Limits ---
    max_samples: int = Field(
        default=0,
        description="Maximum samples to generate (0 = no limit). For cost control.",
    )

    budget_usd: float = Field(
        default=5.75,
        description="Hard billed-cost ceiling for dataset generation; 0 disables it.",
    )
    cost_ledger_path: str = Field(
        default="evals/data/costs/dataset_generation.jsonl",
        description="Append-only usage ledger containing no prompts or credentials.",
    )

    @property
    def panel_model_list(self) -> List[str]:
        """Parse panel_models into a list of model identifiers."""
        return [m.strip() for m in self.panel_models.split(",") if m.strip()]


@lru_cache
def get_dataset_gen_settings() -> DatasetGenSettings:
    """Return the cached DatasetGenSettings singleton."""
    return DatasetGenSettings()
