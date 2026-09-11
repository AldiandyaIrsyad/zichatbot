"""Tests for NLI endpoint wiring, ChatConfig unification, and docker-compose port binding.

Ensures all selectable NLI backends (IndoRoBERTa, mmBERT, zero-shot) resolve
to the canonical `CHAT_NLI_BASE_URL` (default http://localhost:8002) and bind
the shared host port `CHAT_NLI_PORT` (default 8002) so containers can be swapped
plug-and-play without port mismatches or silent 0.5 neutral fallbacks.
"""

from __future__ import annotations

from pathlib import Path
import pytest
import yaml

from app.chat.config import ChatConfig
from app.chat.dependency import build_spec_for_config
from app.guardrails.nli.domain.models import LabelSpace, NLIModelKind


@pytest.fixture
def code_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """Isolate the tests that assert *code* defaults from the operator's config.

    Two separate leaks, which is why both halves are needed:

    * ``ChatConfig(_env_file=None)`` stops pydantic-settings from reading ``.env``
      directly — otherwise a deployment that sets ``CHAT_NLI_BASE_URL`` or picks a
      different backend turns a defaults test red with no defect present.
    * Several ``evals`` modules call ``load_dotenv()`` at import time, which
      exports those same keys into ``os.environ``. That makes the failure
      full-suite-only: the file passes alone and fails when an evals test is
      collected first. Stripping the keys covers it.
    """
    for key in (
        "CHAT_NLI_BASE_URL",
        "CHAT_NLI_MODEL_KIND",
        "CHAT_NLI_PORT",
        "CHAT_NLI_MMBERT_MODEL",
        "CHAT_NLI_ZEROSHOT_MODEL",
    ):
        monkeypatch.delenv(key, raising=False)


class TestNLIConfig:
    """Test ChatConfig canonical NLI endpoint and backward compatibility."""

    def test_nli_base_url_default(self, code_defaults: None) -> None:
        config = ChatConfig(_env_file=None)
        assert config.nli_base_url == "http://localhost:8002"
        assert config.nli_model_kind == "mmbert"

    def test_nli_base_url_env_alias(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CHAT_NLI_BASE_URL", "http://127.0.0.1:9090")
        config = ChatConfig()
        assert config.nli_base_url == "http://127.0.0.1:9090"

    def test_legacy_endpoint_shims(self) -> None:
        config = ChatConfig(nli_base_url="http://127.0.0.1:8002")
        assert config.nli_indo_roberta_base_url == "http://127.0.0.1:8002"
        assert config.nli_mmbert_base_url == "http://127.0.0.1:8002"
        assert config.nli_zeroshot_base_url == "http://127.0.0.1:8002"


class TestBuildSpecForConfig:
    """Test build_spec_for_config uses canonical nli_base_url for all kinds."""

    @pytest.mark.parametrize(
        ("kind", "expected_kind"),
        [
            ("indo_roberta", NLIModelKind.INDO_ROBERTA),
            ("mmbert", NLIModelKind.MMBERT),
            ("zeroshot", NLIModelKind.ZEROSHOT),
        ],
    )
    def test_all_kinds_resolve_to_canonical_nli_base_url(
        self, kind: str, expected_kind: NLIModelKind
    ) -> None:
        custom_url = "http://custom-nli-server:8002"
        config = ChatConfig(nli_model_kind=kind, nli_base_url=custom_url)
        spec = build_spec_for_config(config)
        assert spec.kind == expected_kind
        assert spec.base_url == custom_url

    def test_default_spec_is_mmbert_at_default_port(self, code_defaults: None) -> None:
        # mmBERT is the default backend: IndoNLI fine-tune, macro-F1 0.763 lay /
        # 0.572 expert against indo-roberta's 0.727 / 0.554.
        config = ChatConfig(_env_file=None)
        spec = build_spec_for_config(config)
        assert spec.kind == NLIModelKind.MMBERT
        assert spec.base_url == "http://localhost:8002"
        assert spec.label_space == LabelSpace.THREE_WAY
        assert spec.model_id == "/models/mmbert_nli_id"

    def test_indo_roberta_spec_still_selectable(self, code_defaults: None) -> None:
        config = ChatConfig(_env_file=None, nli_model_kind="indo_roberta")
        spec = build_spec_for_config(config)
        assert spec.kind == NLIModelKind.INDO_ROBERTA
        assert spec.label_space == LabelSpace.THREE_WAY
        assert spec.model_id == "StevenLimcorn/indo-roberta-indonli"

    def test_mmbert_spec_preserves_token_budget_and_labels(self, code_defaults: None) -> None:
        config = ChatConfig(_env_file=None, nli_model_kind="mmbert")
        spec = build_spec_for_config(config)
        assert spec.kind == NLIModelKind.MMBERT
        assert spec.base_url == "http://localhost:8002"
        assert spec.model_id == "/models/mmbert_nli_id"
        assert spec.label_space == LabelSpace.THREE_WAY
        assert spec.max_total_tokens == 2000
        assert spec.max_hypothesis_tokens == 150

    def test_zeroshot_spec_preserves_binary_label_space(self, code_defaults: None) -> None:
        config = ChatConfig(_env_file=None, nli_model_kind="zeroshot")
        spec = build_spec_for_config(config)
        assert spec.kind == NLIModelKind.ZEROSHOT
        assert spec.base_url == "http://localhost:8002"
        assert spec.model_id == "MoritzLaurer/bge-m3-zeroshot-v2.0-c"
        assert spec.label_space == LabelSpace.BINARY


class TestComposeNLIWiring:
    """Test docker-compose.yaml publishes shared CHAT_NLI_PORT 8002 across all NLI services."""

    def test_all_nli_services_bind_same_host_port(self) -> None:
        compose_path = Path(__file__).resolve().parents[3] / "docker-compose.yaml"
        with open(compose_path, "r", encoding="utf-8") as f:
            compose_doc = yaml.safe_load(f)

        services = compose_doc.get("services", {})
        nli_services = {
            "nli-indoroberta": "nli-indoroberta",
            "nli": "nli-mmbert",
            "nli-zeroshot": "nli-zeroshot",
        }

        for service_name, expected_profile in nli_services.items():
            assert service_name in services, f"Service {service_name} missing from docker-compose.yaml"
            service_cfg = services[service_name]
            assert service_cfg.get("profiles") == [expected_profile]
            ports = service_cfg.get("ports", [])
            assert ports == ["127.0.0.1:${CHAT_NLI_PORT:-8002}:7997"], (
                f"Service {service_name} has mismatched port binding: {ports}"
            )
