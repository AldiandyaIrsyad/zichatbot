"""Preflight checks for the dataset-generation panel.

Run this before any full generation run. A panel model that fails silently is
the most expensive failure mode in this pipeline: ``EvaluatorPanel`` is
fail-closed, so a slug that doesn't resolve, a model that rejects
``reasoning: {enabled: false}``, or one that returns empty content all become
a NO vote — and with a ≥4/5 threshold, one such member caps acceptance while
two reject every candidate outright.

Modes:

    # Discover slugs and prices
    python -m evals._dataset_gen.preflight --list nemotron

    # Verify every configured panel model votes correctly in both directions
    python -m evals._dataset_gen.preflight

    # Dump one complete raw response so response-shape assumptions can be
    # checked against reality
    python -m evals._dataset_gen.preflight --raw

    # Settle whether the `session_id` payload field does anything
    python -m evals._dataset_gen.preflight --check-session-id
"""

from __future__ import annotations

import argparse
import asyncio
import json
import secrets
import sys
from typing import Any, Dict, List, Optional, Tuple

import httpx

from evals._dataset_gen.config import DatasetGenSettings, get_dataset_gen_settings
from evals._dataset_gen.panel import EvaluatorPanel

# A trivially true and a trivially false proposition. Any model fit to judge
# a dataset must get both right; a model that passes only the YES case is
# indistinguishable from one that always says YES.
YES_CASE = "Pertanyaan: Apakah 2 + 2 sama dengan 4? Jawab YES jika benar."
NO_CASE = "Pertanyaan: Apakah 2 + 2 sama dengan 5? Jawab YES jika benar."


def _client(settings: DatasetGenSettings) -> httpx.AsyncClient:
    """Build an OpenRouter client from the dataset-gen settings."""
    return httpx.AsyncClient(
        base_url=settings.openrouter_base_url.rstrip("/"),
        headers={"Authorization": f"Bearer {settings.openrouter_api_key}"},
        timeout=httpx.Timeout(120.0, connect=10.0),
    )


async def list_models(settings: DatasetGenSettings, needle: str) -> None:
    """Print live OpenRouter models whose id or name matches ``needle``.

    Use this to resolve a slug from a marketing name before putting it in
    ``panel_models`` — an invalid slug fails closed and silently rejects every
    candidate.
    """
    async with _client(settings) as client:
        resp = await client.get("/models")
        resp.raise_for_status()
        models = resp.json().get("data", [])

    needle_lower = needle.lower()
    matches = [
        m for m in models
        if needle_lower in m.get("id", "").lower() or needle_lower in m.get("name", "").lower()
    ]
    if not matches:
        print(f"No models matching {needle!r} (searched {len(models)} models).")
        return

    print(f"{len(matches)} model(s) matching {needle!r}:\n")
    print(f"{'slug':<55} {'$/M in':>9} {'$/M out':>9} {'ctx':>9}")
    print("-" * 86)
    for m in sorted(matches, key=lambda x: x.get("id", "")):
        pricing = m.get("pricing", {}) or {}

        def per_m(key: str) -> str:
            raw = pricing.get(key)
            try:
                return f"{float(raw) * 1_000_000:.3f}"
            except (TypeError, ValueError):
                return "?"

        ctx = m.get("context_length") or "?"
        print(f"{m.get('id', '?'):<55} {per_m('prompt'):>9} {per_m('completion'):>9} {str(ctx):>9}")
    print(
        "\nNote: a ':free' suffix is a separate, rate-limited variant — pick the paid slug "
        "for reproducible runs."
    )


async def probe_model(
    settings: DatasetGenSettings,
    model: str,
    reasoning_disabled: bool = True,
) -> Dict[str, Any]:
    """Send the YES and NO cases to one model and report what came back.

    Re-probing with ``reasoning_disabled=False`` distinguishes "model is
    broken" from "model rejects the reasoning flag".
    """
    out: Dict[str, Any] = {"model": model, "error": None, "provider": None}

    async with _client(settings) as client:
        for case_name, case_text, expected in (("yes", YES_CASE, True), ("no", NO_CASE, False)):
            payload: Dict[str, Any] = {
                "model": model,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are an evaluator. Answer with ONLY 'YES' or 'NO'. "
                            "Be strict and precise."
                        ),
                    },
                    {"role": "user", "content": case_text},
                ],
                "temperature": settings.panel_temperature,
                "max_tokens": 1024,
                "usage": {"include": True},
            }
            if reasoning_disabled:
                payload["reasoning"] = {"enabled": False}

            try:
                resp = await client.post("/chat/completions", json=payload)
                if resp.status_code != 200:
                    out["error"] = f"HTTP {resp.status_code}: {resp.text[:200]}"
                    return out
                data = resp.json()
            except Exception as exc:  # noqa: BLE001 - reported, not raised
                out["error"] = f"{type(exc).__name__}: {exc}"
                return out

            content = EvaluatorPanel._extract_content(data)
            out["provider"] = data.get("provider") or out["provider"]
            out[f"{case_name}_raw"] = (content or "")[:80]
            out[f"{case_name}_parsed"] = EvaluatorPanel._parse_yes_no(content)
            out[f"{case_name}_ok"] = out[f"{case_name}_parsed"] is expected
            out[f"{case_name}_empty"] = not (content or "").strip()

    return out


async def run_panel_check(settings: DatasetGenSettings) -> int:
    """Probe every configured panel model; return 0 if all pass, 1 otherwise."""
    models = settings.panel_model_list
    print(f"Panel: {len(models)} model(s), acceptance threshold "
          f"{settings.acceptance_threshold}/{len(models)}\n")

    if settings.acceptance_threshold >= len(models) + 1:
        print("FATAL: acceptance_threshold exceeds panel size — nothing can ever be accepted.")
        return 1

    results = await asyncio.gather(*(probe_model(settings, m) for m in models))

    print(f"{'model':<50} {'YES':>5} {'NO':>5} {'provider':<22} note")
    print("-" * 110)
    failures: List[str] = []
    for r in results:
        model = r["model"]
        if r["error"]:
            print(f"{model:<50} {'—':>5} {'—':>5} {'—':<22} ERROR {r['error'][:40]}")
            failures.append(model)
            continue

        yes_ok, no_ok = r.get("yes_ok"), r.get("no_ok")
        note = ""
        if r.get("yes_empty") or r.get("no_empty"):
            note = "EMPTY CONTENT (reasoning budget?)"
        elif not no_ok and yes_ok:
            note = "always-YES? failed the NO case"
        elif not yes_ok and not no_ok:
            note = "always-NO — would reject every candidate"
        print(
            f"{model:<50} {'ok' if yes_ok else 'FAIL':>5} {'ok' if no_ok else 'FAIL':>5} "
            f"{str(r.get('provider') or '?'):<22} {note}"
        )
        if not (yes_ok and no_ok):
            failures.append(model)

    print()
    if failures:
        print(f"{len(failures)} model(s) failed: {', '.join(failures)}")
        print("Re-probe a failing model with --no-reasoning-flag to tell "
              "'model is broken' apart from 'model rejects reasoning:{enabled:false}' "
              "(if that is the cause, add the slug to EvaluatorPanel._reasoning_mandatory).")
        return 1

    print("All panel models resolved and voted correctly in both directions.")
    return 0


async def dump_raw(settings: DatasetGenSettings, model: Optional[str]) -> None:
    """Print one complete response body, so response-shape assumptions can be checked.

    ``EvaluatorPanel._evaluate_single`` keeps only the message content, so
    nothing in the pipeline currently observes the provider that served a call
    or whether any prompt caching occurred.
    """
    target = model or settings.panel_model_list[0]
    async with _client(settings) as client:
        resp = await client.post(
            "/chat/completions",
            json={
                "model": target,
                "messages": [{"role": "user", "content": YES_CASE}],
                "temperature": 0.0,
                "max_tokens": 64,
                "usage": {"include": True},
                "reasoning": {"enabled": False},
            },
        )
    print(f"HTTP {resp.status_code} from {target}\n")
    print(json.dumps(resp.json(), indent=2, ensure_ascii=False))
    print(
        "\nLook for: a top-level 'provider' field, and any cached-token counter inside "
        "'usage' (the exact key varies — do not assume "
        "usage.prompt_tokens_details.cached_tokens)."
    )


def _cached_tokens(data: Dict[str, Any]) -> Optional[int]:
    """Best-effort extraction of a cached-prompt-token count from a response.

    Key names differ between providers, so this searches rather than assuming
    one path. Returns None when nothing cache-shaped is present.
    """
    usage = data.get("usage") or {}
    for path in (
        ("prompt_tokens_details", "cached_tokens"),
        ("input_tokens_details", "cached_tokens"),
        ("cache_read_input_tokens",),
        ("cached_tokens",),
    ):
        node: Any = usage
        for key in path:
            node = node.get(key) if isinstance(node, dict) else None
            if node is None:
                break
        if isinstance(node, (int, float)):
            return int(node)
    return None


async def check_session_id(settings: DatasetGenSettings, model: Optional[str]) -> None:
    """Test whether the ``session_id`` payload field affects caching or routing.

    ``session_id`` is not a parameter this API is known to accept, and
    OpenRouter drops unsupported parameters silently — so it may be a no-op.
    Provider-side caching is automatic and keyed on the prompt prefix, not on
    a session. Sends a long shared prefix twice in each condition and compares
    the cached token count on the second call of each pair.
    """
    target = model or settings.panel_model_list[0]

    async def one(prefix: str, session_id: Optional[str]) -> Tuple[Optional[int], Any, Optional[str]]:
        payload: Dict[str, Any] = {
            "model": target,
            "messages": [
                {"role": "system", "content": prefix},
                {"role": "user", "content": "Jawab hanya dengan kata YES."},
            ],
            "temperature": 0.0,
            "max_tokens": 8,
            "usage": {"include": True},
            "reasoning": {"enabled": False},
        }
        if session_id:
            payload["session_id"] = session_id
        async with _client(settings) as client:
            resp = await client.post("/chat/completions", json=payload)
            if resp.status_code != 200:
                print(f"  HTTP {resp.status_code}: {resp.text[:300]}")
                return None, None, None
            data = resp.json()
        return _cached_tokens(data), data.get("provider"), data.get("id")

    async def condition(label: str, session_id: Optional[str], calls: int = 6) -> int:
        """Run `calls` identical requests on a FRESH prefix; return hits."""
        # A fresh prefix per condition, and enough repetitions to see an
        # intermittent cache. Reusing one prefix across conditions, or judging
        # on a single call, produces a false positive: whichever condition runs
        # second inherits the warm cache the first one created.
        tag = secrets.token_hex(8)
        prefix = f"Dokumen uji {tag}. " + (
            "Konteks peraturan internal UPI untuk pengujian cache. " * 220
        )
        print(f"{label} (fresh prefix, {calls} identical calls):")
        hits = 0
        last_id = None
        for i in range(1, calls + 1):
            cached, provider, gen_id = await one(prefix, session_id)
            last_id = gen_id or last_id
            if cached:
                hits += 1
            print(f"  call {i}: cached={cached!r}  provider={provider!r}")
        if last_id:
            print(f"  generation id: {last_id}  "
                  f"(GET /api/v1/generation?id={last_id} for actual billed cost)")
        print(f"  -> {hits}/{calls} cache hits\n")
        return hits

    print(f"Testing session_id against {target}\n")
    # Run session_id FIRST: if it ran second and 'won', that would be
    # indistinguishable from it simply inheriting a warm cache.
    with_hits = await condition("with session_id", "preflight-fixed-session")
    without_hits = await condition("without session_id", None)

    print("Verdict:")
    if without_hits > 0:
        print(f"  Caching occurs WITHOUT session_id ({without_hits} hits) → it is automatic "
              "and prefix-driven. session_id is not what produces it.")
        if with_hits <= without_hits:
            print("  session_id shows no advantage → do not send it; rely on prefix ordering "
                  "and provider pinning.")
    elif with_hits > 0:
        print(f"  Hits only with session_id ({with_hits} vs 0) → re-run to confirm before "
              "trusting this; a single run cannot separate the effect from cache warm-up.")
    else:
        print("  No caching in either condition. Any cost claim resting on session_id "
              "does not hold.")


def main() -> None:
    """Entry point for the panel preflight."""
    parser = argparse.ArgumentParser(
        description="Preflight checks for the dataset-generation panel (run before a full generation)."
    )
    parser.add_argument("--list", metavar="SUBSTRING",
                        help="List live OpenRouter models matching SUBSTRING, with prices")
    parser.add_argument("--raw", action="store_true",
                        help="Dump one complete raw response body")
    parser.add_argument("--check-session-id", action="store_true",
                        help="Test whether the session_id payload field affects caching/routing")
    parser.add_argument("--model", help="Model to use for --raw / --check-session-id")
    args = parser.parse_args()

    settings = get_dataset_gen_settings()
    if not settings.openrouter_api_key:
        print("ERROR: DATAGEN_OPENROUTER_API_KEY is not set.", file=sys.stderr)
        sys.exit(1)

    if args.list:
        asyncio.run(list_models(settings, args.list))
    elif args.raw:
        asyncio.run(dump_raw(settings, args.model))
    elif args.check_session_id:
        asyncio.run(check_session_id(settings, args.model))
    else:
        sys.exit(asyncio.run(run_panel_check(settings)))


if __name__ == "__main__":
    main()
