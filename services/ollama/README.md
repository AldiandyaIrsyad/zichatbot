# Local Qwen3Guard serving (Ollama)

Qwen3Guard-Gen is the hosted-guard comparison in Experiment 1a. It is **not** on
OpenRouter, and this project's Hugging Face Inference billing is blocked (HTTP
402), so it is served locally as a GGUF through Ollama's OpenAI-compatible API.
The existing `Qwen3GuardClient` speaks to it unchanged — only the base URL
changes to `http://localhost:11434/v1`.

## Deploy the 0.6B (Q8)

```bash
docker compose --profile qwen up -d ollama
docker exec ollama_backend ollama pull hf.co/geoffmunn/Qwen3Guard-Gen-0.6B:Q8_0
docker cp services/ollama/Qwen3Guard-Gen-0.6B.Modelfile ollama_backend:/tmp/Modelfile
docker exec ollama_backend ollama create qwen3guard-gen-0.6b -f /tmp/Modelfile
```

## Why the Modelfile is necessary

The pre-quantized community GGUF (`geoffmunn/Qwen3Guard-Gen-0.6B`) ships an
Ollama chat template hardcoded to **response**-moderation and gated on
`{{ .System }}`. Sent a plain user prompt — which is exactly what the IVM does —
the moderation branch never fires and the model labels **every input Unsafe**,
benign or not. The `Modelfile` keeps the GGUF weights and overrides only the
template with the official **prompt**-moderation branch from
`Qwen/Qwen3Guard-Gen-0.6B` (the one that classifies THE LAST USER's query and
carries the `Jailbreak` category).

## Validate before trusting any number

```bash
# benign JDIH question must be Safe; an injection must be Unsafe/Controversial
curl -s http://localhost:11434/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen3guard-gen-0.6b","messages":[{"role":"user","content":"Apa tugas Majelis Wali Amanat menurut Statuta UPI?"}],"max_tokens":64,"temperature":0}' \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['message']['content'])"
# expect: Safety: Safe / Categories: None
```

`Qwen3GuardClient.parse_verdict` returning a non-None tier is the acceptance
check: if it returns a tier, the template is right.

## Adding the 4B (Q4_K_M) later

No trusted GGUF exists for the 4B, so convert from the official weights:

```bash
# in a llama.cpp checkout
python convert_hf_to_gguf.py Qwen/Qwen3Guard-Gen-4B --outfile qwen3guard-4b-f16.gguf
./llama-quantize qwen3guard-4b-f16.gguf qwen3guard-4b-Q4_K_M.gguf Q4_K_M
```

Then write a Modelfile `FROM ./qwen3guard-4b-Q4_K_M.gguf` reusing the TEMPLATE
block from `Qwen3Guard-Gen-0.6B.Modelfile`, `ollama create qwen3guard-gen-4b`,
and append it to the roster:

```bash
QWEN_MODELS="Qwen3Guard-Gen-0.6B,qwen3guard-gen-4b" ./scripts/run_exp1a.sh
```

The 4B is heavier than the 0.6B; uncomment the GPU reservation on the `ollama`
service (and stop the reranker/NLI if the 8 GB card is tight) to offload it,
or accept CPU latency for the one-off eval.
