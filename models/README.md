# models/

Local model artifacts referenced by the services in `../services/` and mounted
read-only into the `prompt-guard` container (`./models:/models:ro`).

## Layout

- `prompt_guard_id/` — the locally fine-tuned Indonesian prompt-guard
  checkpoint (HF snapshot: `config.json`, `model.safetensors`,
  `tokenizer*`, `training_args.bin`). **Gitignored** — copy it in from
  `skripsi_app/models/prompt_guard_id/` at deploy time. Serve it with
  `docker compose --profile finetuned up -d prompt-guard-ft` (port 7999).
- `prompt_guard/` — dropped. The base-model label mapping is now resolved in
  code by `services/prompt_guard/main.py`, so no patched config mount is needed.

## Deploy

```bash
# from skripsi_app (thesis archive) — bring the fine-tune into this repo
cp -r ../skripsi_app/models/prompt_guard_id ./models/
```
