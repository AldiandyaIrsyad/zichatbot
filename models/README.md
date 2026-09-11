# models/

Local model artifacts referenced by the services in `../services/` and mounted
read-only into the `prompt-guard` and `nli` containers (`./models:/models:ro`).

## Layout

- `prompt_guard_id/`: the locally fine-tuned Indonesian prompt-guard
  checkpoint (HF snapshot: `config.json`, `model.safetensors`,
  `tokenizer*`, `training_args.bin`). **Gitignored** (copy it in from
  `skripsi_app/models/prompt_guard_id/` at deploy time). Copy the serving files
  only; the sibling `checkpoints/` directory is 8.5 GB of training state the
  server never reads, so the deployed copy is ~1.1 GB:

  ```bash
  mkdir -p models/prompt_guard_id
  cp ../skripsi_app/models/prompt_guard_id/{config.json,model.safetensors,\
     tokenizer.json,tokenizer_config.json,training_args.bin} models/prompt_guard_id/
  ```

  Serve it by swapping profiles: it replaces the stock guard on the same port
  rather than running alongside it, so no app config changes:

  ```bash
  docker compose stop prompt-guard
  COMPOSE_PROFILES=guard-ft docker compose up -d prompt-guard-ft
  ```

  Without this directory the container starts and then exits: the model path is
  handed to the HF hub client, which rejects it with "Repo id must be in the
  form 'repo_name' or 'namespace/repo_name'".
- `mmbert_nli_id/`: the mmBERT-small fine-tune on IndoNLI train (3-way
  entailment/neutral/contradiction). Produced by
  `python -m evals._train.train_nli_mmbert --output models/mmbert_nli_id`.
  **Gitignored**. All NLI containers share host port 8002 (`CHAT_NLI_PORT`), so
  serve it by stopping the default container first:
  ```bash
  docker compose stop nli-indoroberta
  COMPOSE_PROFILES=guard-ft,nli-mmbert docker compose up -d nli
  ```
  and select it in `.env` with `CHAT_NLI_MODEL_KIND=mmbert`.
- `prompt_guard/`: dropped. The base-model label mapping is now resolved in
  code by `services/prompt_guard/main.py`, so no patched config mount is needed.

## Deploy

```bash
# from skripsi_app (thesis archive): bring the fine-tune into this repo
cp -r ../skripsi_app/models/prompt_guard_id ./models/

# train the NLI fine-tune (requires torch + transformers + GPU)
python -m evals._train.train_nli_mmbert --output models/mmbert_nli_id --device cuda
```
