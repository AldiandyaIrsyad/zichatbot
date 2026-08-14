# scripts/

Operational tooling, grouped by purpose. Run everything from the repository
root so `evals/data/…` and `app/…` resolve.

## Categories

| Folder | What's here |
| --- | --- |
| `scrapers/` | Download the JDIH/UPI PDF corpus (Playwright + cloudscraper), fill gaps, smoke-test the fetchers. `.browser_profile/` is a transient Playwright session (gitignored). |
| `ingestion/` | Upload PDFs to the app and re-ingest documents that changed on disk (`bulk_upload_pdfs.py`, `reingest_affected_docs.py`). |
| `db/` | Database lifecycle helpers (`reset_databases.py`). |
| `data/` | Corpus hygiene (`fix_filenames.py`, `find_and_delete_wrong_duplicates.py`). |
| `experiments/` | Shell wrappers that run the eval harnesses with preflight checks (`run_exp1a.sh`, `run_exp1b.sh`, `run_subset_abc.sh`). |

## Example

```bash
# scrape the corpus
.venv/bin/python scripts/scrapers/download_pdfs_playwright.py

# upload PDFs for ingestion
.venv/bin/python scripts/ingestion/bulk_upload_pdfs.py

# run an experiment with a preflight (services must be up)
./scripts/experiments/run_exp1a.sh
```

Experiment runners log to `logs/` and write results under `evals/data/results/`.
