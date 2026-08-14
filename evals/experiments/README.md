# Experiments — mitigating wrong-document selection

Six experiments on the **ten audited questions** from
[`../results/demo_answer_audit.md`](../results/demo_answer_audit.md), where 5 of
10 answers came from a document whose title made no sense as their source — and
every one of those five was a *Keputusan Rektor* (an administrative decision
about named people) rather than a *Peraturan Rektor* (a general rule).

Generation and judging use **`deepseek/deepseek-v4-flash-0731`**.

| | experiment | verdict |
|---|---|---|
| [E1](E1_chunk_dilution.md) | Is hierarchical chunking diluting the vectors? | **No** — real but small, and removing the tag changes no outcome |
| [E2](E2_judge.md) | Can an LLM judge score this? | **As a screen only** — 7/10 exact even with a fixed prompt |
| [E3](E3_doctype_prior.md) | Down-weight decrees | **Ship, gated** — decrees on top 5/10 → 3/10 |
| [E4](E4_title_channel.md) | Document-title channel | **Ship as a document prior, not RRF** — 2/6 → 3/6 correct |
| [E5](E5_hyde.md) | HyDE | **Not first** — same gain as E3 at three LLM calls per query |
| [E6](E6_json_context.md) | JSON context with title and type | **Ship the type *instruction*, not the JSON alone** — structure is only the enabler |

## Headline

| intervention | decree on top | expected doc first |
|---|---|---|
| baseline (production) | 5/10 | 2/6 |
| E3 doctype prior | 3/10 | 2/6 |
| E4 title channel, RRF fusion | 5/10 | 2/6 |
| E4b title channel, document prior | 3/10 | 3/6 |
| **E3 + E4b** | **0/10** | **3/6** |
| E5 HyDE | 3/10 | 2/6 |

## Recommended stack

Three layers, cheapest first. Each is independently reversible and none touches
the chunk embeddings, so the hierarchical-chunking results stay valid.

1. **E3 document-type prior** — a regex. Stops a general question being answered
   by one programme's decree. Gate it on queries carrying no document reference,
   or it will hurt the many subset_a-style questions that legitimately want a
   decree.
2. **E4b document-title prior** — 922 cached vectors as a multiplicative
   document-level boost (*not* RRF fusion, which does nothing). Recovers
   documents whose chunks never matched.
3. **E6 C3 structured context** — JSON context carrying `jenis_dokumen`, plus
   one instruction that a decree cannot answer a general question. Makes a wrong
   retrieval less harmful when the first two still miss.

Not recommended: E5 HyDE (same gain as E3 at three LLM calls per query), E4 as
RRF fusion (no effect), removing the breadcrumb tag (E1 — costs structure, fixes
nothing).

## Running

Needs Postgres, Qdrant and Infinity up; not the app.

```bash
.venv/bin/python -m evals.experiments.e1_chunk_dilution   # local
.venv/bin/python -m evals.experiments.e2_judge            # ~40 calls
.venv/bin/python -m evals.experiments.e3_retrieval        # HyDE costs
.venv/bin/python -m evals.experiments.e6_json_context     # ~60 calls
```

`_bench.py` is the shared harness: it retrieves once and lets each intervention
reorder the same candidates, so the numbers are comparable.

## How these are scored

Two objective measures, chosen because E2 showed the judge is not reliable
enough to be the score:

- **decree on top** — how often the winning document is a Keputusan. Every
  mismatch in the audit was one, so this is the failure-rate proxy.
- **expected document first** — for the 6 of 10 questions where an obviously
  correct document exists in the corpus. The expectations are my judgement and
  are strict: for the "how much is tuition" questions, several tuition
  regulations are defensible answers and only one is counted correct.

The judge's verdict is reported alongside, never alone.

## The honest limit

**Ten questions, and the "expected document" labels are mine.** These
experiments can separate an intervention that clearly helps from one that
clearly does not, on the population that actually fails. They cannot produce an
effect size, and every one of them should be re-run against a labelled student
question set before any of it reaches the thesis.

The existing `subset_a` is the wrong population for this: all 115 of its
questions name an institution and 72 name a specific document, which is exactly
the case where these failures do not occur.
