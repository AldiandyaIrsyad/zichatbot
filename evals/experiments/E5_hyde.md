# E5 — HyDE

**Result: helps the document class, not the document.** Decrees on top 5/10 →
3/10, but `expected doc first` unchanged at 2/6. Consistent with the earlier
probe: HyDE moves recall, not ranking — at three LLM calls per query.

    python -m evals.experiments.e3_retrieval     # spends API credit

## Setup

Fused exactly as `search_service.py:107-133` would: HyDE drives the dense
channel only, the sparse vector stays the raw query's, and the two rankings are
combined by RRF. Generation model `deepseek/deepseek-v4-flash-0731`.

## Result

| intervention | decree on top | expected doc first |
|---|---|---|
| baseline | 5/10 | 2/6 |
| **E5 HyDE** | **3/10** | 2/6 |
| E3 doctype prior | 3/10 | 2/6 |
| E3 + E4b | **0/10** | **3/6** |

HyDE matches the doctype prior on the decree measure and neither improves the
expected-document measure. Between runs it moved between 3/10 and 4/10, so
treat the difference from baseline as directional.

## Why it half-works

From the earlier probe, the passage generated for *"Biaya kuliah per semester
berapa?"*:

> *"Biaya kuliah yang harus dibayar oleh mahasiswa per semester ditetapkan
> sesuai dengan ketentuan yang berlaku di institusi pendidikan terkait…"*

Generic institutional prose. It contains none of the vocabulary that would
discriminate the tariff regulation from a decree that mentions a fee — no
"Kelompok Tarif UKT", no "golongan", no "kemampuan ekonomi". It helps because
generic policy language resembles a regulation more than a five-word question
does, which is why the *class* improves while the *document* does not.

On the SMS register it degenerates entirely: *"Biaya smstr brp ya?"* expanded to
*"Biaya semester berapa ya?"*. HyDE inherits the query-understanding problem.

## The cost

Three generations on **every** query, including the majority that never needed
it — real latency and money per request. E3 is a regex and E4b is one extra
vector search against 922 cached vectors; both achieve at least as much.

There is also prior evidence against enabling it globally: on subset_a, HyDE
costs 5 points of hit@1 with reranking on and 21 points without
(`data/v2/results/exp2b_hyde_rerank.csv`). Those are document-naming questions,
where expansion can only dilute the identifiers that make them answerable.

## The connection worth noting

`CHAT_HYDE_CONTEXT_ENABLED=false` disables a **grounded** mode that injects
real KB document titles into the HyDE system prompt (`hyde_expander.py:86`).
Grounded HyDE would plausibly produce "Kelompok Tarif UKT" *because it would be
shown the titles* — making it the title channel routed through an LLM, at three
generations per query instead of one vector lookup.

## Verdict

**Not first, and not globally.** E3 and E4b get more for far less. If HyDE is
revisited, gate it on queries carrying no document identifier and measure the
marginal gain **on top of** the title channel, since grounded HyDE and the title
channel are substantially the same signal.
