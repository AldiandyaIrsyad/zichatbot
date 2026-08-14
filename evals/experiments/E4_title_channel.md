# E4 — A document-title retrieval channel

**Result: the shape matters more than the idea.** As a rival chunk ranking
fused by RRF it does nothing (5/10 decrees, unchanged). As a **multiplicative
document prior** it is the only intervention that improved `expected doc first`
— 2/6 → **3/6**, with the correct document's rank improving on three questions
and worsening on none.

    python -m evals.experiments.e3_retrieval     # local, no LLM

## The idea

The title never reaches ranking. Chunks are embedded as `breadcrumbs + body`,
the Qdrant payload has no title field, and the reranker sees only chunk text.
A separate index of the 922 document titles can surface a document whose
*chunks* never matched — the failure no reranking change can reach.

922 vectors, ~3.8 MB, already cached in `cache/title_emb.npz`.

## Two shapes, very different outcomes

| variant | how the title score is used | decree on top | expected doc first |
|---|---|---|---|
| baseline | — | 5/10 | 2/6 |
| **E4** RRF fusion | title ranking as a rival chunk ranking | 5/10 | 2/6 |
| **E4b** document prior | `score x (1 + w x title_sim(doc))` | 3/10 | **3/6** |
| E3 + E4b | prior, then decree penalty | **0/10** | **3/6** |

RRF gives the base ranking equal weight, and against a strong base ranking that
is enough to wash the title signal out entirely. Title similarity is a property
of the **document**, not a competing opinion about chunks, so it belongs as a
multiplier on that document's chunks. `w = 1.0` and `w = 2.0` gave identical
headline numbers.

## Rank of the correct document, baseline → E3+E4b

| question | before | after |
|---|---|---|
| Berapa biaya UKT? | #11 | **#4** |
| Biaya kuliah per semester berapa? | **not retrieved** | **#7** |
| Bagaimana proses penentuan golongan tarif UKT? | #2 | **#1** |
| Apa syarat memperoleh keringanan biaya pendidikan? | #1 | #1 |
| Apakah ada tarif khusus untuk parkir kendaraan? | #1 | #1 |
| Bagaimana syarat pendaftaran stiker izin parkir? | #3 | #3 |

Improved on three, unchanged on three, worse on none. And note row 2: the
channel **recovers a document that was not retrieved at all**, which is the
half of the problem probe 1 identified as unreachable by reranking.

## One tuning trap worth recording

The first run used `TITLE_TOP_DOCS = 5` and showed no effect. The correct UKT
document ranks **#6** by title for one of these questions, so the top-5 cut
discarded exactly the case the channel exists to fix. Widened to 15.

A cut-off tuned on the wrong end of the distribution can make a working idea
look dead.

## What it still does not fix

For the two "how much is tuition" questions the correct document reaches #4 and
#7 but not #1. What wins instead is *69 Tahun 2025 — Peninjauan Tarif UKT* and
*12252-UN40-HK-2018 — Biaya Pendidikan Mahasiswa Baru* — both genuinely
reasonable tuition regulations, arguably better answers than my strict labels
allow. Among regulations the ranker still cannot tell which tuition rule is the
apt one.

## Verdict

**Ship E4b (document prior), not E4 (RRF fusion).** Combined with E3 it takes
decrees on top from 5/10 to 0/10 and improves the correct document's rank on
half the questions. It needs no re-ingestion, no chunk-embedding change, and
therefore leaves the hierarchical chunking results intact.

**Caveat:** ten questions. The `w` parameter is untuned and the risk that a
document prior drags in plausible-titled documents for queries where the chunk
evidence was already right is real and unmeasured here.
