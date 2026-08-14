# Post-prasidang probes — findings

Six feasibility probes, run 12 August 2026 against the live corpus (922
documents, 25,315 parent chunks, 58,466 child chunks) and the production
Qdrant collection.

These are **feasibility probes, not experiments**. Retrieval numbers come from
the 115 labelled rows of `data/v2/subset_a.csv`, which is small enough that a
two-point move is noise. Where a result is inside noise, it says so.

| # | Issue | Verdict |
|---|---|---|
| 1 | Title priority | **Add a separate document-title retrieval channel.** Do *not* touch chunk embeddings — that costs 32% of within-document discrimination. The reranker fix does not solve the UKT case |
| 2 | Distinguishable LLM abstention | **Implement**, and add a code-level empty-context guard, which matters more |
| 3 | Amendments | **Confirmed.** A quarter are article-level. Deleting superseded documents would lose law still in force |
| 4 | Recency penalty | **Do not implement.** Helps 0 queries, hurts 21 |
| 5 | HyDE | **Complementary, not a substitute.** Fixes retrieval recall (4/7 → 6/7 retrieved), fixes ranking not at all (2/7 → 2/7). Enable conditionally, if at all |
| 6 | Document-level score aggregation | **Do not implement.** Fixes the UKT anecdote (#11 → #2) but costs 18 points of hit@1 on labelled data |
| 7 | Answer/source audit (10 questions) | **5/10 mismatched — and every one came from a *Keputusan* (decree), every good answer from a *Peraturan* (rule).** The type is free to read off the title |

---

## 1. Title priority

### The problem is real

The title never reaches ranking. The embedded string is
`breadcrumbs + "\n\n" + body` (`ingest_worker.py:201`), the Qdrant payload has
no title field (`qdrant_store.py:93-104`), and the reranker sees only
`child.text` (`search_service.py:161`). The title is fetched at
`search_service.py:186` — *after* ranking — purely to fill `source_title`.

Measuring it: cosine between the query and the title of the document the
baseline ranks first.

| query set | n | median cos | below 0.45 |
|---|---|---|---|
| subset_a | 115 | 0.650 | **5.2%** |
| student, formal | 50 | 0.415 | **66.0%** |
| student, colloquial | 50 | 0.399 | 76.0% |
| student, SMS-abbreviated | 50 | 0.286 | 100.0% |

subset_a questions are *derived from* the documents and reuse their wording, so
their titles match. Real student phrasing is different, and the match collapses
— for two thirds of even the **formal** student questions, the top document's
title has little to do with what was asked. That is the "UKT figure inside an
outbound-student decree" failure, and it is the common case for realistic
queries, not an edge case.

(The SMS row is 100% but should not be read as title mismatch: BGE-M3 barely
embeds `Wjb lglsr ijz sma gk?` at all, so *every* similarity is low.)

### What helps

Four rankings over the same retrieved candidates — no re-ingestion:

| variant | hit@1 | hit@3 | hit@5 | MRR |
|---|---|---|---|---|
| **A** baseline (rerank on `child.text`) | 0.765 | 0.835 | 0.896 | 0.817 |
| **B** rerank on `title\n` + `child.text` | **0.783** | **0.878** | **0.922** | **0.841** |
| C blend, w=0.1 | 0.748 | 0.852 | 0.896 | 0.812 |
| C blend, w=0.2 | 0.730 | 0.852 | 0.896 | 0.804 |
| C blend, w=0.3 | 0.722 | 0.870 | 0.913 | 0.802 |
| D title similarity only | 0.600 | 0.791 | 0.852 | 0.711 |

**B is better on all four metrics. C is worse on hit@1 at every weight.** A
linear blend of the rerank score with a title cosine is the wrong shape — the
cross-encoder already combines title and body better than a weighted sum does,
once you let it see both. D confirms titles alone cannot retrieve.

### Honest caveat on those numbers

B's margin is **not separable from noise at n=115**:

- MRR delta `+0.024`, 95% paired bootstrap CI `[-0.007, +0.057]` — includes zero
- hit@1: B better on 5 queries, worse on 3, tied on 107. Sign test p = 0.73

### The UKT demo — and B does not fix it

`python -m evals.probes.demo_title` → [results/demo_title.md](results/demo_title.md)

Asking **"Berapa biaya UKT?"** today returns, in order:

1. *1282-UN40-KM.02.02-2026 — Peserta Program Magang Luar Negeri…*
2. *346-UN40-KM.02.02-2026 — Peserta Program Outbound Student Mobility Ke Sookmyung…*
   > "**Biaya yang harus dibayar**: Rp 500.000,- per semester"
3. *1313-UN40-KM.02.02-2026 — Peserta Program Outbound Student Mobility…*

The document that actually sets UKT by income bracket — *003 Tahun 2022 —
Kelompok Kemampuan Ekonomi Orang Tua/Wali … Kelompok Tarif Uang Kuliah
Tunggal* — is nowhere in the top 3. Exactly the case you described.

**Variant B leaves this unchanged.** Top document identical on 4 of the 5
tuition questions. Showing the reranker the title is not enough to demote a
chunk that says "Rp 500.000 per semester" for a query about per-semester cost.

Digging in, there are **two different bugs**:

| question | right doc in top-50? | rank under A | rank under B |
|---|---|---|---|
| Berapa biaya UKT? | yes (12 chunks) | #11 of 12 | #11 |
| Berapa tarif UKT yang harus saya bayar? | yes | #5 | #6 |
| UKT saya masuk golongan berapa? | yes | #1 | #1 |
| Biaya kuliah per semester berapa? | **no** | — | — |
| Biaya smstr brp ya? | **no** | — | — |

For the colloquial phrasings the right document is **never retrieved** — no
reranking change can reach it. For "Berapa biaya UKT?" it *is* retrieved, 12
chunks of it, and the cross-encoder still puts it 11th of 12.

### What does fix it

Dense-only ranking over all 6,788 chunks of the 44 candidate documents,
embedded two ways — as today, and with the title prepended. Position of the
correct UKT document:

| question | today | title in embedding |
|---|---|---|
| Berapa biaya UKT? | #6 | #5 |
| Biaya kuliah per semester berapa? | #14 | **#1** |
| Berapa tarif UKT yang harus saya bayar? | #3 | **#1** |
| UKT saya masuk golongan berapa? | #1 | #1 |
| Biaya smstr brp ya? | #10 | **#2** |

Read as direction, not benchmark: dense only, no sparse channel, no RRF, 44
documents rather than 922. But it answers the question the reranker variants
cannot — the title carries the signal, and it has to be **in the vector**, not
bolted on afterwards.

### Two ways to put the title in — and they are not the same thing

**What the simulation above tested (one vector):**

```python
embed(f"{title}\n{breadcrumbs}\n\n{body}")     # one vector per chunk
```

**The alternative (two vectors, weighted):**

```python
v = normalise((1 - w) * embed(body) + w * embed(title))   # or two stored vectors
score = (1 - w) * cos(q, v_body) + w * cos(q, v_title)
```

These behave differently, and the difference is exactly what the UKT case
turns on. BGE-M3 is a transformer, so `embed(title + body) ≠ embed(title) +
embed(body)`. In the concatenated form the title tokens are *in the attention
window* while the body is encoded — "Rp 500.000 per semester" gets a different
representation when it sits under "Outbound Student Mobility ke Sookmyung"
than under "Kelompok Tarif Uang Kuliah Tunggal". The title changes what the
chunk **means**.

A weighted sum cannot do that. `embed(body)` is computed without ever seeing
the title, so blending only *moves the point* in embedding space; the chunk's
own representation is already fixed and still ambiguous. For a failure whose
whole nature is "this number is meaningless without knowing which document
it's in", that distinction is the mechanism.

Related evidence, though not a clean test of the two-vector idea: probe 1's
variant C blended a title cosine into the *rerank* score and was worse than
baseline at every weight (0.748 / 0.730 / 0.722 vs 0.765). That blended a
cross-encoder score with a bi-encoder cosine, which is a different and messier
thing than blending two bi-encoder cosines at retrieval — so it is suggestive,
not conclusive.

| | one vector (concatenate) | two vectors (weighted) |
|---|---|---|
| title can disambiguate the body | **yes** — attention | no — body vector already fixed |
| weight tunable after indexing | no, baked in | **yes**, runtime knob |
| cost to try | full re-ingest, 58,466 chunks | **~922 title embeddings** (already cached) |
| Qdrant schema | unchanged | new named vector, or app-side blend |
| also improves BM25 | **yes** (title words become lexically searchable) | no |
| reversible | re-ingest again | change one number |

**The two-vector version is much cheaper to evaluate, and Qdrant already
supports it natively.** `ensure_collection` (`qdrant_store.py:36-70`) declares
named vectors `dense` + `bm25`, and `hybrid_search` (`:112-176`) already fuses
prefetches with server-side RRF. Adding a third named `title` vector and a
third prefetch is a small change — no re-chunking, only 922 title embeddings to
compute, and `w` stays adjustable. That makes it the right thing to try
**first**, precisely because it is refutable in an afternoon.

The concatenation route has one more thing going for it that is easy to miss:
**the codebase already does this, one level down.** `_build_breadcrumb_tag`
(`app/thesis/chunking/logic.py:404-417`) prepends `"BAB II > Pasal 5\n\n"` to
every child chunk before embedding, and `logic.py:458` is where that happens
(`final_child_text = breadcrumb_tag + child_text`). The document title is
simply the missing top level of that same hierarchy. Prepending it is not a
new mechanism, it is finishing an existing one.

Caveat on my own numbers: the simulation was **dense-only**. Concatenating the
title also changes the sparse/BM25 vector — title words become lexically
searchable, which for keyword-ish queries could help more than the dense gain
shown, or could dilute the chunk's own terms. Untested either way.

### Concatenating costs within-document resolution — measured

The worry that touching the embedding would undo what hierarchical chunking
buys is correct, and it is measurable. Over 3,551 chunks in 40 documents,
embedded both ways:

| | intra-doc cosine | inter-doc cosine | separation |
|---|---|---|---|
| current (`breadcrumbs + body`) | 0.606 | 0.516 | +0.090 |
| title prepended | **0.800** | 0.594 | +0.206 |

Prepending the title makes every chunk in a document **much more similar to
every other chunk in the same document** (0.606 → 0.800). At document level
that is the point — the document becomes a tighter, more findable cluster,
which is why the UKT document jumped to #1. At chunk level it is the problem.
Discrimination *inside* a document:

| | std of similarity within a doc | top1−top2 margin |
|---|---|---|
| current | 0.0479 | 0.0209 |
| title prepended | **0.0326** (−32%) | 0.0189 (−10%) |

A third of the within-document signal disappears. Picking *which* Pasal answers
the question is exactly what hierarchical chunking is for, and what Small-to-Big
and the RAM citation step depend on. The chunker's own docstring already warns
about this failure mode for breadcrumbs — "without diluting the embedding with a
repeated constant string" (`logic.py:404-417`) — and a document title is a
constant string repeated across every chunk of that document.

Concatenation trades chunk-level precision for document-level recall. Since the
thesis conclusion rests on the chunk-level side, that is the wrong trade.

### Recommended: a separate document-title channel

The failure is **document selection**, not chunk selection. The right chunk was
never the problem — "Rp 500.000 per semester" is a perfectly good chunk. The
wrong *document* won. So the fix belongs at document level, not baked into
58,466 chunk vectors.

Ranking the 922 **document titles alone**, chunk embeddings untouched:

| question | by title | by chunks today |
|---|---|---|
| UKT saya masuk golongan berapa? | **#1** of 922 | #1 |
| Berapa tarif UKT yang harus saya bayar? | **#2** | #5 |
| Biaya kuliah per semester berapa? | **#4** | **not retrieved** |
| Berapa biaya UKT? | **#6** | #11 of 12 |
| Biaya smstr brp ya? | #18 | **not retrieved** |
| Gmn cara byr ukt? | #191 | — |

The top titles it returns are *Peninjauan Tarif UKT*, *Pembebasan Biaya UKT*,
*Biaya Pendidikan Mahasiswa Baru* — tuition documents, every one, and not a
single outbound-student decree. The title channel separates "documents about
tuition" from "decrees that happen to mention a fee". That is the missing
signal, and it is available without touching a single chunk vector.

**The design:**

1. A second, small Qdrant collection: one point per document, dense vector of
   the title. 922 points, ~3.8 MB — against ~240 MB to hang a title vector off
   every chunk. The embeddings already exist in `cache/title_emb.npz`.
2. In `SearchService.search`, query it alongside the existing hybrid search and
   fuse by **rank** (RRF), not by score — variant C shows a linear score blend
   is the wrong shape.
3. `SEARCH_TITLE_TOP_N` and the fusion weight in a new `SearchSettings`
   (`app/kb/config.py`), defaulting to **off**.

**Why this shape and not the other:**

- `child_text` is untouched, so exp2a's hierarchical-vs-fixed comparison stays
  valid. Nothing already run is invalidated.
- The production collection is untouched, so existing exp2 / exp4 numbers stay
  the baseline instead of becoming incomparable history.
- It is **additive and ablatable** — a clean extra row ("hierarchical chunking +
  document-title prior" vs "hierarchical chunking alone") rather than a change
  that forces a settled conclusion to be re-defended.
- Reversible: a config flag, not a re-ingest.

**Still unproven.** Six queries is a demo, not a result. Before this goes in the
thesis it needs the full labelled set, and the ablation must check the obvious
risk — that a document-level prior drags in plausible-titled documents for
queries where the chunk evidence was already right. It does nothing for the SMS
register (#191), which no index change will fix.

Variant B (`search_service.py:161`, plus hoisting the `get_pdfs_by_ids` fetch
from `:185-187`) is three lines, needs no re-indexing, and moves every aggregate
metric the right way — worth taking alongside, but it does not fix the UKT case.

Do **not** implement variant C — the score blend is worse than baseline at
every weight tested.

### The alternative, for the record

Concatenating at embed time, `app/kb/application/ingest_worker.py:201`
(`child_texts = [c.text for c in all_children]` → title-prefixed), is the
stronger document-level fix and also improves BM25, but it costs a full
re-ingest of 58,466 chunks, the 32% within-document resolution measured above,
and the validity of every chunking result already run. Keep it as the fallback
if the title channel cannot disambiguate the UKT case.

---

## 2. Distinguishable LLM abstention

### Two findings, and the second is the bigger one

40 questions × 3 context conditions × 2 prompts, `qwen/qwen3-14b`,
temperature 0. P1 is the production system prompt with the open-ended
abstention clause (`prompt_maker.py:27-31`) replaced by a mandate to reply with
exactly `"Maaf, saya tidak menemukan dokumen relevan."`

| prompt | context | exact sentence | refused outright |
|---|---|---|---|
| P0 current | relevant | 0% | 0% |
| P0 current | mismatched | 0% | 27.5% |
| P0 current | **empty** | 0% | **5.0%** |
| P1 mandated | relevant | 10% | 10% |
| P1 mandated | mismatched | **100%** | 100% |
| P1 mandated | **empty** | **55%** | 55% |

**Finding A — the mandate works when there is irrelevant context.** 100% exact
emission, 40/40. That gives `chat_service` a string it can detect and tag, which
is what was asked for. Cost: 10% over-refusal (4 of 40 refused despite having
the gold passage). That is a real price and should be watched, but it is small
and the failure is safe — an unnecessary "I found nothing" beats a fabrication.

**Finding B — with *no* context, the current prompt fabricates 95% of the
time.** P0 refused outright in only 2 of 40 empty-context cases. The other 38
answered from parametric memory. Verbatim, with zero context supplied:

> *"Berdasarkan Keputusan Rektor, penetapan Panitia RIKSA BAHASA XIX
> Universitas Pendidikan Indonesia Tahun 2025 dilakukan dengan menunjuk Ketua,
> Sekretaris..."*

The mandate roughly halves this (55% correctly refuse) but **45% still
fabricate**. So the prompt alone does not fix empty context.

### The change

Prompt (fixes the mismatched-context case, gives us a detectable sentinel):
swap the clause at `prompt_maker.py:27-31` for the mandated wording in
`probe2_refusal.py::_MANDATED_CLAUSE`.

Code (fixes the empty-context case, which the prompt cannot): `chat_service.py`
runs the deep retrieval at `:404` with **no emptiness check** — the top-3
pre-check at `:394-397` is skipped entirely when `skip_ivm` is set. Refuse
before calling the LLM when the deep retrieval comes back empty.

Plumbing, so the two are actually distinguishable in the UI:

- `chat_service.py:570` currently emits `reason ∈ {unsafe, irrelevant,
  internal}`. Add `llm_abstained`, set when the response matches the sentinel.
- `app/templates/pages/index.html:118-126` renders one badge for both `unsafe`
  and `irrelevant`. Split it, and add the new reason.
- `reason` is **not persisted** — `Message` (`app/chat/domain/models.py:45-69`)
  has no column for it, so after a reload a refusal renders exactly like a
  normal answer. This is arguably the worst of the three gaps and needs a
  migration (there is no Alembic; `create_all` will not add a column to an
  existing table).

---

## 3. Amendments

### Documents do amend each other, and a quarter of it is surgical

Scanning all 25,315 parent chunks for *operative* clauses:

| relation | documents | clauses |
|---|---|---|
| `full_repeal` (`dicabut dan dinyatakan tidak berlaku`) | 82 | 110 |
| `partial_amend` (`Ketentuan Pasal N diubah`) | 25 | 103 |
| `delete` (`Pasal N dihapus`) | 11 | 39 |
| `insert` (`Di antara Pasal N dan N+1 disisipkan`) | 8 | 12 |

**27 documents perform article-level surgery only, 80 perform whole-document
repeal only, 2 do both. Article-level is 26.6% of all amending instruments.**

Real examples, verified against the source text:

> *"Ketentuan Pasal 5 diubah, sehingga ketentuan Pasal 5 berbunyi sebagai
> berikut"* — `36 tahun 2020 - Perubahan Atas Peraturan Rektor Nomor 2151-UN40-HK-2019`
>
> *"Ketentuan Pasal 6 dihapus. Ketentuan Pasal 7 diubah..."* — same document
>
> *"Di antara Pasal 14 dan Pasal 15 disisipkan dua pasal baru yaitu Pasal 14A
> dan Pasal 148..."* — same document (`Pasal 148` is OCR damage for `Pasal 14B`)

### Worked examples — what gets patched, what gets left

`python -m evals.probes.demo_amendments` → [results/demo_amendments.md](results/demo_amendments.md)

Six amendments have their target in the corpus *and* operate article by
article. For each, the articles the amendment touches, set against every
article the original contains.

**Parking, 2019 → 2020.** The clearest case: the original's article list
survives OCR completely (35 of 35).

| | Pasal |
|---|---|
| rewritten | 5, 7, 10, 14, 15, 16, 23, 24, 28, 33 |
| deleted | 6 |
| inserted (new) | 14A, 14B |
| **untouched** | **1, 2, 3, 4, 8, 9, 11, 12, 13, 17–22, 25, 26, 27, 29, 30, 31, 32, 34, 35** |

> *"Ketentuan Pasal 5 diubah, sehingga ketentuan Pasal 5 berbunyi sebagai berikut…"*
> *"Ketentuan Pasal 6 dihapus."*
> *"Di antara Pasal 14 dan Pasal 15 disisipkan dua pasal baru yaitu Pasal 14A dan Pasal 14B…"*

`36 tahun 2020 - Perubahan Atas Peraturan Rektor Nomor 2151-UN40-HK-2019
tentang Pengelolaan Perparkiran` touches **13 of 35 articles**. The other
**24 exist only in the 2019 document.** Delete it and Pasal 1–4, 8–13, 17–22
and the rest are gone — the 2020 amendment never restates them.

**Staffing, 2015 → 2022.** `043 Tahun 2022 - Perubahan Ketiga Atas Peraturan
Rektor Nomor 7739-UN40-HK-2015 tentang Sistem Pengelolaan Pegawai` rewrites
Pasal 1, 12, 15, 17 and deletes Pasal 13 — **5 of 52 recovered articles
(72% of an implied 72)**. Forty-eight untouched. Note the "Ketiga": this is the
*third* amendment to the same 2015 regulation, so the current law is spread
across at least four documents.

**Organisational structure, 2015 → 2017.** `6323-UN40-HK-2017 - Perubahan Atas
Peraturan Rektor Nomor 6489-UN40-HK-2015 tentang Struktur Organisasi dan Tata
Kerja` touches Pasal 1, 7, 126, 148, 149, 161, 174, 184, 190A, 193 — **10 of
165 recovered articles (85% of an implied 195)**. 157 untouched.

Three further cases (`28 Tahun 2024`, `3 Tahun 2023`, `4601-UN40-HK-2015`) are
listed in the demo but their **originals are too damaged to enumerate** — one
2017 scan yields 6 "Pasal" mentions and a single parseable number out of an
implied 138 articles. The amendments' own clauses are still readable and name
their targets explicitly; it is the originals that cannot be inventoried. That
is worth knowing on its own: the amendment graph is recoverable, the article
inventory of the older scans often is not.

### This answers "why can't we delete old stuff"

In every case above the amendment is a patch, not a replacement. It rewrites a
handful of articles and is silent about the rest — and the rest stay in force.
The current law for parking is *2019 minus Pasal 6, with 5/7/10/14/15/16/23/24/
28/33 replaced and 14A/14B added*. Neither document states that on its own.

### Two traps worth recording

- **Naive greps overcount by 8×.** `sebagaimana telah diubah` matches **879 of
  922 documents**, because it appears in the *considerans* of nearly everything
  (`"Peraturan Pemerintah Nomor 26 Tahun 2015 ... sebagaimana telah diubah
  dengan Peraturan Pemerintah Nomor 8 Tahun 2020"`). That describes some other
  law's history, not what this document does. The probe excludes it and counts
  only operative clauses.
- **A `Perubahan Atas` title does not mean article surgery.** 121 titles
  declare it; only 25 of those actually rewrite articles, and 95 contain no
  operative clause at all — they are Keputusan Rektor replacing an attached
  list of names wholesale. Title alone is not a usable signal.

Of the 121 declared amendments, **46 have their target in our corpus** and 69
amend documents we never ingested (mostly national law).

---

## 4. Recency penalty

### Do not implement

`score × exp(−λ × age)` applied to the same cached candidates. `λ=0` reproduces
probe 1 variant A exactly (0.765), which is the harness sanity check.

| λ | hit@1 | hit@3 | hit@5 | MRR | queries helped | queries hurt |
|---|---|---|---|---|---|---|
| **0.00** | **0.765** | 0.835 | 0.896 | **0.817** | — | — |
| 0.05 | 0.583 | 0.704 | 0.809 | 0.679 | **0** | **21** |
| 0.10 | 0.530 | 0.696 | 0.783 | 0.638 | 1 | 28 |
| 0.20 | 0.504 | 0.617 | 0.730 | 0.599 | 2 | 32 |

At the gentlest setting tested, a recency prior **helps zero queries and breaks
21 of 115**. This is not a tuning problem — there is no λ between 0 and 0.05
worth hunting for, because the mechanism is wrong.

Robustness: treating the 50 documents with no parseable year as *oldest*
instead of newest gives 0.600 / 0.548 at λ=0.05 / 0.10 — still far below 0.765.
The verdict does not depend on that choice.

### Why it fails

The corpus is heavily skewed recent — 89 of 112 gold documents are 2025–2026.
A decay therefore barely separates the documents that dominate the corpus while
catastrophically burying the 20 gold documents that are three or more years
old. And the queries people actually ask often name a *specific historical*
document:

> *"Apa saja cakupan Program Pengembangan Dosen dan Kapasitas Institusi..."* —
> gold document is from 2015, displaced by a 2025 document at λ=0.1.

A second artifact worth knowing: documents with no parseable year escape the
penalty and float upward, so one displacement was "gold 2025 displaced by a
document with no year at all".

### Year metadata, if it is ever wanted for something else

Year is recoverable for **872 of 922 documents (94.6%)** — 870 from the title
prefix (`26 Tahun 2025 - ...`, `0118-UN40-HK-2015 - ...`), 2 from the
`description` provenance code. Range 2011–2026. The 50 failures are titles with
a stray space inside the code (`1207-UN40.R4- PL.05.05-2026`).

Parse the **prefix before the first `" - "`**, never a global regex: subject
text carries its own years (`"...Laporan Akhir Tahun UPI Tahun 2025"` sits on a
2026 document).

### The interaction with probe 3

Probe 3 predicted the tension and the data confirms the direction, though not
through the path expected. Of the gold documents, 8 are amended by something
else in the corpus, but **none of those amendments are article-level**, so in
this sample the "partially superseded document is still valid" case did not
drive the failure. Recency failed for the simpler reason above. The probe 3
argument still stands on its own evidence: 27 documents in the corpus are
partially amended, and burying them would lose the articles that were never
touched.

---

## 5. HyDE (follow-up question)

**None of the numbers above used HyDE.** It is off in production
(`CHAT_HYDE_ENABLED=false`, "enable only after paired validation") and my
`Retriever` embeds the raw query and calls `hybrid_search` directly, so probes
1–4 are all HyDE-off. That matches the deployed system.

### Why it is off

| condition | hit@1 | MRR | source |
|---|---|---|---|
| off, no rerank | 0.704 | 0.784 | `exp2b_hyde_rerank.csv` |
| **on**, no rerank | **0.496** | 0.578 | same |
| off, rerank | 0.765 | 0.810 | same |
| **on**, rerank | 0.713 | 0.756 | same |
| on, RRF fusion (current code) | 0.765 | 0.808 | `exp2b_hyde_fusion_v4.csv` |

Naive HyDE costs 21 points of hit@1. The RRF-fusion form now in
`search_service.py:107-133` is neutral. Hence disabled.

**But every one of those rows is subset_a**, whose questions are written *from*
the documents and already carry the nomor, the tahun and the document's own
vocabulary — the one case where HyDE has nothing to add and can only dilute.
`diagnose_hyde.py` says as much in its docstring and measures the identifier
loss directly. So that evidence does not settle your question.

### On the UKT case, HyDE half-works

`python -m evals.probes.probe5_hyde`. Rank of the correct tariff document
after reranking:

| question | HyDE off | HyDE only | RRF fusion |
|---|---|---|---|
| UKT saya masuk golongan berapa? | **#1** | **#1** | **#1** |
| Gmn cara byr ukt? | **#1** | **#1** | **#1** |
| Berapa tarif UKT yang harus saya bayar? | #5 | #5 | #5 |
| Berapa biaya UKT? | #11 | #9 | #11 |
| Biaya kuliah per semester berapa? | **not retrieved** | **#19** | #20 |
| Berapa sih bayaran kuliah per semester? | **not retrieved** | **#21** | #27 |
| Biaya smstr brp ya? | not retrieved | not retrieved | not retrieved |
| **document retrieved at all** | **4/7** | **6/7** | **6/7** |
| **document ranked first** | **2/7** | **2/7** | **2/7** |

**Your intuition is half right, and it is the useful half.** HyDE fixes the
*retrieval miss* — it pulls the tariff document into the candidate set for two
queries where it previously never appeared at all. That is real, and it is
exactly the failure mode probe 1 identified as unreachable by any reranking
change.

It does **not** fix the *ranking*. The document arrives at #19–#27 and the
outbound decrees still hold the top spot. Ranked-first is 2/7 in all three
conditions — identical. HyDE gets the document into the room; it does not get
it to the front.

### Why it only half-works

Look at what it actually generated for "Biaya kuliah per semester berapa?":

> *"Biaya kuliah yang harus dibayar oleh mahasiswa per semester ditetapkan
> sesuai dengan ketentuan yang berlaku di institusi pendidikan terkait. Besaran
> biaya tersebut dapat berbeda-beda tergantung pada program studi…"*

Generic institutional prose. No "Kelompok Tarif UKT", no "golongan", no
"kemampuan ekonomi orang tua" — none of the vocabulary that would actually
discriminate the tariff document from a decree that mentions a fee. It helps
because the generic terms overlap the right document more than a five-word
question does, not because it found the right words.

And on the SMS register it degenerates completely — for "Biaya smstr brp ya?"
it emitted *"Biaya semester berapa ya?"*, just the question re-spelled. HyDE
inherits the query-understanding problem rather than solving it.

### The connection worth noticing

`CHAT_HYDE_CONTEXT_ENABLED=false` disables a **grounded** mode that injects
actual KB document titles and descriptions into the HyDE system prompt
(`hyde_expander.py:86`). Grounded HyDE would plausibly generate "Kelompok Tarif
UKT" *because it would be shown the document titles*.

Which means grounded HyDE is, in effect, the document-title channel routed
through an LLM. The same signal, obtained by paraphrasing titles into a passage
at a cost of three generations per query, instead of one extra vector search
against 922 cached vectors. That is a strong argument for taking the title
channel first: it is the cheap, direct form of the thing HyDE would be doing
indirectly.

### Verdict

Not a substitute for the title channel, and not free — HyDE is **three LLM
calls on every user query**, paid in latency and money on every request,
including the majority that never needed it. But it is complementary: it
attacks recall, the title channel attacks document-level ranking.

If you want it, the shape that makes sense is **conditional**: enable HyDE only
for queries that carry no document identifier (no nomor, no tahun, no ALL-CAPS
code) — which is exactly the population where subset_a's negative result does
not apply and where these two queries live. That is testable with
`diagnose_hyde.py`'s existing identifier-retention machinery. Do not enable it
globally; the subset_a rows above show what that costs.

---

## 6. Ranking documents by how much of them matched

The UKT diagnosis showed the tariff document contributing **12 of the 50**
candidate chunks and still ranking 11th, because `doc_ranking` takes the first
chunk to surface and throws away breadth. Ranking documents by how much of them
matched looked like the cheapest possible fix — pure post-processing over
probe 1's cached scores, no index, no embeddings, no LLM.

On the UKT queries it works beautifully:

| aggregator | Berapa biaya UKT? | Berapa tarif UKT…? |
|---|---|---|
| first occurrence (current) | #11 | #5 |
| max chunk score | #11 | #5 |
| **RRF over chunk ranks** | **#2** | **#2** |
| **chunk count** | **#2** | **#2** |

On the 115 labelled queries it falls apart:

| aggregator | hit@1 | hit@3 | hit@5 | MRR | better / worse |
|---|---|---|---|---|---|
| first_occurrence (current) | **0.765** | 0.835 | 0.896 | **0.817** | — |
| max_score | 0.765 | 0.835 | 0.896 | 0.817 | 0 / 0 |
| sum_top3 | 0.678 | **0.861** | 0.904 | 0.782 | 6 / 16 |
| rrf_ranks | 0.583 | 0.826 | **0.904** | 0.714 | 9 / 30 |
| mean_top3 | 0.565 | 0.730 | 0.800 | 0.674 | 2 / 25 |
| chunk_count | 0.504 | 0.809 | 0.896 | 0.672 | 9 / 39 |

Breadth moves the right document *into* the top 5 slightly more often (hit@5
0.896 → 0.904) while knocking it *off* the top spot far more often (hit@1
0.765 → 0.583). Precision at rank 1 is what the user experiences.

I expected this to be query-type dependent — breadth should suit topical
questions and hurt pinpoint ones — so I split subset_a by whether the question
names a document (a nomor, a tahun, a UN40 code):

| | first_occurrence | rrf_ranks | delta |
|---|---|---|---|
| names a document (n=72) | 0.778 | 0.542 | **−0.236** |
| topical only (n=43) | 0.744 | 0.651 | **−0.093** |

The direction of the hypothesis holds — breadth is *less* harmful for topical
queries — but it is still negative in both. **The hypothesis does not survive.**
Document-level aggregation fixes the UKT anecdote and costs accuracy everywhere
I can measure it. Do not implement it.

Worth recording why the anecdote misled: a subset_a question like *"Berapa
nomor Keputusan Rektor yang menetapkan peserta program Outbound Student
Mobility…"* is answered by exactly one chunk. A document with one excellent
chunk **is** the right answer, and counting how many other chunks it
contributed is noise. The UKT question is the opposite kind. Both kinds are
real; no single ranking function serves both.

---

## 7. Answer / source-title audit — the document *type* is the tell

`python -m evals.probes.demo_answer_audit` →
[results/demo_answer_audit.md](results/demo_answer_audit.md)

Ten student questions through the real pipeline — hybrid retrieval, BGE
reranking, Small-to-Big hydration, production system prompt — then the answer
placed next to the title of the document it came from. **5 mismatched, 2
partial, 3 coherent.** Verdicts by reading all ten; the LLM judge was kept but
is not reliable (it passed question 1 because answer and title agree with each
other, missing that both are wrong for the question).

| # | question | answer | source | verdict |
|---|---|---|---|---|
| 1 | Berapa biaya UKT? | Rp 500.000 for magang participants | *Peserta Program Magang Luar Negeri…* | mismatch |
| 2 | Biaya kuliah per semester berapa? | Rp 500.000 for outbound participants | *Peserta Outbound … Sookmyung* | mismatch |
| 5 | Sanksi keterlambatan bayar UKT? | not stated, then speculates | *Peserta Outbound … Sookmyung* | mismatch |
| 8 | Syarat dokumen daftar ulang? | "see pmb.upi.edu" | *Peserta Lulus Seleksi Pascasarjana* | mismatch |
| 10 | Kapan mulai perkuliahan maba? | vague, no date | *Inbound Student Mobility … UTM* | mismatch |
| 3 | Penentuan golongan tarif UKT? | correct content | *69 Tahun 2025 Peninjauan Tarif UKT* | partial |
| 7 | Syarat stiker parkir? | must hold a SIM | *36 tahun 2020 Perubahan … Perparkiran* | partial |
| 4 | Syarat keringanan biaya? | full, correct | *41 Tahun 2023 Pemberian Keringanan* | ok |
| 6 | Tarif parkir mahasiswa? | real tariffs from Lampiran V | *2151 Pengelolaan Perparkiran* | ok |
| 9 | Max buku dipinjam? | correctly says not specified | *7565 Standar Mutu* | ok |

### The pattern

| source document type | ok | partial | mismatch |
|---|---|---|---|
| Peraturan Rektor (a general rule) | 3 | 2 | **0** |
| Keputusan Rektor (a decision about named people) | 0 | 0 | **5** |

**Every mismatch came from a Keputusan; every acceptable answer came from a
Peraturan.** No crossover in ten cases.

A *Peraturan* sets a general rule. A *Keputusan* decides something about named
people or one event — who joins an exchange programme, who passed a selection.
A student's general question needs the rule, but the decree usually matches the
words better, because it states a concrete figure where the rule states a
formula. `Rp 500.000` beats *"ditetapkan berdasarkan kemampuan ekonomi orang
tua"* on a query about cost, every time. That is the whole UKT failure in one
sentence, and it explains why it recurs across unrelated topics.

**63% of the corpus (582 of 922) is Keputusan.** The class that answers these
questions badly is the majority class.

And the distinction is **free**: it is in the title's own numbering.
`41 Tahun 2023` and `2151-UN40-HK-2019` are rules; `346-UN40-KM.02.02-2026`
and `549-UN40-TM.01.04-2026` are decrees. No embedding, no model call, one
regex over a string already in the database.

### Two side findings

- **Question 7 shows amendment retrieval working.** The answer came from the
  2020 amending document, out of the inserted Pasal 14B that probe 3 found. The
  system found law that exists only in an amendment — which is the positive
  case for keeping superseded documents.
- **Question 6 shows the staleness risk.** It quotes parking tariffs from the
  2019 regulation, and probe 3 established the 2020 amendment rewrote Pasal 5,
  7, 10, 14, 15, 16, 23, 24, 28 and 33 of that document. Whether the quoted
  tariffs are still current is not something the system can currently know or
  tell the user.

---

## How to actually get the right document

Six interventions tested. Five are dead:

| intervention | result |
|---|---|
| reranker sees the title (B) | does not fix UKT; aggregate gain inside noise |
| blend title cosine into the score (C) | worse at every weight |
| concatenate title into chunk text | fixes document level, **costs 32% of within-document discrimination**, needs full re-ingest, invalidates the chunking results |
| recency prior | helps 0 queries, hurts 21 |
| HyDE | fixes recall 4/7 → 6/7, ranking 2/7 → 2/7, three LLM calls per query |
| document-level score aggregation | fixes the anecdote, −18 points hit@1 on labelled data |

**The pattern in the failures is the answer.** Every one of them tried to make a
*chunk-level* representation answer a *document-level* question:

- concatenation put document identity **into** the chunk vector → the chunk
  vectors stopped discriminating chunks
- aggregation let chunk scores **vote on** document identity → pinpoint
  questions, where one chunk is the whole answer, broke
- reranking with the title asked the cross-encoder to **weigh** document
  identity against chunk match → it preferred the chunk match, correctly, since
  that is what a cross-encoder is trained to judge

Document identity and passage relevance are different questions. They need
different representations, decided separately and combined by rank — not one
vector, one score, or one model asked to serve both.

That is the same principle the chunking already follows. Hierarchical chunking
keeps parent and child as separate objects for separate jobs (child to match,
parent to read). Retrieval currently has no equivalent: there is no document
object at all, only chunks that happen to share a `doc_id`. **Adding a real
document-level representation is finishing the hierarchy, not bolting something
on.**

Concretely, and in cost order:

0. **A document-type prior — the cheapest thing on this list.** Down-weight
   *Keputusan Rektor* for general questions, or surface the type in the UI so a
   user can see the answer came from one exchange programme's decree. One regex
   over the title, no embedding, no model. In the ten-question audit this alone
   separates every mismatch from every good answer.
1. **A document channel.** One point per document, embedding its title (and
   description, and later possibly a generated one-line summary). 922 points,
   ~3.8 MB, embeddings already cached. Query it in parallel with the chunk
   search and fuse by RRF, so neither channel can veto the other. Behind an
   off-by-default flag. This is the only intervention where the signal was
   demonstrated without a measured cost: ranking the 922 titles alone puts the
   tariff document at #1, #2, #4 and #6 for four of the UKT questions,
   including the two the chunk channel never retrieves at all.
2. **Conditional HyDE**, only for queries carrying no document reference. It
   attacks recall, which the document channel may already cover; measure the
   marginal gain before paying three generations per query.
3. **Nothing else from this list.**

### The honest blocker

**Every number in this document that could disqualify an intervention comes
from subset_a, and subset_a cannot represent the failing population.** All 115
of its questions carry an institutional identifier; 72 name a specific
document. They are written *from* the documents, so they arrive already
speaking the corpus's language — which is exactly why the title cosine is 0.650
there and 0.415 for a formal student question.

The queries that fail are the ones I cannot score: the 150 student questions
are unlabelled, so for them I can only report that a ranking changed, never
that it improved.

So the highest-value next step is not a retrieval change at all. It is **50–100
labelled student questions** — real phrasing, gold `doc_id` — because without
them the document channel cannot be validated, conditional HyDE cannot be
tuned, and the next promising idea will fail the same way probe 6 just did:
convincing on an anecdote, negative on the only data available, and unmeasurable
on the data that matters.

---

## What to do next

1. **Add a separate document-title channel** (probe 1) — a 922-point Qdrant
   collection fused by RRF, behind an off-by-default flag. Do **not** touch the
   chunk embeddings: concatenating the title costs 32% of within-document
   discrimination and invalidates every chunking result already run. Variant B
   (the reranker change) is three lines and can ship alongside, but it does not
   fix the UKT case either.
2. **Ship the empty-context guard** (probe 2, finding B) — this is a
   hallucination bug, not a UX nicety, and the prompt cannot fix it.
3. **Ship the mandated sentence plus the `llm_abstained` reason and badge**
   (probe 2, finding A), accepting ~10% over-refusal.
4. **Do not ship a recency prior** (probe 4).
5. **Consider HyDE only as a conditional second step** (probe 5) — gated on
   queries with no document identifier, after the document channel is in. Three
   LLM calls per query, fixes recall but not ranking: second move, not first.
6. **Do not add document-level score aggregation** (probe 6).
7. **Add the Peraturan/Keputusan distinction** (§7) — as a retrieval prior, or
   at minimum as a label in the answer's source citation. It is a regex, and in
   the audit it separated all five mismatches from all five good answers.
7. **Label 50–100 student questions.** This is the real blocker. Every
   disqualifying number here comes from subset_a, whose 115 questions all name
   an institution and 72 of which name a specific document — the population
   where these failures do not occur. Without labelled realistic queries, the
   document channel cannot be validated and the next good idea will fail the
   same way probe 6 did.
5. If probe 1 or probe 2 is going into the thesis, promote it to
   `app/thesis/_eval/` and run it against the full labelled set with the
   protocol in `app/thesis/_eval/README.md`. Neither number here carries a
   confidence interval that would survive review.
