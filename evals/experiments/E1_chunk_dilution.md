# E1 — Is hierarchical chunking diluting the embedding?

**Question asked:** is the wrong-document problem caused by hierarchical
semantic chunking watering down the child vectors?

**Answer: no.** The breadcrumb tag does dilute, measurably, but removing it does
not change which document wins. It is not the mechanism behind the failure.

    python -m evals.experiments.e1_chunk_dilution     # no LLM, no cost

## What was tested

Every child chunk is embedded as `"BAB II > Pasal 5\n\n" + body`
(`app/thesis/chunking/logic.py:458`). Two hypotheses:

1. **Dilution** — the tag is near-constant within a section, so chunks in the
   same section drift together and lose discrimination.
2. **Asymmetry** — a *Peraturan* is structured (BAB / Pasal) and gets rich
   breadcrumbs; a *Keputusan* is a flat decree (`MEMUTUSKAN: KESATU / KEDUA`)
   and gets almost none. If the tag dilutes, it would dilute regulations
   specifically — the exact class that loses to decrees in the audit.

17,096 chunks from the 94 documents in the ten questions' candidate pool were
embedded twice: as stored, and with the tag stripped by the production
chunker's own `_strip_breadcrumb_tag`, so the strip is exactly the inverse of
what ingestion did.

## Result 1 — the asymmetry hypothesis is wrong

| document type | mean share of chunk that is tag | chunks carrying a tag |
|---|---|---|
| Keputusan (decree) | 13.4% | 92% |
| Peraturan (rule) | 15.9% | 97% |

Nearly identical. Decrees are **not** structurally advantaged by carrying less
breadcrumb. The hypothesis fails on its own terms.

That is worth knowing on its own: decrees turn out to have breadcrumbs too,
because the parser finds *some* structure in `MEMUTUSKAN / KESATU / KEDUA`.

## Result 2 — dilution is real but small

| | intra-document cosine |
|---|---|
| with tag (as shipped) | 0.5954 |
| tag stripped | 0.5520 |
| **effect of the tag** | **+0.0434** |

The tag does pull a document's chunks together. For scale, prepending the
document *title* moved the same measure by **+0.194** (probe 1) — four and a
half times as much. The breadcrumb tag is a minor contributor.

## Result 3 — and it does not change the outcome

| | with tag | tag stripped |
|---|---|---|
| decree on top | 2/10 | 2/10 |
| mean rank of the expected document | 4.5 | 3.5 |

Per question, rank of the document that should have won:

| question | with tag | stripped |
|---|---|---|
| Berapa biaya UKT? | #6 | #3 |
| Biaya kuliah per semester berapa? | #16 | #13 |
| Bagaimana proses penentuan golongan tarif UKT? | #1 | #1 |
| Apa syarat memperoleh keringanan biaya pendidikan? | #1 | #1 |
| Apakah ada tarif khusus untuk parkir kendaraan? | #1 | #1 |
| Bagaimana syarat pendaftaran stiker izin parkir? | #2 | #2 |

Stripping helps marginally on two questions and changes nothing on four. The
same documents win either way.

*(These ranks are better than production's because this is dense-only over 94
documents, not hybrid over 922. Only the with/against comparison is meaningful
here, not the absolute positions.)*

## Verdict

**Do not remove the breadcrumb tag.** It costs a small amount of
within-document discrimination and buys the structural signal — "Pasal 5" —
that hierarchical chunking exists to provide. Removing it would surrender that
for a rank improvement that does not change any outcome.

The wrong-document problem is not a chunking problem. It is a document-selection
problem, and E3/E4 address it where it lives.
