# E3 — Down-weight decrees (Keputusan) for general questions

**Result: the single most effective intervention tested, and it is a regex.**
Decrees on top drop from **5/10 to 3/10**; combined with E4b, to **0/10**.

    python -m evals.experiments.e3_retrieval     # local, no LLM

## The idea

The audit found perfect separation: all five mismatches came from a *Keputusan
Rektor*, all five acceptable answers from a *Peraturan Rektor*.

- A **Peraturan** states a general rule that applies to everyone.
- A **Keputusan** decides something about named people or one event — who joins
  an exchange programme, who passed a selection.

A student's general question needs the rule. But the decree usually matches the
words better, because it states a concrete figure where the rule states a
formula: `Rp 500.000` beats *"ditetapkan berdasarkan kemampuan ekonomi orang
tua"* on a query about cost, every time.

## The implementation

The type is in the title's own numbering — no embedding, no model call:

```python
_PERATURAN_NUM = re.compile(r"^\s*\d+\s+[Tt]ahun\s+(?:19|20)\d{2}\b")
_HK_CODE       = re.compile(r"[-/]HK[-/.]", re.I)

def doc_type(title):
    head = title.split(" - ")[0]
    return "Peraturan" if (_PERATURAN_NUM.match(title) or _HK_CODE.search(head)) \
        else "Keputusan"
```

`41 Tahun 2023` and `2151-UN40-HK-2019` are rules. `346-UN40-KM.02.02-2026` and
`549-UN40-TM.01.04-2026` are decrees. Applied as a multiplicative penalty of
**0.5** on the min-max normalised rerank score.

## Result

| intervention | decree on top | expected doc first |
|---|---|---|
| baseline | 5/10 | 2/6 |
| **E3 doctype prior** | **3/10** | 2/6 |
| E3 + E4b title prior | **0/10** | **3/6** |

Two of the five decree wins flip to regulations on the prior alone; all five
flip when combined with the title prior.

## The catch

**Making the decree lose does not make the right rule win.** `expected doc
first` stays at 2/6 for E3 alone. For "Berapa biaya UKT?" the top result moves
from *Peserta Program Magang Luar Negeri* to *031 Tahun 2022 — Penyelenggaraan
Program Percepatan Studi* — a regulation, but not the tariff one.

So this is a **harm-reduction** intervention. It stops the system answering a
general question with one programme's decree, which is the failure that
produces confidently wrong numbers. It does not, by itself, find the right
document.

## Risks worth testing before shipping

- **Questions that genuinely want a decree.** "Siapa saja peserta program
  outbound ke Sookmyung?" *should* return the decree. A flat penalty would hurt
  it. subset_a is full of such questions (72 of 115 name a specific document),
  which is where the −18pt result in probe 6 came from for a different
  intervention — the same trap applies here.
- The penalty value 0.5 is a guess, not tuned.

**Recommended shape:** condition the penalty on the query. Apply it only when
the question carries no document reference (no nomor, no tahun, no UN40 code) —
the same gate proposed for HyDE. That population is exactly the ten questions
here, and exactly where subset_a's evidence does not apply.

## Verdict

**Ship it, gated on identifier-free queries.** It is a regex over a string
already in the database, it addresses the failure mode the audit identified,
and it is trivially reversible. Validate on labelled student questions before
enabling globally.
