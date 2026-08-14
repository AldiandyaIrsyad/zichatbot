# E6 — Give the LLM the document title and type, as structured context

**Result: JSON alone changes nothing. JSON *plus one instruction about document
types* produces the best answers on the flagship cases — but the gain is
qualitative and n=5, not a countable win.**

    python -m evals.experiments.e6_json_context     # ~60 calls

## The idea

`prompt_maker.build_context_block` emits flat prose — `Sumber 1`, page,
breadcrumbs — and **deliberately excludes the document title**. So when a decree
about one exchange programme is retrieved, the model has no way to know that is
what it is reading. It sees "Rp 500.000 per semester" with no indication of
whose semester.

Three presentations of the *same* retrieved context, on **baseline retrieval**
(the hard case, a decree on top for 5 of 10 questions):

| | |
|---|---|
| **C1 flat** | what production does now |
| **C2 json** | JSON records carrying `judul_dokumen` and `jenis_dokumen` |
| **C3 json + rule** | the same, plus one instruction: a `Keputusan` about a named programme cannot answer a general question — say who it applies to, or say you have no general rule |

## The counts say nothing

Of the 5 questions where a decree is on top, how often did the answer say who
the terms apply to, or decline?

| condition | scoped or declined | judge flagged |
|---|---|---|
| C1 flat | 3/5 | 5/10 |
| C2 json | 2/5 | 6/10 |
| C3 json + rule | 3/5 | 7/10 |

JSON alone is slightly *worse*. C3 recovers to baseline. On these numbers the
intervention does nothing.

**The metric is the problem.** It fires on phrases like "peserta program", and
deepseek already scopes its answers without being asked — unlike the flat
`Rp 500.000` the metric was designed to catch. So it cannot see the difference
that actually exists.

## Reading the answers, the difference is real

**"Berapa biaya UKT?"** — source is *Peserta Program Magang Luar Negeri*:

> **C1 flat** — "biaya UKT yang wajib dibayarkan oleh mahasiswa peserta program
> magang atau mobilitas luar negeri adalah sebesar Rp500.000 untuk satu semester."

> **C2 json** — near-identical.

> **C3 json + rule** — "besaran UKT yang disebutkan adalah Rp500.000 untuk satu
> semester. **Namun, perlu dicatat bahwa angka ini hanya tercantum dalam
> beberapa keputusan rektor yang bersifat khusus**, yaitu untuk peserta program
> magang […]"

C1 and C2 scope the figure to a population. **C3 additionally signals that no
general rule was found** — that the number exists only inside special decrees.
That is the difference between an answer a student might mistake for the tuition
rate and one that tells them this is not it.

Same pattern on *"Biaya kuliah per semester berapa?"*: C3 adds "berlaku bagi
mahasiswa peserta program mobilitas … **yang disebutkan dalam masing-masing
keputusan**".

Against it: on *"Apa saja syarat dokumen untuk daftar ulang?"* C1 opens with
"saya tidak menemukan informasi rinci" while C2 and C3 lead with the
pmb.upi.edu pointer — arguably a regression, since the honest framing comes
first in C1.

## What this actually shows

1. **Structure alone does not help.** Handing the model JSON with a title field
   changed nothing; C2 tracks C1 throughout. The model does not spontaneously
   reason about whether a source *type* fits the question.
2. **The instruction is what works**, and the JSON exists to make it
   actionable — the rule can only refer to `jenis_dokumen` if `jenis_dokumen`
   is in the context. Structure is the enabler, not the intervention.
3. **The model matters more than expected.** deepseek-v4-flash scopes its
   answers unprompted; the flat, unscoped `Rp 500.000` in the original audit
   came from a weaker generation. Part of the original failure was the model.

## Verdict

**Ship C3, not C2** — JSON context *with* the document-type instruction. It is a
prompt change plus a context-format change, no retrieval work, and it produces
the only answers in this whole set that tell a student the number they are
seeing is not the general rule.

But treat it as a **second line of defence**, not the fix. It makes a wrong
retrieval less harmful; E3 and E4b stop the wrong retrieval happening. Combining
them is untested here — E6 ran on baseline retrieval deliberately, to isolate
whether generation can recover on its own.

**Caveats.** Five decree cases. The scoping metric is too crude to score this
and the judge (E2: 7/10 exact) cannot arbitrate it either, so the conclusion
rests on reading ten answers. A labelled set with an explicit "does the answer
flag the absence of a general rule" annotation would settle it.
