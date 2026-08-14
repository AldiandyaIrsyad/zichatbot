# E2 — Can an LLM judge this reliably enough to score the other experiments?

**Answer: as a screen, yes. As the score, no.** A better prompt fixes the
systematic failure but exact agreement stays at 7/10, so the retrieval
experiments are scored on objective measures instead and the judge is used only
to flag.

    python -m evals.experiments.e2_judge     # ~40 calls

## Why calibrate first

The qwen judge in the original audit got 2 of 10 wrong, and both errors were
structural rather than random:

- **Question 1** it passed. The answer said *"biaya UKT bagi mahasiswa peserta
  program magang luar negeri adalah Rp 500.000"* and the title was about a
  magang programme, so answer and title agreed **with each other** — and the
  judge scored that agreement. It never asked whether either was right for the
  question, which was a general one.
- **Question 7** it failed. The system had correctly used an amending document
  whose title describes *what it amends* rather than what it contains.

Both come from comparing the answer to the title. So prompt v2 forces the
comparison to be question-to-title first, and adds an explicit note about
`Perubahan Atas` documents.

## Result

Reference: ten verdicts assigned by hand.

| judge | exact 3-way | flag agreement | self-consistent |
|---|---|---|---|
| qwen3-14b, prompt v1 | 6/10 | 8/10 | — |
| deepseek-v4-flash, prompt v1 | 7/10 | 7/10 | 9/10 |
| **deepseek-v4-flash, prompt v2** | **7/10** | 7/10 | **10/10** |

Per question:

| # | manual | qwen v1 | ds v1 | ds v2 | |
|---|---|---|---|---|---|
| 1 | MISMATCH | OK | MISMATCH | MISMATCH | v2 fixes the systematic error |
| 2 | MISMATCH | MISMATCH | MISMATCH | MISMATCH | |
| 3 | PARTIAL | MISMATCH | OK | OK | disagrees |
| 4 | OK | OK | OK | OK | |
| 5 | MISMATCH | MISMATCH | MISMATCH | MISMATCH | |
| 6 | OK | OK | OK | OK | |
| 7 | PARTIAL | MISMATCH | OK | OK | disagrees |
| 8 | MISMATCH | MISMATCH | OK | MISMATCH | v2 fixes |
| 9 | OK | MISMATCH | OK | MISMATCH | v2 breaks |
| 10 | MISMATCH | MISMATCH | MISMATCH | MISMATCH | |

**v2 fixes both of qwen's structural errors** (1 and 8) and is perfectly stable
across repeats. It still disagrees on 3 and 7 — where it calls PARTIAL cases OK,
which is defensible, those are genuinely borderline — and it newly breaks 9,
where the system correctly reported that a limit is not specified.

## Verdict

Prompt v2 on deepseek-v4-flash is **good enough to flag, not to score**. Its
remaining disagreements cluster on the PARTIAL band, which is exactly where a
binary flag is least meaningful.

Consequences for the other experiments:

- Primary metrics are objective and need no judge: **is the winning document a
  Keputusan or a Peraturan**, and **did the expected document win**.
- The judge is reported alongside as a secondary signal.

**Caveat:** n=10. This is enough to catch a judge that is systematically wrong
— which it did — and nowhere near enough to establish a reliability figure. Any
thesis use needs a proper labelled set and Cohen's kappa against two annotators.
