"""E2 — can an LLM judge this reliably enough to score the other experiments?

The qwen judge in the original audit got 2 of 10 wrong, and both errors were
structural rather than random:

* **Question 1** it passed. The answer said "biaya UKT bagi mahasiswa peserta
  program magang luar negeri adalah Rp 500.000" and the title was about a
  magang programme, so answer and title agreed *with each other* — and the
  judge scored that agreement. It never asked whether either was right for the
  question, which was a general one about UKT.
* **Question 7** it failed. The system had correctly used an amending document
  whose title describes *what it amends* rather than what it contains.

Both failures come from the judge comparing the answer to the title. So this
tests two prompts: the original, and one that forces the comparison to be
question-to-title first, with an explicit note about amending documents.

Ten hand-assigned labels are the reference. Ten items cannot establish a
reliability figure — the point is to catch a judge that is systematically
wrong before trusting it to score five other experiments.

    python -m evals.experiments.e2_judge

Writes results/e2_judge.json. ~40 LLM calls.
"""

from __future__ import annotations

import asyncio
import csv
import json
from typing import Dict, List, Tuple

from evals.probes._common import RESULTS_DIR, quiet_logs
from evals.probes.demo_answer_audit import JUDGE_SYSTEM as JUDGE_V1, MANUAL_VERDICTS
from ._bench import MODEL, doc_type

JUDGE_V2 = (
    "Anda mengaudit sistem pencarian dokumen hukum kampus (UPI). Nilai SATU "
    "hal saja: apakah dokumen sumber itu jenis dokumen yang tepat untuk "
    "menjawab PERTANYAAN tersebut.\n\n"
    "Urutan penilaian yang WAJIB Anda ikuti:\n"
    "1. Baca PERTANYAAN. Tentukan apakah itu pertanyaan umum (berlaku bagi "
    "semua mahasiswa) atau pertanyaan tentang satu orang/kegiatan tertentu.\n"
    "2. Baca JUDUL dokumen sumber. Tentukan apakah dokumen itu aturan umum "
    "(Peraturan Rektor) atau keputusan administratif tentang orang/kegiatan "
    "tertentu (Keputusan Rektor: 'Peserta Program ...', 'Peserta Lulus "
    "Seleksi ...').\n"
    "3. Pertanyaan umum yang dijawab oleh keputusan administratif tentang satu "
    "program adalah TIDAK_MASUK_AKAL, meskipun angka dalam jawabannya benar "
    "untuk program itu.\n\n"
    "JANGAN menilai apakah jawaban konsisten dengan judul. Jawaban dan judul "
    "bisa saja cocok satu sama lain tetapi keduanya salah untuk pertanyaannya "
    "— itu tetap TIDAK_MASUK_AKAL.\n\n"
    "Catatan: dokumen berjudul 'Perubahan Atas Peraturan ...' adalah aturan "
    "umum yang sah. Judulnya menyebut aturan yang diubahnya, bukan isinya, "
    "jadi jangan menolaknya hanya karena judul tidak menyebut topik "
    "pertanyaan.\n\n"
    "Jawab HANYA satu baris:\n"
    "VERDICT: MASUK_AKAL|TIDAK_MASUK_AKAL|SEBAGIAN — <alasan satu kalimat>"
)

TEMPLATE = (
    "PERTANYAAN: {question}\n\n"
    "JAWABAN SISTEM: {answer}\n\n"
    "JUDUL DOKUMEN SUMBER: {title}\n\n"
    "Apakah dokumen itu sumber yang tepat untuk pertanyaan tersebut?"
)

# The judge's three labels collapse onto the manual ones for scoring: the
# decision that matters is "would this have been flagged".
TO_MANUAL = {
    "MASUK_AKAL": "OK",
    "SEBAGIAN": "PARTIAL",
    "TIDAK_MASUK_AKAL": "MISMATCH",
}


def parse(text: str) -> Tuple[str, str]:
    line = text.strip().splitlines()[0] if text.strip() else ""
    body = line.split("VERDICT:", 1)[-1].strip()
    for tag in ("TIDAK_MASUK_AKAL", "SEBAGIAN", "MASUK_AKAL"):
        if body.upper().startswith(tag):
            return tag, body[len(tag):].lstrip(" —-:").strip()
    return "UNPARSED", line[:140]


async def judge_all(llm, system: str, rows: List[dict], repeats: int = 2
                    ) -> List[List[str]]:
    """Verdicts per row, repeated to expose the judge's own instability."""
    out: List[List[str]] = []
    for r in rows:
        verdicts = []
        for _ in range(repeats):
            resp = await llm.chat(
                [{"role": "system", "content": system},
                 {"role": "user", "content": TEMPLATE.format(
                     question=r["question"], answer=r["answer"][:700],
                     title=r["source_title"])}],
                temperature=0.0, max_tokens=140,
            )
            verdicts.append(parse(resp)[0])
            await asyncio.sleep(1.0)
        out.append(verdicts)
    return out


async def main() -> None:
    quiet_logs()
    from dotenv import load_dotenv

    load_dotenv(".env")
    from evals._shared.clients import get_llm_client_from_env

    print(f"E2 — judge calibration  [{MODEL}]\n")

    with open(RESULTS_DIR / "demo_answer_audit.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    manual = [MANUAL_VERDICTS[n][0] for n in range(1, len(rows) + 1)]

    llm = get_llm_client_from_env(model=MODEL)
    try:
        v1 = await judge_all(llm, JUDGE_V1, rows)
        v2 = await judge_all(llm, JUDGE_V2, rows)
    finally:
        await llm.aclose()

    def score(verdicts: List[List[str]]) -> dict:
        first = [TO_MANUAL.get(v[0], v[0]) for v in verdicts]
        exact = sum(1 for a, b in zip(first, manual) if a == b)
        # "Would it have been flagged" — the decision that actually matters.
        flag_pred = [v != "OK" for v in first]
        flag_true = [m != "OK" for m in manual]
        agree = sum(1 for a, b in zip(flag_pred, flag_true) if a == b)
        stable = sum(1 for v in verdicts if len(set(v)) == 1)
        return {"exact_3way": exact, "flagged_agreement": agree,
                "self_consistent": stable, "n": len(manual),
                "labels": first}

    s1, s2 = score(v1), score(v2)
    qwen = [r["verdict"] for r in rows]
    qwen_first = [TO_MANUAL.get(v, v) for v in qwen]
    qwen_exact = sum(1 for a, b in zip(qwen_first, manual) if a == b)
    qwen_flag = sum(1 for a, b in zip(qwen_first, manual)
                    if (a != "OK") == (b != "OK"))

    print(f"  {'judge':28s} {'exact 3-way':>12s} {'flag agree':>11s} {'stable':>8s}")
    print(f"  {'qwen3-14b, prompt v1':28s} {qwen_exact:>9d}/10 {qwen_flag:>8d}/10 "
          f"{'n/a':>8s}")
    print(f"  {'deepseek, prompt v1':28s} {s1['exact_3way']:>9d}/10 "
          f"{s1['flagged_agreement']:>8d}/10 {s1['self_consistent']:>5d}/10")
    print(f"  {'deepseek, prompt v2':28s} {s2['exact_3way']:>9d}/10 "
          f"{s2['flagged_agreement']:>8d}/10 {s2['self_consistent']:>5d}/10")

    print(f"\n  {'#':>2} {'manual':9s} {'qwen v1':9s} {'ds v1':9s} {'ds v2':9s} question")
    for i, r in enumerate(rows):
        mark = "" if s2["labels"][i] == manual[i] else "   <-- v2 differs"
        print(f"  {i+1:2d} {manual[i]:9s} {qwen_first[i]:9s} {s1['labels'][i]:9s} "
              f"{s2['labels'][i]:9s} {r['question'][:40]}{mark}")

    out = {
        "model": MODEL,
        "reference": "10 manual verdicts (demo_answer_audit.MANUAL_VERDICTS)",
        "qwen_prompt_v1": {"exact_3way": qwen_exact, "flagged_agreement": qwen_flag},
        "deepseek_prompt_v1": s1,
        "deepseek_prompt_v2": s2,
        "manual": manual,
        "caveat": "n=10. This catches a systematically wrong judge; it cannot "
                  "establish a reliability figure.",
    }
    (RESULTS_DIR / "e2_judge.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  Written to {RESULTS_DIR / 'e2_judge.json'}")


if __name__ == "__main__":
    asyncio.run(main())
