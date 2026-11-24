"""Experiment 6, step 4 (needs a person): blind A/B check that does not rely on CometKiwi.

Make a sheet:   python improved_router/spot_check.py make  --lang hi --n 200
Score it:       python improved_router/spot_check.py score --lang hi

"make" takes the strings where predictive + reactive (Experiment 3) and the improved C+B router
(NLLB + GPT-4o-mini, gate v2, both sending ~10% of strings to the LLM) deliver DIFFERENT
translations, samples n of them, and shuffles which router is shown as A or B. The reviewer
fills the "better" column with A, B or same (and may add notes), without knowing which is which.
The key that says which router is A is written to a separate file; don't open it before scoring.

"score" reads the filled sheet and the key and reports how often each router won,
with a 95% interval (Wilson) for the share of non-tied judgements won by C+B.
"""
import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from improved_router.common import RES, load, matched_theta, outcomes, route, score  # noqa: E402
from router.data import catalogs, load_source, read_lines  # noqa: E402

OUT = RES / "spot_check"
NMT, LLM, GATE, TARGET = "nllb", "gpt4o", "v2", 10


def make(lang, n, seed):
    df, ref, ext = load(NMT)
    o = outcomes(df, NMT, LLM, GATE)
    final = {}
    for p in ("predictive_reactive", "CB_post_nmt_gain"):
        s = score(p, ref, ext, LLM, GATE)
        final[p] = route(p, s, matched_theta(p, s, o, TARGET), o)[1]
    paths = {(c.system, c.tier, c.lang): c.path for c in catalogs()}
    nmt_txt, llm_txt = read_lines(paths[("nllb", "output", lang)]), read_lines(paths[("gpt4o", "repaired", lang)])
    src = load_source()
    m = (df.lang == lang).values
    rows = []
    for i in np.flatnonzero(m):
        li = df.line_idx.values[i]
        a = llm_txt[li] if final["predictive_reactive"][i] else nmt_txt[li]
        b = llm_txt[li] if final["CB_post_nmt_gain"][i] else nmt_txt[li]
        if a != b:
            rows.append((li, src[li], a, b))
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(rows), min(n, len(rows)), replace=False)
    sheet, key = [], []
    for k, j in enumerate(pick):
        li, s, ref_t, cb_t = rows[j]
        flip = bool(rng.integers(2))
        sheet.append({"id": k + 1, "english": s, "translation_A": cb_t if flip else ref_t,
                      "translation_B": ref_t if flip else cb_t, "better": "", "notes": ""})
        key.append({"id": k + 1, "line_idx": li, "A_is": "C+B" if flip else "reference"})
    OUT.mkdir(exist_ok=True)
    pd.DataFrame(sheet).to_csv(OUT / f"sheet_{lang}.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(key).to_csv(OUT / f"key_{lang}_DO_NOT_OPEN.csv", index=False)
    print(f"{len(rows)} {lang} strings differ between the two routers; sheet with {len(sheet)} rows: "
          f"{OUT / f'sheet_{lang}.csv'}")


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


def score_sheet(lang):
    sheet = pd.read_csv(OUT / f"sheet_{lang}.csv", encoding="utf-8-sig")
    key = pd.read_csv(OUT / f"key_{lang}_DO_NOT_OPEN.csv")
    d = sheet.merge(key, on="id")
    b = d.better.astype(str).str.strip().str.upper()
    d = d[b.isin(["A", "B", "SAME"])].assign(better=b)
    cb = ((d.better == "A") & (d.A_is == "C+B")) | ((d.better == "B") & (d.A_is == "reference"))
    gd = ((d.better == "A") & (d.A_is == "reference")) | ((d.better == "B") & (d.A_is == "C+B"))
    k, n = int(cb.sum()), int(cb.sum() + gd.sum())
    lo, hi = wilson(k, n)
    print(f"{len(d)} judged: C+B better {k}, reference router better {int(gd.sum())}, same {int((d.better == 'SAME').sum())}")
    print(f"C+B wins {100 * k / max(n, 1):.0f}% of non-tied judgements, 95% interval [{100 * lo:.0f}%, {100 * hi:.0f}%] "
          "(above 50% throughout = C+B is genuinely better to a human)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["make", "score"])
    ap.add_argument("--lang", default="hi")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    make(a.lang, a.n, a.seed) if a.action == "make" else score_sheet(a.lang)
