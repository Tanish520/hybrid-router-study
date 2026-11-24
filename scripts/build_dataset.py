"""Build the router dataset: one row per (English string, target language).

For each row: the 12 Stage 1 features, the duplicate group (for grouped CV),
and for every candidate system its verifier result (v1, v2) and its per-string
CometKiwi score. Output: results/datasets/router_dataset.csv.gz (122,736 rows).

Systems: the four NMT engines with all 12 languages, and the two LLMs of the
main comparison (baseline and IGSC-repaired tiers).
"""
import csv
import gzip
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from router.data import LANGS, ROOT, catalogs, load_source, read_lines  # noqa: E402
from router.features import FEATURES, features  # noqa: E402
from router.paths import DATASETS, PAIR_SCORES, ROUTER_DATASET, ensure  # noqa: E402
from router.scores import pair_id  # noqa: E402
from router.verifier import check  # noqa: E402

SYSTEMS = {  # column prefix -> (system, tier)
    "nllb": ("nllb", "output"), "google": ("google", "output"),
    "microsoft": ("microsoft", "output"), "madlad": ("madlad", "output"),
    "gpt4o_base": ("gpt4o", "baseline"), "gpt4o_igsc": ("gpt4o", "repaired"),
    "llama_base": ("llama_8b", "baseline"), "llama_igsc": ("llama_8b", "repaired"),
}


def load_scores():
    with gzip.open(PAIR_SCORES, "rt", encoding="utf-8") as f:
        return {r["pair_id"]: float(r["cometkiwi"]) for r in csv.DictReader(f)}


def main():
    src = load_source()
    rows_df = pd.read_csv(ROOT / "data" / "manifest" / "rows.csv", keep_default_na=False)
    group = dict(zip(rows_df.line_idx, rows_df.dup_group))
    scores = load_scores()
    paths = {(c.system, c.tier, c.lang): c.path for c in catalogs()}

    records = []
    for lang in LANGS:
        hyps = {p: read_lines(paths[(s, t, lang)]) for p, (s, t) in SYSTEMS.items()}
        for i, s in enumerate(src):
            r = {"line_idx": i, "lang": lang, "group": group[i]}
            r.update(features(s, lang))
            for p, h in hyps.items():
                r[f"{p}_c1"] = int(not check(s, h[i], lang, "v1"))
                r[f"{p}_c2"] = int(not check(s, h[i], lang, "v2"))
                r[f"{p}_ck"] = scores[pair_id(s, h[i])]
            records.append(r)
    df = pd.DataFrame(records)
    assert len(df) == 12 * 10228 and list(df.columns[3:15]) == FEATURES
    ensure(DATASETS)
    df.to_csv(ROUTER_DATASET, index=False, compression="gzip")
    print(f"{len(df):,} rows, {df.group.nunique():,} source groups, {len(df.columns)} columns")


if __name__ == "__main__":
    main()
