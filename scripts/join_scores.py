"""Join the Colab CometKiwi pair scores back to every catalog row and validate them.

Input:  scoring/scores/<pairs stem>/part-*.csv (downloaded from Google Drive)
Output: results/00_cometkiwi_scores/cometkiwi_pair_scores.csv.gz   pair_id, cometkiwi (one row per distinct pair)
        results/00_cometkiwi_scores/validation_vs_published_means.csv       per catalog: rows scored, our mean, published mean, difference
Exits with status 1 if any fully scored catalog differs from its published mean by more than --tolerance.
"""
import argparse
import csv
import gzip
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from router.data import ROOT, catalogs, load_source, read_lines  # noqa: E402
from router.paths import SCORES, ensure  # noqa: E402
from router.scores import pair_id, published_means  # noqa: E402


def load_pair_scores(scores_dir):
    scores = {}
    for part in sorted(Path(scores_dir).glob("*/part-*.csv")):
        with open(part, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                scores[r["pair_id"]] = float(r["cometkiwi"])
    return scores


def validate(scores, tolerance):
    src, published, rows = load_source(), published_means(), []
    for c in catalogs():
        hyp = read_lines(c.path)
        got = [scores.get(pair_id(s, h)) for s, h in zip(src, hyp)]
        have = [g for g in got if g is not None]
        k = (c.paradigm, c.system, c.tier, c.lang)
        ours = sum(have) / len(have) if len(have) == len(got) else None
        pub = published.get(k)
        diff = None if ours is None or pub is None else ours - pub
        rows.append({"paradigm": c.paradigm, "system": c.system, "tier": c.tier, "lang": c.lang,
                     "rows_scored": len(have), "rows": len(got),
                     "ours": None if ours is None else round(ours, 6), "published": pub,
                     "diff": None if diff is None else round(diff, 6),
                     "ok": None if diff is None else abs(diff) <= tolerance})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores-dir", default=str(ROOT / "scoring" / "scores"))
    ap.add_argument("--tolerance", type=float, default=0.002)
    args = ap.parse_args()
    scores = load_pair_scores(args.scores_dir)
    if not scores:
        sys.exit(f"no score files under {args.scores_dir}")
    out = ensure(SCORES)
    with gzip.open(out / "cometkiwi_pair_scores.csv.gz", "wt", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["pair_id", "cometkiwi"])
        for k in sorted(scores):
            w.writerow([k, scores[k]])
    rows = validate(scores, args.tolerance)
    with open(out / "validation_vs_published_means.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    done = [r for r in rows if r["ok"] is not None]
    bad = [r for r in done if not r["ok"]]
    print(f"{len(scores):,} pair scores; {len(done)}/{len(rows)} catalogs fully scored; "
          f"{len(bad)} outside ±{args.tolerance} of the published mean")
    if done:
        print(f"largest difference: {max(abs(r['diff']) for r in done):.6f}")
    for r in bad:
        print(f"  MISMATCH {r['paradigm']}/{r['system']}/{r['tier']}/{r['lang']}: "
              f"ours {r['ours']} vs published {r['published']}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
