"""Experiment 1: run the six-invariant verifier (v1 and v2) over all 174 catalogs.

Outputs
  results/01_structural_audit/failure_labels.csv.gz  every failing (catalog, line, version) with its issues
  results/01_structural_audit/catalog_summary.csv    per catalog and version: counts per check, defects, rate
"""
import csv
import gzip
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from router.data import catalogs, load_source, read_lines  # noqa: E402
from router.paths import AUDIT, ensure  # noqa: E402
from router.verifier import SIX_INVARIANTS, VERSIONS, accelerator_failure_kind, check  # noqa: E402

RESULTS = AUDIT
CHECKS = ("empty_output",) + SIX_INVARIANTS + ("accel_dropped", "accel_escaped", "accel_detached")


def main():
    ensure(RESULTS)
    src = load_source()
    summary = []
    with gzip.open(RESULTS / "failure_labels.csv.gz", "wt", newline="", encoding="utf-8") as gz:
        w = csv.writer(gz)
        w.writerow(["version", "paradigm", "system", "tier", "lang", "line_idx", "issues"])
        for cat in catalogs():
            hyp = read_lines(cat.path)
            for version in VERSIONS:
                counts, defective = Counter(), 0
                for i, (s, h) in enumerate(zip(src, hyp)):
                    issues = check(s, h, cat.lang, version)
                    if issues:
                        defective += 1
                        counts.update(issues)
                        if version == "v2" and "broken_accelerator" in issues:
                            counts["accel_" + accelerator_failure_kind(h)] += 1
                        w.writerow([version, cat.paradigm, cat.system, cat.tier, cat.lang, i, ";".join(issues)])
                summary.append({"version": version, "paradigm": cat.paradigm, "system": cat.system,
                                "tier": cat.tier, "lang": cat.lang, "lines": len(hyp),
                                **{c: counts[c] for c in CHECKS}, "defective": defective,
                                "defect_rate_pct": round(100 * defective / len(hyp), 4)})
    with open(RESULTS / "catalog_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)
    print(f"audited {len(summary) // len(VERSIONS)} catalogs x {len(VERSIONS)} verifier versions")


if __name__ == "__main__":
    main()
