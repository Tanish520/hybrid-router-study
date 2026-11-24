"""Build the stable row manifest and the catalog inventory.

Outputs
  data/manifest/rows.csv      one row per source line: row_id, line_idx, source,
                              text hash, duplicate group and occurrence, artifact flags
  data/manifest/catalogs.csv  one row per catalog: paradigm, system, tier, lang,
                              path, sha256, empty lines
"""
import csv
import hashlib
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from router.data import INDIC, LANGS, LLM_SYSTEMS, NMT_SYSTEMS, ROOT, catalogs, load_source, read_lines  # noqa: E402
from router.verifier import PLACEHOLDER_REGEX, has_accelerator_v1, has_accelerator_v2  # noqa: E402

OUT = ROOT / "data" / "manifest"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    src = load_source()
    first_seen, occurrence = {}, defaultdict(int)
    with open(OUT / "rows.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["row_id", "line_idx", "text_sha1", "dup_group", "dup_occurrence",
                    "has_placeholder", "has_accelerator_v1", "has_accelerator_v2", "source"])
        for i, s in enumerate(src):
            h = hashlib.sha1(s.encode("utf-8")).hexdigest()
            first_seen.setdefault(h, i)
            occurrence[h] += 1
            w.writerow([f"kde4-{i:05d}", i, h, f"kde4-{first_seen[h]:05d}", occurrence[h],
                        int(bool(PLACEHOLDER_REGEX.search(s))), int(has_accelerator_v1(s)),
                        int(has_accelerator_v2(s)), s])

    cats = catalogs()
    with open(OUT / "catalogs.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["paradigm", "system", "tier", "lang", "suite", "path", "sha256", "empty_lines"])
        for c in cats:
            lines = read_lines(c.path)
            w.writerow([c.paradigm, c.system, c.tier, c.lang, c.suite,
                        str(c.path.relative_to(ROOT)), hashlib.sha256(c.path.read_bytes()).hexdigest(),
                        sum(1 for x in lines if not x.strip())])

    expected = len(NMT_SYSTEMS) * len(LANGS) - (len(LANGS) - len(INDIC)) + len(LLM_SYSTEMS) * 2 * len(LANGS)
    assert len(cats) == expected == 174, (len(cats), expected)
    dups = sum(1 for v in occurrence.values() if v > 1)
    print(f"rows: {len(src)} ({len(occurrence)} distinct, {dups} texts repeated); catalogs: {len(cats)}")


if __name__ == "__main__":
    main()
