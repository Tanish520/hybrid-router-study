"""Write the distinct (source, translation) pairs that CometKiwi must score.

Outputs in scoring/data/ (upload this folder to Google Drive for Colab):
  pairs_main.jsonl.gz    pairs from the main comparison: NLLB + GPT-4o-mini + Llama-8B (both tiers)
  pairs_rest.jsonl.gz    every other distinct pair from the remaining catalogs
  parity_check.jsonl.gz  two whole catalogs with published upstream means, scored first
  manifest.json          counts and SHA-256 of each file
"""
import gzip
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from router.data import ROOT, catalogs, load_source, read_lines  # noqa: E402
from router.scores import pair_id, published_means  # noqa: E402

OUT = ROOT / "scoring" / "data"
MAIN = {("nmt", "nllb", "output"), ("llm", "gpt4o", "baseline"), ("llm", "gpt4o", "repaired"),
        ("llm", "llama_8b", "baseline"), ("llm", "llama_8b", "repaired")}
# one NMT and one LLM catalog; their means must match the published ones before the long run
PARITY = [("nmt", "nllb", "output", "de"), ("llm", "gpt4o", "baseline", "de")]


def write_jsonl_gz(path, items):
    # mtime=0 keeps the gzip bytes identical across runs, so the SHA-256 is stable
    with open(path, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as gz:
        for it in items:
            gz.write((json.dumps(it, ensure_ascii=False) + "\n").encode("utf-8"))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    src = load_source()
    main_pairs, rest_pairs = {}, {}
    parity, published = [], published_means()
    for c in catalogs():
        hyp = read_lines(c.path)
        target = main_pairs if (c.paradigm, c.system, c.tier) in MAIN else rest_pairs
        for s, h in zip(src, hyp):
            target.setdefault(pair_id(s, h), (s, h))
        k = (c.paradigm, c.system, c.tier, c.lang)
        if k in PARITY:
            parity.append({"catalog": "/".join(k), "expected_mean": published[k],
                           "rows": [{"src": s, "mt": h} for s, h in zip(src, hyp)]})
    for pid in main_pairs:
        rest_pairs.pop(pid, None)

    files = {
        "pairs_main.jsonl.gz": [{"pair_id": k, "src": s, "mt": h} for k, (s, h) in sorted(main_pairs.items())],
        "pairs_rest.jsonl.gz": [{"pair_id": k, "src": s, "mt": h} for k, (s, h) in sorted(rest_pairs.items())],
        "parity_check.jsonl.gz": parity,
    }
    manifest = {}
    for name, items in files.items():
        write_jsonl_gz(OUT / name, items)
        manifest[name] = {"items": len(items), "sha256": hashlib.sha256((OUT / name).read_bytes()).hexdigest()}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    for name, m in manifest.items():
        print(f"{name}: {m['items']:,} items")


if __name__ == "__main__":
    main()
