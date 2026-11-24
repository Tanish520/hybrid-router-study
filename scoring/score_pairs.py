"""Score (source, translation) pairs with CometKiwi, resumably, in chunks.

Used by notebooks/cometkiwi_scoring.ipynb on a Colab GPU. Self-contained (only
the standard library, plus `comet` inside load_cometkiwi) so it can be copied
to Google Drive next to the pair files and imported from there.

Input files (written by scripts/export_pairs.py), gzipped JSON lines:
  pairs_*.jsonl.gz      {"pair_id", "src", "mt"}, one line per distinct pair
  parity_check.jsonl.gz {"catalog", "expected_mean", "rows": [{"src", "mt"}, ...]}

Output: <out_dir>/<pairs file stem>/part-00000.csv, ... with header pair_id,cometkiwi.
A finished chunk is never rescored, so a disconnected Colab session resumes.
"""
import csv
import gzip
import json
import os
import time

MODEL_NAME = "Unbabel/wmt22-cometkiwi-da"


def load_cometkiwi(batch_size=64, gpus=1):
    """Return predict(list_of_{src,mt}) -> list of floats, using the real model."""
    from comet import download_model, load_from_checkpoint
    model = load_from_checkpoint(download_model(MODEL_NAME))

    def predict(rows):
        out = model.predict(rows, batch_size=batch_size, gpus=gpus, progress_bar=False)
        return [float(s) for s in out.scores]
    return predict


def read_jsonl_gz(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def parity_check(predict, path, tolerance=0.002):
    """Score whole catalogs (every row, in order) and compare with the published means."""
    report = []
    for item in read_jsonl_gz(path):
        rows = [{"src": r["src"], "mt": r["mt"]} for r in item["rows"]]
        scores = predict(rows)
        mean = sum(scores) / len(scores)
        diff = mean - item["expected_mean"]
        report.append({"catalog": item["catalog"], "rows": len(rows), "ours": round(mean, 6),
                       "published": item["expected_mean"], "diff": round(diff, 6),
                       "ok": abs(diff) <= tolerance})
    return report


def score_file(predict, pairs_path, out_dir, chunk_size=20000, log=print):
    """Score every pair in pairs_path, writing one CSV per chunk. Returns pairs scored now."""
    stem = os.path.basename(pairs_path).split(".")[0]
    part_dir = os.path.join(out_dir, stem)
    os.makedirs(part_dir, exist_ok=True)
    pairs = list(read_jsonl_gz(pairs_path))
    n_chunks = (len(pairs) + chunk_size - 1) // chunk_size
    scored_now, t0 = 0, time.time()
    for k in range(n_chunks):
        part = os.path.join(part_dir, f"part-{k:05d}.csv")
        if os.path.exists(part):
            continue
        chunk = pairs[k * chunk_size:(k + 1) * chunk_size]
        scores = predict([{"src": p["src"], "mt": p["mt"]} for p in chunk])
        if len(scores) != len(chunk):
            raise RuntimeError(f"chunk {k}: {len(scores)} scores for {len(chunk)} pairs")
        tmp = part + ".tmp"
        with open(tmp, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["pair_id", "cometkiwi"])
            for p, s in zip(chunk, scores):
                w.writerow([p["pair_id"], repr(s)])
        os.replace(tmp, part)  # a chunk file exists only once it is complete
        scored_now += len(chunk)
        rate = scored_now / max(time.time() - t0, 1e-9)
        left = sum(1 for j in range(k + 1, n_chunks)
                   if not os.path.exists(os.path.join(part_dir, f"part-{j:05d}.csv")))
        log(f"{stem}: chunk {k + 1}/{n_chunks} done, {rate:.0f} pairs/s, "
            f"about {left * chunk_size / max(rate, 1e-9) / 60:.0f} min left")
    return scored_now


def time_sample(predict, pairs_path, n=2000):
    """Score the first n pairs once and return pairs per second (for an ETA)."""
    rows = []
    for p in read_jsonl_gz(pairs_path):
        rows.append({"src": p["src"], "mt": p["mt"]})
        if len(rows) == n:
            break
    t0 = time.time()
    predict(rows)
    return len(rows) / (time.time() - t0)
