import csv
import gzip
import json

from router.scores import pair_id, published_means
from scoring.score_pairs import parity_check, score_file


def fake_predict(rows):
    # deterministic stand-in for CometKiwi: depends only on (src, mt)
    return [((len(r["src"]) * 7 + len(r["mt"]) * 3) % 100) / 100 for r in rows]


def write_pairs(path, pairs):
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for s, m in pairs:
            f.write(json.dumps({"pair_id": pair_id(s, m), "src": s, "mt": m}) + "\n")


def test_pair_id_is_stable_and_order_sensitive():
    assert pair_id("Cancel", "रद्द करें") == pair_id("Cancel", "रद्द करें")
    assert pair_id("a", "bc") != pair_id("ab", "c")
    assert len(pair_id("x", "y")) == 20


def test_score_file_writes_every_pair_and_resumes(tmp_path):
    pairs = [(f"src {i}", f"mt {i * i}") for i in range(25)]
    write_pairs(tmp_path / "pairs_demo.jsonl.gz", pairs)
    n = score_file(fake_predict, str(tmp_path / "pairs_demo.jsonl.gz"), str(tmp_path / "out"),
                   chunk_size=10, log=lambda *_: None)
    assert n == 25
    parts = sorted((tmp_path / "out" / "pairs_demo").glob("part-*.csv"))
    assert len(parts) == 3
    got = {}
    for p in parts:
        with open(p) as f:
            got.update({r["pair_id"]: float(r["cometkiwi"]) for r in csv.DictReader(f)})
    expected = dict(zip([pair_id(s, m) for s, m in pairs],
                        fake_predict([{"src": s, "mt": m} for s, m in pairs])))
    assert got == expected
    # a second run must not rescore finished chunks
    calls = []
    score_file(lambda rows: calls.append(1) or fake_predict(rows), str(tmp_path / "pairs_demo.jsonl.gz"),
               str(tmp_path / "out"), chunk_size=10, log=lambda *_: None)
    assert calls == []


def test_parity_check_compares_with_expected_mean(tmp_path):
    rows = [{"src": "Open", "mt": "Öffnen"}, {"src": "Cancel", "mt": "Abbrechen"}]
    mean = sum(fake_predict(rows)) / 2
    with gzip.open(tmp_path / "parity.jsonl.gz", "wt", encoding="utf-8") as f:
        f.write(json.dumps({"catalog": "x", "expected_mean": mean, "rows": rows}) + "\n")
        f.write(json.dumps({"catalog": "y", "expected_mean": mean + 0.01, "rows": rows}) + "\n")
    report = parity_check(fake_predict, str(tmp_path / "parity.jsonl.gz"))
    assert [r["ok"] for r in report] == [True, False]


def test_published_means_cover_all_174_catalogs():
    pub = published_means()
    assert len(pub) == 174
    assert abs(pub[("nmt", "nllb", "output", "de")] - 0.777472835609607) < 1e-12
    assert pub[("llm", "gpt4o", "baseline", "de")] == 0.7868
