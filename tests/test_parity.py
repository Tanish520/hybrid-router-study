"""v1 must reproduce the published upstream audit exactly (all 120 LLM catalogs)."""
import json

import pytest

from router.data import LLM_REPO, catalogs, load_source, read_lines
from router.verifier import check

LOG = LLM_REPO / "reports" / "audit_findings_log.jsonl"
pytestmark = pytest.mark.skipif(not LOG.exists(), reason="run scripts/fetch_upstream.sh first")


def test_v1_reproduces_published_audit_log():
    published = {}
    with open(LOG, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            published[(r["model"], r["lang"], r["line_idx"])] = r["issues"]

    src = load_source()
    ours = {}
    for cat in catalogs():
        if cat.paradigm != "llm":
            continue
        hyp = read_lines(cat.path)
        for i, (s, h) in enumerate(zip(src, hyp)):
            issues = check(s, h, cat.lang, "v1")
            if issues:
                ours[(cat.suite, cat.lang, i)] = issues

    assert len(ours) == len(published) == 46818
    assert ours == published
