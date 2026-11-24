"""Per-string CometKiwi scores: pair identifiers and the published upstream means.

CometKiwi reads only (source, translation), so every distinct pair is scored
once and the score is shared by every catalog row that contains that pair.
"""
import csv
import hashlib

from router.data import LLM_REPO, NMT_REPO

MODEL_NAME = "Unbabel/wmt22-cometkiwi-da"

# NMT repo display names (metrics/cometkiwi/cometkiwi_scores.csv) -> our system ids
NMT_PUBLISHED_NAMES = {
    "google_translate": "google", "microsoft_translator": "microsoft",
    "nllb200": "nllb", "madlad400": "madlad", "indictrans2": "indictrans2",
}


def pair_id(src, mt):
    """Stable 20-hex-character identifier for one (source, translation) pair."""
    return hashlib.sha1((src + "\x1f" + mt).encode("utf-8")).hexdigest()[:20]


def published_means():
    """{(paradigm, system, tier, lang): published CometKiwi mean} from both repos."""
    out = {}
    with open(NMT_REPO / "metrics" / "cometkiwi" / "cometkiwi_scores.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out[("nmt", NMT_PUBLISHED_NAMES[r["system"]], "output", r["lang"])] = float(r["cometkiwi"])
    with open(LLM_REPO / "metrics" / "cometkiwi" / "cometkiwi_scores.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            tier, system = r["system"].split("_", 1)
            out[("llm", system, tier, r["language"])] = float(r["cometkiwi"])
    return out
