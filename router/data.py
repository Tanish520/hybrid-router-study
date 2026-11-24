"""Load the two upstream benchmark repositories as one aligned table.

Rows are aligned by line position (both repos ship the same en.txt, SHA-256
464d6582...). A catalog is rejected, not truncated, if its line count differs.
"""
import hashlib
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXTERNAL = ROOT / "external"
NMT_REPO = EXTERNAL / "cloud-native-nmt-benchmark"
LLM_REPO = EXTERNAL / "cloud-native-llm-benchmark"
SOURCE_SHA256 = "464d65825b81856c4cb600638fca87c3b3abc1692239679e1ec093677bfe281f"
N_ROWS = 10228
LANGS = ("ar", "bn", "de", "es", "fr", "hi", "ja", "mr", "pa", "ta", "te", "zh")
INDIC = ("bn", "hi", "mr", "pa", "ta", "te")

NMT_SYSTEMS = {  # system id -> folder in the NMT repo
    "google": "baseline_google_translate",
    "microsoft": "microsoft_translator",
    "nllb": "nllb200",
    "madlad": "madlad400",
    "indictrans2": "indictrans2",
}
LLM_SYSTEMS = ("gemma_9b", "gpt4o", "llama_8b", "mistral_nemo", "qwen_7b")


@dataclass(frozen=True)
class Catalog:
    paradigm: str   # "nmt" or "llm"
    system: str     # e.g. "nllb", "llama_8b"
    tier: str       # "output" for NMT, "baseline" or "repaired" for LLMs
    lang: str
    path: Path

    @property
    def suite(self):
        return self.system if self.paradigm == "nmt" else f"{self.tier}_{self.system}"


def catalogs():
    out = []
    for system, folder in NMT_SYSTEMS.items():
        for lang in LANGS:
            p = NMT_REPO / "data" / folder / f"{lang}.txt"
            if p.exists():
                out.append(Catalog("nmt", system, "output", lang, p))
    for system in LLM_SYSTEMS:
        for tier in ("baseline", "repaired"):
            for lang in LANGS:
                p = LLM_REPO / "data" / f"{tier}_{system}" / f"{lang}.txt"
                if p.exists():
                    out.append(Catalog("llm", system, tier, lang, p))
    return out


class CatalogError(ValueError):
    pass


def read_lines(path):
    """Strict UTF-8, LF only, exactly N_ROWS lines (one trailing newline allowed)."""
    raw = Path(path).read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as e:
        raise CatalogError(f"{path}: invalid UTF-8 ({e})") from e
    if "\r" in text:
        raise CatalogError(f"{path}: contains CR characters")
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    if len(lines) != N_ROWS:
        raise CatalogError(f"{path}: {len(lines)} lines, expected {N_ROWS}")
    return lines


def load_source():
    for repo in (NMT_REPO, LLM_REPO):
        p = repo / "data" / "source" / "en.txt"
        digest = hashlib.sha256(p.read_bytes()).hexdigest()
        if digest != SOURCE_SHA256:
            raise CatalogError(f"{p}: SHA-256 {digest} != pinned {SOURCE_SHA256}")
    return read_lines(NMT_REPO / "data" / "source" / "en.txt")
