"""Seven cheap features of the NMT output, used by the extension's Stage 2 score.

Stage 1 sees only the English string. Once NMT has translated it, its
output is available for free, so Stage 2 can look at it before deciding whether to
escalate. Each feature is a plain string computation (microseconds, no model).
"""
import re

OUTPUT_FEATURES = ["len_ratio", "tok_ratio", "copy", "latin_frac", "digits_match", "end_punct_match", "uniq_ratio"]
_DIGITS = re.compile(r"\d+")


def output_features(src: str, hyp: str) -> dict:
    hyp_tokens, src_tokens = hyp.split(), src.split()
    letters = sum(ch.isalpha() for ch in hyp)
    latin = sum(ch.isascii() and ch.isalpha() for ch in hyp)
    return {
        "len_ratio": len(hyp) / max(len(src), 1),                       # far too long or short: dropped/added text
        "tok_ratio": len(hyp_tokens) / max(len(src_tokens), 1),
        "copy": int(hyp.strip() == src.strip()),                       # NMT gave the English back unchanged
        "latin_frac": latin / letters if letters else 0.0,             # English letters left in the output
        "digits_match": int(sorted(_DIGITS.findall(src)) == sorted(_DIGITS.findall(hyp))),
        "end_punct_match": int(src.rstrip()[-1:] == hyp.rstrip()[-1:]),
        "uniq_ratio": len(set(hyp_tokens)) / len(hyp_tokens) if hyp_tokens else 0.0,  # repeated words
    }
