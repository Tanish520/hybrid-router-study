"""Stage 1 features: 12 deterministic features of the English source and target locale.

Follows the reference design ("token count, %1/&/XML presence, camelCase
density, NTT match, and script tier"), completed to 12 with simple surface
statistics. No translation, reference or neural score is used, so the
features are available before any model runs.
"""
import re

from router.verifier import PLACEHOLDER_REGEX, has_accelerator_v2, is_legitimate_ntt

FEATURES = [
    "n_chars",          # source length in characters
    "n_tokens",         # whitespace tokens
    "has_placeholder",  # %1, %s, {name} (the verifier's placeholder pattern)
    "has_accelerator",  # keyboard accelerator, tokenized KDE form "& X" included
    "has_xml",          # XML/HTML tag or entity (<b>, & amp;)
    "camel_density",    # camelCase boundaries per token
    "ntt_match",        # non-translatable token per the verifier's NTT rules
    "digit_ratio",      # share of characters that are digits
    "punct_ratio",      # share of characters that are punctuation/symbols
    "upper_ratio",      # share of letters that are upper case
    "identifier_like",  # code identifier / path: underscore, camelCase or dotted name without spaces
    "script_tier",      # target script group: 0 Latin, 1 CJK, 2 Arabic, 3 Indic
]
SCRIPT_TIER = {"de": 0, "es": 0, "fr": 0, "ja": 1, "zh": 1, "ar": 2,
               "bn": 3, "hi": 3, "mr": 3, "pa": 3, "ta": 3, "te": 3}

XML_REGEX = re.compile(r'<[^>]+>|&\s*[a-zA-Z0-9._-]+;')
CAMEL_REGEX = re.compile(r'[a-z][A-Z]')
IDENT_REGEX = re.compile(r'^\S*(_[A-Za-z0-9]|[a-z][A-Z]|[A-Za-z]\.[A-Za-z]|/)\S*$')


def features(src, lang):
    s = src.strip()
    tokens = s.split()
    n = max(len(s), 1)
    letters = [c for c in s if c.isalpha()]
    return {
        "n_chars": len(s),
        "n_tokens": len(tokens),
        "has_placeholder": int(bool(PLACEHOLDER_REGEX.search(s))),
        "has_accelerator": int(has_accelerator_v2(s)),
        "has_xml": int(bool(XML_REGEX.search(s))),
        "camel_density": len(CAMEL_REGEX.findall(s)) / max(len(tokens), 1),
        "ntt_match": int(is_legitimate_ntt(s, lang)),
        "digit_ratio": sum(c.isdigit() for c in s) / n,
        "punct_ratio": sum((not c.isalnum()) and (not c.isspace()) for c in s) / n,
        "upper_ratio": sum(c.isupper() for c in letters) / max(len(letters), 1),
        "identifier_like": int(bool(IDENT_REGEX.match(s))),
        "script_tier": SCRIPT_TIER[lang],
    }
