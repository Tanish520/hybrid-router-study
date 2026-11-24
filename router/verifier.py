"""The six-invariant structural verifier from the upstream LLM benchmark.

v1 is the original verifier exactly as published in
cloud-native-llm-benchmark@6f31ddd (scripts/audit_and_capture_findings.py,
vendored unchanged in third_party/). tests/test_parity.py checks that v1
reproduces the published audit_findings_log.jsonl record for record.

v2 keeps all six checks and changes only what is wrong for this corpus:
  1. Accelerators. OPUS KDE4 is tokenized, so "&Close" is stored as "& Close"
     and the upstream pattern (a letter directly after "&") matches 1 of the
     10,228 sources. v2 allows one optional space, accepts a letter in any
     script, and ignores the literal ampersand "& &" (KDE's "&&").
  2. Empty outputs are flagged as "empty_output" (row integrity, kept separate
     from the six invariants).
Catalog-level integrity (exact line count, strict UTF-8) is enforced by the
loader in router/data.py, not here.
"""
import re

VERSIONS = ("v1", "v2")
SIX_INVARIANTS = (
    "script_contamination", "broken_placeholder", "broken_accelerator",
    "encoding_corruption", "conversational_chatter", "copy_fallback",
)

# ---- copied verbatim from upstream ---------------------------------------
FORBIDDEN_SCRIPT = {
    "ar": re.compile(r'[一-鿿가-힯Ѐ-ӿ]'),
    "bn": re.compile(r'[一-鿿가-힯Ѐ-ӿ؀-ۿ]'),
    "de": re.compile(r'[一-鿿가-힯Ѐ-ӿ؀-ۿঀ-৿]'),
    "es": re.compile(r'[一-鿿가-힯Ѐ-ӿ؀-ۿঀ-৿]'),
    "fr": re.compile(r'[一-鿿가-힯Ѐ-ӿ؀-ۿঀ-৿]'),
    "hi": re.compile(r'[一-鿿가-힯Ѐ-ӿ؀-ۿ]'),
    "ja": re.compile(r'[Ѐ-ӿ؀-ۿ가-힯]'),
    "mr": re.compile(r'[一-鿿가-힯Ѐ-ӿ؀-ۿ]'),
    "pa": re.compile(r'[一-鿿가-힯Ѐ-ӿ؀-ۿ]'),
    "ta": re.compile(r'[一-鿿가-힯Ѐ-ӿ؀-ۿ]'),
    "te": re.compile(r'[一-鿿가-힯Ѐ-ӿ؀-ۿ]'),
    "zh": re.compile(r'[Ѐ-ӿ؀-ۿ가-힯]'),
}
LANGS = tuple(FORBIDDEN_SCRIPT)

CHATTER_REGEX = re.compile(r'(?i)^(here is the translation|translation:|note:|as an ai|translate into|arabic:|spanish:|bengali:)')
PLACEHOLDER_REGEX = re.compile(r'%\d+|%[a-zA-Z]|\{[^}]+\}')
XML_ENTITY_REGEX = re.compile(r'&\s*[a-zA-Z0-9._-]+;')
ACCELERATOR_REGEX = re.compile(r'&[^\s\d_;]')
REPLACEMENT_CHAR_REGEX = re.compile(r'�')
FORMULA_REGEX = re.compile(r'^[A-Z0-9_]+\([^)]*\)$')
POSIX_FMT_REGEX = re.compile(r'^%[a-zA-Z](\s*\([^)]*\))?$')
XML_ONLY_REGEX = re.compile(r'^&\s*[a-zA-Z0-9._-]+;$')
CAMEL_OR_SNAKE_REGEX = re.compile(r'([a-z][A-Z]|_[a-zA-Z])')
X11_COLOR_REGEX = re.compile(r'^[A-Za-z]+[1-4]$')
UA_STRING_REGEX = re.compile(r'^UADescription\s*\(.*\)$')
TZ_PATH_REGEX = re.compile(r'^[A-Za-z]+/\s*[A-Za-z_]+$')
EURO_COGNATE_PHRASE_REGEX = re.compile(r'^(\d+\s+Points|Position\s+[xyz]:?|Pixmap:?|Kig\s+figure|Dithering\s+PPI)$', re.IGNORECASE)

KNOWN_NTTS = {
    'SCSI', 'S-ATA', 'IEEE 1394', 'SD/ MMC', 'RITOn', 'kcm_partition', 'KTraderClient',
    'USB', 'PCI', 'BIOS', 'IPv4', 'IPv6', 'UUID', 'HDMI', 'TCP', 'UDP', 'DHCP', 'DNS',
    'RAM', 'CPU', 'GPU', 'Samba', 'OpenVPN', 'Cisco', 'ALSA', 'MIME', 'UTF-8', 'ASCII',
    'HTML', 'HTTP', 'HTTPS', 'LDAP', 'IMAP', 'POP3', 'SMTP', 'X11', 'OpenGL',
    'Iptc', 'NoPCL3GUI', 'SDP UUid:', 'KMobileTools EngineXP', 'Linediff: A< - > B'
}


def has_accelerator_v1(text):
    return bool(ACCELERATOR_REGEX.search(XML_ENTITY_REGEX.sub('', text)))


def is_legitimate_ntt(src_str, lang_code=""):
    s = src_str.strip()
    if s in KNOWN_NTTS:
        return True
    if (FORMULA_REGEX.match(s) or POSIX_FMT_REGEX.match(s) or XML_ONLY_REGEX.match(s)
            or X11_COLOR_REGEX.match(s) or UA_STRING_REGEX.match(s) or TZ_PATH_REGEX.match(s)):
        return True
    if re.match(r'^[A-Z0-9\s/\-\.]{2,12}$', s) and not s.isalpha():
        return True
    if s.isupper() and len(s) <= 6:
        return True
    if lang_code in ("de", "es", "fr"):
        if EURO_COGNATE_PHRASE_REGEX.match(s):
            return True
        if " " not in s and not CAMEL_OR_SNAKE_REGEX.search(s) and re.match(r'^[A-Za-z\-]+:?$', s):
            return True
    return False
# ---- end of upstream copy ------------------------------------------------

LITERAL_AMPERSAND_REGEX = re.compile(r'&\s?&')
ACCELERATOR_V2_REGEX = re.compile(r'&\s?[^\W\d_]')


def has_accelerator_v2(text):
    cleaned = XML_ENTITY_REGEX.sub('', text)
    cleaned = LITERAL_AMPERSAND_REGEX.sub('', cleaned)
    return bool(ACCELERATOR_V2_REGEX.search(cleaned))


AMP_ESCAPE_REGEX = re.compile(r'&amp;', re.IGNORECASE)


def accelerator_failure_kind(hyp):
    """Why a v2 broken_accelerator fired: 'escaped' (&amp; written instead of &),
    'detached' (an & remains but not before a letter, e.g. trailing or '(&)'),
    or 'dropped' (no & left at all). 'dropped' is the conservative lower bound."""
    if AMP_ESCAPE_REGEX.search(hyp):
        return "escaped"
    rest = XML_ENTITY_REGEX.sub('', LITERAL_AMPERSAND_REGEX.sub('', hyp))
    return "detached" if "&" in rest else "dropped"


def check(src, hyp, lang, version="v2"):
    """Return the list of failed checks for one translation (empty = pass)."""
    if version not in VERSIONS:
        raise ValueError(f"unknown verifier version {version!r}")
    issues = []
    if version == "v2" and not hyp.strip():
        issues.append("empty_output")

    if FORBIDDEN_SCRIPT[lang].search(hyp):
        issues.append("script_contamination")

    if sorted(PLACEHOLDER_REGEX.findall(src)) != sorted(PLACEHOLDER_REGEX.findall(hyp)):
        issues.append("broken_placeholder")

    has_acc = has_accelerator_v1 if version == "v1" else has_accelerator_v2
    if has_acc(src) and not has_acc(hyp):
        issues.append("broken_accelerator")

    if REPLACEMENT_CHAR_REGEX.search(hyp):
        issues.append("encoding_corruption")

    if CHATTER_REGEX.search(hyp):
        issues.append("conversational_chatter")

    src_s, hyp_s = src.strip(), hyp.strip()
    if src_s == hyp_s and len(src_s) > 3 and any(c.isalpha() for c in src_s):
        if not is_legitimate_ntt(src_s, lang):
            issues.append("copy_fallback")
    return issues
