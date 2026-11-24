from router.verifier import check


# --- v1 must behave exactly like the original verifier ---------------------

def test_v1_misses_tokenized_accelerator():
    # OPUS KDE4 stores "&Close Project" as "& Close Project"; upstream misses it.
    assert "broken_accelerator" not in check("& Close Project", "Projekt schließen", "de", "v1")


def test_v1_flags_placeholder_split():
    assert "broken_placeholder" in check("Save as %1", "% 1 के रूप में सहेजें", "hi", "v1")


def test_v1_copy_fallback_and_ntt_exemption():
    assert "copy_fallback" in check("Layer Settings", "Layer Settings", "hi", "v1")
    assert "copy_fallback" not in check("HTTPS", "HTTPS", "hi", "v1")


# --- v2 fixes -------------------------------------------------------------

def test_v2_detects_tokenized_accelerator_dropped():
    assert "broken_accelerator" in check("& Close Project", "Projekt schließen", "de", "v2")


def test_v2_accepts_accelerator_kept_with_or_without_space():
    assert check("& Close Project", "Projekt & schließen", "de", "v2") == []
    assert check("& Close Project", "&Projekt schließen", "de", "v2") == []


def test_v2_accepts_accelerator_on_non_latin_letter():
    assert "broken_accelerator" not in check("& File", "& फ़ाइल", "hi", "v2")
    assert "broken_accelerator" not in check("& File", "ファイル(&F)", "ja", "v2")


def test_v2_mid_string_accelerator():
    assert "broken_accelerator" in check("Match & case", "Groß-/Kleinschreibung beachten", "de", "v2")


def test_v2_literal_ampersand_is_not_an_accelerator():
    # KDE writes a literal ampersand as "&&", tokenized to "& &".
    assert "broken_accelerator" not in check("Date & & Time", "Datum und Uhrzeit", "de", "v2")


def test_v2_xml_entity_is_not_an_accelerator():
    assert "broken_accelerator" not in check("Using & ktts;", "Verwendung von KTTS", "de", "v2")
    assert "broken_accelerator" not in check("& QIT; Explained", "QIT erklärt", "de", "v2")


def test_v2_target_entity_does_not_count_as_kept_accelerator():
    assert "broken_accelerator" in check("& Open", "Öffnen & amp;", "de", "v2")


def test_accelerator_failure_kinds():
    from router.verifier import accelerator_failure_kind as kind
    assert kind("Projekt schließen") == "dropped"
    assert kind("स्थाय &") == "detached"
    assert kind("अपडेट करें (&)") == "detached"
    assert kind("&amp; 高さに合わせる") == "escaped"


def test_v2_empty_output_flagged():
    assert "empty_output" in check("Cancel", "", "hi", "v2")
    assert "empty_output" in check("Cancel", "   ", "hi", "v2")
    assert "empty_output" not in check("Cancel", "", "hi", "v1")


def test_v2_keeps_other_checks_identical():
    for src, hyp, lang in [
        ("Save as %1", "% 1 के रूप में सहेजें", "hi"),
        ("Layer Settings", "Layer Settings", "hi"),
        ("Cancel", "Here is the translation: रद्द करें", "hi"),
        ("Cancel", "Отмена", "hi"),
        ("Cancel", "रद्द�", "hi"),
    ]:
        assert check(src, hyp, lang, "v1") == check(src, hyp, lang, "v2")
