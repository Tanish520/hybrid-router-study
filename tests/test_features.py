from router.features import FEATURES, SCRIPT_TIER, features


def test_exactly_twelve_features_in_fixed_order():
    assert len(FEATURES) == 12
    assert list(features("Cancel", "hi")) == FEATURES


def test_simple_label():
    f = features("Cancel", "hi")
    assert f["n_chars"] == 6 and f["n_tokens"] == 1
    assert f["has_placeholder"] == 0 and f["has_accelerator"] == 0 and f["has_xml"] == 0
    assert f["script_tier"] == SCRIPT_TIER["hi"] == 3


def test_placeholder_and_accelerator():
    f = features("& Location of %1", "de")
    assert f["has_placeholder"] == 1
    assert f["has_accelerator"] == 1          # tokenized KDE accelerator "& L"
    assert features("Date & & Time", "de")["has_accelerator"] == 0   # literal ampersand
    assert features("{searchTerms} found", "de")["has_placeholder"] == 1


def test_xml_and_identifier_and_camel():
    assert features("Using & ktts;", "de")["has_xml"] == 1
    assert features("<b>Bold</b>", "de")["has_xml"] == 1
    f = features("rectMeanKineticEnergyVariance", "fr")
    assert f["camel_density"] == 4.0 and f["identifier_like"] == 1
    assert features("kcm_partition", "fr")["identifier_like"] == 1
    assert features("Open file", "fr")["identifier_like"] == 0


def test_ntt_and_ratios():
    assert features("HTTPS", "ja")["ntt_match"] == 1
    assert features("Cancel", "ja")["ntt_match"] == 0
    f = features("IPv6: 128", "zh")
    assert 0 < f["digit_ratio"] < 1 and 0 < f["punct_ratio"] < 1 and 0 < f["upper_ratio"] < 1


def test_script_tiers_cover_all_languages():
    assert set(SCRIPT_TIER) == {"ar", "bn", "de", "es", "fr", "hi", "ja", "mr", "pa", "ta", "te", "zh"}
    assert {SCRIPT_TIER[x] for x in ("de", "es", "fr")} == {0}
