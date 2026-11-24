"""Tests for Experiment 6 (improved router): NMT-output features and routing rules."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from improved_router.common import route  # noqa: E402
from improved_router.output_features import OUTPUT_FEATURES, output_features  # noqa: E402


def test_output_features_basic():
    f = output_features("Save as %1", "% 1 के रूप में सहेजें")
    assert list(f) == OUTPUT_FEATURES
    assert f["copy"] == 0 and f["latin_frac"] == 0.0 and f["digits_match"] == 1


def test_copy_and_latin():
    f = output_features("Codec from", "Codec from")
    assert f["copy"] == 1 and f["latin_frac"] == 1.0 and f["len_ratio"] == 1.0


def test_empty_output():
    f = output_features("Open", "")
    assert f["len_ratio"] == 0 and f["uniq_ratio"] == 0.0 and f["latin_frac"] == 0.0


def test_digits_mismatch_and_repetition():
    f = output_features("Page 2 of 10", "Seite 2 von 2 2")
    assert f["digits_match"] == 0 and f["uniq_ratio"] < 1


def _o(n_ok, r_ok):
    n = len(n_ok)
    return {"n_ok": np.array(n_ok), "r_ok": np.array(r_ok), "n_ck": np.zeros(n), "r_ck": np.ones(n),
            "repair": np.zeros(n, bool)}


def test_reference_policy_has_no_fallback():
    o = _o([True, True], [False, True])
    sent, fin, nc = route("predictive_reactive", np.array([1.0, 0.0]), 0.5, o)
    assert sent.tolist() == [True, False] and fin.tolist() == [True, False] and nc.tolist() == [False, True]


def test_gated_bypass_falls_back_to_passing_nmt():
    # row 0: bypassed, LLM fails the checks, NMT passes -> NMT is called and kept
    # row 1: bypassed, both fail -> LLM output kept
    o = _o([True, False], [False, False])
    sent, fin, nc = route("A_gated_bypass", np.array([1.0, 1.0]), 0.5, o)
    assert sent.tolist() == [True, True] and fin.tolist() == [False, True] and nc.tolist() == [True, True]


def test_post_nmt_policy_always_calls_nmt_and_escalates_failures():
    o = _o([False, True, True], [True, True, False])
    sent, fin, nc = route("CB_post_nmt_gain", np.array([0.0, 1.0, 1.0]), 0.5, o)
    assert nc.all()
    assert sent.tolist() == [True, True, True]       # check failure, high score, high score
    assert fin.tolist() == [True, True, False]       # row 2: LLM fails, NMT passes -> keep NMT
