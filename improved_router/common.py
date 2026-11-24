"""Shared helpers for Experiment 6, the improved router.

Reads the router dataset and the Experiment 2 predictions; writes only to
results/06_improved_router/. Experiments 1-5 are not changed.

Policies compared (NMT engine N, LLM L, safety gate = verifier v1 or v2):
  predictive_reactive  the Experiment 3 policy (reference)
  A_gated_bypass       the same, but the LLM output is also checked; if it fails and
                       N's output passes, N's output is kept (needs one extra N call)
  B_gain_bypass        A, but Stage 1 ranks strings by the predicted quality GAIN of
                       L over N (CometKiwi(L) - CometKiwi(N)) instead of N's risk
  C_post_nmt_risk      N always runs; strings that fail the checks OR have a high
                       Stage 2 risk (12 source + 7 output features) go to L; LLM gated
  CB_post_nmt_gain     C, but Stage 2 ranks by the predicted gain (source + output)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from router.paths import IMPROVED, ROUTER_DATASET, ensure, oof_path  # noqa: E402,F401

RES, FIG = ensure(IMPROVED), ensure(IMPROVED / "figures")
SEM = 0.6
PAIRS = [("nllb", "gpt4o"), ("nllb", "llama"), ("google", "gpt4o"), ("google", "llama")]
GATES = ("v1", "v2")
LLM_NAME = {"gpt4o": "GPT-4o-mini", "llama": "Llama-3.1-8B"}
POLICIES = ["predictive_reactive", "A_gated_bypass", "B_gain_bypass", "C_post_nmt_risk", "CB_post_nmt_gain"]
LABEL = {"predictive_reactive": "predictive + reactive (Exp. 3)", "A_gated_bypass": "A: + check LLM output",
         "B_gain_bypass": "B: A + rank by predicted gain", "C_post_nmt_risk": "C: Stage 2 risk with NMT-output clues",
         "CB_post_nmt_gain": "C+B: Stage 2 gain with NMT-output clues"}
BYPASS = {"predictive_reactive", "A_gated_bypass", "B_gain_bypass"}


def load(nmt):
    df = pd.read_csv(ROUTER_DATASET)
    ref = pd.read_csv(oof_path(nmt))
    ext = pd.read_csv(RES / f"stage2_scores_{nmt}.csv.gz")
    for x in (ref, ext):
        assert (x.line_idx.values == df.line_idx.values).all() and (x.lang.values == df.lang.values).all()
    return df, ref, ext


def outcomes(df, nmt, llm, gate):
    c = "c1" if gate == "v1" else "c2"
    return {"n_ck": df[f"{nmt}_ck"].values, "n_ok": df[f"{nmt}_{c}"].values == 1,
            "r_ck": df[f"{llm}_igsc_ck"].values, "r_ok": df[f"{llm}_igsc_{c}"].values == 1,
            "repair": df[f"{llm}_base_c1"].values == 0}   # IGSC ran when the baseline failed (as in Exp 3)


def score(policy, ref, ext, llm, gate):
    if policy in ("predictive_reactive", "A_gated_bypass"):
        return ref["p_combined" if gate == "v1" else "p_combined_v2"].values
    return {"B_gain_bypass": ext[f"gain_src_{llm}"], "C_post_nmt_risk": ext[f"post_risk_{gate}"],
            "CB_post_nmt_gain": ext[f"gain_post_{llm}"]}[policy].values


def route(policy, s, theta, o):
    """Returns per-row booleans: sent to LLM, final output from LLM, NMT called."""
    n = len(s)
    hi = s >= theta
    n_fail = ~o["n_ok"]
    if policy in BYPASS:
        bypass = hi
        sent, nmt_called = bypass | n_fail, ~bypass
    else:
        bypass = np.zeros(n, bool)
        sent, nmt_called = n_fail | hi, np.ones(n, bool)
    final_llm = sent.copy()
    if policy != "predictive_reactive":           # also check the LLM output; fall back to NMT if it passes
        llm_bad = sent & ~o["r_ok"]
        nmt_called = nmt_called | (bypass & ~o["r_ok"])
        final_llm = sent & ~(llm_bad & o["n_ok"])
    return sent, final_llm, nmt_called


def per_row(o, sent, final_llm):
    ck = np.where(final_llm, o["r_ck"], o["n_ck"])
    ok = np.where(final_llm, o["r_ok"], o["n_ok"])
    return {"ck": ck, "ok": ok.astype(float), "fail": ((~ok) | (ck < SEM)).astype(float)}


def summary(name, o, sent, final_llm, nmt_called, theta=np.nan):
    r = per_row(o, sent, final_llm)
    return {"policy": name, "theta": theta, "llm_share_pct": 100 * sent.mean(),
            "mean_cometkiwi": r["ck"].mean(), "struct_pass_pct": 100 * r["ok"].mean(),
            "fail_pct": 100 * r["fail"].mean(),
            "llm_calls_per_string": (sent * (1 + o["repair"])).mean(), "nmt_calls_per_string": nmt_called.mean()}


def fixed_policies(o):
    n = len(o["n_ck"])
    ones, zeros = np.ones(n, bool), np.zeros(n, bool)
    oracle = (o["r_ok"] & ~o["n_ok"]) | ((o["r_ok"] == o["n_ok"]) & (o["r_ck"] > o["n_ck"]))
    return {"static_nmt": (zeros, zeros, ones), "static_llm_igsc": (ones, ones, zeros),
            "reactive_cascade": (~o["n_ok"], ~o["n_ok"], ones), "oracle (upper bound)": (oracle, oracle, ones)}


def thetas(s, k=401):
    return np.unique(np.concatenate([np.quantile(s, np.linspace(0, 1, k)), [np.inf]]))


def matched_theta(policy, s, o, target_pct):
    """theta whose total LLM share is closest to target_pct (fair comparison at equal LLM use)."""
    best = min(thetas(s), key=lambda t: abs(100 * route(policy, s, t, o)[0].mean() - target_pct))
    return best
