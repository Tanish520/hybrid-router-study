"""Experiment 3: the five routing policies and the theta sweep (Pareto frontier).

Offline replay: every policy selects saved outputs. "LLM+IGSC" selects the saved
repaired line, which IGSC produced from that LLM's own baseline (not from the NMT
output). The Stage 1 risk is the out-of-fold prediction from train_router.py, so
no row is scored by a model that saw its English string during training.

Policies, for NMT engine N and LLM L:
  static_nmt          N for every string
  zero_shot_llm       L baseline for every string
  static_llm_igsc     L repaired for every string
  reactive_cascade    N, then the six checks (v1); failures go to L repaired
  predictive_reactive risk >= theta goes straight to L repaired; the rest as reactive_cascade
Reference points: rule_router (placeholder or accelerator -> L repaired, the rest reactive),
random bypass at the same LLM share (expected value), and an oracle upper bound.

Cost is counted in model calls; the dollar/latency model comes with the commit replay.
  NMT calls per string; LLM calls per string = generation + 1 IGSC repair call when the
  LLM baseline failed the checks (that is when IGSC ran).

Outputs (results/03_routing_policies/): policies_<nmt>_<llm>_<gate>.csv (all points), summary.md,
figures/pareto_<nmt>_<llm>_<gate>.png
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from router.paths import POLICIES, ROUTER_DATASET, ensure, oof_path  # noqa: E402

SEM = 0.6
PAIRS = [("nllb", "gpt4o"), ("nllb", "llama"), ("google", "gpt4o"), ("google", "llama")]
LLM_NAME = {"gpt4o": "GPT-4o-mini", "llama": "Llama-3.1-8B"}
BLUE, ORANGE, INK, INK2, GRID, MUTED = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df", "#8a8984"


def outcomes(df, nmt, llm):
    """Per-row arrays for the NMT path and the LLM paths."""
    return {
        "n_ck": df[f"{nmt}_ck"].values, "n_c1": df[f"{nmt}_c1"].values, "n_c2": df[f"{nmt}_c2"].values,
        "b_ck": df[f"{llm}_base_ck"].values, "b_c1": df[f"{llm}_base_c1"].values, "b_c2": df[f"{llm}_base_c2"].values,
        "r_ck": df[f"{llm}_igsc_ck"].values, "r_c1": df[f"{llm}_igsc_c1"].values, "r_c2": df[f"{llm}_igsc_c2"].values,
        "repair": 1 - df[f"{llm}_base_c1"].values,   # IGSC ran only on baseline lines failing the checks
    }


def summarize(name, choose_llm, nmt_called, o, theta=np.nan, llm_kind="igsc"):
    """choose_llm: bool per row (final output from the LLM path); nmt_called: bool per row."""
    p = "r" if llm_kind == "igsc" else "b"
    ck = np.where(choose_llm, o[f"{p}_ck"], o["n_ck"])
    c1 = np.where(choose_llm, o[f"{p}_c1"], o["n_c1"])
    c2 = np.where(choose_llm, o[f"{p}_c2"], o["n_c2"])
    llm_calls = choose_llm * (1 + (o["repair"] if llm_kind == "igsc" else 0))
    return {"policy": name, "theta": theta,
            "mean_cometkiwi": ck.mean(), "struct_pass_v1_pct": 100 * c1.mean(), "struct_pass_v2_pct": 100 * c2.mean(),
            "fail_combined_pct": 100 * ((c1 == 0) | (ck < SEM)).mean(),
            "fail_combined_v2_pct": 100 * ((c2 == 0) | (ck < SEM)).mean(),
            "llm_dispatch_pct": 100 * choose_llm.mean(), "llm_calls_per_string": llm_calls.mean(),
            "nmt_calls_per_string": nmt_called.mean()}


def run_pair(df, oof, nmt, llm, gate="v1"):
    """gate: which verifier the safety gate (Stage 2) uses, 'v1' (original) or 'v2' (corpus-aware).
    The Stage 1 risk is trained on the matching combined label."""
    o = outcomes(df, nmt, llm)
    n = len(df)
    ones, zeros = np.ones(n, bool), np.zeros(n, bool)
    c = "c1" if gate == "v1" else "c2"
    nmt_fail = o[f"n_{c}"] == 0
    rows = [
        summarize("static_nmt", zeros, ones, o),
        summarize("zero_shot_llm", ones, zeros, o, llm_kind="base"),
        summarize("static_llm_igsc", ones, zeros, o),
        summarize("reactive_cascade", nmt_fail, ones, o),
    ]
    rule = ((df.has_placeholder == 1) | (df.has_accelerator == 1)).values
    rows.append(summarize("rule_router", rule | nmt_fail, ~rule, o))
    # oracle upper bound: per string, the better of NMT and LLM+IGSC by (passes checks, CometKiwi)
    pick_llm = (o[f"r_{c}"] > o[f"n_{c}"]) | ((o[f"r_{c}"] == o[f"n_{c}"]) & (o["r_ck"] > o["n_ck"]))
    rows.append(summarize("oracle (upper bound)", pick_llm, ones, o))
    # theta sweep for predictive_reactive, plus random bypass at the same bypass share
    risk = oof["p_combined" if gate == "v1" else "p_combined_v2"].values
    thetas = np.unique(np.concatenate([[0.0, 1.0001], np.quantile(risk, np.linspace(0, 1, 201))]))
    react = rows[3]
    stat = rows[2]
    for t in thetas:
        bypass = risk >= t
        rows.append(summarize("predictive_reactive", bypass | nmt_fail, ~bypass, o, theta=t))
        q = bypass.mean()  # random bypass: expected value of bypassing a random share q
        rnd = {k: (q * stat[k] + (1 - q) * react[k]) if isinstance(stat[k], float) else stat[k] for k in stat}
        rnd.update({"policy": "random_bypass", "theta": t,
                    "llm_dispatch_pct": q * 100 + (1 - q) * react["llm_dispatch_pct"],
                    "nmt_calls_per_string": 1 - q,
                    "llm_calls_per_string": q * stat["llm_calls_per_string"] + (1 - q) * react["llm_calls_per_string"]})
        rows.append(rnd)
    return pd.DataFrame(rows)


def pareto_figure(res, nmt, llm, gate="v1"):
    fixed = res[~res.policy.isin(["predictive_reactive", "random_bypass"])].set_index("policy")
    pr = res[res.policy == "predictive_reactive"].sort_values("llm_dispatch_pct")
    rb = res[res.policy == "random_bypass"].sort_values("llm_dispatch_pct")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, col, ylab in [(axes[0], "mean_cometkiwi", "Mean CometKiwi of the final output"),
                          (axes[1], f"struct_pass_{gate}_pct", f"Structural pass rate, verifier {gate} (%)")]:
        ax.plot(rb.llm_dispatch_pct, rb[col], color=ORANGE, lw=2, ls="--", label="random bypass + reactive")
        ax.plot(pr.llm_dispatch_pct, pr[col], color=BLUE, lw=2, label="predictive + reactive (θ sweep)")
        # label offsets in points (dx, dy, alignment), chosen so the clustered points near 0% and 100% stay readable
        labels = {"static_nmt": ("static NMT", 0, -14, "left"),
                  "reactive_cascade": ("reactive cascade", 2, 9, "left"),
                  "rule_router": ("rule router", 8, -12, "left"),
                  "zero_shot_llm": ("zero-shot LLM", -8, -14, "right"),
                  "static_llm_igsc": ("static LLM+IGSC", -8, 9, "right"),
                  "oracle (upper bound)": ("oracle (upper bound)", 8, -4, "left")}
        for k, (lab, dx, dy, ha) in labels.items():
            r = fixed.loc[k]
            ax.scatter(r.llm_dispatch_pct, r[col], s=46, color=INK if k != "oracle (upper bound)" else MUTED,
                       zorder=5, edgecolor="#fcfcfb", linewidth=1.5)
            ax.annotate(lab, (r.llm_dispatch_pct, r[col]), textcoords="offset points", xytext=(dx, dy),
                        fontsize=8, color=INK2, ha=ha)
        ax.set_xlim(-6, 106)
        ax.margins(y=0.1)
        ax.set_xlabel("Strings sent to the LLM (%)", color=INK2)
        ax.set_ylabel(ylab, color=INK2)
        ax.grid(color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(colors=INK2, labelsize=9)
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle(f"Quality vs LLM use: NMT = {nmt.upper()}, LLM = {LLM_NAME[llm]}, safety gate = verifier {gate} "
                 f"(offline replay of saved outputs, 122,736 strings)", fontsize=10, x=0.01, ha="left", color=INK)
    fig.tight_layout()
    fig.savefig(ensure(POLICIES / "figures") / f"pareto_{nmt}_{llm}_{gate}.png", dpi=200, facecolor="#fcfcfb")
    plt.close(fig)


def headline(res, gate="v1"):
    """Smallest LLM share at which predictive_reactive matches static LLM+IGSC on both quality measures."""
    col = f"struct_pass_{gate}_pct"
    stat = res[res.policy == "static_llm_igsc"].iloc[0]
    pr = res[(res.policy == "predictive_reactive")
             & (res.mean_cometkiwi >= stat.mean_cometkiwi) & (res[col] >= stat[col])]
    return None if pr.empty else pr.sort_values("llm_dispatch_pct").iloc[0]


def main():
    df = pd.read_csv(ROUTER_DATASET)
    md = ["# Experiment 3: routing policies (offline replay of saved outputs)", "",
          "Risk = out-of-fold LightGBM prediction (combined label matching the gate's verifier). LLM calls include "
          "the IGSC repair call when the LLM baseline failed the checks. Each pair is run twice: safety gate = "
          "verifier v1 (the original checker, as published) and verifier v2 (corpus-aware accelerator fix).", ""]
    for gate in ("v1", "v2"):
        for nmt, llm in PAIRS:
            oof = pd.read_csv(oof_path(nmt))
            assert (oof.line_idx.values == df.line_idx.values).all() and (oof.lang.values == df.lang.values).all()
            res = run_pair(df, oof, nmt, llm, gate)
            res.to_csv(ensure(POLICIES) / f"policies_{nmt}_{llm}_{gate}.csv", index=False)
            pareto_figure(res, nmt, llm, gate)
            fixed = res[~res.policy.isin(["predictive_reactive", "random_bypass"])]
            pr = res[res.policy == "predictive_reactive"]
            picks = [pr.iloc[(pr.llm_dispatch_pct - x).abs().argmin()] for x in (10, 20, 30)]
            table = pd.concat([fixed, pd.DataFrame(picks)])
            md += [f"## Gate {gate}: NMT = {nmt}, LLM = {LLM_NAME[llm]}", "",
                   "| policy | θ | mean CometKiwi | struct pass v1 % | struct pass v2 % | fail (struct v1 or CK<0.6) % "
                   "| fail (struct v2 or CK<0.6) % | LLM dispatch % | LLM calls/string | NMT calls/string |",
                   "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
            for r in table.itertuples():
                th = "" if np.isnan(r.theta) else f"{r.theta:.3f}"
                md.append(f"| {r.policy} | {th} | {r.mean_cometkiwi:.4f} | {r.struct_pass_v1_pct:.2f} | "
                          f"{r.struct_pass_v2_pct:.2f} | {r.fail_combined_pct:.2f} | {r.fail_combined_v2_pct:.2f} | "
                          f"{r.llm_dispatch_pct:.1f} | {r.llm_calls_per_string:.3f} | {r.nmt_calls_per_string:.3f} |")
            h = headline(res, gate)
            md += ["", ("Predictive+reactive matches static LLM+IGSC on both mean CometKiwi and structural pass "
                        f"({gate}) at θ = {h.theta:.3f}, sending {h.llm_dispatch_pct:.1f}% of strings to the LLM."
                        if h is not None else
                        "Predictive+reactive does not reach static LLM+IGSC on both measures at any θ."), ""]
            print(f"gate {gate} {nmt}/{llm}: done")
    (POLICIES / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
