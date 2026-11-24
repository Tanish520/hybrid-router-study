"""95% confidence intervals for Experiments 2 and 3 (grouped bootstrap).

Resampling unit: the English source text (all 12 languages and duplicates of a
string are drawn together), the same grouping as the cross-validation, so rows
that are not independent are not counted as independent.

Experiment 3: for each policy at a fixed operating point (θ fixed on the full
data), the mean CometKiwi, structural pass rates and combined failure rate,
plus paired differences between policies. A difference is called significant
when its 95% interval excludes zero.
Experiment 2: AUC of LightGBM and of the rule baseline, and their difference.

Outputs: results/05_confidence_intervals/bootstrap_ci.csv and bootstrap_ci.md
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from router.paths import CONFIDENCE, ROUTER_DATASET, ensure, oof_path  # noqa: E402

B3, B2, SEED, SEM = 1000, 200, 0, 0.6
PAIRS = [("nllb", "gpt4o"), ("nllb", "llama")]
BYPASS_SHARES = (0.10, 0.20, 0.30)


def policy_rows(df, oof, nmt, llm, gate):
    """Per-row outcome vectors for each policy (final CometKiwi, pass v1, pass v2, LLM used)."""
    c = "c1" if gate == "v1" else "c2"
    n_fail = (df[f"{nmt}_{c}"] == 0).values
    risk = oof["p_combined" if gate == "v1" else "p_combined_v2"].values
    rule = ((df.has_placeholder == 1) | (df.has_accelerator == 1)).values
    N = {k: df[f"{nmt}_{k}"].values.astype(float) for k in ("ck", "c1", "c2")}
    R = {k: df[f"{llm}_igsc_{k}"].values.astype(float) for k in ("ck", "c1", "c2")}
    Bz = {k: df[f"{llm}_base_{k}"].values.astype(float) for k in ("ck", "c1", "c2")}

    def pick(use_llm, llm_out=R):
        out = {k: np.where(use_llm, llm_out[k], N[k]) for k in ("ck", "c1", "c2")}
        out["fail"] = ((out[c] == 0) | (out["ck"] < SEM)).astype(float)
        out["llm"] = use_llm.astype(float)
        return out

    n = len(df)
    pol = {"static_nmt": pick(np.zeros(n, bool)), "zero_shot_llm": pick(np.ones(n, bool), Bz),
           "static_llm_igsc": pick(np.ones(n, bool)), "reactive_cascade": pick(n_fail),
           "rule_router": pick(rule | n_fail)}
    for q in BYPASS_SHARES:
        theta = np.quantile(risk, 1 - q)
        bypass = risk >= theta
        pol[f"predictive_{int(q * 100)}"] = pick(bypass | n_fail)
        # random bypass of the same share, as an expected value per row
        stat, react = pol["static_llm_igsc"], pol["reactive_cascade"]
        share = bypass.mean()
        pol[f"random_{int(q * 100)}"] = {k: share * stat[k] + (1 - share) * react[k] for k in stat}
    return pol


def group_weights(groups, B, seed):
    codes, uniq = pd.factorize(groups)
    G = len(uniq)
    rng = np.random.default_rng(seed)
    W = rng.multinomial(G, np.full(G, 1.0 / G), size=B).astype(np.float64)   # B x G draw counts
    return codes, G, W


def boot_mean(x, codes, G, W):
    """Bootstrap distribution of the row mean of x under group resampling."""
    s = np.bincount(codes, weights=x, minlength=G)
    m = np.bincount(codes, minlength=G).astype(float)
    return (W @ s) / (W @ m)


def ci(dist, point):
    lo, hi = np.percentile(dist, [2.5, 97.5])
    return point, lo, hi


def experiment3(df, codes, G, W):
    rows = []
    for gate in ("v1", "v2"):
        for nmt, llm in PAIRS:
            oof = pd.read_csv(oof_path(nmt))
            pol = policy_rows(df, oof, nmt, llm, gate)
            for name, p in pol.items():
                for metric, scale in (("ck", 1), ("c1", 100), ("c2", 100), ("fail", 100), ("llm", 100)):
                    pt, lo, hi = ci(boot_mean(p[metric], codes, G, W) * scale, p[metric].mean() * scale)
                    rows.append({"experiment": 3, "gate": gate, "nmt": nmt, "llm": llm, "item": name,
                                 "metric": metric, "estimate": pt, "ci_low": lo, "ci_high": hi})
            diffs = [("reactive_cascade", "static_llm_igsc"), ("reactive_cascade", "static_nmt"),
                     ("static_llm_igsc", "static_nmt")]
            diffs += [(f"predictive_{int(q * 100)}", f"random_{int(q * 100)}") for q in BYPASS_SHARES]
            diffs += [(f"predictive_{int(q * 100)}", "reactive_cascade") for q in BYPASS_SHARES]
            for a, b in diffs:
                for metric, scale in (("ck", 1), ("c1", 100), ("c2", 100), ("fail", 100)):
                    d = pol[a][metric] - pol[b][metric]
                    pt, lo, hi = ci(boot_mean(d, codes, G, W) * scale, d.mean() * scale)
                    rows.append({"experiment": 3, "gate": gate, "nmt": nmt, "llm": llm, "item": f"{a} − {b}",
                                 "metric": metric, "estimate": pt, "ci_low": lo, "ci_high": hi,
                                 "significant": not (lo <= 0 <= hi)})
    return rows


def experiment2(df, codes, G, W2):
    rows = []
    rule = ((df.has_placeholder == 1) | (df.has_accelerator == 1)).values.astype(float)
    for nmt in ("nllb", "google"):
        oof = pd.read_csv(oof_path(nmt))
        labels = {"struct": 1 - df[f"{nmt}_c1"], "sem": (df[f"{nmt}_ck"] < SEM).astype(int),
                  "combined": ((1 - df[f"{nmt}_c1"]) | (df[f"{nmt}_ck"] < SEM)).astype(int),
                  "combined_v2": ((1 - df[f"{nmt}_c2"]) | (df[f"{nmt}_ck"] < SEM)).astype(int)}
        for name, y in labels.items():
            p = oof[f"p_{name}"].values
            y = y.values
            a_m, a_r, dd = [], [], []
            for w in W2:
                rw = w[codes]
                am = roc_auc_score(y, p, sample_weight=rw)
                ar = roc_auc_score(y, rule, sample_weight=rw)
                a_m.append(am), a_r.append(ar), dd.append(am - ar)
            pm, pr = roc_auc_score(y, p), roc_auc_score(y, rule)
            for item, dist, pt in (("lightgbm_auc", a_m, pm), ("rule_auc", a_r, pr), ("lightgbm − rule", dd, pm - pr)):
                lo, hi = np.percentile(dist, [2.5, 97.5])
                rows.append({"experiment": 2, "gate": "", "nmt": nmt, "llm": "", "item": item, "metric": name,
                             "estimate": pt, "ci_low": lo, "ci_high": hi,
                             "significant": (not (lo <= 0 <= hi)) if "−" in item else np.nan})
        print(f"experiment 2 {nmt}: done", flush=True)
    return rows


def fmt(r, digits):
    return f"{r.estimate:.{digits}f} [{r.ci_low:.{digits}f}, {r.ci_high:.{digits}f}]"


def main():
    df = pd.read_csv(ROUTER_DATASET)
    codes, G, W = group_weights(df.group.values, B3, SEED)
    rows = experiment3(df, codes, G, W)
    print("experiment 3: done", flush=True)
    rows += experiment2(df, codes, G, W[:B2])
    res = pd.DataFrame(rows)
    res.to_csv(ensure(CONFIDENCE) / "bootstrap_ci.csv", index=False)

    md = ["# 95% confidence intervals (grouped bootstrap by English source string)", "",
          f"Experiment 3: {B3} resamples; Experiment 2: {B2} resamples. Format: estimate [2.5%, 97.5%]. "
          "A difference is significant when its interval excludes 0.", ""]
    names = {"ck": "mean CometKiwi", "c1": "struct pass v1 %", "c2": "struct pass v2 %",
             "fail": "fail (struct or CK<0.6) %", "llm": "LLM dispatch %"}
    for gate in ("v1", "v2"):
        for nmt, llm in PAIRS:
            d = res[(res.experiment == 3) & (res.gate == gate) & (res.nmt == nmt) & (res.llm == llm)]
            md += [f"## Experiment 3, gate {gate}: NMT = {nmt}, LLM = {llm}", "",
                   "| policy | " + " | ".join(names.values()) + " |", "|---|" + "---:|" * len(names)]
            for item in d[~d.item.str.contains("−")].item.unique():
                cells = [fmt(d[(d.item == item) & (d.metric == m)].iloc[0], 4 if m == "ck" else 2) for m in names]
                md.append(f"| {item} | " + " | ".join(cells) + " |")
            md += ["", "| difference | mean CometKiwi | struct pass v1 % | struct pass v2 % | fail % |",
                   "|---|---:|---:|---:|---:|"]
            for item in d[d.item.str.contains("−")].item.unique():
                cells = []
                for m in ("ck", "c1", "c2", "fail"):
                    r = d[(d.item == item) & (d.metric == m)].iloc[0]
                    cells.append(fmt(r, 4 if m == "ck" else 2) + (" ✓" if r.significant else ""))
                md.append(f"| {item} | " + " | ".join(cells) + " |")
            md.append("")
    md += ["## Experiment 2: AUC (✓ = LightGBM − rule significant)", "",
           "| NMT | label | LightGBM AUC | rule AUC | LightGBM − rule |", "|---|---|---:|---:|---:|"]
    d2 = res[res.experiment == 2]
    for (nmt, label), g in d2.groupby(["nmt", "metric"], sort=False):
        m = g[g.item == "lightgbm_auc"].iloc[0]
        r = g[g.item == "rule_auc"].iloc[0]
        diff = g[g.item == "lightgbm − rule"].iloc[0]
        md.append(f"| {nmt} | {label} | {fmt(m, 3)} | {fmt(r, 3)} | {fmt(diff, 3)}{' ✓' if diff.significant else ''} |")
    (CONFIDENCE / "bootstrap_ci.md").write_text("\n".join(md) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
