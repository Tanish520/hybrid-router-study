"""Experiment 6, step 2: compare the improved policies with the Experiment 3 policies (offline replay).

For every pair (NMT, LLM) and safety gate (v1, v2):
  * theta sweep for each policy (quality vs share of strings sent to the LLM);
  * fair comparison at EQUAL LLM use (10 / 20 / 30% of strings sent to the LLM);
  * grouped bootstrap CIs (1,000 resamples of English source groups, as in Experiment 3)
    for C+B minus predictive + reactive, minus static LLM+IGSC, minus reactive;
  * an independent check that does not use CometKiwi: chrF against Google Translate as a
    pseudo-reference (NLLB pairs only, since Google is not one of the routed systems there).

Outputs (results/06_improved_router/): policies_<nmt>_<llm>_<gate>.csv, matched.csv,
bootstrap_ci.csv, beat_static_llm.csv, summary.md, figures/pareto_<nmt>_<llm>_<gate>.png
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from improved_router.common import (FIG, GATES, LABEL, LLM_NAME, PAIRS, POLICIES, RES, ROOT, fixed_policies, load,  # noqa: E402
                    matched_theta, outcomes, per_row, route, score, summary, thetas)

sys.path.insert(0, str(ROOT / "scripts"))
from bootstrap import boot_mean, group_weights  # noqa: E402
from router.data import LANGS, catalogs, read_lines  # noqa: E402

TARGETS = (10, 20, 30)
B, SEED = 1000, 0
COLOR = {"predictive_reactive": "#8a8984", "A_gated_bypass": "#eb6834", "B_gain_bypass": "#d4a72c",
         "C_post_nmt_risk": "#1baf7a", "CB_post_nmt_gain": "#2a78d6"}
LLM_SYS = {"gpt4o": "gpt4o", "llama": "llama_8b"}


def chrf_table(df):
    """Sentence chrF of NLLB and each LLM+IGSC output against Google Translate (cached)."""
    path = RES / "chrf_vs_google.csv.gz"
    if path.exists():
        return pd.read_csv(path)
    from sacrebleu.metrics import CHRF
    chrf = CHRF()
    paths = {(c.system, c.tier, c.lang): c.path for c in catalogs()}
    cols = {"nllb": ("nllb", "output"), "gpt4o": ("gpt4o", "repaired"), "llama": ("llama_8b", "repaired")}
    out = {k: [] for k in cols}
    for lang in LANGS:
        ref = read_lines(paths[("google", "output", lang)])
        for k, key in cols.items():
            hyp = read_lines(paths[(key[0], key[1], lang)])
            out[k] += [chrf.sentence_score(h, [r]).score for h, r in zip(hyp, ref)]
        print(f"  chrF {lang} done", flush=True)
    t = pd.DataFrame({"line_idx": df.line_idx, "lang": df.lang, **{f"chrf_{k}": v for k, v in out.items()}})
    t.to_csv(path, index=False, compression="gzip")
    return t


def ci(dist, point):
    lo, hi = np.percentile(dist, [2.5, 97.5])
    return point, lo, hi


def figure(res, nmt, llm, gate):
    fixed = res[~res.policy.isin(POLICIES)].set_index("policy")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, col, ylab in [(axes[0], "mean_cometkiwi", "Mean CometKiwi of the final output (higher is better)"),
                          (axes[1], "fail_pct", f"Broken or bad meaning, % (gate {gate}; lower is better)")]:
        for p in POLICIES:
            d = res[res.policy == p].sort_values("llm_share_pct")
            ax.plot(d.llm_share_pct, d[col], color=COLOR[p], lw=2.2 if p == "CB_post_nmt_gain" else 1.5,
                    ls="--" if p == "predictive_reactive" else "-", label=LABEL[p])
        offsets = {"static_nmt": (4, -14, "left"), "static_llm_igsc": (-4, -14, "right"),
                   "reactive_cascade": (8, 4, "left"), "oracle (upper bound)": (6, 6, "left")}
        for k, lab in [("static_nmt", "static NMT"), ("static_llm_igsc", "static LLM+IGSC"),
                       ("reactive_cascade", "reactive"), ("oracle (upper bound)", "oracle")]:
            r = fixed.loc[k]
            dx, dy, ha = offsets[k]
            ax.scatter(r.llm_share_pct, r[col], s=40, color="#0b0b0b", zorder=5, edgecolor="#fcfcfb")
            ax.annotate(lab, (r.llm_share_pct, r[col]), textcoords="offset points", xytext=(dx, dy), fontsize=8,
                        color="#52514e", ha=ha)
        ax.set_xlim(-4, 104)
        ax.margins(y=0.12)
        ax.set_xlabel("Strings sent to the LLM (%)", color="#52514e")
        ax.set_ylabel(ylab, color="#52514e", fontsize=9)
        ax.grid(color="#e4e3df", lw=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=5, frameon=False, fontsize=8)
    fig.suptitle(f"Improved router (Experiment 6): NMT = {nmt.upper()}, LLM = {LLM_NAME[llm]}, "
                 f"gate {gate}; offline replay, 122,736 strings", fontsize=10, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(FIG / f"pareto_{nmt}_{llm}_{gate}.png", dpi=200, facecolor="#fcfcfb")
    plt.close(fig)


def main():
    matched, boots, heads = [], [], []
    chrf = None
    for nmt in ("nllb", "google"):
        df, ref, ext = load(nmt)
        codes, G, W = group_weights(df.group.values, B, SEED)
        if nmt == "nllb":
            chrf = chrf_table(df)
        for llm in ("gpt4o", "llama"):
            if (nmt, llm) not in PAIRS:
                continue
            for gate in GATES:
                o = outcomes(df, nmt, llm, gate)
                rows, arrays = [], {}
                for name, (sent, fin, nc) in fixed_policies(o).items():
                    rows.append(summary(name, o, sent, fin, nc))
                    arrays[name] = (sent, fin)
                for p in POLICIES:
                    s = score(p, ref, ext, llm, gate)
                    rows += [summary(p, o, *route(p, s, t, o), theta=t) for t in thetas(s)]
                    for tgt in TARGETS:
                        t = matched_theta(p, s, o, tgt)
                        sent, fin, nc = route(p, s, t, o)
                        arrays[(p, tgt)] = (sent, fin)
                        r = summary(p, o, sent, fin, nc, theta=t)
                        r.update(nmt=nmt, llm=llm, gate=gate, target_pct=tgt)
                        if chrf is not None and nmt == "nllb":
                            r["mean_chrf_vs_google"] = np.where(fin, chrf[f"chrf_{llm}"], chrf.chrf_nllb).mean()
                        matched.append(r)
                res = pd.DataFrame(rows)
                res.to_csv(RES / f"policies_{nmt}_{llm}_{gate}.csv", index=False)
                figure(res, nmt, llm, gate)
                stat = res[res.policy == "static_llm_igsc"].iloc[0]
                for p in POLICIES:
                    d = res[(res.policy == p) & (res.mean_cometkiwi >= stat.mean_cometkiwi)
                            & (res.struct_pass_pct >= stat.struct_pass_pct)]
                    heads.append({"nmt": nmt, "llm": llm, "gate": gate, "policy": p,
                                  "min_llm_share_to_beat_static_llm":
                                      None if d.empty else round(d.llm_share_pct.min(), 1)})
                # bootstrap: C+B against the Experiment 3 policies (paired: same rows, same resamples)
                comps = [(f"C+B vs predictive+reactive @ {t}%", ("CB_post_nmt_gain", t),
                          ("predictive_reactive", t)) for t in TARGETS]
                comps += [("C+B @ 10% vs static LLM+IGSC", ("CB_post_nmt_gain", 10), "static_llm_igsc"),
                          ("C+B @ 10% vs reactive cascade", ("CB_post_nmt_gain", 10), "reactive_cascade")]
                for name, a, b in comps:
                    ra, rb = per_row(o, *arrays[a]), per_row(o, *arrays[b])
                    rec = {"nmt": nmt, "llm": llm, "gate": gate, "comparison": name,
                           "llm_share_a_pct": 100 * arrays[a][0].mean(), "llm_share_b_pct": 100 * arrays[b][0].mean()}
                    metrics = [("cometkiwi", "ck", 1), ("struct_pass_pts", "ok", 100), ("fail_pts", "fail", 100)]
                    for lab, k, scale in metrics:
                        x = scale * (ra[k] - rb[k])
                        rec[lab], rec[f"{lab}_lo"], rec[f"{lab}_hi"] = ci(boot_mean(x, codes, G, W), x.mean())
                    if nmt == "nllb":
                        ca = np.where(arrays[a][1], chrf[f"chrf_{llm}"], chrf.chrf_nllb)
                        cb = np.where(arrays[b][1], chrf[f"chrf_{llm}"], chrf.chrf_nllb)
                        x = ca - cb
                        rec["chrf"], rec["chrf_lo"], rec["chrf_hi"] = ci(boot_mean(x, codes, G, W), x.mean())
                    boots.append(rec)
                print(f"{nmt}/{llm}/{gate} done", flush=True)
    m, bt, hd = pd.DataFrame(matched), pd.DataFrame(boots), pd.DataFrame(heads)
    m.to_csv(RES / "matched.csv", index=False)
    bt.to_csv(RES / "bootstrap_ci.csv", index=False)
    hd.to_csv(RES / "beat_static_llm.csv", index=False)
    write_summary(m, bt, hd)


def write_summary(m, bt, hd):
    f = lambda v, lo, hi, d: f"{v:+.{d}f} [{lo:+.{d}f}, {hi:+.{d}f}]"  # noqa: E731
    md = ["# Experiment 6: improved router", "",
          "Offline replay of the saved outputs, as in Experiment 3. Every policy is compared at the same "
          "share of strings sent to the LLM. CIs: grouped bootstrap, 1,000 resamples of English source groups.", ""]
    for gate in GATES:
        md += [f"## Safety gate {gate}", ""]
        for nmt, llm in PAIRS:
            d = m[(m.nmt == nmt) & (m.llm == llm) & (m.gate == gate)]
            md += [f"### NMT = {nmt}, LLM = {LLM_NAME[llm]}", "",
                   "| target LLM share | policy | actual LLM share % | mean CometKiwi | struct pass % | "
                   "broken or bad meaning % | NMT calls/string |" + (" chrF vs Google |" if nmt == "nllb" else ""),
                   "|---:|---|---:|---:|---:|---:|---:|" + ("---:|" if nmt == "nllb" else "")]
            for r in d.itertuples():
                md.append(f"| {r.target_pct}% | {LABEL[r.policy]} | {r.llm_share_pct:.1f} | {r.mean_cometkiwi:.4f} | "
                          f"{r.struct_pass_pct:.2f} | {r.fail_pct:.2f} | {r.nmt_calls_per_string:.3f} |"
                          + (f" {r.mean_chrf_vs_google:.2f} |" if nmt == "nllb" else ""))
            md += ["", "| comparison | Δ CometKiwi | Δ struct pass (points) | Δ broken or bad (points) |"
                   + (" Δ chrF vs Google |" if nmt == "nllb" else ""), "|---|---|---|---|" + ("---|" if nmt == "nllb" else "")]
            for r in bt[(bt.nmt == nmt) & (bt.llm == llm) & (bt.gate == gate)].itertuples():
                md.append(f"| {r.comparison} | {f(r.cometkiwi, r.cometkiwi_lo, r.cometkiwi_hi, 4)} | "
                          f"{f(r.struct_pass_pts, r.struct_pass_pts_lo, r.struct_pass_pts_hi, 2)} | "
                          f"{f(r.fail_pts, r.fail_pts_lo, r.fail_pts_hi, 2)} |"
                          + (f" {f(r.chrf, r.chrf_lo, r.chrf_hi, 2)} |" if nmt == "nllb" else ""))
            h = hd[(hd.nmt == nmt) & (hd.llm == llm) & (hd.gate == gate)]
            md += ["", "Smallest LLM share at which each policy matches static LLM+IGSC on both mean CometKiwi and "
                   "structural pass: " + "; ".join(
                       f"{LABEL[r.policy]}: " + ("never" if pd.isna(r.min_llm_share_to_beat_static_llm)
                                                 else f"{r.min_llm_share_to_beat_static_llm}%")
                       for r in h.itertuples()), ""]
    (RES / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
