"""Experiment 4: CI/CD commit replay (modelled build time and cost per pull request).

A synthetic PR changes n English strings (n = 10, 50, 250, 1000), drawn at random
from the 10,228 KDE4 strings; each must be translated into all 12 languages.
These are synthetic PR-like workloads, not observed repository history.
Changed strings are new by definition, so the translation memory misses them;
TM behaviour is identical for every policy and therefore left out.

Routing per (string, language) uses the real saved outcomes: whether NLLB fails
the six checks (v1), whether the LLM baseline needed an IGSC repair, and the
out-of-fold Stage 1 risk. Timing and cost come from config/replay_profiles.json,
where every parameter is labelled measured or assumed.

Timing model (per PR):
  NMT path   : cold_nmt (if cold) + n_nmt * s_per_translation, then the verifier.
  LLM calls  : k calls with concurrency c take ceil(k / c) * latency.
               IGSC repair is a second round for the baselines that failed the checks.
  Bypassed strings (predictive) start on the LLM at once (after its cold start);
  escalated strings join when the NMT path and the verifier are done.
  build time = overhead + max(NMT path end, LLM end).
Cost (per PR): NMT machine time and GPU time billed per second including cold start;
API calls billed per token (estimated from character counts).

Outputs (results/04_cicd_replay/): summary.csv, summary.md, coldstart_sensitivity.csv,
nmt_replicas4.csv, figures/build_time_cdf_nllb_<llm>_<gate>.png
"""
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from router.data import LANGS, ROOT, catalogs, load_source, read_lines  # noqa: E402
from router.paths import REPLAY, ROUTER_DATASET, ensure, oof_path  # noqa: E402

CFG = json.loads((ROOT / "config" / "replay_profiles.json").read_text())
NMT = "nllb"
POLICIES = ["static_nmt", "zero_shot_llm", "static_llm_igsc", "reactive_cascade", "predictive_reactive"]
LABEL = {"static_nmt": "static NMT", "zero_shot_llm": "zero-shot LLM", "static_llm_igsc": "static LLM+IGSC",
         "reactive_cascade": "reactive cascade", "predictive_reactive": "predictive + reactive (10% bypass)"}
COLOR = {"static_nmt": "#2a78d6", "zero_shot_llm": "#eb6834", "static_llm_igsc": "#1baf7a",
         "reactive_cascade": "#eda100", "predictive_reactive": "#e87ba4"}   # reference palette slots 1-5
SCRIPT = {"de": "latin", "es": "latin", "fr": "latin", "ja": "cjk", "zh": "cjk", "ar": "arabic",
          "bn": "indic", "hi": "indic", "mr": "indic", "pa": "indic", "ta": "indic", "te": "indic"}
LLM_COL = {"gpt4o": "gpt4o", "llama": "llama"}
LLM_SYSTEM = {"gpt4o": "gpt4o", "llama": "llama_8b"}


def v(x):
    return x["value"] if isinstance(x, dict) else x


def per_row_table(llm, gate="v1"):
    """One row per (string, language): routing facts and token estimates for the LLM calls."""
    df = pd.read_csv(ROUTER_DATASET)
    oof = pd.read_csv(oof_path(NMT))
    assert (oof.line_idx.values == df.line_idx.values).all()
    src = load_source()
    paths = {(c.system, c.tier, c.lang): c.path for c in catalogs()}
    out_chars = {}
    for lang in LANGS:
        b = read_lines(paths[(LLM_SYSTEM[llm], "baseline", lang)])
        r = read_lines(paths[(LLM_SYSTEM[llm], "repaired", lang)])
        out_chars[lang] = (np.array([len(x) for x in b]), np.array([len(x) for x in r]))
    cpt = CFG["token_chars_per_token"]
    src_tok = np.array([max(1, math.ceil(len(s) / cpt["latin"])) for s in src])
    t = pd.DataFrame({"line_idx": df.line_idx, "lang": df.lang,
                      "nmt_fail": (df[f"{NMT}_{'c1' if gate == 'v1' else 'c2'}"] == 0).values,
                      "repair": (df[f"{LLM_COL[llm]}_base_c1"] == 0).values,
                      "risk": (oof.p_combined if gate == "v1" else oof.p_combined_v2).values})
    t["src_tok"] = src_tok[t.line_idx]
    gen_out, rep_out = np.zeros(len(t)), np.zeros(len(t))
    for lang in LANGS:
        m = (t.lang == lang).values
        bc, rc = out_chars[lang]
        gen_out[m] = np.ceil(bc[t.line_idx[m]] / cpt[SCRIPT[lang]])
        rep_out[m] = np.ceil(rc[t.line_idx[m]] / cpt[SCRIPT[lang]])
    t["gen_out_tok"], t["rep_out_tok"] = gen_out, rep_out
    return t


def batch_time(k, prof):
    return 0.0 if k == 0 else math.ceil(k / prof["concurrency"]) * v(prof["latency_s_per_call"])


def simulate_pr(rows, policy, theta, llm, prof, cold_nmt, cold_llm):
    """rows: per-job DataFrame slice for one PR. Returns (build seconds, cost USD, llm calls, nmt calls)."""
    nmt_cfg = CFG["nmt"]
    n = len(rows)
    if policy == "static_nmt":
        bypass, use_nmt = np.zeros(n, bool), np.ones(n, bool)
    elif policy in ("zero_shot_llm", "static_llm_igsc"):
        bypass, use_nmt = np.ones(n, bool), np.zeros(n, bool)
    elif policy == "reactive_cascade":
        bypass, use_nmt = np.zeros(n, bool), np.ones(n, bool)
    else:  # predictive_reactive
        bypass = rows.risk.values >= theta
        use_nmt = ~bypass
    escalate = use_nmt & rows.nmt_fail.values & (policy != "static_nmt")
    repair_on = policy != "zero_shot_llm"
    rep = rows.repair.values & repair_on
    kb, kb_rep = int(bypass.sum()), int((bypass & rep).sum())
    ke, ke_rep = int(escalate.sum()), int((escalate & rep).sum())
    n_nmt = int(use_nmt.sum())

    stage1 = n * v(CFG["stage1_s_per_job"]) if policy == "predictive_reactive" else 0.0
    t_nmt = 0.0
    if n_nmt:
        reps = v(nmt_cfg["replicas"])   # NMT machines working in parallel, each starting at the same time
        t_nmt = (v(nmt_cfg["cold_start_s"]) if cold_nmt else 0.0) + n_nmt * v(nmt_cfg["s_per_translation"]) / reps
        t_nmt += n_nmt * v(CFG["verifier_s_per_job"])
    t_ready = stage1 + t_nmt
    llm_cold = v(prof["cold_start_s"]) if cold_llm else 0.0
    w_b = batch_time(kb, prof) + batch_time(kb_rep, prof)
    w_e = batch_time(ke, prof) + batch_time(ke_rep, prof)
    end_llm = 0.0
    if kb:
        end_llm = stage1 + llm_cold + w_b
    if ke:
        end_llm = max(end_llm, max(stage1 + llm_cold, t_ready)) + w_e
    build = v(CFG["pipeline_overhead_s"]) + max(t_ready, end_llm)

    cost = 0.0
    if n_nmt:  # every replica is billed for its cold start; the translation work itself is shared
        cost = ((v(nmt_cfg["cold_start_s"]) if cold_nmt else 0.0) * v(nmt_cfg["replicas"])
                + n_nmt * v(nmt_cfg["s_per_translation"])) / 3600 * v(nmt_cfg["usd_per_hour"])
    calls = kb + ke + kb_rep + ke_rep
    if calls:
        if "usd_per_hour" in prof:  # GPU container billed per second, including its cold start
            cost += ((llm_cold + w_b + w_e) / 3600) * v(prof["usd_per_hour"])
        else:  # API billed per token
            sel = bypass | escalate
            gen_in = (v(prof["prompt_tokens"]) + rows.src_tok.values[sel]).sum()
            gen_out = rows.gen_out_tok.values[sel].sum()
            rsel = sel & rep
            rep_in = (v(prof["repair_prompt_tokens"]) + rows.src_tok.values[rsel]).sum()
            rep_out = rows.rep_out_tok.values[rsel].sum()
            cost += ((gen_in + rep_in) * v(prof["usd_per_1m_input_tokens"])
                     + (gen_out + rep_out) * v(prof["usd_per_1m_output_tokens"])) / 1e6
    return build, cost, calls, n_nmt


def run(llm, prof, profiles, sizes, n_prs, theta, seed=0, gate="v1"):
    t = per_row_table(llm, gate)
    by_line = {i: g.index.values for i, g in t.groupby("line_idx")}
    lines = np.array(sorted(by_line))
    rng = np.random.default_rng(seed)
    records = []
    for size in sizes:
        prs = [rng.choice(lines, size, replace=False) for _ in range(n_prs)]
        for pname, p_cold in profiles.items():
            colds = rng.random((n_prs, 2)) < p_cold
            for policy in POLICIES:
                for j, pr in enumerate(prs):
                    rows = t.loc[np.concatenate([by_line[i] for i in pr])]
                    b, c, calls, nn = simulate_pr(rows, policy, theta, llm, prof, colds[j, 0], colds[j, 1])
                    records.append({"gate": gate, "llm": llm, "pr_size": size, "start": pname, "policy": policy,
                                    "pr": j, "build_s": b, "cost_usd": c, "llm_calls": calls, "nmt_calls": nn})
    return pd.DataFrame(records)


def summarize(raw):
    g = raw.groupby(["gate", "llm", "pr_size", "start", "policy"])
    s = g.agg(p50_s=("build_s", "median"), p90_s=("build_s", lambda x: np.percentile(x, 90)),
              p99_s=("build_s", lambda x: np.percentile(x, 99)), mean_cost_usd=("cost_usd", "mean"),
              mean_llm_calls=("llm_calls", "mean"), mean_nmt_calls=("nmt_calls", "mean")).reset_index()
    return s


def cdf_figure(raw, llm, gate, start="mixed"):
    sizes = CFG["pr_sizes"]
    fig, axes = plt.subplots(1, len(sizes), figsize=(15, 3.9), sharey=True)
    for ax, size in zip(axes, sizes):
        d = raw[(raw.gate == gate) & (raw.llm == llm) & (raw.pr_size == size) & (raw.start == start)]
        for policy in POLICIES:
            x = np.sort(d[d.policy == policy].build_s.values)
            ax.plot(x, np.arange(1, len(x) + 1) / len(x), color=COLOR[policy], lw=2, label=LABEL[policy])
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_title(f"{size} changed strings × 12 languages", fontsize=9, loc="left", color="#0b0b0b")
        ax.set_xlabel("Build time (s, log scale)", color="#52514e")
        ax.grid(color="#e4e3df", lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(colors="#52514e", labelsize=8)
    axes[0].set_ylabel("Share of PRs finished", color="#52514e")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8, loc="lower center", ncol=len(POLICIES))
    name = {"gpt4o": "GPT-4o-mini (API)", "llama": "Llama-3.1-8B (serverless A100)"}[llm]
    fig.suptitle(f"Modelled CI build time per PR, NMT = NLLB, LLM = {name}, safety gate {gate}, {start} start "
                 f"(30% of PRs hit cold services); synthetic PRs, 1,000 per size",
                 fontsize=10, x=0.01, ha="left", color="#0b0b0b")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(ensure(REPLAY / "figures") / f"build_time_cdf_nllb_{llm}_{gate}.png", dpi=200, facecolor="#fcfcfb")
    plt.close(fig)


def main():
    oof = pd.read_csv(oof_path(NMT))
    # bypass about 10% of jobs, as in Experiment 3; risk label matches the gate's verifier
    thetas = {"v1": float(np.quantile(oof.p_combined, 0.90)), "v2": float(np.quantile(oof.p_combined_v2, 0.90))}
    sizes, n_prs, profiles = CFG["pr_sizes"], CFG["prs_per_size"], CFG["start_profiles"]
    raws, sens, rep = [], [], []
    for gate, theta in thetas.items():
        for llm, prof in CFG["llm_profiles"].items():
            raws.append(run(llm, prof, profiles, sizes, n_prs, theta, CFG["seed"], gate))
            print(f"gate {gate} {llm}: simulated", flush=True)
        # cold-start sensitivity (Llama GPU container): 20 / 60 / 180 s, all-cold profile, 200 PRs per size
        for cs in (20.0, 60.0, 180.0):
            prof = json.loads(json.dumps(CFG["llm_profiles"]["llama"]))
            prof["cold_start_s"]["value"] = cs
            x = summarize(run("llama", prof, {"cold": 1.0}, sizes, 200, theta, CFG["seed"], gate))
            x["llm_cold_start_s"] = cs
            sens.append(x)
        # NMT scale-out sensitivity: 4 NMT machines instead of the measured single runner, mixed start
        saved = CFG["nmt"]["replicas"]["value"]
        CFG["nmt"]["replicas"]["value"] = 4
        for llm, prof in CFG["llm_profiles"].items():
            x = summarize(run(llm, prof, {"mixed": CFG["start_profiles"]["mixed"]}, sizes, 200, theta,
                              CFG["seed"], gate))
            x["nmt_replicas"] = 4
            rep.append(x)
        CFG["nmt"]["replicas"]["value"] = saved
    raw = pd.concat(raws)
    s = summarize(raw)
    s.to_csv(ensure(REPLAY) / "summary.csv", index=False)
    pd.concat(sens).to_csv(REPLAY / "coldstart_sensitivity.csv", index=False)
    pd.concat(rep).to_csv(REPLAY / "nmt_replicas4.csv", index=False)
    for gate in thetas:
        for llm in CFG["llm_profiles"]:
            cdf_figure(raw, llm, gate)

    md = ["# Experiment 4: CI/CD commit replay (modelled; synthetic PRs)", "",
          "NMT = NLLB. Predictive + reactive bypasses about 10% of jobs "
          f"(θ = {thetas['v1']:.3f} with the v1 gate, {thetas['v2']:.3f} with the v2 gate). "
          "Parameters and their sources: config/replay_profiles.json (measured vs assumed).", ""]
    order = {p: i for i, p in enumerate(POLICIES)}
    for gate in thetas:
        for llm in CFG["llm_profiles"]:
            for start in ("warm", "mixed", "cold"):
                md += [f"## Gate {gate}, LLM = {llm}, {start} start", "",
                       "| PR size | policy | median build s | p90 s | p99 s | mean cost per PR (USD) | LLM calls | NMT calls |",
                       "|---:|---|---:|---:|---:|---:|---:|---:|"]
                d = s[(s.gate == gate) & (s.llm == llm) & (s.start == start)].copy()
                d["o"] = d.policy.map(order)
                for r in d.sort_values(["pr_size", "o"]).itertuples():
                    md.append(f"| {r.pr_size} | {r.policy} | {r.p50_s:.1f} | {r.p90_s:.1f} | {r.p99_s:.1f} | "
                              f"{r.mean_cost_usd:.5f} | {r.mean_llm_calls:.0f} | {r.mean_nmt_calls:.0f} |")
                md.append("")
    (REPLAY / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
