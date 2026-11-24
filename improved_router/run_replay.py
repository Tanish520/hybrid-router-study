"""Experiment 6, step 3: build time and cost per pull request for the improved router (C+B).

Reuses the Experiment 4 model unchanged (scripts/replay.py: same synthetic PRs,
same timing and cost parameters from config/replay_profiles.json, NMT = NLLB, mixed start).
C+B is a reactive cascade whose escalation set is "fails the checks OR predicted gain >= theta";
NMT always runs, so the LLM-output fallback needs no extra NMT call. Its Stage 2 scoring time
(features + prediction per job) is added to the build time. Theta is chosen so that C+B sends
the same share of strings to the LLM as predictive + reactive at 10% bypass (Experiment 4).

Output: results/06_improved_router/replay.{csv,md}
"""
import sys

import numpy as np
import pandas as pd

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from improved_router.common import RES, ROOT, ROUTER_DATASET, matched_theta, outcomes, route  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
import replay  # noqa: E402

POLICIES = ["static_llm_igsc", "reactive_cascade", "predictive_reactive", "CB_post_nmt_gain"]


def main():
    cfg = replay.CFG
    df = pd.read_csv(ROUTER_DATASET)
    ext = pd.read_csv(RES / "stage2_scores_nllb.csv.gz")
    stage_s = replay.v(cfg["stage1_s_per_job"])
    records, notes = [], []
    for gate in ("v1", "v2"):
        for llm, prof in cfg["llm_profiles"].items():
            t = replay.per_row_table(llm, gate)
            theta_g = float(np.quantile(t.risk, 0.90))          # the Experiment 4 setting
            o = outcomes(df, "nllb", llm, gate)
            share_g = 100 * ((t.risk.values >= theta_g) | t.nmt_fail.values).mean()
            s = ext[f"gain_post_{llm}"].values
            th = matched_theta("CB_post_nmt_gain", s, o, share_g)
            sent = route("CB_post_nmt_gain", s, th, o)[0]
            t["esc_cb"] = sent
            notes.append(f"gate {gate}, {llm}: predictive + reactive sends {share_g:.1f}% of jobs to the LLM; "
                         f"C+B sends {100 * sent.mean():.1f}%")
            by_line = {i: g.index.values for i, g in t.groupby("line_idx")}
            lines = np.array(sorted(by_line))
            rng = np.random.default_rng(cfg["seed"])
            p_cold = cfg["start_profiles"]["mixed"]
            for size in cfg["pr_sizes"]:
                prs = [rng.choice(lines, size, replace=False) for _ in range(cfg["prs_per_size"])]
                colds = rng.random((len(prs), 2)) < p_cold
                for j, pr in enumerate(prs):
                    rows = t.loc[np.concatenate([by_line[i] for i in pr])]
                    for p in POLICIES:
                        if p == "CB_post_nmt_gain":
                            b, c, calls, nn = replay.simulate_pr(rows.assign(nmt_fail=rows.esc_cb.values),
                                                                 "reactive_cascade", 0, llm, prof, *colds[j])
                            b += len(rows) * stage_s
                        else:
                            b, c, calls, nn = replay.simulate_pr(rows, p, theta_g, llm, prof, *colds[j])
                        records.append({"gate": gate, "llm": llm, "pr_size": size, "policy": p, "pr": j,
                                        "build_s": b, "cost_usd": c, "llm_calls": calls,
                                        "needs_llm": calls > 0})
            print(notes[-1], flush=True)
    raw = pd.DataFrame(records)
    s = raw.groupby(["gate", "llm", "pr_size", "policy"]).agg(
        p50_s=("build_s", "median"), p90_s=("build_s", lambda x: np.percentile(x, 90)),
        mean_cost_usd=("cost_usd", "mean"), mean_llm_calls=("llm_calls", "mean"),
        prs_needing_llm_pct=("needs_llm", lambda x: 100 * x.mean())).reset_index()
    s.to_csv(RES / "replay.csv", index=False)
    order = {p: i for i, p in enumerate(POLICIES)}
    md = ["# Experiment 6: build time and cost per PR (Experiment 4 model, NLLB, mixed start)", "",
          *[f"- {n}" for n in notes], "",
          "C+B is compared with predictive + reactive at the same share of strings sent to the LLM.", ""]
    for gate in ("v1", "v2"):
        for llm in cfg["llm_profiles"]:
            md += [f"## Gate {gate}, LLM = {llm}", "",
                   "| PR size | policy | median build s | p90 s | mean cost per PR (USD) | LLM calls | PRs needing the LLM % |",
                   "|---:|---|---:|---:|---:|---:|---:|"]
            d = s[(s.gate == gate) & (s.llm == llm)].copy()
            d["o"] = d.policy.map(order)
            for r in d.sort_values(["pr_size", "o"]).itertuples():
                md.append(f"| {r.pr_size} | {r.policy} | {r.p50_s:.1f} | {r.p90_s:.1f} | {r.mean_cost_usd:.5f} | "
                          f"{r.mean_llm_calls:.0f} | {r.prs_needing_llm_pct:.1f} |")
            md.append("")
    (RES / "replay.md").write_text("\n".join(md) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
