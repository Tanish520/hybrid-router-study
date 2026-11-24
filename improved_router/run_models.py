"""Experiment 6, step 1: train the new scores (out-of-fold) with the Experiment 2 setup.

Same LightGBM settings (imported from scripts/train_router.py, never tuned), same grouped
5-fold split (every row of one English text in one fold), so each string is scored by a
model that never saw it.

Per NMT engine N (nllb, google):
  post_risk_v1 / post_risk_v2   N fails the checks (v1/v2) or CometKiwi(N) < 0.6, from the
                                12 source features + 7 features of N's output (Stage 2)
Per (N, LLM L):
  gain_src_<L>                  CometKiwi(L+IGSC) - CometKiwi(N), from the 12 source features
  gain_post_<L>                 the same target, from source + output features

Outputs (results/06_improved_router/): stage2_scores_<nmt>.csv.gz, model_metrics.{json,md}
"""
import json
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from improved_router.common import RES, ROOT, ROUTER_DATASET, SEM, oof_path  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from router.data import LANGS, catalogs, load_source, read_lines  # noqa: E402
from router.features import FEATURES, features  # noqa: E402
from train_router import PARAMS, TIERS  # noqa: E402
from improved_router.output_features import OUTPUT_FEATURES, output_features  # noqa: E402

NMT_SYSTEM = {"nllb": ("nllb", "output"), "google": ("google", "output")}
LLMS = ("gpt4o", "llama")


def output_frame(nmt, src):
    paths = {(c.system, c.tier, c.lang): c.path for c in catalogs()}
    rows = []
    for lang in LANGS:
        hyp = read_lines(paths[(*NMT_SYSTEM[nmt], lang)])
        rows += [output_features(s, h) for s, h in zip(src, hyp)]
    return pd.DataFrame(rows, columns=OUTPUT_FEATURES)


def cv(X, y, groups, regression=False):
    params = dict(PARAMS, objective="regression") if regression else PARAMS
    out = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=5, shuffle=True, random_state=0).split(X, y, groups):
        m = (lgb.LGBMRegressor if regression else lgb.LGBMClassifier)(**params)
        m.fit(X.iloc[tr], y[tr], categorical_feature=["script_tier"])
        out[te] = m.predict(X.iloc[te]) if regression else m.predict_proba(X.iloc[te])[:, 1]
    return out


def time_stage2(X, y, src, hyps, langs, n=2000):
    """Microseconds per string for source features + output features + one prediction."""
    m = lgb.LGBMClassifier(**PARAMS).fit(X, y, categorical_feature=["script_tier"])
    t0 = time.perf_counter()
    rows = []
    for s, h, lang in zip(src[:n], hyps[:n], langs[:n]):
        r = features(s, lang)
        r.update(output_features(s, h))
        rows.append(r)
    t1 = time.perf_counter()
    F = pd.DataFrame(rows)[X.columns]
    F["script_tier"] = pd.Categorical(F.script_tier, categories=TIERS)
    m.booster_.predict(F)            # batch prediction, as in the Experiment 2 timing
    t2 = time.perf_counter()
    return {"features_us": 1e6 * (t1 - t0) / n, "predict_us": 1e6 * (t2 - t1) / n}


def main():
    df = pd.read_csv(ROUTER_DATASET)
    src = load_source()
    groups = df.group.values
    X = df[FEATURES].copy()
    X["script_tier"] = pd.Categorical(X.script_tier, categories=TIERS)
    metrics = {}
    for nmt in NMT_SYSTEM:
        ref = pd.read_csv(oof_path(nmt))
        out = output_frame(nmt, src)
        Xh = pd.concat([X, out], axis=1)
        res = pd.DataFrame({"line_idx": df.line_idx, "lang": df.lang})
        sem = (df[f"{nmt}_ck"] < SEM).values
        m = metrics[nmt] = {}
        for gate, c in (("v1", "c1"), ("v2", "c2")):
            y = ((df[f"{nmt}_{c}"] == 0).values | sem).astype(int)
            res[f"post_risk_{gate}"] = cv(Xh, y, groups)
            ref_col = "p_combined" if gate == "v1" else "p_combined_v2"
            m[f"combined_{gate}_auc"] = {"stage1_source_only": round(roc_auc_score(y, ref[ref_col]), 4),
                                         "with_output_features": round(roc_auc_score(y, res[f"post_risk_{gate}"]), 4)}
            print(f"{nmt} combined_{gate}: AUC {m[f'combined_{gate}_auc']}", flush=True)
        p_sem = cv(Xh, sem.astype(int), groups)
        m["sem_auc"] = {"stage1_source_only": round(roc_auc_score(sem, ref.p_sem), 4),
                        "with_output_features": round(roc_auc_score(sem, p_sem), 4)}
        for llm in LLMS:
            g = (df[f"{llm}_igsc_ck"] - df[f"{nmt}_ck"]).values
            res[f"gain_src_{llm}"] = cv(X, g, groups, regression=True)
            res[f"gain_post_{llm}"] = cv(Xh, g, groups, regression=True)
            m[f"gain_{llm}_spearman"] = {
                "stage1_risk_score": round(spearmanr(ref.p_combined_v2, g).statistic, 4),
                "gain_source_only": round(spearmanr(res[f"gain_src_{llm}"], g).statistic, 4),
                "gain_with_output_features": round(spearmanr(res[f"gain_post_{llm}"], g).statistic, 4)}
            print(f"{nmt}/{llm} gain: {m[f'gain_{llm}_spearman']}", flush=True)
        res.to_csv(RES / f"stage2_scores_{nmt}.csv.gz", index=False, compression="gzip")
        if nmt == "nllb":
            paths = {(c.system, c.tier, c.lang): c.path for c in catalogs()}
            hyps = sum((read_lines(paths[("nllb", "output", lang)]) for lang in LANGS), [])
            y = ((df.nllb_c2 == 0).values | sem).astype(int)
            idx = np.random.default_rng(0).choice(len(df), 2000, replace=False)
            metrics["stage2_timing"] = time_stage2(Xh, y, [src[i] for i in df.line_idx.values[idx]],
                                                   [hyps[i] for i in idx], df.lang.values[idx])
            print("timing", metrics["stage2_timing"], flush=True)
    (RES / "model_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

    md = ["# Experiment 6: Stage 2 scores (out-of-fold, grouped 5-fold, Experiment 2 LightGBM settings)", "",
          "## Does seeing the NMT output help predict failure? (ROC-AUC, higher is better)", "",
          "| NMT | label | Stage 1 (source only) | + 7 NMT-output features |", "|---|---|---:|---:|"]
    for nmt in NMT_SYSTEM:
        for k in ("sem_auc", "combined_v1_auc", "combined_v2_auc"):
            v = metrics[nmt][k]
            md.append(f"| {nmt} | {k.replace('_auc', '')} | {v['stage1_source_only']:.3f} | {v['with_output_features']:.3f} |")
    md += ["", "## Does the score rank strings by how much the LLM would improve them? (Spearman with the true gain)", "",
           "| NMT | LLM | Stage 1 risk score | gain, source only | gain, + output features |", "|---|---|---:|---:|---:|"]
    for nmt in NMT_SYSTEM:
        for llm in LLMS:
            v = metrics[nmt][f"gain_{llm}_spearman"]
            md.append(f"| {nmt} | {llm} | {v['stage1_risk_score']:.3f} | {v['gain_source_only']:.3f} | "
                      f"{v['gain_with_output_features']:.3f} |")
    t = metrics["stage2_timing"]
    md += ["", f"Stage 2 cost per string (laptop CPU): features {t['features_us']:.1f} µs, "
               f"prediction {t['predict_us']:.1f} µs (batch).", ""]
    (RES / "model_metrics.md").write_text("\n".join(md))


if __name__ == "__main__":
    main()
