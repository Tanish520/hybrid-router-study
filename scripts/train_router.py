"""Experiment 2: Stage 1 source-only risk predictor (LightGBM, grouped 5-fold CV).

Predicts whether the NMT output will fail, from the 12 source/locale features only.
Labels (fixed before training):
  struct    the six checks fail (verifier v1)
  sem       CometKiwi of the NMT output < 0.6
  combined  struct OR sem   (primary label: CometKiwi and C = 1)
  combined_v2  the same with verifier v2 (used when the safety gate is v2)
Sensitivity: struct with verifier v2; sem at 0.5 and 0.7.

Grouped 5-fold CV: every row of the same English text (all 12 languages and
duplicates) is in one fold, so the model is always tested on unseen strings.
LightGBM settings are fixed in advance and never tuned on test folds.

Outputs (results/02_risk_predictor/): oof_predictions_<nmt>.csv.gz (out-of-fold risk per row),
metrics.json, figures/shap_<nmt>.png
"""
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from router.data import load_source  # noqa: E402
from router.features import FEATURES, features  # noqa: E402
from router.paths import PREDICTOR, ROUTER_DATASET, ensure, oof_path  # noqa: E402

PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_child_samples=50,
              n_estimators=300, subsample=0.8, subsample_freq=1, colsample_bytree=0.9,
              random_state=0, verbose=-1)
SEM_THRESHOLD = 0.6
NMT_ENGINES = ["nllb", "google"]
TIER_NAME = {0: "Latin", 1: "CJK", 2: "Arabic", 3: "Indic"}
TIERS = sorted(TIER_NAME)


def labels(df, nmt):
    return {
        "struct": 1 - df[f"{nmt}_c1"],
        "sem": (df[f"{nmt}_ck"] < SEM_THRESHOLD).astype(int),
        "combined": ((1 - df[f"{nmt}_c1"]) | (df[f"{nmt}_ck"] < SEM_THRESHOLD)).astype(int),
        "struct_v2": 1 - df[f"{nmt}_c2"],
        "combined_v2": ((1 - df[f"{nmt}_c2"]) | (df[f"{nmt}_ck"] < SEM_THRESHOLD)).astype(int),
        "sem_0.5": (df[f"{nmt}_ck"] < 0.5).astype(int),
        "sem_0.7": (df[f"{nmt}_ck"] < 0.7).astype(int),
    }


def cross_validate(X, y, groups, with_shap=False):
    oof, shap = np.zeros(len(y)), (np.zeros((len(y), X.shape[1])) if with_shap else None)
    folds = np.zeros(len(y), dtype=int)
    for k, (tr, te) in enumerate(GroupKFold(n_splits=5, shuffle=True, random_state=0).split(X, y, groups)):
        model = lgb.LGBMClassifier(**PARAMS)
        model.fit(X.iloc[tr], y.iloc[tr], categorical_feature=["script_tier"])
        oof[te] = model.predict_proba(X.iloc[te])[:, 1]
        if with_shap:  # TreeSHAP on a 5,000-row sample of the test fold (drop the bias column)
            idx = te if len(te) <= 5000 else np.random.default_rng(k).choice(te, 5000, replace=False)
            shap[idx] = model.booster_.predict(X.iloc[idx], pred_contrib=True)[:, :-1]
        folds[te] = k
        print(f"    fold {k + 1}/5 done", flush=True)
    if with_shap:
        shap = shap[np.abs(shap).sum(axis=1) > 0]
    return oof, shap, folds


def rule_score(X):
    """Rule baseline: strings with a placeholder or accelerator are risky."""
    return ((X.has_placeholder == 1) | (X.has_accelerator == 1)).astype(float)


def evaluate(y, p):
    return {"prevalence": round(float(y.mean()), 4), "roc_auc": round(roc_auc_score(y, p), 4),
            "pr_auc": round(average_precision_score(y, p), 4), "brier": round(brier_score_loss(y, p), 4)}


def time_stage1(df, X, y):
    """CPU cost of Stage 1: feature extraction + prediction, per string."""
    model = lgb.LGBMClassifier(**PARAMS).fit(X, y, categorical_feature=["script_tier"])
    src = load_source()
    sample = df.sample(10000, random_state=1)
    t0 = time.perf_counter()
    feats = pd.DataFrame([features(src[i], lang) for i, lang in zip(sample.line_idx, sample.lang)])[FEATURES]
    feats["script_tier"] = pd.Categorical(feats["script_tier"], categories=TIERS)
    t1 = time.perf_counter()
    model.predict_proba(feats)
    t2 = time.perf_counter()
    one = feats.iloc[[0]]
    t3 = time.perf_counter()
    for _ in range(200):
        model.predict_proba(one)
    t4 = time.perf_counter()
    return {"features_us_per_string": round((t1 - t0) / len(sample) * 1e6, 2),
            "predict_us_per_string_batched": round((t2 - t1) / len(sample) * 1e6, 2),
            "predict_ms_single_call": round((t4 - t3) / 200 * 1e3, 3)}


def shap_figure(shap, nmt, label):
    imp = pd.Series(np.abs(shap).mean(axis=0), index=FEATURES).sort_values()
    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.barh(imp.index, imp.values, color="#2a78d6", height=0.6)
    for i, v in enumerate(imp.values):
        ax.text(v, i, f" {v:.3f}", va="center", fontsize=8, color="#0b0b0b")
    ax.set_xlabel("Mean |SHAP value| (log-odds contribution to predicted NMT failure)", color="#52514e")
    ax.set_title(f"What drives predicted {nmt.upper()} failure risk ({label} label, out-of-fold)",
                 fontsize=10, loc="left", color="#0b0b0b")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors="#52514e", labelsize=9)
    fig.tight_layout()
    fig.savefig(ensure(PREDICTOR / "figures") / f"shap_{nmt}.png", dpi=200, facecolor="#fcfcfb")


def main():
    df = pd.read_csv(ROUTER_DATASET)
    X = df[FEATURES].copy()
    X["script_tier"] = pd.Categorical(X["script_tier"], categories=TIERS)
    groups = df["group"]
    report = {}
    for nmt in NMT_ENGINES:
        out = df[["line_idx", "lang"]].copy()
        report[nmt] = {}
        for name, y in labels(df, nmt).items():
            oof, shap, folds = cross_validate(X, y, groups, with_shap=(name == "combined"))
            out[f"p_{name}"] = oof
            out["fold"] = folds
            m = {"lightgbm": evaluate(y, oof), "rule_placeholder_or_accel": evaluate(y, rule_score(X))}
            m["lightgbm_by_script"] = {TIER_NAME[t]: evaluate(y[df.script_tier == t], oof[df.script_tier == t])
                                       for t in sorted(TIER_NAME)}
            report[nmt][name] = m
            if name == "combined":
                shap_figure(shap, nmt, name)
                report[nmt]["shap_mean_abs_combined"] = dict(
                    sorted(zip(FEATURES, np.abs(shap).mean(axis=0).round(4).tolist()), key=lambda x: -x[1]))
            print(f"{nmt:7s} {name:9s} prevalence {m['lightgbm']['prevalence']:.3f}  "
                  f"AUC {m['lightgbm']['roc_auc']:.3f} (rule {m['rule_placeholder_or_accel']['roc_auc']:.3f})  "
                  f"PR-AUC {m['lightgbm']['pr_auc']:.3f} (rule {m['rule_placeholder_or_accel']['pr_auc']:.3f})")
        out.to_csv(oof_path(nmt), index=False, compression="gzip")
    report["settings"] = {"lightgbm": PARAMS, "folds": "GroupKFold(5, shuffle, seed 0) by source text",
                          "sem_threshold": SEM_THRESHOLD}
    (ensure(PREDICTOR) / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    report["stage1_timing_cpu"] = time_stage1(df, X, labels(df, "nllb")["combined"])
    report["settings"] = {"lightgbm": PARAMS, "folds": "GroupKFold(5, shuffle, seed 0) by source text",
                          "sem_threshold": SEM_THRESHOLD}
    (ensure(PREDICTOR) / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["stage1_timing_cpu"]))


if __name__ == "__main__":
    main()
