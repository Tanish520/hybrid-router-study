# 02 · Stage 1 risk predictor

## Question
Before anything is translated, can a cheap model predict from the English source alone whether NMT will fail?

## Why it was run
Stage 1 of the router sends predicted high-risk strings straight to the LLM. That only helps if the predictor
is better than a simple rule, and if it is cheap enough to run on every string in a CI/CD build.

## How it was produced
`scripts/train_router.py` (about 12 minutes). It uses:
- **Model:** LightGBM, with 12 source/locale features (`router/features.py`): length, token count, placeholder,
  accelerator, XML, camelCase density, non-translatable-term match, digit, punctuation and uppercase ratios,
  identifier-like, and the target script tier.
- **Labels,** fixed before training, for NLLB and for Google:
  - `struct`: verifier v1 fails;
  - `sem`: CometKiwi < 0.6;
  - `combined`: either (the primary label);
  - `struct_v2` / `combined_v2`: the same with verifier v2;
  - sensitivity labels at CometKiwi < 0.5 and < 0.7.
- **Validation:** grouped 5-fold cross-validation. Every copy of an English string (all 12 languages and
  duplicates) stays in one fold, so every prediction is for a string the model never saw. Model settings were
  fixed in advance, not tuned.
- **Baseline:** the rule "risky if the source has a placeholder or an accelerator".
- **Explanation:** TreeSHAP on the `combined` label.

## Files

| File | Contents |
|---|---|
| `oof_predictions_nllb.csv.gz`, `oof_predictions_google.csv.gz` | Out-of-fold risk per string–language pair: `p_struct`, `p_sem`, `p_combined`, `p_struct_v2`, `p_combined_v2`, `p_sem_0.5`, `p_sem_0.7`, and `fold`. These are the Stage 1 scores used in Experiments 3–6 |
| `metrics.json` | Per engine and label: prevalence, ROC-AUC, PR-AUC and Brier score for LightGBM, the rule baseline and each script tier; mean absolute SHAP; model settings; CPU timing |
| `figures/shap_nllb.png`, `figures/shap_google.png` | Feature importance (mean absolute SHAP) |

## Result (NLLB; confidence intervals in [05](../05_confidence_intervals/))

| Label | Failure rate | LightGBM AUC | Rule AUC | Difference [95% CI] |
|---|---:|---:|---:|---|
| struct | 1.9% | 0.972 | 0.854 | +0.118 [0.104, 0.137] |
| sem (CometKiwi < 0.6) | 7.5% | 0.679 | 0.501 | +0.178 [0.166, 0.189] |
| combined | 9.2% | 0.727 | 0.572 | +0.155 [0.146, 0.165] |
| combined (v2) | 11.2% | 0.771 | 0.649 | +0.122 [0.114, 0.128] |

- **Structural failure is easy to predict; low quality is hard.** The source shows whether there is a
  placeholder to lose, but not whether its meaning is ambiguous.
- **Cost:** 9.5 µs for features plus 3.0 µs for prediction per string on a laptop CPU, against a target of
  under 100 µs.
- **Main SHAP drivers:** target script tier, string length, placeholder presence. These are associations,
  not causes.
- Google as the NMT engine gives similar results.
