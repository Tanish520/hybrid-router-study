# 05 · Confidence intervals

## Question
Are the differences reported in Experiments 2 and 3 real, or could they come from which strings happen to be
in the corpus?

## Why it was run
Many policy differences are a fraction of a percentage point. Without intervals, a reader cannot tell a real
effect from sampling noise.

## How it was produced
`scripts/bootstrap.py` (about 1 minute) runs a **grouped bootstrap**. It resamples English source strings
(with all their languages and duplicates together) with replacement, and recomputes every metric:
- Experiment 3: 1,000 resamples;
- Experiment 2: 200 resamples (each requires recomputing AUCs).

Differences are paired: both policies are evaluated on the same resampled strings. An interval is the 2.5th
to 97.5th percentile, and a difference is significant when its interval excludes 0.

## Files

| File | Contents |
|---|---|
| `bootstrap_ci.csv` | `experiment`, `gate`, `nmt`, `llm`, `item` (policy or comparison), `metric`, `estimate`, `ci_low`, `ci_high`, `significant` |
| `bootstrap_ci.md` | The same as readable tables per experiment, gate and model pair |

## Key intervals
- **Reactive cascade (gate v2) minus static LLM + IGSC** (NLLB + GPT-4o-mini): structural pass +0.81 points
  [0.74, 0.88]; mean CometKiwi −0.0038 [−0.0045, −0.0032].
- **Stage 1 minus random bypass at 10% LLM share** (gate v1): mean CometKiwi +0.0012 [0.0010, 0.0014]. Stage 1
  is significantly better than chance at every share.
- **LightGBM minus rule baseline** (AUC, NLLB): +0.118 [0.104, 0.137] for struct, +0.178 [0.166, 0.189] for
  sem, +0.155 [0.146, 0.165] for combined.
