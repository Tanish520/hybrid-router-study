# 06 · Improved router

## Question
Experiment 3 showed that Stage 1 bypass lets unchecked LLM errors through and adds little quality. Can three
targeted changes fix this, without changing the two-stage structure?

## Why it was run
Experiment 3 exposed three weaknesses, and each change targets one:

| Change | Weakness | Fix |
|---|---|---|
| **A · Gate the LLM output** | Bypassed LLM outputs are never verified | If the LLM output fails the verifier and the NMT output passes, deliver the NMT output |
| **B · Predict gain, not risk** | Strings where NMT is weak are often hard for the LLM too | Rank strings by predicted CometKiwi(LLM) − CometKiwi(NMT) instead of NMT failure risk |
| **C · Stage 2 reads the NMT output** | The gate ignores the translation it already has | Add 7 cheap output features: length ratio, token ratio, copied source, leftover Latin letters, digits match, end punctuation match, repeated words |

The main candidate is **C + B with A's gate**. NMT translates every string, failures go to the LLM, and the
strings with the highest predicted gain (from source + output features) are escalated too. The LLM output is
verified as well.

## How it was produced
Same LightGBM settings, same grouped 5-fold split and same PR replay model as Experiments 2–4. Code is in
[`improved_router/`](../../improved_router/).

| Step | Command | What it does |
|---|---|---|
| 1 | `python improved_router/run_models.py` | Out-of-fold Stage 2 risk and gain scores (about 15 min) |
| 2 | `python improved_router/run_policies.py` | θ sweep, comparison at **equal LLM share**, grouped bootstrap (1,000), and chrF check |
| 3 | `python improved_router/run_replay.py` | Cost and build time per PR with the Experiment 4 model |
| 4 | `python improved_router/spot_check.py make` / `score` | Blind A/B sheet for human raters (written to `spot_check/`, not committed) |

**Independent check.** The gain model is trained on CometKiwi and judged by CometKiwi. As a check that does not
use CometKiwi, the NLLB pairs are also scored with sentence chrF against Google Translate as a pseudo-reference.
Google is not one of the routed systems in those pairs.

## Files

| File | Contents |
|---|---|
| `stage2_scores_<nmt>.csv.gz` | Out-of-fold scores per pair: `post_risk_v1/v2` (source + output features), `gain_src_<llm>`, `gain_post_<llm>` |
| `model_metrics.json` / `.md` | AUC with and without output features; Spearman correlation of each score with the true gain; Stage 2 CPU time |
| `policies_<nmt>_<llm>_<gate>.csv` | θ sweep for the five policies (reference, A, B, C, C+B) and the fixed policies |
| `matched.csv` | Every policy at 10 / 20 / 30% LLM share, including chrF for NLLB pairs |
| `bootstrap_ci.csv` | Paired 95% intervals: C+B vs the Experiment 3 policy, vs static LLM + IGSC, vs reactive cascade |
| `beat_static_llm.csv` | Smallest LLM share at which each policy matches static LLM + IGSC on both CometKiwi and structural pass |
| `chrf_vs_google.csv.gz` | Sentence chrF of NLLB and each LLM output against Google Translate |
| `summary.md` | All of the above as readable tables |
| `replay.csv` / `replay.md` | Cost and build time per PR |
| `figures/pareto_<nmt>_<llm>_<gate>.png` | Quality (left) and broken-or-poor rate (right, lower is better) against LLM share |

## Result (NLLB + GPT-4o-mini, gate v2, per 1,000 string–language pairs)

| Router | Sent to LLM | Structurally broken | Broken or poor | Mean CometKiwi |
|---|---:|---:|---:|---:|
| Static LLM + IGSC | 1,000 | 18 | 83 | 0.7877 |
| Reactive cascade | 41 | 10 | 84 | 0.7838 |
| Predictive + reactive (Experiment 3) | 101 | 13 | 84 | 0.7846 |
| **C + B** | **99** | **10** | **73** | **0.7882** |

**Against the Experiment 3 router at the same LLM share:**
- 11.5 fewer broken-or-poor translations per 1,000 [10.6, 12.5];
- CometKiwi +0.0037 [+0.0034, +0.0039];
- chrF +0.97 [+0.87, +1.06].

**Matching static LLM + IGSC** on both CometKiwi and structural pass takes 8.3% LLM share with C+B, against
53.4% for the Experiment 3 router (with gate v1: 6.9% against 50.8%).

**Other pairs** (broken or poor per 1,000, 10% LLM share, gate v2):

| Pair | Experiment 3 router | C + B |
|---|---:|---:|
| NLLB + Llama | 115 | 87 |
| Google + GPT-4o-mini | 70 | 68 |
| Google + Llama | 113 | 93 |

**Signal quality:**
- Output features raise the NLLB combined_v2 AUC from 0.771 to 0.823.
- The Spearman correlation with the true gain rises from 0.045 (the Experiment 3 risk score) to 0.173 (GPT-4o-mini).

**Cost and build time:**
- About 4% more per PR than the Experiment 3 router, and 13–16% longer builds, because NMT must translate
  every string first.
- Still about 4.5× cheaper than static LLM + IGSC.

## Limitations
- **vs static LLM + IGSC:** mean CometKiwi is a tie [−0.0001, +0.0012], not a win. chrF prefers static
  GPT-4o-mini, plausibly because its wording is closer to Google's.
- **Google + Llama:** no gain over the plain reactive cascade.
- **Offline replay only.** A blind human A/B evaluation (`spot_check.py`) is the next step.
