# 03 · Routing policies

## Question
For a given LLM budget, which routing policy delivers the best translations?

## Why it was run
This is the central experiment. It compares the two-stage router with the obvious alternatives (NMT only,
LLM only, check-then-escalate) on the same strings.

## How it was produced
`scripts/policies.py`. It is an **offline replay**: each policy only *selects* between saved outputs, so all
policies are compared on identical translations at no cost. Pairs: NMT ∈ {NLLB, Google} × LLM ∈ {GPT-4o-mini,
Llama-3.1-8B}. Each pair is run with gate v1 and gate v2.

| Policy | Rule |
|---|---|
| `static_nmt` | NMT for every string |
| `zero_shot_llm` | LLM first attempt for every string |
| `static_llm_igsc` | LLM with IGSC repair for every string |
| `reactive_cascade` | NMT, then the verifier; failures go to the LLM with IGSC |
| `predictive_reactive` | Stage 1 risk ≥ θ goes straight to the LLM; the rest follows the reactive cascade. θ is swept from "bypass nothing" to "bypass everything" |
| `rule_router` | Placeholder or accelerator in the source goes to the LLM; the rest is reactive (baseline) |
| `random_bypass` | Same LLM share as `predictive_reactive`, strings chosen at random (tests whether Stage 1 beats chance) |
| `oracle (upper bound)` | Picks the better saved output per string with hindsight; not deployable |

LLM calls count one generation plus one IGSC repair call when the first attempt failed the verifier.

## Files

| File | Contents |
|---|---|
| `policies_<nmt>_<llm>_<gate>.csv` | Every policy and every θ point: mean CometKiwi, structural pass under v1 and v2, broken-or-poor rate, LLM share, LLM and NMT calls per string |
| `summary.md` | Readable tables for each pair and gate, plus the smallest LLM share at which `predictive_reactive` matches all-LLM |
| `figures/pareto_<nmt>_<llm>_<gate>.png` | Quality (left) and structural pass (right) against the share of strings sent to the LLM |

## Result (NLLB + GPT-4o-mini, gate v2)

| Policy | LLM share | Mean CometKiwi | Structural pass (v2) |
|---|---:|---:|---:|
| Static NMT | 0% | 0.7834 | 95.85% |
| Static LLM + IGSC | 100% | 0.7877 | 98.22% |
| **Reactive cascade** | **4.1%** | 0.7838 | **99.03%** |
| Predictive + reactive, 10% bypass | 10.7% | 0.7847 | 98.69% |
| Predictive + reactive, 30% bypass | 30.2% | 0.7868 | 98.53% |
| Oracle (upper bound) | 37.6% | 0.8071 | 99.03% |

- **The cascade is more reliable than all-LLM** (+0.81 points [0.74, 0.88]) while using the LLM for 4.1% of
  strings. Every string gets two chances: NMT first, then the LLM if a check fails. Mean CometKiwi is slightly
  lower (−0.0038 [−0.0045, −0.0032]).
- **Stage 1 beats random bypass on quality** at every LLM share, but the gains are small.
- **Stage 1 bypass lowers structural pass** by 0.3–0.5 points with GPT-4o-mini and 2–3 points with Llama. In
  the figures the predictive curve falls below the random one on the right panel. The predictor sends exactly
  the strings with placeholders and accelerators, which the LLM handles worst, and those bypassed LLM outputs
  are never checked. Experiment 6 addresses this.
- **The best policy depends on the models paired.** With Llama, or with Google as the NMT, any bypass lowers
  quality and the reactive cascade alone is best.
