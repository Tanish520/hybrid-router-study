# 04 · CI/CD commit replay: cost and build time per pull request

## Question
In a continuous-integration pipeline that translates new UI strings into 12 languages on every pull request
(PR), what does each policy cost, and how long does the build wait?

## Why it was run
Quality per string (Experiment 3) is only half the story. A localization step in CI/CD has a budget and sits
on the build's critical path, where LLM cold starts and limited NMT throughput matter.

## How it was produced
`scripts/replay.py` (about 2 minutes) simulates 1,000 synthetic PRs of 10, 50, 250 and 1,000 changed strings.
Each string is translated into 12 languages. Three start profiles are used: warm, mixed (30% cold) and cold.
Routing uses the real saved outcomes (verifier result, IGSC repair needed, Stage 1 risk). Stage 1 bypass is
set to about 10% of jobs.

Timing and price parameters are in [`config/replay_profiles.json`](../../config/replay_profiles.json). Each
one is labelled:
- **measured:** NLLB throughput and start-up on an AWS m5.large, Llama per-request time from repair telemetry,
  Stage 1 and verifier CPU time, and the GPT-4o-mini list price;
- **assumed**, with a sensitivity range: GPT-4o-mini latency, GPU cold start, GPU price.

The PRs are synthetic, drawn from the corpus, not taken from real repository history.

## Files

| File | Contents |
|---|---|
| `summary.csv` / `summary.md` | Per gate, LLM, PR size, start profile and policy: median, p90 and p99 build time, mean cost per PR, LLM and NMT calls |
| `coldstart_sensitivity.csv` | Llama on a serverless GPU with cold starts of 20, 60 and 180 s |
| `nmt_replicas4.csv` | The same replay with 4 NMT replicas instead of 1 |
| `figures/build_time_cdf_nllb_<llm>_<gate>.png` | Distribution of build time per policy |

## Result (gate v2, mixed start)

| Per PR | Static LLM + IGSC | Reactive cascade |
|---|---:|---:|
| Cost, 50 strings, GPT-4o-mini | $0.0197 | $0.0030 (6.5× less) |
| Cost, 1,000 strings, GPT-4o-mini | $0.402 | $0.064 (6.3× less) |
| Cost, 1,000 strings, Llama on serverless A100 | $0.425 | $0.074 (5.7× less) |
| Median build time, 50 strings, 1 NMT replica | 36 s | 71 s |
| Median build time, 50 strings, 4 NMT replicas | 36 s | 23 s |

- **About six times cheaper.** Only about 4% of jobs reach the LLM.
- **Faster only with NMT scale-out.** A 16-way concurrent LLM API beats a single NMT runner working through
  600 translations; 4 NMT replicas reverse this.
- **Cold starts are rarely avoided.** The share of cascade PRs needing at least one LLM call is 12.5% for 1
  string, 74% for 10 strings and 99.7% for 50 strings. Design implication: take escalations off the critical
  path. Ship the verified NMT output and repair escalations in a follow-up step, or keep one LLM instance warm.
