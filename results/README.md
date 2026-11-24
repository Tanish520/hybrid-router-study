# Results

Every experiment has its own folder. Each folder's README says what question the experiment answers, why
it was run, how to regenerate it, what every file contains, and what the main result is. Figures are kept
inside the folder of the experiment that produced them.

All numbers come from an **offline replay of saved translations**. No translation was regenerated. The data
are 10,228 KDE4 UI strings × 12 target languages = **122,736 string–language pairs**.

| Folder | Question | Headline result | Produced by |
|---|---|---|---|
| [00_cometkiwi_scores](00_cometkiwi_scores/) | Can per-string quality scores be recovered, given that the benchmarks publish only averages? | 1,779,672 translations scored; all 174 catalog means reproduce the published values | `scripts/export_pairs.py` → `notebooks/cometkiwi_scoring.ipynb` → `scripts/join_scores.py` |
| [01_structural_audit](01_structural_audit/) | How often does each NMT system and LLM break placeholders, accelerators, script and format? | NMT drops placeholders, LLMs invent them; the original accelerator check misses tokenized `& X` accelerators | `scripts/run_audit.py`, `scripts/exp1_report.py` |
| [02_risk_predictor](02_risk_predictor/) | Can a cheap model predict NMT failure from the English source alone (Stage 1)? | AUC 0.972 for structural failure, 0.679 for low quality; 12.5 µs per string | `scripts/train_router.py` |
| [03_routing_policies](03_routing_policies/) | Which routing policy gives the best quality for the LLM budget? | Reactive cascade: 99.03% structural pass using the LLM for 4.1% of strings (all-LLM: 98.22%) | `scripts/policies.py` |
| [04_cicd_replay](04_cicd_replay/) | What does each policy cost, and how long does a CI/CD build wait, per pull request? | Cascade is 5.7–6.5× cheaper than all-LLM; faster only when NMT is scaled out | `scripts/replay.py` |
| [05_confidence_intervals](05_confidence_intervals/) | Are the differences in Experiments 2 and 3 real, or sampling noise? | All headline differences have 95% intervals that exclude zero | `scripts/bootstrap.py` |
| [06_improved_router](06_improved_router/) | Can the router be improved by gating LLM output, predicting gain and reading NMT-output signals? | 11.5 fewer broken-or-poor translations per 1,000 at equal LLM use; matches all-LLM quality with 8% instead of 53% LLM use | `improved_router/*.py` |
| [datasets](datasets/) | Shared input for Experiments 2–6 | One row per string–language pair: features, verifier outcomes and CometKiwi for each system | `scripts/build_dataset.py` |

## Terms used throughout

| Term | Meaning |
|---|---|
| Verifier v1 | The original six-invariant structural verifier from the LLM benchmark, unchanged |
| Verifier v2 | The same six invariants with a corpus-aware accelerator rule (tokenized `& X`) and an empty-output flag |
| Structural pass | The translation passes all six invariants |
| Low quality / "sem" | CometKiwi below 0.6 (fixed before any training) |
| Broken or poor ("fail combined") | Structural failure **or** CometKiwi below 0.6 |
| IGSC | Invariant-Guided Self-Correction: one repair round in which the LLM is told which invariant it broke |
| Gate | The verifier used as the Stage 2 safety check (v1 or v2); file names end in `_v1` / `_v2` |
| LLM share / dispatch | Percentage of string–language pairs whose final output comes from, or was sent to, the LLM |
