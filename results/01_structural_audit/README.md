# 01 · Structural audit of NMT and LLM output

## Question
How often does each system break the structure of a UI string? This covers placeholders, keyboard
accelerators, script, encoding, chatter and untranslated copies.

## Why it was run
UI strings contain code-like tokens: `%1`, `{name}` and `&File`. If a translation drops, alters or invents
one, the application can show broken text or lose a keyboard shortcut. The audit measures these failures for
5 NMT systems and 5 LLMs (each LLM before and after one IGSC self-repair round). It also defines the Stage 2
safety gate used by the router.

## How it was produced
`scripts/run_audit.py` runs the six-invariant verifier, both v1 and v2 (`router/verifier.py`), over all 174
catalogs (about 7 s on a laptop CPU). `scripts/exp1_report.py` aggregates the results into tables, examples
and figures.

The six invariants are:
1. script purity;
2. placeholder parity;
3. accelerator kept;
4. no encoding corruption (U+FFFD);
5. no conversational chatter;
6. no copy of the source (with exemptions for names that should not be translated).

**v1 vs v2.** OPUS KDE4 stores `&Close` tokenized as `& Close`.
- The v1 accelerator rule expects no space, so it detects an accelerator in only **1** of the 10,228 sources,
  although **780** have one.
- v2 changes only that rule (it allows one space, accepts letters in any script, and ignores a literal `& &`).
  It also flags empty outputs.
- v1 is reported as the published baseline, and v2 beside it as a corpus-aware correction.

## Files

| File | Contents |
|---|---|
| `failure_labels.csv.gz` | Every failing (verifier version, catalog, line) with its list of issues |
| `catalog_summary.csv` | Per catalog and verifier version: count per check, defective lines, defect rate (%) |
| `system_defect_rates.csv` / `.md` | Per system: defect rate with v1 and v2, over all 12 languages and over the 6 Indic languages |
| `placeholder_accelerator_subsets.csv` | Failure rate restricted to sources that contain a placeholder (356) or an accelerator (780) |
| `examples.md` | Side-by-side source/output examples for each check |
| `figures/defect_rates_v1_v2.png` | Defect rate per system under v1 and v2 |
| `figures/defect_heatmap_v2.png` | v2 defect rate per system × language |

## Result

**Defect rate (share of translations failing at least one invariant):**

| NMT system | v1 | v2 | LLM (baseline → IGSC-repaired) | v1 | v2 |
|---|---:|---:|---|---:|---:|
| Google | 1.28% | 6.25% | GPT-4o-mini | 0.95% → 0.22% | 2.51% → 1.78% |
| NLLB-600M | 1.87% | 4.15% | Llama-3.1-8B | 2.45% → 0.12% | 8.34% → 6.06% |
| Microsoft | 3.34% | 7.71% | Qwen2.5-7B | 5.11% → 1.54% | 6.03% → 2.49% |
| IndicTrans2 (6 languages) | 3.54% | 7.64% | Mistral-Nemo | 8.87% → 0.25% | 14.09% → 5.55% |
| MADLAD-3B | 3.72% | 9.03% | Gemma-2-9B | 17.83% → 0.81% | 21.91% → 5.02% |

- **IGSC works.** One repair round brings every LLM below 1.6% under v1.
- **The two families fail in opposite ways.** 96% of the 8,987 NMT placeholder errors are omissions. NLLB
  typically writes `% 1` for `%1`, and a one-line normalisation repairs 94–95% of these. LLMs invent
  placeholders on strings that have none: Mistral-Nemo did so 9,820 times, Llama-3.1-8B 1,529 times.
- **Accelerators are lost widely once they are detected (v2).** Share of accelerator sources failing (LLM
  first attempts; IGSC-repaired outputs are within about 1 point): Qwen 12.6%, GPT-4o-mini 20.5%, NLLB 30.2%,
  Gemma 54.1%, Microsoft 57.5%, Google 65.3%, MADLAD 69.3%, Mistral 69.6%, Llama 77.9%. Llama additionally
  writes `&amp;` in 15.4% of cases.
- **v2 changes the ranking.** For example, repaired Llama goes from 0.12% (v1) to 6.06% (v2).
