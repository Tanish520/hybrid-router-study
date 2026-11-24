# 00 · Per-string CometKiwi scores

## Question
Both upstream benchmarks computed CometKiwi for every translation but published only one average per catalog.
Can the per-string scores be recovered exactly?

## Why it was needed
Training a risk predictor (Experiment 2) and comparing routing policies (Experiment 3) both need a quality
label for every individual translation. A catalog average cannot tell which strings were poor.

## How it was produced
Nothing is translated again. CometKiwi only reads the English source and the saved translation.

1. `scripts/export_pairs.py` lists the 692,115 distinct (source, translation) pairs among the 1,779,672 catalog rows.
2. `notebooks/cometkiwi_scoring.ipynb` scores them on a Google Colab T4 GPU (about 1.5 h). It uses the upstream
   model and versions: `Unbabel/wmt22-cometkiwi-da`, `unbabel-comet 2.2.7`, `transformers 4.57.6`. Before the long
   run, a parity check scores two complete catalogs and stops if their means differ from the published values.
3. `scripts/join_scores.py` maps every score back to all 1,779,672 rows and compares each of the 174 catalog
   means with its published value.

## Files

| File | Contents |
|---|---|
| `cometkiwi_pair_scores.csv.gz` | `pair_id`, `cometkiwi` for 692,115 distinct pairs. `pair_id` = first 20 hex characters of SHA-1(source + U+001F + translation), see `router/scores.py` |
| `validation_vs_published_means.csv` | One row per catalog: `paradigm`, `system`, `tier`, `lang`, rows scored, our mean, published mean, difference, and `ok` (within ±0.002) |

## Result
- **All 174 catalog means match the published values.**
- NMT catalogs match to within rounding (5+ decimal places). LLM catalogs differ by at most 0.000103, only
  because the LLM benchmark publishes means rounded to 4 decimals.
- Parity example: NLLB German, 0.777473 here against 0.7774728 published.

So the per-string scores are the same numbers the benchmarks averaged, now available string by string.
