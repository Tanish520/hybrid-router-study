# datasets · Router dataset

`router_dataset.csv.gz` is the shared input for Experiments 2–6. It is produced by `scripts/build_dataset.py`
from the upstream catalogs, the verifier and the per-string scores in [00](../00_cometkiwi_scores/).

**One row per string–language pair:** 10,228 strings × 12 languages = 122,736 rows.

| Columns | Meaning |
|---|---|
| `line_idx`, `lang`, `group` | Source line, target language, duplicate group (identical English text shares a group; used for grouped CV and bootstrap) |
| `n_chars` … `script_tier` | The 12 Stage 1 features (`router/features.py`) |
| `<system>_c1`, `<system>_c2` | 1 if the output passes verifier v1 / v2 |
| `<system>_ck` | CometKiwi score of the output |

Systems: `nllb`, `google`, `microsoft`, `madlad` (NMT) and `gpt4o_base`, `gpt4o_igsc`, `llama_base`,
`llama_igsc` (LLM first attempt and after IGSC repair).
