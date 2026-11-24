# Experiment 6: Stage 2 scores (out-of-fold, grouped 5-fold, Experiment 2 LightGBM settings)

## Does seeing the NMT output help predict failure? (ROC-AUC, higher is better)

| NMT | label | Stage 1 (source only) | + 7 NMT-output features |
|---|---|---:|---:|
| nllb | sem | 0.679 | 0.742 |
| nllb | combined_v1 | 0.727 | 0.787 |
| nllb | combined_v2 | 0.771 | 0.823 |
| google | sem | 0.692 | 0.734 |
| google | combined_v1 | 0.723 | 0.779 |
| google | combined_v2 | 0.843 | 0.875 |

## Does the score rank strings by how much the LLM would improve them? (Spearman with the true gain)

| NMT | LLM | Stage 1 risk score | gain, source only | gain, + output features |
|---|---|---:|---:|---:|
| nllb | gpt4o | 0.045 | 0.115 | 0.173 |
| nllb | llama | 0.075 | 0.198 | 0.243 |
| google | gpt4o | 0.065 | 0.088 | 0.146 |
| google | llama | 0.078 | 0.193 | 0.231 |

Stage 2 cost per string (laptop CPU): features 14.3 µs, prediction 6.7 µs (batch).
