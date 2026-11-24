# Experiment 6: build time and cost per PR (Experiment 4 model, NLLB, mixed start)

- gate v1, gpt4o: predictive + reactive sends 10.3% of jobs to the LLM; C+B sends 10.4%
- gate v1, llama: predictive + reactive sends 10.3% of jobs to the LLM; C+B sends 10.2%
- gate v2, gpt4o: predictive + reactive sends 10.7% of jobs to the LLM; C+B sends 10.8%
- gate v2, llama: predictive + reactive sends 10.7% of jobs to the LLM; C+B sends 10.8%

C+B is compared with predictive + reactive at the same share of strings sent to the LLM.

## Gate v1, LLM = gpt4o

| PR size | policy | median build s | p90 s | mean cost per PR (USD) | LLM calls | PRs needing the LLM % |
|---:|---|---:|---:|---:|---:|---:|
| 10 | static_llm_igsc | 11.4 | 12.2 | 0.00412 | 121 | 100.0 |
| 10 | reactive_cascade | 18.6 | 48.6 | 0.00090 | 2 | 43.9 |
| 10 | predictive_reactive | 17.4 | 47.3 | 0.00120 | 13 | 92.5 |
| 10 | CB_post_nmt_gain | 19.4 | 49.4 | 0.00124 | 13 | 100.0 |
| 50 | static_llm_igsc | 36.2 | 36.2 | 0.01977 | 606 | 100.0 |
| 50 | reactive_cascade | 70.6 | 100.6 | 0.00265 | 12 | 94.4 |
| 50 | predictive_reactive | 64.8 | 94.5 | 0.00417 | 65 | 100.0 |
| 50 | CB_post_nmt_gain | 73.8 | 103.0 | 0.00435 | 64 | 100.0 |
| 250 | static_llm_igsc | 157.0 | 157.8 | 0.09976 | 3029 | 100.0 |
| 250 | reactive_cascade | 329.6 | 358.8 | 0.01316 | 62 | 100.0 |
| 250 | predictive_reactive | 297.9 | 327.5 | 0.02071 | 324 | 100.0 |
| 250 | CB_post_nmt_gain | 342.5 | 372.5 | 0.02167 | 322 | 100.0 |
| 1000 | static_llm_igsc | 611.4 | 612.2 | 0.40146 | 12114 | 100.0 |
| 1000 | reactive_cascade | 1298.7 | 1327.9 | 0.05436 | 252 | 100.0 |
| 1000 | predictive_reactive | 1164.4 | 1193.9 | 0.08456 | 1301 | 100.0 |
| 1000 | CB_post_nmt_gain | 1351.7 | 1380.9 | 0.08825 | 1290 | 100.0 |

## Gate v1, LLM = llama

| PR size | policy | median build s | p90 s | mean cost per PR (USD) | LLM calls | PRs needing the LLM % |
|---:|---|---:|---:|---:|---:|---:|
| 10 | static_llm_igsc | 9.3 | 69.3 | 0.02378 | 123 | 100.0 |
| 10 | reactive_cascade | 18.3 | 65.5 | 0.00983 | 2 | 43.9 |
| 10 | predictive_reactive | 18.1 | 65.9 | 0.01921 | 13 | 92.5 |
| 10 | CB_post_nmt_gain | 19.2 | 65.9 | 0.02058 | 13 | 100.0 |
| 50 | static_llm_igsc | 23.9 | 83.9 | 0.04250 | 615 | 100.0 |
| 50 | reactive_cascade | 69.9 | 99.9 | 0.02342 | 12 | 94.4 |
| 50 | predictive_reactive | 66.9 | 94.0 | 0.02667 | 64 | 100.0 |
| 50 | CB_post_nmt_gain | 71.8 | 101.3 | 0.02650 | 63 | 100.0 |
| 250 | static_llm_igsc | 96.8 | 156.3 | 0.12282 | 3072 | 100.0 |
| 250 | reactive_cascade | 327.7 | 357.2 | 0.03318 | 59 | 100.0 |
| 250 | predictive_reactive | 297.3 | 326.8 | 0.04132 | 318 | 100.0 |
| 250 | CB_post_nmt_gain | 335.3 | 364.8 | 0.04145 | 314 | 100.0 |
| 1000 | static_llm_igsc | 369.2 | 428.7 | 0.42197 | 12293 | 100.0 |
| 1000 | reactive_cascade | 1292.4 | 1322.4 | 0.06248 | 242 | 100.0 |
| 1000 | predictive_reactive | 1163.2 | 1192.6 | 0.09360 | 1280 | 100.0 |
| 1000 | CB_post_nmt_gain | 1323.3 | 1352.8 | 0.09603 | 1262 | 100.0 |

## Gate v2, LLM = gpt4o

| PR size | policy | median build s | p90 s | mean cost per PR (USD) | LLM calls | PRs needing the LLM % |
|---:|---|---:|---:|---:|---:|---:|
| 10 | static_llm_igsc | 11.4 | 12.2 | 0.00412 | 121 | 100.0 |
| 10 | reactive_cascade | 18.6 | 48.6 | 0.00099 | 5 | 74.3 |
| 10 | predictive_reactive | 17.6 | 47.6 | 0.00122 | 13 | 87.7 |
| 10 | CB_post_nmt_gain | 19.4 | 49.4 | 0.00126 | 14 | 99.9 |
| 50 | static_llm_igsc | 36.2 | 36.2 | 0.01977 | 606 | 100.0 |
| 50 | reactive_cascade | 71.4 | 101.4 | 0.00309 | 26 | 100.0 |
| 50 | predictive_reactive | 65.1 | 94.7 | 0.00425 | 67 | 100.0 |
| 50 | CB_post_nmt_gain | 73.8 | 103.8 | 0.00443 | 67 | 100.0 |
| 250 | static_llm_igsc | 157.0 | 157.8 | 0.09976 | 3029 | 100.0 |
| 250 | reactive_cascade | 332.8 | 362.8 | 0.01539 | 130 | 100.0 |
| 250 | predictive_reactive | 298.7 | 328.1 | 0.02112 | 335 | 100.0 |
| 250 | CB_post_nmt_gain | 343.3 | 373.3 | 0.02210 | 335 | 100.0 |
| 1000 | static_llm_igsc | 611.4 | 612.2 | 0.40146 | 12114 | 100.0 |
| 1000 | reactive_cascade | 1313.1 | 1342.3 | 0.06326 | 526 | 100.0 |
| 1000 | predictive_reactive | 1167.6 | 1195.6 | 0.08621 | 1345 | 100.0 |
| 1000 | CB_post_nmt_gain | 1354.1 | 1383.3 | 0.08995 | 1342 | 100.0 |

## Gate v2, LLM = llama

| PR size | policy | median build s | p90 s | mean cost per PR (USD) | LLM calls | PRs needing the LLM % |
|---:|---|---:|---:|---:|---:|---:|
| 10 | static_llm_igsc | 9.3 | 69.3 | 0.02378 | 123 | 100.0 |
| 10 | reactive_cascade | 18.7 | 65.5 | 0.01622 | 5 | 74.3 |
| 10 | predictive_reactive | 18.3 | 65.9 | 0.01886 | 13 | 87.7 |
| 10 | CB_post_nmt_gain | 19.2 | 65.9 | 0.02061 | 13 | 100.0 |
| 50 | static_llm_igsc | 23.9 | 83.9 | 0.04250 | 615 | 100.0 |
| 50 | reactive_cascade | 70.4 | 100.4 | 0.02513 | 25 | 100.0 |
| 50 | predictive_reactive | 67.2 | 94.1 | 0.02676 | 67 | 100.0 |
| 50 | CB_post_nmt_gain | 71.8 | 101.8 | 0.02663 | 67 | 100.0 |
| 250 | static_llm_igsc | 96.8 | 156.3 | 0.12282 | 3072 | 100.0 |
| 250 | reactive_cascade | 329.6 | 359.6 | 0.03546 | 128 | 100.0 |
| 250 | predictive_reactive | 297.7 | 327.2 | 0.04171 | 330 | 100.0 |
| 250 | CB_post_nmt_gain | 335.8 | 365.8 | 0.04208 | 333 | 100.0 |
| 1000 | static_llm_igsc | 369.2 | 428.7 | 0.42197 | 12293 | 100.0 |
| 1000 | reactive_cascade | 1300.9 | 1330.4 | 0.07151 | 517 | 100.0 |
| 1000 | predictive_reactive | 1164.9 | 1193.3 | 0.09507 | 1327 | 100.0 |
| 1000 | CB_post_nmt_gain | 1325.6 | 1355.2 | 0.09856 | 1339 | 100.0 |

