# Experiment 3: routing policies (offline replay of saved outputs)

Risk = out-of-fold LightGBM prediction (combined label matching the gate's verifier). LLM calls include the IGSC repair call when the LLM baseline failed the checks. Each pair is run twice: safety gate = verifier v1 (the original checker, as published) and verifier v2 (corpus-aware accelerator fix).

## Gate v1: NMT = nllb, LLM = GPT-4o-mini

| policy | θ | mean CometKiwi | struct pass v1 % | struct pass v2 % | fail (struct v1 or CK<0.6) % | fail (struct v2 or CK<0.6) % | LLM dispatch % | LLM calls/string | NMT calls/string |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| static_nmt |  | 0.7834 | 98.13 | 95.85 | 9.23 | 11.23 | 0.0 | 0.000 | 1.000 |
| zero_shot_llm |  | 0.7877 | 99.05 | 97.49 | 7.41 | 8.83 | 100.0 | 1.000 | 0.000 |
| static_llm_igsc |  | 0.7877 | 99.78 | 98.22 | 6.89 | 8.30 | 100.0 | 1.010 | 0.000 |
| reactive_cascade |  | 0.7834 | 99.92 | 97.60 | 7.64 | 9.66 | 1.9 | 0.021 | 1.000 |
| rule_router |  | 0.7837 | 99.91 | 98.33 | 7.64 | 9.06 | 11.4 | 0.117 | 0.889 |
| oracle (upper bound) |  | 0.8074 | 99.92 | 98.43 | 3.16 | 4.56 | 37.5 | 0.379 | 1.000 |
| predictive_reactive | 0.160 | 0.7850 | 99.89 | 97.81 | 7.27 | 9.09 | 9.8 | 0.104 | 0.905 |
| predictive_reactive | 0.115 | 0.7860 | 99.88 | 98.05 | 7.07 | 8.69 | 20.2 | 0.208 | 0.800 |
| predictive_reactive | 0.091 | 0.7869 | 99.86 | 98.25 | 6.90 | 8.34 | 30.2 | 0.308 | 0.700 |

Predictive+reactive matches static LLM+IGSC on both mean CometKiwi and structural pass (v1) at θ = 0.063, sending 51.1% of strings to the LLM.

## Gate v1: NMT = nllb, LLM = Llama-3.1-8B

| policy | θ | mean CometKiwi | struct pass v1 % | struct pass v2 % | fail (struct v1 or CK<0.6) % | fail (struct v2 or CK<0.6) % | LLM dispatch % | LLM calls/string | NMT calls/string |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| static_nmt |  | 0.7834 | 98.13 | 95.85 | 9.23 | 11.23 | 0.0 | 0.000 | 1.000 |
| zero_shot_llm |  | 0.7482 | 97.55 | 91.66 | 16.57 | 21.49 | 100.0 | 1.000 | 0.000 |
| static_llm_igsc |  | 0.7531 | 99.88 | 93.94 | 14.60 | 19.55 | 100.0 | 1.025 | 0.000 |
| reactive_cascade |  | 0.7826 | 99.96 | 97.60 | 7.78 | 9.83 | 1.9 | 0.020 | 1.000 |
| rule_router |  | 0.7791 | 99.93 | 93.98 | 8.74 | 13.69 | 11.4 | 0.118 | 0.889 |
| oracle (upper bound) |  | 0.8032 | 99.96 | 96.49 | 3.76 | 7.04 | 33.1 | 0.338 | 1.000 |
| predictive_reactive | 0.160 | 0.7814 | 99.94 | 97.39 | 8.20 | 10.39 | 9.8 | 0.102 | 0.905 |
| predictive_reactive | 0.115 | 0.7799 | 99.93 | 97.07 | 8.60 | 11.04 | 20.2 | 0.208 | 0.800 |
| predictive_reactive | 0.091 | 0.7776 | 99.93 | 96.76 | 9.17 | 11.85 | 30.2 | 0.309 | 0.700 |

Predictive+reactive matches static LLM+IGSC on both mean CometKiwi and structural pass (v1) at θ = 1.000, sending 1.9% of strings to the LLM.

## Gate v1: NMT = google, LLM = GPT-4o-mini

| policy | θ | mean CometKiwi | struct pass v1 % | struct pass v2 % | fail (struct v1 or CK<0.6) % | fail (struct v2 or CK<0.6) % | LLM dispatch % | LLM calls/string | NMT calls/string |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| static_nmt |  | 0.7939 | 98.72 | 93.75 | 6.44 | 10.99 | 0.0 | 0.000 | 1.000 |
| zero_shot_llm |  | 0.7877 | 99.05 | 97.49 | 7.41 | 8.83 | 100.0 | 1.000 | 0.000 |
| static_llm_igsc |  | 0.7877 | 99.78 | 98.22 | 6.89 | 8.30 | 100.0 | 1.010 | 0.000 |
| reactive_cascade |  | 0.7939 | 99.86 | 94.88 | 5.51 | 10.06 | 1.3 | 0.016 | 1.000 |
| rule_router |  | 0.7946 | 99.84 | 98.28 | 5.55 | 6.97 | 11.7 | 0.121 | 0.889 |
| oracle (upper bound) |  | 0.8070 | 99.86 | 97.34 | 3.38 | 5.76 | 25.5 | 0.259 | 1.000 |
| predictive_reactive | 0.129 | 0.7935 | 99.83 | 95.45 | 5.66 | 9.71 | 10.0 | 0.106 | 0.905 |
| predictive_reactive | 0.086 | 0.7932 | 99.82 | 95.94 | 5.77 | 9.38 | 19.9 | 0.206 | 0.805 |
| predictive_reactive | 0.067 | 0.7927 | 99.82 | 96.35 | 5.88 | 9.11 | 29.8 | 0.306 | 0.704 |

Predictive+reactive matches static LLM+IGSC on both mean CometKiwi and structural pass (v1) at θ = 1.000, sending 1.3% of strings to the LLM.

## Gate v1: NMT = google, LLM = Llama-3.1-8B

| policy | θ | mean CometKiwi | struct pass v1 % | struct pass v2 % | fail (struct v1 or CK<0.6) % | fail (struct v2 or CK<0.6) % | LLM dispatch % | LLM calls/string | NMT calls/string |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| static_nmt |  | 0.7939 | 98.72 | 93.75 | 6.44 | 10.99 | 0.0 | 0.000 | 1.000 |
| zero_shot_llm |  | 0.7482 | 97.55 | 91.66 | 16.57 | 21.49 | 100.0 | 1.000 | 0.000 |
| static_llm_igsc |  | 0.7531 | 99.88 | 93.94 | 14.60 | 19.55 | 100.0 | 1.025 | 0.000 |
| reactive_cascade |  | 0.7934 | 99.94 | 94.93 | 5.56 | 10.13 | 1.3 | 0.014 | 1.000 |
| rule_router |  | 0.7899 | 99.91 | 93.97 | 6.64 | 11.59 | 11.7 | 0.121 | 0.889 |
| oracle (upper bound) |  | 0.8063 | 99.94 | 94.87 | 3.30 | 8.15 | 25.4 | 0.259 | 1.000 |
| predictive_reactive | 0.129 | 0.7903 | 99.92 | 94.72 | 6.46 | 11.13 | 10.0 | 0.103 | 0.905 |
| predictive_reactive | 0.086 | 0.7871 | 99.91 | 94.64 | 7.30 | 11.98 | 19.9 | 0.204 | 0.805 |
| predictive_reactive | 0.067 | 0.7837 | 99.90 | 94.56 | 8.15 | 12.85 | 29.8 | 0.305 | 0.704 |

Predictive+reactive matches static LLM+IGSC on both mean CometKiwi and structural pass (v1) at θ = 1.000, sending 1.3% of strings to the LLM.

## Gate v2: NMT = nllb, LLM = GPT-4o-mini

| policy | θ | mean CometKiwi | struct pass v1 % | struct pass v2 % | fail (struct v1 or CK<0.6) % | fail (struct v2 or CK<0.6) % | LLM dispatch % | LLM calls/string | NMT calls/string |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| static_nmt |  | 0.7834 | 98.13 | 95.85 | 9.23 | 11.23 | 0.0 | 0.000 | 1.000 |
| zero_shot_llm |  | 0.7877 | 99.05 | 97.49 | 7.41 | 8.83 | 100.0 | 1.000 | 0.000 |
| static_llm_igsc |  | 0.7877 | 99.78 | 98.22 | 6.89 | 8.30 | 100.0 | 1.010 | 0.000 |
| reactive_cascade |  | 0.7838 | 99.91 | 99.03 | 7.55 | 8.35 | 4.1 | 0.044 | 1.000 |
| rule_router |  | 0.7837 | 99.91 | 98.34 | 7.63 | 9.05 | 11.4 | 0.117 | 0.889 |
| oracle (upper bound) |  | 0.8071 | 99.89 | 99.03 | 3.25 | 4.07 | 37.6 | 0.380 | 1.000 |
| predictive_reactive | 0.271 | 0.7846 | 99.89 | 98.74 | 7.39 | 8.42 | 9.9 | 0.103 | 0.910 |
| predictive_reactive | 0.132 | 0.7857 | 99.87 | 98.59 | 7.17 | 8.33 | 19.8 | 0.204 | 0.805 |
| predictive_reactive | 0.101 | 0.7868 | 99.86 | 98.53 | 6.92 | 8.13 | 30.2 | 0.309 | 0.700 |

Predictive+reactive matches static LLM+IGSC on both mean CometKiwi and structural pass (v2) at θ = 0.064, sending 53.7% of strings to the LLM.

## Gate v2: NMT = nllb, LLM = Llama-3.1-8B

| policy | θ | mean CometKiwi | struct pass v1 % | struct pass v2 % | fail (struct v1 or CK<0.6) % | fail (struct v2 or CK<0.6) % | LLM dispatch % | LLM calls/string | NMT calls/string |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| static_nmt |  | 0.7834 | 98.13 | 95.85 | 9.23 | 11.23 | 0.0 | 0.000 | 1.000 |
| zero_shot_llm |  | 0.7482 | 97.55 | 91.66 | 16.57 | 21.49 | 100.0 | 1.000 | 0.000 |
| static_llm_igsc |  | 0.7531 | 99.88 | 93.94 | 14.60 | 19.55 | 100.0 | 1.025 | 0.000 |
| reactive_cascade |  | 0.7822 | 99.96 | 98.18 | 7.90 | 9.39 | 4.1 | 0.043 | 1.000 |
| rule_router |  | 0.7791 | 99.93 | 93.99 | 8.73 | 13.68 | 11.4 | 0.118 | 0.889 |
| oracle (upper bound) |  | 0.8022 | 99.91 | 98.18 | 3.97 | 5.62 | 31.9 | 0.327 | 1.000 |
| predictive_reactive | 0.271 | 0.7806 | 99.94 | 96.29 | 8.36 | 11.39 | 9.9 | 0.102 | 0.910 |
| predictive_reactive | 0.132 | 0.7788 | 99.93 | 95.35 | 8.86 | 12.65 | 19.8 | 0.203 | 0.805 |
| predictive_reactive | 0.101 | 0.7771 | 99.93 | 95.01 | 9.29 | 13.37 | 30.2 | 0.310 | 0.700 |

Predictive+reactive matches static LLM+IGSC on both mean CometKiwi and structural pass (v2) at θ = 1.000, sending 4.1% of strings to the LLM.

## Gate v2: NMT = google, LLM = GPT-4o-mini

| policy | θ | mean CometKiwi | struct pass v1 % | struct pass v2 % | fail (struct v1 or CK<0.6) % | fail (struct v2 or CK<0.6) % | LLM dispatch % | LLM calls/string | NMT calls/string |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| static_nmt |  | 0.7939 | 98.72 | 93.75 | 6.44 | 10.99 | 0.0 | 0.000 | 1.000 |
| zero_shot_llm |  | 0.7877 | 99.05 | 97.49 | 7.41 | 8.83 | 100.0 | 1.000 | 0.000 |
| static_llm_igsc |  | 0.7877 | 99.78 | 98.22 | 6.89 | 8.30 | 100.0 | 1.010 | 0.000 |
| reactive_cascade |  | 0.7949 | 99.85 | 98.35 | 5.48 | 6.84 | 6.3 | 0.066 | 1.000 |
| rule_router |  | 0.7946 | 99.84 | 98.28 | 5.55 | 6.97 | 11.7 | 0.121 | 0.889 |
| oracle (upper bound) |  | 0.8064 | 99.85 | 98.35 | 3.51 | 4.94 | 26.5 | 0.269 | 1.000 |
| predictive_reactive | 0.289 | 0.7945 | 99.84 | 98.28 | 5.57 | 6.99 | 9.9 | 0.104 | 0.910 |
| predictive_reactive | 0.105 | 0.7941 | 99.82 | 98.26 | 5.71 | 7.13 | 19.9 | 0.206 | 0.805 |
| predictive_reactive | 0.075 | 0.7935 | 99.81 | 98.25 | 5.85 | 7.27 | 29.8 | 0.305 | 0.705 |

Predictive+reactive matches static LLM+IGSC on both mean CometKiwi and structural pass (v2) at θ = 1.000, sending 6.3% of strings to the LLM.

## Gate v2: NMT = google, LLM = Llama-3.1-8B

| policy | θ | mean CometKiwi | struct pass v1 % | struct pass v2 % | fail (struct v1 or CK<0.6) % | fail (struct v2 or CK<0.6) % | LLM dispatch % | LLM calls/string | NMT calls/string |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| static_nmt |  | 0.7939 | 98.72 | 93.75 | 6.44 | 10.99 | 0.0 | 0.000 | 1.000 |
| zero_shot_llm |  | 0.7482 | 97.55 | 91.66 | 16.57 | 21.49 | 100.0 | 1.000 | 0.000 |
| static_llm_igsc |  | 0.7531 | 99.88 | 93.94 | 14.60 | 19.55 | 100.0 | 1.025 | 0.000 |
| reactive_cascade |  | 0.7926 | 99.93 | 95.96 | 6.00 | 9.34 | 6.3 | 0.064 | 1.000 |
| rule_router |  | 0.7899 | 99.91 | 93.97 | 6.64 | 11.59 | 11.7 | 0.121 | 0.889 |
| oracle (upper bound) |  | 0.8055 | 99.92 | 95.96 | 3.53 | 7.34 | 25.4 | 0.260 | 1.000 |
| predictive_reactive | 0.289 | 0.7907 | 99.91 | 94.17 | 6.51 | 11.30 | 9.9 | 0.101 | 0.910 |
| predictive_reactive | 0.105 | 0.7874 | 99.90 | 93.96 | 7.35 | 12.30 | 19.9 | 0.203 | 0.805 |
| predictive_reactive | 0.075 | 0.7838 | 99.89 | 93.96 | 8.25 | 13.20 | 29.8 | 0.304 | 0.705 |

Predictive+reactive matches static LLM+IGSC on both mean CometKiwi and structural pass (v2) at θ = 1.000, sending 6.3% of strings to the LLM.

