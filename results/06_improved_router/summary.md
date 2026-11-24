# Experiment 6: improved router

Offline replay of the saved outputs, as in Experiment 3. Every policy is compared at the same share of strings sent to the LLM. CIs: grouped bootstrap, 1,000 resamples of English source groups.

## Safety gate v1

### NMT = nllb, LLM = GPT-4o-mini

| target LLM share | policy | actual LLM share % | mean CometKiwi | struct pass % | broken or bad meaning % | NMT calls/string | chrF vs Google |
|---:|---|---:|---:|---:|---:|---:|---:|
| 10% | predictive + reactive (Exp. 3) | 10.0 | 0.7850 | 99.89 | 7.27 | 0.902 | 64.59 |
| 20% | predictive + reactive (Exp. 3) | 20.0 | 0.7860 | 99.88 | 7.08 | 0.802 | 65.86 |
| 30% | predictive + reactive (Exp. 3) | 29.9 | 0.7869 | 99.86 | 6.90 | 0.702 | 66.88 |
| 10% | A: + check LLM output | 10.0 | 0.7850 | 99.92 | 7.25 | 0.903 | 64.58 |
| 20% | A: + check LLM output | 20.0 | 0.7860 | 99.92 | 7.06 | 0.803 | 65.85 |
| 30% | A: + check LLM output | 29.9 | 0.7868 | 99.92 | 6.87 | 0.703 | 66.86 |
| 10% | B: A + rank by predicted gain | 9.9 | 0.7865 | 99.92 | 6.96 | 0.918 | 65.09 |
| 20% | B: A + rank by predicted gain | 20.0 | 0.7881 | 99.92 | 6.60 | 0.815 | 66.65 |
| 30% | B: A + rank by predicted gain | 30.0 | 0.7891 | 99.92 | 6.40 | 0.713 | 67.89 |
| 10% | C: Stage 2 risk with NMT-output clues | 10.1 | 0.7873 | 99.92 | 6.66 | 1.000 | 65.17 |
| 20% | C: Stage 2 risk with NMT-output clues | 20.0 | 0.7883 | 99.92 | 6.47 | 1.000 | 66.23 |
| 30% | C: Stage 2 risk with NMT-output clues | 30.0 | 0.7886 | 99.92 | 6.41 | 1.000 | 67.11 |
| 10% | C+B: Stage 2 gain with NMT-output clues | 10.1 | 0.7886 | 99.92 | 6.43 | 1.000 | 65.55 |
| 20% | C+B: Stage 2 gain with NMT-output clues | 19.9 | 0.7900 | 99.92 | 6.18 | 1.000 | 66.79 |
| 30% | C+B: Stage 2 gain with NMT-output clues | 30.0 | 0.7908 | 99.92 | 6.09 | 1.000 | 67.87 |

| comparison | Δ CometKiwi | Δ struct pass (points) | Δ broken or bad (points) | Δ chrF vs Google |
|---|---|---|---|---|
| C+B vs predictive+reactive @ 10% | +0.0036 [+0.0033, +0.0039] | +0.03 [+0.02, +0.05] | -0.84 [-0.93, -0.74] | +0.96 [+0.86, +1.05] |
| C+B vs predictive+reactive @ 20% | +0.0040 [+0.0037, +0.0044] | +0.04 [+0.03, +0.06] | -0.90 [-1.01, -0.79] | +0.92 [+0.78, +1.06] |
| C+B vs predictive+reactive @ 30% | +0.0039 [+0.0035, +0.0043] | +0.05 [+0.04, +0.07] | -0.81 [-0.93, -0.70] | +1.00 [+0.84, +1.14] |
| C+B @ 10% vs static LLM+IGSC | +0.0010 [+0.0003, +0.0015] | +0.14 [+0.11, +0.17] | -0.46 [-0.62, -0.27] | -3.01 [-3.26, -2.74] |
| C+B @ 10% vs reactive cascade | +0.0052 [+0.0049, +0.0055] | +0.00 [+0.00, +0.00] | -1.21 [-1.30, -1.12] | +1.86 [+1.77, +1.94] |

Smallest LLM share at which each policy matches static LLM+IGSC on both mean CometKiwi and structural pass: predictive + reactive (Exp. 3): 50.8%; A: + check LLM output: 57.1%; B: A + rank by predicted gain: 16.8%; C: Stage 2 risk with NMT-output clues: 12.8%; C+B: Stage 2 gain with NMT-output clues: 6.9%

### NMT = nllb, LLM = Llama-3.1-8B

| target LLM share | policy | actual LLM share % | mean CometKiwi | struct pass % | broken or bad meaning % | NMT calls/string | chrF vs Google |
|---:|---|---:|---:|---:|---:|---:|---:|
| 10% | predictive + reactive (Exp. 3) | 10.0 | 0.7813 | 99.94 | 8.22 | 0.902 | 63.02 |
| 20% | predictive + reactive (Exp. 3) | 20.0 | 0.7800 | 99.93 | 8.58 | 0.802 | 62.89 |
| 30% | predictive + reactive (Exp. 3) | 29.9 | 0.7777 | 99.93 | 9.16 | 0.702 | 62.31 |
| 10% | A: + check LLM output | 10.0 | 0.7813 | 99.96 | 8.20 | 0.903 | 63.03 |
| 20% | A: + check LLM output | 20.0 | 0.7800 | 99.96 | 8.56 | 0.802 | 62.90 |
| 30% | A: + check LLM output | 29.9 | 0.7777 | 99.96 | 9.13 | 0.703 | 62.32 |
| 10% | B: A + rank by predicted gain | 9.9 | 0.7838 | 99.96 | 7.49 | 0.918 | 63.97 |
| 20% | B: A + rank by predicted gain | 20.0 | 0.7838 | 99.96 | 7.50 | 0.815 | 64.64 |
| 30% | B: A + rank by predicted gain | 30.0 | 0.7829 | 99.96 | 7.72 | 0.713 | 64.74 |
| 10% | C: Stage 2 risk with NMT-output clues | 10.1 | 0.7842 | 99.96 | 7.43 | 1.000 | 63.63 |
| 20% | C: Stage 2 risk with NMT-output clues | 20.0 | 0.7820 | 99.96 | 8.10 | 1.000 | 63.11 |
| 30% | C: Stage 2 risk with NMT-output clues | 30.0 | 0.7794 | 99.96 | 8.71 | 1.000 | 62.46 |
| 10% | C+B: Stage 2 gain with NMT-output clues | 9.9 | 0.7859 | 99.96 | 7.00 | 1.000 | 64.25 |
| 20% | C+B: Stage 2 gain with NMT-output clues | 20.0 | 0.7855 | 99.96 | 7.09 | 1.000 | 64.78 |
| 30% | C+B: Stage 2 gain with NMT-output clues | 30.1 | 0.7843 | 99.96 | 7.41 | 1.000 | 64.75 |

| comparison | Δ CometKiwi | Δ struct pass (points) | Δ broken or bad (points) | Δ chrF vs Google |
|---|---|---|---|---|
| C+B vs predictive+reactive @ 10% | +0.0046 [+0.0042, +0.0050] | +0.02 [+0.01, +0.04] | -1.23 [-1.34, -1.11] | +1.23 [+1.13, +1.33] |
| C+B vs predictive+reactive @ 20% | +0.0056 [+0.0051, +0.0060] | +0.03 [+0.01, +0.05] | -1.49 [-1.63, -1.34] | +1.89 [+1.74, +2.02] |
| C+B vs predictive+reactive @ 30% | +0.0066 [+0.0061, +0.0071] | +0.03 [+0.02, +0.06] | -1.75 [-1.91, -1.59] | +2.44 [+2.29, +2.59] |
| C+B @ 10% vs static LLM+IGSC | +0.0328 [+0.0319, +0.0338] | +0.08 [+0.05, +0.11] | -7.60 [-7.89, -7.33] | +11.63 [+11.35, +11.90] |
| C+B @ 10% vs reactive cascade | +0.0033 [+0.0030, +0.0036] | +0.00 [+0.00, +0.00] | -0.79 [-0.88, -0.70] | +0.86 [+0.78, +0.94] |

Smallest LLM share at which each policy matches static LLM+IGSC on both mean CometKiwi and structural pass: predictive + reactive (Exp. 3): 1.9%; A: + check LLM output: 1.9%; B: A + rank by predicted gain: 1.9%; C: Stage 2 risk with NMT-output clues: 1.9%; C+B: Stage 2 gain with NMT-output clues: 1.9%

### NMT = google, LLM = GPT-4o-mini

| target LLM share | policy | actual LLM share % | mean CometKiwi | struct pass % | broken or bad meaning % | NMT calls/string |
|---:|---|---:|---:|---:|---:|---:|
| 10% | predictive + reactive (Exp. 3) | 10.0 | 0.7935 | 99.83 | 5.66 | 0.905 |
| 20% | predictive + reactive (Exp. 3) | 20.1 | 0.7932 | 99.82 | 5.77 | 0.802 |
| 30% | predictive + reactive (Exp. 3) | 30.0 | 0.7927 | 99.82 | 5.88 | 0.702 |
| 10% | A: + check LLM output | 10.0 | 0.7935 | 99.86 | 5.64 | 0.906 |
| 20% | A: + check LLM output | 20.1 | 0.7932 | 99.86 | 5.74 | 0.803 |
| 30% | A: + check LLM output | 30.0 | 0.7927 | 99.86 | 5.85 | 0.704 |
| 10% | B: A + rank by predicted gain | 10.1 | 0.7946 | 99.86 | 5.53 | 0.910 |
| 20% | B: A + rank by predicted gain | 19.9 | 0.7946 | 99.86 | 5.57 | 0.810 |
| 30% | B: A + rank by predicted gain | 30.1 | 0.7942 | 99.86 | 5.64 | 0.708 |
| 10% | C: Stage 2 risk with NMT-output clues | 10.1 | 0.7943 | 99.86 | 5.47 | 1.000 |
| 20% | C: Stage 2 risk with NMT-output clues | 20.0 | 0.7940 | 99.86 | 5.58 | 1.000 |
| 30% | C: Stage 2 risk with NMT-output clues | 30.0 | 0.7936 | 99.86 | 5.70 | 1.000 |
| 10% | C+B: Stage 2 gain with NMT-output clues | 9.9 | 0.7953 | 99.86 | 5.35 | 1.000 |
| 20% | C+B: Stage 2 gain with NMT-output clues | 20.1 | 0.7956 | 99.86 | 5.32 | 1.000 |
| 30% | C+B: Stage 2 gain with NMT-output clues | 29.9 | 0.7955 | 99.86 | 5.35 | 1.000 |

| comparison | Δ CometKiwi | Δ struct pass (points) | Δ broken or bad (points) |
|---|---|---|---|
| C+B vs predictive+reactive @ 10% | +0.0018 [+0.0016, +0.0021] | +0.03 [+0.02, +0.04] | -0.31 [-0.39, -0.24] |
| C+B vs predictive+reactive @ 20% | +0.0024 [+0.0022, +0.0027] | +0.04 [+0.03, +0.06] | -0.45 [-0.54, -0.37] |
| C+B vs predictive+reactive @ 30% | +0.0027 [+0.0024, +0.0030] | +0.04 [+0.03, +0.06] | -0.53 [-0.63, -0.44] |
| C+B @ 10% vs static LLM+IGSC | +0.0077 [+0.0072, +0.0082] | +0.08 [+0.06, +0.10] | -1.53 [-1.68, -1.38] |
| C+B @ 10% vs reactive cascade | +0.0015 [+0.0013, +0.0016] | +0.00 [+0.00, +0.00] | -0.16 [-0.22, -0.10] |

Smallest LLM share at which each policy matches static LLM+IGSC on both mean CometKiwi and structural pass: predictive + reactive (Exp. 3): 1.3%; A: + check LLM output: 1.3%; B: A + rank by predicted gain: 1.3%; C: Stage 2 risk with NMT-output clues: 1.3%; C+B: Stage 2 gain with NMT-output clues: 1.3%

### NMT = google, LLM = Llama-3.1-8B

| target LLM share | policy | actual LLM share % | mean CometKiwi | struct pass % | broken or bad meaning % | NMT calls/string |
|---:|---|---:|---:|---:|---:|---:|
| 10% | predictive + reactive (Exp. 3) | 10.0 | 0.7903 | 99.92 | 6.46 | 0.905 |
| 20% | predictive + reactive (Exp. 3) | 20.1 | 0.7870 | 99.91 | 7.32 | 0.802 |
| 30% | predictive + reactive (Exp. 3) | 30.0 | 0.7836 | 99.90 | 8.16 | 0.702 |
| 10% | A: + check LLM output | 10.0 | 0.7903 | 99.94 | 6.44 | 0.905 |
| 20% | A: + check LLM output | 20.1 | 0.7870 | 99.94 | 7.30 | 0.803 |
| 30% | A: + check LLM output | 30.0 | 0.7836 | 99.94 | 8.13 | 0.703 |
| 10% | B: A + rank by predicted gain | 9.9 | 0.7928 | 99.94 | 5.81 | 0.913 |
| 20% | B: A + rank by predicted gain | 20.0 | 0.7909 | 99.94 | 6.25 | 0.810 |
| 30% | B: A + rank by predicted gain | 30.0 | 0.7886 | 99.94 | 6.76 | 0.710 |
| 10% | C: Stage 2 risk with NMT-output clues | 10.1 | 0.7915 | 99.94 | 6.14 | 1.000 |
| 20% | C: Stage 2 risk with NMT-output clues | 20.0 | 0.7882 | 99.94 | 7.07 | 1.000 |
| 30% | C: Stage 2 risk with NMT-output clues | 30.0 | 0.7848 | 99.94 | 7.92 | 1.000 |
| 10% | C+B: Stage 2 gain with NMT-output clues | 10.1 | 0.7933 | 99.94 | 5.69 | 1.000 |
| 20% | C+B: Stage 2 gain with NMT-output clues | 20.0 | 0.7916 | 99.94 | 6.09 | 1.000 |
| 30% | C+B: Stage 2 gain with NMT-output clues | 29.9 | 0.7894 | 99.94 | 6.59 | 1.000 |

| comparison | Δ CometKiwi | Δ struct pass (points) | Δ broken or bad (points) |
|---|---|---|---|
| C+B vs predictive+reactive @ 10% | +0.0030 [+0.0026, +0.0033] | +0.02 [+0.01, +0.04] | -0.77 [-0.88, -0.66] |
| C+B vs predictive+reactive @ 20% | +0.0046 [+0.0042, +0.0051] | +0.03 [+0.01, +0.05] | -1.24 [-1.39, -1.10] |
| C+B vs predictive+reactive @ 30% | +0.0058 [+0.0052, +0.0063] | +0.04 [+0.02, +0.06] | -1.57 [-1.74, -1.40] |
| C+B @ 10% vs static LLM+IGSC | +0.0402 [+0.0393, +0.0412] | +0.06 [+0.04, +0.08] | -8.91 [-9.20, -8.64] |
| C+B @ 10% vs reactive cascade | -0.0001 [-0.0003, +0.0001] | +0.00 [+0.00, +0.00] | +0.13 [+0.07, +0.19] |

Smallest LLM share at which each policy matches static LLM+IGSC on both mean CometKiwi and structural pass: predictive + reactive (Exp. 3): 1.3%; A: + check LLM output: 1.3%; B: A + rank by predicted gain: 1.3%; C: Stage 2 risk with NMT-output clues: 1.3%; C+B: Stage 2 gain with NMT-output clues: 1.3%

## Safety gate v2

### NMT = nllb, LLM = GPT-4o-mini

| target LLM share | policy | actual LLM share % | mean CometKiwi | struct pass % | broken or bad meaning % | NMT calls/string | chrF vs Google |
|---:|---|---:|---:|---:|---:|---:|---:|
| 10% | predictive + reactive (Exp. 3) | 10.1 | 0.7846 | 98.73 | 8.44 | 0.907 | 64.35 |
| 20% | predictive + reactive (Exp. 3) | 20.1 | 0.7857 | 98.59 | 8.32 | 0.802 | 65.59 |
| 30% | predictive + reactive (Exp. 3) | 30.0 | 0.7867 | 98.54 | 8.13 | 0.702 | 66.68 |
| 10% | A: + check LLM output | 10.1 | 0.7846 | 99.03 | 8.17 | 0.918 | 64.30 |
| 20% | A: + check LLM output | 20.1 | 0.7858 | 99.03 | 7.93 | 0.816 | 65.52 |
| 30% | A: + check LLM output | 30.0 | 0.7868 | 99.03 | 7.69 | 0.716 | 66.60 |
| 10% | B: A + rank by predicted gain | 10.0 | 0.7863 | 99.03 | 7.79 | 0.936 | 64.90 |
| 20% | B: A + rank by predicted gain | 20.1 | 0.7880 | 99.03 | 7.43 | 0.830 | 66.47 |
| 30% | B: A + rank by predicted gain | 30.1 | 0.7890 | 99.03 | 7.22 | 0.727 | 67.76 |
| 10% | C: Stage 2 risk with NMT-output clues | 9.9 | 0.7869 | 99.03 | 7.61 | 1.000 | 64.77 |
| 20% | C: Stage 2 risk with NMT-output clues | 20.1 | 0.7882 | 99.03 | 7.32 | 1.000 | 65.99 |
| 30% | C: Stage 2 risk with NMT-output clues | 30.0 | 0.7886 | 99.03 | 7.25 | 1.000 | 66.89 |
| 10% | C+B: Stage 2 gain with NMT-output clues | 9.9 | 0.7882 | 99.03 | 7.29 | 1.000 | 65.32 |
| 20% | C+B: Stage 2 gain with NMT-output clues | 20.0 | 0.7900 | 99.03 | 7.01 | 1.000 | 66.63 |
| 30% | C+B: Stage 2 gain with NMT-output clues | 30.1 | 0.7907 | 99.03 | 6.91 | 1.000 | 67.78 |

| comparison | Δ CometKiwi | Δ struct pass (points) | Δ broken or bad (points) | Δ chrF vs Google |
|---|---|---|---|---|
| C+B vs predictive+reactive @ 10% | +0.0037 [+0.0034, +0.0039] | +0.30 [+0.26, +0.34] | -1.15 [-1.25, -1.06] | +0.97 [+0.87, +1.06] |
| C+B vs predictive+reactive @ 20% | +0.0042 [+0.0039, +0.0046] | +0.44 [+0.38, +0.49] | -1.31 [-1.43, -1.20] | +1.04 [+0.91, +1.17] |
| C+B vs predictive+reactive @ 30% | +0.0040 [+0.0036, +0.0043] | +0.49 [+0.44, +0.54] | -1.22 [-1.35, -1.10] | +1.10 [+0.94, +1.24] |
| C+B @ 10% vs static LLM+IGSC | +0.0006 [-0.0001, +0.0012] | +0.81 [+0.74, +0.88] | -1.02 [-1.19, -0.82] | -3.24 [-3.49, -2.97] |
| C+B @ 10% vs reactive cascade | +0.0044 [+0.0041, +0.0047] | +0.00 [+0.00, +0.00] | -1.06 [-1.14, -0.97] | +1.49 [+1.41, +1.56] |

Smallest LLM share at which each policy matches static LLM+IGSC on both mean CometKiwi and structural pass: predictive + reactive (Exp. 3): 53.4%; A: + check LLM output: 43.4%; B: A + rank by predicted gain: 17.9%; C: Stage 2 risk with NMT-output clues: 14.4%; C+B: Stage 2 gain with NMT-output clues: 8.3%

### NMT = nllb, LLM = Llama-3.1-8B

| target LLM share | policy | actual LLM share % | mean CometKiwi | struct pass % | broken or bad meaning % | NMT calls/string | chrF vs Google |
|---:|---|---:|---:|---:|---:|---:|---:|
| 10% | predictive + reactive (Exp. 3) | 10.1 | 0.7805 | 96.22 | 11.47 | 0.907 | 62.70 |
| 20% | predictive + reactive (Exp. 3) | 20.1 | 0.7788 | 95.35 | 12.66 | 0.802 | 62.42 |
| 30% | predictive + reactive (Exp. 3) | 30.0 | 0.7772 | 95.02 | 13.35 | 0.702 | 62.01 |
| 10% | A: + check LLM output | 10.1 | 0.7813 | 98.18 | 9.66 | 0.942 | 62.97 |
| 20% | A: + check LLM output | 20.1 | 0.7800 | 98.18 | 10.03 | 0.848 | 62.85 |
| 30% | A: + check LLM output | 30.0 | 0.7786 | 98.18 | 10.40 | 0.751 | 62.51 |
| 10% | B: A + rank by predicted gain | 10.0 | 0.7833 | 98.18 | 9.14 | 0.939 | 63.61 |
| 20% | B: A + rank by predicted gain | 20.0 | 0.7834 | 98.18 | 9.10 | 0.839 | 64.27 |
| 30% | B: A + rank by predicted gain | 30.1 | 0.7827 | 98.18 | 9.26 | 0.738 | 64.61 |
| 10% | C: Stage 2 risk with NMT-output clues | 9.9 | 0.7838 | 98.18 | 8.99 | 1.000 | 63.44 |
| 20% | C: Stage 2 risk with NMT-output clues | 20.1 | 0.7824 | 98.18 | 9.45 | 1.000 | 63.19 |
| 30% | C: Stage 2 risk with NMT-output clues | 30.0 | 0.7801 | 98.18 | 10.04 | 1.000 | 62.62 |
| 10% | C+B: Stage 2 gain with NMT-output clues | 9.9 | 0.7852 | 98.18 | 8.66 | 1.000 | 63.92 |
| 20% | C+B: Stage 2 gain with NMT-output clues | 19.9 | 0.7852 | 98.18 | 8.66 | 1.000 | 64.50 |
| 30% | C+B: Stage 2 gain with NMT-output clues | 30.0 | 0.7840 | 98.18 | 8.95 | 1.000 | 64.62 |

| comparison | Δ CometKiwi | Δ struct pass (points) | Δ broken or bad (points) | Δ chrF vs Google |
|---|---|---|---|---|
| C+B vs predictive+reactive @ 10% | +0.0047 [+0.0044, +0.0050] | +1.96 [+1.76, +2.16] | -2.80 [-3.03, -2.59] | +1.22 [+1.13, +1.32] |
| C+B vs predictive+reactive @ 20% | +0.0064 [+0.0059, +0.0068] | +2.83 [+2.58, +3.05] | -4.00 [-4.26, -3.71] | +2.07 [+1.94, +2.20] |
| C+B vs predictive+reactive @ 30% | +0.0069 [+0.0064, +0.0073] | +3.16 [+2.90, +3.41] | -4.40 [-4.69, -4.12] | +2.61 [+2.45, +2.76] |
| C+B @ 10% vs static LLM+IGSC | +0.0321 [+0.0312, +0.0330] | +4.24 [+3.95, +4.54] | -10.89 [-11.31, -10.51] | +11.30 [+11.01, +11.56] |
| C+B @ 10% vs reactive cascade | +0.0030 [+0.0028, +0.0033] | +0.00 [+0.00, +0.00] | -0.73 [-0.81, -0.64] | +0.74 [+0.67, +0.80] |

Smallest LLM share at which each policy matches static LLM+IGSC on both mean CometKiwi and structural pass: predictive + reactive (Exp. 3): 4.1%; A: + check LLM output: 4.1%; B: A + rank by predicted gain: 4.1%; C: Stage 2 risk with NMT-output clues: 4.1%; C+B: Stage 2 gain with NMT-output clues: 4.1%

### NMT = google, LLM = GPT-4o-mini

| target LLM share | policy | actual LLM share % | mean CometKiwi | struct pass % | broken or bad meaning % | NMT calls/string |
|---:|---|---:|---:|---:|---:|---:|
| 10% | predictive + reactive (Exp. 3) | 10.1 | 0.7945 | 98.28 | 6.99 | 0.907 |
| 20% | predictive + reactive (Exp. 3) | 19.9 | 0.7941 | 98.26 | 7.13 | 0.805 |
| 30% | predictive + reactive (Exp. 3) | 30.0 | 0.7935 | 98.25 | 7.27 | 0.702 |
| 10% | A: + check LLM output | 10.1 | 0.7946 | 98.35 | 6.92 | 0.923 |
| 20% | A: + check LLM output | 19.9 | 0.7941 | 98.35 | 7.05 | 0.821 |
| 30% | A: + check LLM output | 30.0 | 0.7935 | 98.35 | 7.19 | 0.719 |
| 10% | B: A + rank by predicted gain | 10.0 | 0.7948 | 98.35 | 6.88 | 0.939 |
| 20% | B: A + rank by predicted gain | 20.0 | 0.7946 | 98.35 | 6.94 | 0.829 |
| 30% | B: A + rank by predicted gain | 30.0 | 0.7943 | 98.35 | 7.01 | 0.727 |
| 10% | C: Stage 2 risk with NMT-output clues | 9.9 | 0.7949 | 98.35 | 6.85 | 1.000 |
| 20% | C: Stage 2 risk with NMT-output clues | 20.0 | 0.7948 | 98.35 | 6.89 | 1.000 |
| 30% | C: Stage 2 risk with NMT-output clues | 30.0 | 0.7941 | 98.35 | 7.04 | 1.000 |
| 10% | C+B: Stage 2 gain with NMT-output clues | 9.9 | 0.7954 | 98.35 | 6.76 | 1.000 |
| 20% | C+B: Stage 2 gain with NMT-output clues | 20.1 | 0.7956 | 98.35 | 6.71 | 1.000 |
| 30% | C+B: Stage 2 gain with NMT-output clues | 30.0 | 0.7955 | 98.35 | 6.72 | 1.000 |

| comparison | Δ CometKiwi | Δ struct pass (points) | Δ broken or bad (points) |
|---|---|---|---|
| C+B vs predictive+reactive @ 10% | +0.0008 [+0.0007, +0.0010] | +0.08 [+0.06, +0.10] | -0.23 [-0.27, -0.18] |
| C+B vs predictive+reactive @ 20% | +0.0015 [+0.0013, +0.0018] | +0.10 [+0.08, +0.12] | -0.42 [-0.49, -0.34] |
| C+B vs predictive+reactive @ 30% | +0.0020 [+0.0017, +0.0023] | +0.11 [+0.09, +0.13] | -0.55 [-0.64, -0.46] |
| C+B @ 10% vs static LLM+IGSC | +0.0077 [+0.0072, +0.0082] | +0.14 [+0.11, +0.17] | -1.54 [-1.70, -1.40] |
| C+B @ 10% vs reactive cascade | +0.0005 [+0.0004, +0.0006] | +0.00 [+0.00, +0.00] | -0.08 [-0.12, -0.04] |

Smallest LLM share at which each policy matches static LLM+IGSC on both mean CometKiwi and structural pass: predictive + reactive (Exp. 3): 6.3%; A: + check LLM output: 6.3%; B: A + rank by predicted gain: 6.3%; C: Stage 2 risk with NMT-output clues: 6.3%; C+B: Stage 2 gain with NMT-output clues: 6.3%

### NMT = google, LLM = Llama-3.1-8B

| target LLM share | policy | actual LLM share % | mean CometKiwi | struct pass % | broken or bad meaning % | NMT calls/string |
|---:|---|---:|---:|---:|---:|---:|
| 10% | predictive + reactive (Exp. 3) | 10.1 | 0.7906 | 94.14 | 11.34 | 0.907 |
| 20% | predictive + reactive (Exp. 3) | 19.9 | 0.7874 | 93.96 | 12.30 | 0.805 |
| 30% | predictive + reactive (Exp. 3) | 30.0 | 0.7837 | 93.96 | 13.23 | 0.702 |
| 10% | A: + check LLM output | 10.1 | 0.7916 | 95.96 | 9.62 | 0.965 |
| 20% | A: + check LLM output | 19.9 | 0.7885 | 95.96 | 10.43 | 0.865 |
| 30% | A: + check LLM output | 30.0 | 0.7848 | 95.96 | 11.35 | 0.763 |
| 10% | B: A + rank by predicted gain | 10.0 | 0.7923 | 95.96 | 9.45 | 0.964 |
| 20% | B: A + rank by predicted gain | 20.0 | 0.7911 | 95.96 | 9.70 | 0.865 |
| 30% | B: A + rank by predicted gain | 29.9 | 0.7892 | 95.96 | 10.10 | 0.767 |
| 10% | C: Stage 2 risk with NMT-output clues | 9.9 | 0.7921 | 95.96 | 9.47 | 1.000 |
| 20% | C: Stage 2 risk with NMT-output clues | 20.0 | 0.7896 | 95.96 | 10.20 | 1.000 |
| 30% | C: Stage 2 risk with NMT-output clues | 30.0 | 0.7859 | 95.96 | 11.10 | 1.000 |
| 10% | C+B: Stage 2 gain with NMT-output clues | 9.9 | 0.7928 | 95.96 | 9.33 | 1.000 |
| 20% | C+B: Stage 2 gain with NMT-output clues | 20.0 | 0.7916 | 95.96 | 9.60 | 1.000 |
| 30% | C+B: Stage 2 gain with NMT-output clues | 29.9 | 0.7896 | 95.96 | 10.03 | 1.000 |

| comparison | Δ CometKiwi | Δ struct pass (points) | Δ broken or bad (points) |
|---|---|---|---|
| C+B vs predictive+reactive @ 10% | +0.0021 [+0.0019, +0.0024] | +1.82 [+1.69, +1.96] | -2.01 [-2.15, -1.86] |
| C+B vs predictive+reactive @ 20% | +0.0042 [+0.0038, +0.0046] | +2.00 [+1.85, +2.16] | -2.70 [-2.90, -2.52] |
| C+B vs predictive+reactive @ 30% | +0.0059 [+0.0054, +0.0064] | +2.01 [+1.86, +2.16] | -3.20 [-3.42, -2.99] |
| C+B @ 10% vs static LLM+IGSC | +0.0397 [+0.0388, +0.0406] | +2.02 [+1.88, +2.18] | -10.22 [-10.52, -9.91] |
| C+B @ 10% vs reactive cascade | +0.0001 [+0.0000, +0.0003] | +0.00 [+0.00, +0.00] | -0.01 [-0.05, +0.03] |

Smallest LLM share at which each policy matches static LLM+IGSC on both mean CometKiwi and structural pass: predictive + reactive (Exp. 3): 6.3%; A: + check LLM output: 6.3%; B: A + rank by predicted gain: 6.3%; C: Stage 2 risk with NMT-output clues: 6.3%; C+B: Stage 2 gain with NMT-output clues: 6.3%

