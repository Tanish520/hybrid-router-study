import numpy as np
import pandas as pd

import scripts.replay as rp

PROF = {"concurrency": 4, "latency_s_per_call": 1.0, "cold_start_s": 10.0, "usd_per_hour": 3600.0}


def rows(nmt_fail, repair, risk):
    n = len(nmt_fail)
    return pd.DataFrame({"nmt_fail": nmt_fail, "repair": repair, "risk": risk,
                         "src_tok": [3] * n, "gen_out_tok": [3] * n, "rep_out_tok": [3] * n})


def test_batch_time_rounds_up_by_concurrency():
    assert rp.batch_time(0, PROF) == 0.0
    assert rp.batch_time(4, PROF) == 1.0
    assert rp.batch_time(5, PROF) == 2.0


def test_static_llm_igsc_counts_repairs_and_cold_start():
    r = rows([False] * 8, [True, True] + [False] * 6, [0.0] * 8)
    build, cost, calls, nmt = rp.simulate_pr(r, "static_llm_igsc", 0.5, "llama", PROF, cold_nmt=False, cold_llm=True)
    # 8 generations (2 rounds of 4) + 2 repairs (1 round) after a 10 s cold start, plus overhead
    assert calls == 10 and nmt == 0
    assert build == rp.v(rp.CFG["pipeline_overhead_s"]) + 10 + 2 + 1
    assert abs(cost - (10 + 3)) < 1e-9          # 3600 $/h = 1 $/s, billed including the cold start


def test_reactive_escalates_only_nmt_failures_after_nmt():
    r = rows([True, False, False, False], [False] * 4, [0.0] * 4)
    build, cost, calls, nmt = rp.simulate_pr(r, "reactive_cascade", 0.5, "llama", PROF, cold_nmt=False, cold_llm=False)
    t_nmt = 4 * rp.v(rp.CFG["nmt"]["s_per_translation"]) + 4 * rp.v(rp.CFG["verifier_s_per_job"])
    assert calls == 1 and nmt == 4
    assert abs(build - (rp.v(rp.CFG["pipeline_overhead_s"]) + t_nmt + 1.0)) < 1e-9


def test_predictive_bypass_skips_nmt_for_high_risk():
    r = rows([False] * 4, [False] * 4, [0.9, 0.1, 0.1, 0.1])
    _, _, calls, nmt = rp.simulate_pr(r, "predictive_reactive", 0.5, "llama", PROF, cold_nmt=False, cold_llm=False)
    assert calls == 1 and nmt == 3


def test_static_nmt_never_calls_the_llm():
    r = rows([True] * 4, [True] * 4, [0.9] * 4)
    _, _, calls, nmt = rp.simulate_pr(r, "static_nmt", 0.5, "llama", PROF, cold_nmt=False, cold_llm=True)
    assert calls == 0 and nmt == 4
