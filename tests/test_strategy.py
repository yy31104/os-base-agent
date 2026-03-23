from __future__ import annotations

import pandas as pd

from os_base_agent.strategy import buyback, risk_day, thresholds


def test_thresholds_fallback_when_history_is_short() -> None:
    hist = pd.DataFrame({"gap": [0.001, 0.002], "early_ret": [-0.001, 0.0]})
    thr_g, thr_e = thresholds(hist, min_history=5, fallback_thr_gap=0.01, fallback_thr_early=-0.02)
    assert thr_g == 0.01
    assert thr_e == -0.02


def test_thresholds_quantile_on_recent_lookback() -> None:
    hist = pd.DataFrame(
        {
            "gap": [0.0, 0.01, 0.02, 0.03],
            "early_ret": [-0.03, -0.02, -0.01, 0.0],
        }
    )
    thr_g, thr_e = thresholds(hist, lookback=3, q_gap=0.5, q_early=0.5, min_history=1)
    assert abs(thr_g - 0.02) < 1e-12
    assert abs(thr_e - (-0.01)) < 1e-12


def test_risk_day_obeys_buffers() -> None:
    assert not risk_day(gap=0.0101, early_ret=0.0, thr_gap=0.01, thr_early=-0.01, gap_buffer_bp=2.0)
    assert risk_day(gap=0.0102, early_ret=0.0, thr_gap=0.01, thr_early=-0.01, gap_buffer_bp=2.0)
    assert risk_day(gap=0.0, early_ret=-0.0102, thr_gap=0.01, thr_early=-0.01, early_buffer_bp=2.0)


def test_buyback_rule() -> None:
    assert buyback(0.0)
    assert buyback(0.01)
    assert not buyback(-0.0001)
