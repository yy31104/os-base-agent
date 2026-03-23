from __future__ import annotations
import numpy as np
import pandas as pd

def thresholds(hist: pd.DataFrame, lookback=252, q_gap=0.80, q_early=0.30,
               min_history=60, fallback_thr_gap=0.006, fallback_thr_early=-0.001):
    if hist is None or len(hist) < min_history:
        return float(fallback_thr_gap), float(fallback_thr_early)
    g = hist["gap"].to_numpy()[-min(lookback,len(hist)):]
    e = hist["early_ret"].to_numpy()[-min(lookback,len(hist)):]
    return float(np.quantile(g,q_gap)), float(np.quantile(e,q_early))

def risk_day(gap, early_ret, thr_gap, thr_early, gap_buffer_bp=2.0, early_buffer_bp=2.0):
    buf_gap = gap_buffer_bp*1e-4
    buf_early = early_buffer_bp*1e-4
    return (gap >= thr_gap + buf_gap) or (early_ret <= thr_early - buf_early)

def buyback(cc_ret: float) -> bool:
    return cc_ret >= 0.0
