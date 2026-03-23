from __future__ import annotations
import pandas as pd
import numpy as np
from zoneinfo import ZoneInfo

NY = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")

def to_ny(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    if not isinstance(idx, pd.DatetimeIndex):
        idx = pd.to_datetime(idx, errors="coerce")
    if idx.tz is not None:
        return idx.tz_convert(NY)
    hrs = pd.Series(idx.hour)
    med = float(hrs.median()) if len(hrs) else 0.0
    assume_utc = 11 <= med <= 20
    if assume_utc:
        return idx.tz_localize(UTC).tz_convert(NY)
    return idx.tz_localize(NY, ambiguous="infer", nonexistent="shift_forward")

def within_rth(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df
    df = df.copy()
    df.index = to_ny(df.index)
    return df.between_time("09:30", "15:59")

def nearest_bar(df: pd.DataFrame, hhmm: str, max_diff_min: int = 2):
    if df.empty: return None
    df = df.copy()
    df.index = to_ny(df.index)
    target = pd.to_datetime(hhmm).time()
    tmin = target.hour*60 + target.minute
    mins = df.index.hour*60 + df.index.minute
    dist = np.abs(mins - tmin)
    i = int(dist.argmin())
    if int(dist[i]) > max_diff_min:
        return None
    return df.iloc[i]
