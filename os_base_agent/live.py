from __future__ import annotations
import yfinance as yf
import pandas as pd
from .tz import within_rth, nearest_bar

def fetch_intraday(symbol: str) -> pd.DataFrame:
    df = yf.download(symbol, period="1d", interval="1m", prepost=False, progress=False, auto_adjust=False)
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    df = df.rename(columns={c: c.strip().lower() for c in df.columns})
    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df.dropna(subset=["open","high","low","close"])
    return within_rth(df)

def prev_close(symbol: str) -> float | None:
    d1 = yf.download(symbol, period="7d", interval="1d", progress=False, auto_adjust=False)
    if d1 is None or d1.empty:
        return None
    if isinstance(d1.columns, pd.MultiIndex):
        d1.columns = [c[0] for c in d1.columns]
    d1 = d1.rename(columns={c: c.strip().lower() for c in d1.columns}).dropna()
    if len(d1) < 2:
        return None
    return float(d1["close"].iloc[-2])

def daily_points(df_rth: pd.DataFrame):
    b930 = nearest_bar(df_rth, "09:30")
    b935 = nearest_bar(df_rth, "09:35")
    if b930 is None or b935 is None:
        return None
    return float(b930["open"]), float(b935["close"]), float(df_rth["close"].iloc[-1])
