from __future__ import annotations
import os
import pandas as pd

def load(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame(columns=["date","gap","early_ret"])
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    return df

def append(path: str, date, gap: float, early_ret: float) -> None:
    df = load(path)
    new = pd.DataFrame([{"date": pd.to_datetime(date), "gap": float(gap), "early_ret": float(early_ret)}])
    df = pd.concat([df, new], ignore_index=True).sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
