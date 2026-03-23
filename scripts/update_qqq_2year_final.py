from __future__ import annotations

import argparse
import os
import shutil
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import certifi
import pandas as pd
import yfinance as yf


NY = ZoneInfo("America/New_York")


def ensure_ca_bundle() -> None:
    bundle = certifi.where()
    if any(ord(ch) > 127 for ch in bundle):
        target_dir = r"C:\os_agent_certs"
        target_bundle = os.path.join(target_dir, "cacert.pem")
        os.makedirs(target_dir, exist_ok=True)
        if not os.path.exists(target_bundle):
            shutil.copyfile(bundle, target_bundle)
        bundle = target_bundle
    os.environ["CURL_CA_BUNDLE"] = bundle
    os.environ["SSL_CERT_FILE"] = bundle
    os.environ["REQUESTS_CA_BUNDLE"] = bundle


def normalize_time_col(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    time_col = None
    for c in ["Time(EST)", "Time", "Datetime", "Date"]:
        if c in df.columns:
            time_col = c
            break
    if time_col is None:
        raise ValueError("Missing time column.")
    if time_col != "Time(EST)":
        df = df.rename(columns={time_col: "Time(EST)"})
        time_col = "Time(EST)"
    return df, time_col


def parse_local_ny(ts: pd.Series) -> pd.Series:
    dt = pd.to_datetime(ts, errors="coerce")
    if getattr(dt.dt, "tz", None) is None:
        dt = dt.dt.tz_localize(NY, ambiguous="infer", nonexistent="shift_forward")
    else:
        dt = dt.dt.tz_convert(NY)
    return dt


def parse_downloaded_ny(ts: pd.Series) -> pd.Series:
    dt = pd.to_datetime(ts, errors="coerce")
    if getattr(dt.dt, "tz", None) is None:
        dt = dt.dt.tz_localize("UTC")
    dt = dt.dt.tz_convert(NY)
    return dt


def format_time_est(ts: pd.Series) -> pd.Series:
    return (
        ts.dt.year.astype(str)
        + "/"
        + ts.dt.month.astype(str)
        + "/"
        + ts.dt.day.astype(str)
        + " "
        + ts.dt.hour.astype(str)
        + ":"
        + ts.dt.minute.astype(str).str.zfill(2)
    )


def load_existing(path: Path) -> tuple[pd.DataFrame, pd.Series, pd.Timestamp]:
    df = pd.read_csv(path)
    df, time_col = normalize_time_col(df)
    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    ts = parse_local_ny(df[time_col])
    if ts.isna().all():
        raise ValueError("Failed to parse existing timestamps.")
    last_ts = ts.dropna().max()
    return df, ts, last_ts


def fetch_new(symbol: str, periods: list[str]) -> pd.DataFrame:
    raw = pd.DataFrame()
    for p in periods:
        raw = yf.download(
            symbol,
            period=p,
            interval="1m",
            prepost=True,
            auto_adjust=False,
            progress=False,
        )
        if raw is not None and not raw.empty:
            break
    if raw is None or raw.empty:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw = raw.reset_index()
    dt_col = "Datetime" if "Datetime" in raw.columns else "Date"
    raw = raw.rename(columns={dt_col: "Time(EST)"})
    ts = parse_downloaded_ny(raw["Time(EST)"])
    out = pd.DataFrame(
        {
            "Time(EST)": format_time_est(ts),
            "Open": raw["Open"],
            "High": raw["High"],
            "Low": raw["Low"],
            "Close": raw["Close"],
            "Volume": raw["Volume"],
            "_ts": ts,
        }
    )
    return out


def main() -> int:
    ensure_ca_bundle()

    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/QQQ_2Year_Final.csv")
    ap.add_argument("--symbol", default="QQQ")
    ap.add_argument("--period", default="8d")
    ap.add_argument("--backup", action="store_true")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    existing, existing_ts, last_ts = load_existing(csv_path)
    periods = [args.period]
    for p in ["8d", "7d", "6d"]:
        if p not in periods:
            periods.append(p)
    downloaded = fetch_new(args.symbol, periods)
    if downloaded.empty:
        print("No downloaded rows.")
        return 1

    add = downloaded[downloaded["_ts"] > last_ts].copy()
    add = add.dropna(subset=["_ts"])
    if add.empty:
        print(f"No new rows. Last existing: {last_ts}")
        return 0

    existing["_ts"] = existing_ts
    merged = pd.concat([existing, add], ignore_index=True)
    merged = merged.dropna(subset=["_ts"])
    merged = merged.sort_values("_ts").drop_duplicates(subset=["_ts"], keep="last")
    merged["Time(EST)"] = format_time_est(merged["_ts"])
    merged = merged[["Time(EST)", "Open", "High", "Low", "Close", "Volume"]]

    if args.backup:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = csv_path.with_suffix(f".bak_{stamp}.csv")
        shutil.copy2(csv_path, backup)
        print(f"Backup: {backup}")

    merged.to_csv(csv_path, index=False)
    print(f"Updated: {csv_path}")
    print(f"Added rows: {len(add)}")
    print(f"New last row: {merged.iloc[-1].to_dict()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
