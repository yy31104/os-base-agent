from __future__ import annotations
import argparse
import pandas as pd
import datetime as dt

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    args=ap.parse_args()

    df=pd.read_csv(args.csv)
    df=df.rename(columns={c: c.strip().lower().replace("(est)","").replace(" ","") for c in df.columns})
    df["dt"]=pd.to_datetime(df["time"], format="%Y/%m/%d %H:%M", errors="coerce")
    df=df.dropna(subset=["dt"]).sort_values("dt").reset_index(drop=True)
    df["time_hm"]=df["dt"].dt.strftime("%H:%M")
    r=df[(df["dt"].dt.time >= dt.time(9,30)) & (df["dt"].dt.time <= dt.time(15,59))].copy()
    r["date"]=pd.to_datetime(r["dt"].dt.date)

    bar_930 = r[r["time_hm"]=="09:30"].groupby("date").first().reset_index().rename(columns={"open":"open_930"})
    bar_935 = r[r["time_hm"]=="09:35"].groupby("date").first().reset_index().rename(columns={"close":"close_935"})
    last_bar = r.groupby("date").last().reset_index().rename(columns={"close":"close_today"})

    daily = bar_930[["date","open_930"]].merge(bar_935[["date","close_935"]], on="date", how="inner") \
                                        .merge(last_bar[["date","close_today"]], on="date", how="inner") \
                                        .sort_values("date").reset_index(drop=True)
    daily["prev_close"]=daily["close_today"].shift(1)
    daily=daily.dropna(subset=["prev_close"]).copy()
    daily["gap"]=daily["open_930"]/daily["prev_close"] - 1.0
    daily["early_ret"]=daily["close_935"]/daily["open_930"] - 1.0

    out = daily[["date","gap","early_ret"]].copy()
    out.to_csv(args.out, index=False)
    print("Wrote", args.out, "rows:", len(out))

if __name__=="__main__":
    main()
