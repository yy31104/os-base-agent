from __future__ import annotations
import argparse
import pandas as pd
import numpy as np
import math
import datetime as dt

from os_base_agent.strategy import thresholds, risk_day, buyback

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--initial-cash", type=float, default=10000.0)
    ap.add_argument("--cost-bps", type=float, default=2.0)
    ap.add_argument("--lookback", type=int, default=252)
    ap.add_argument("--q-gap", type=float, default=0.80)
    ap.add_argument("--q-early", type=float, default=0.30)
    ap.add_argument("--gap-buffer-bp", type=float, default=2.0)
    ap.add_argument("--early-buffer-bp", type=float, default=2.0)
    ap.add_argument("--out", default="outputs/os_base_backtest_report_v2.xlsx")
    args=ap.parse_args()

    fee_rate=args.cost_bps*1e-4

    df=pd.read_csv(args.csv)
    df=df.rename(columns={c: c.strip().lower().replace("(est)","").replace(" ","") for c in df.columns})
    df["dt"]=pd.to_datetime(df["time"], format="%Y/%m/%d %H:%M", errors="coerce")
    df=df.dropna(subset=["dt"]).sort_values("dt").reset_index(drop=True)
    df["time_hm"]=df["dt"].dt.strftime("%H:%M")
    r=df[(df["dt"].dt.time >= dt.time(9,30)) & (df["dt"].dt.time <= dt.time(15,59))].copy()
    r["date"]=pd.to_datetime(r["dt"].dt.date)

    bar_930 = r[r["time_hm"]=="09:30"].groupby("date").first().reset_index().rename(columns={"open":"open_930"})
    bar_935 = r[r["time_hm"]=="09:35"].groupby("date").first().reset_index().rename(columns={"close":"close_935"})
    bar_940 = r[r["time_hm"]=="09:40"].groupby("date").first().reset_index().rename(columns={"open":"open_940"})
    last_bar = r.groupby("date").last().reset_index().rename(columns={"close":"close_today"})

    daily = bar_930[["date","open_930"]].merge(bar_935[["date","close_935"]], on="date", how="inner") \
                                        .merge(bar_940[["date","open_940"]], on="date", how="inner") \
                                        .merge(last_bar[["date","close_today"]], on="date", how="inner") \
                                        .sort_values("date").reset_index(drop=True)
    daily["prev_close"]=daily["close_today"].shift(1)
    daily=daily.dropna(subset=["prev_close"]).copy()
    daily["gap"]=daily["open_930"]/daily["prev_close"] - 1.0
    daily["early_ret"]=daily["close_935"]/daily["open_930"] - 1.0
    daily["cc_ret"]=daily["close_today"]/daily["prev_close"] - 1.0

    hist_full=daily[["date","gap","early_ret"]].copy()

    cash=float(args.initial_cash); shares=0
    def buy_max(price):
        nonlocal cash, shares
        max_sh=int(math.floor(cash/(price*(1+fee_rate))))
        if max_sh<=0: return 0
        notional=max_sh*price
        cash -= (notional + fee_rate*notional)
        shares += max_sh
        return max_sh
    def sell_all(price):
        nonlocal cash, shares
        if shares<=0: return 0
        notional=shares*price
        cash += (notional - fee_rate*notional)
        sold=shares
        shares=0
        return sold

    rows=[]
    for i in range(len(daily)):
        day=daily.iloc[i]
        date=day["date"]
        hist = hist_full[hist_full["date"] < date]
        thr_g, thr_e = thresholds(hist, lookback=args.lookback, q_gap=args.q_gap, q_early=args.q_early)
        risk = risk_day(float(day["gap"]), float(day["early_ret"]), thr_g, thr_e, args.gap_buffer_bp, args.early_buffer_bp)

        buy930=sell940=buyclose=0; bb=False
        if shares==0:
            buy930=buy_max(float(day["open_930"]))
        if risk and shares>0:
            sell940=sell_all(float(day["open_940"]))
        if risk and shares==0:
            bb=buyback(float(day["cc_ret"]))
            if bb:
                buyclose=buy_max(float(day["close_today"]))

        eq=cash + shares*float(day["close_today"])
        rows.append({"date":date,"risk_day":bool(risk),"gap":float(day["gap"]),"early_ret":float(day["early_ret"]),
                     "thr_gap":thr_g,"thr_early":thr_e,"open_930":float(day["open_930"]),"open_940":float(day["open_940"]),
                     "close_today":float(day["close_today"]),"cc_ret":float(day["cc_ret"]),
                     "buy_shares_930":int(buy930),"sell_shares_940":int(sell940),"buyback":bool(bb),"buyback_shares_close":int(buyclose),
                     "cash":float(cash),"shares":int(shares),"equity":float(eq)})
    out=pd.DataFrame(rows)
    out["equity_ret"]=out["equity"].pct_change().fillna(0.0)

    # buy&hold baseline (buy max at first open)
    bh_cash=float(args.initial_cash)
    first=float(out["open_930"].iloc[0])
    bh_sh=int(math.floor(bh_cash/(first*(1+fee_rate))))
    bh_cash -= (bh_sh*first + fee_rate*(bh_sh*first))
    out["bh_equity"]=bh_cash + bh_sh*out["close_today"]
    out["bh_ret"]=out["bh_equity"].pct_change().fillna(0.0)

    def wstats(ret, n):
        x=np.asarray(ret.tail(n), float)
        eq=np.cumprod(1+x)
        total=float(eq[-1]-1)
        peak=np.maximum.accumulate(eq)
        mdd=float(np.min(eq/peak-1))
        return total,mdd
    windows=[("1M",21),("2M",42),("3M",63),("6M",126),("1Y",252),("All",len(out))]
    sums=[]
    for name,n in windows:
        if len(out)<n: continue
        tot,mdd=wstats(out["equity_ret"], n)
        btot,bmdd=wstats(out["bh_ret"], n)
        sums.append({"window":name,"start":str(out["date"].iloc[-n].date()),"end":str(out["date"].iloc[-1].date()),
                     "OS_total":tot,"BH_total":btot,"OS_mdd":mdd,"BH_mdd":bmdd})
    summary=pd.DataFrame(sums)

    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with pd.ExcelWriter(args.out, engine="openpyxl") as w:
        summary.to_excel(w, sheet_name="summary", index=False)
        out.to_excel(w, sheet_name="daily_log", index=False)

    print(summary)
    print("Wrote", args.out)

if __name__=="__main__":
    main()
