from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from os_base_agent.history import append as append_hist
from os_base_agent.history import load as load_hist
from os_base_agent.live import daily_points, fetch_intraday, prev_close
from os_base_agent.market_day import is_nyse_trading_day
from os_base_agent.paper import buy_max, equity, load as load_state, save as save_state, sell_all
from os_base_agent.runtime import (
    ensure_ca_bundle,
    load_day_state,
    load_last_close,
    load_notify_state,
    mark_sent,
    pct_text,
    safe_tg_send,
    save_day_state,
    validate_config,
)
from os_base_agent.strategy import risk_day, thresholds
from os_base_agent.tz import nearest_bar
from scripts import update_qqq_2year_final as minute_updater

NY = ZoneInfo("America/New_York")
MORNING_CUTOFF = "09:45"


def position_state(st) -> tuple[str, str]:
    if st is None:
        return "UNKNOWN", "未知"
    if st.shares > 0:
        return "HOLD", f"持有 {st.shares} 股"
    return "FLAT", "空仓"


def account_line(st, mark_price: float, initial_cash: float, day_start_equity: float) -> str:
    if st is None:
        return "仓位: UNKNOWN | 净值 N/A"
    pos_code, _ = position_state(st)
    eq = equity(st, mark_price)
    cum_ret = eq / initial_cash - 1.0
    today_ret = 0.0 if day_start_equity <= 0 else eq / day_start_equity - 1.0
    return f"仓位: {pos_code} {st.shares}股 | 净值 ${eq:,.2f} (累计{pct_text(cum_ret)} | 今日{pct_text(today_ret)})"


def resolve_day_start_equity(
    day_state: dict,
    date_key: str,
    st,
    initial_cash: float,
    mark_price: float,
) -> float:
    if day_state.get("date") == date_key:
        try:
            return float(day_state.get("day_start_equity"))
        except (TypeError, ValueError):
            pass
    if st is None:
        return float(initial_cash)
    return float(equity(st, mark_price))


def signal_compare_line(
    gap: float,
    early: float,
    thr_g: float,
    thr_e: float,
    gap_buf_bp: float,
    early_buf_bp: float,
) -> str:
    gap_gate = thr_g + gap_buf_bp * 1e-4
    early_gate = thr_e - early_buf_bp * 1e-4
    gap_cmp = ">=" if gap >= gap_gate else "<"
    early_cmp = "<=" if early <= early_gate else ">"
    return f"信号: gap {gap*100:+.3f}% {gap_cmp} {gap_gate*100:+.3f}% | early {early*100:+.3f}% {early_cmp} {early_gate*100:+.3f}%"


def build_open_msg(
    symbol: str,
    date_key: str,
    conclusion: str,
    account: str,
    signal_line: str,
    price_line: str,
    note: str | None = None,
) -> str:
    lines = [
        f"【OS_base 开盘】{symbol} {date_key}",
        f"结论: {conclusion}",
        account,
        signal_line,
        price_line,
    ]
    if note:
        lines.append(f"备注: {note}")
    return "\n".join(lines)


def build_exec_msg(
    symbol: str,
    date_key: str,
    phase: str,
    action_line: str,
    account: str,
    exec_price_line: str,
    reason_line: str,
    note: str | None = None,
) -> str:
    lines = [
        f"【OS_base {phase}】{symbol} {date_key}",
        f"动作: {action_line}",
        account,
        f"执行价: {exec_price_line}",
        f"原因: {reason_line}",
    ]
    if note:
        lines.append(f"备注: {note}")
    return "\n".join(lines)


def _to_ny_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    idx = pd.to_datetime(out.index, errors="coerce")
    if idx.tz is None:
        idx = idx.tz_localize(NY)
    else:
        idx = idx.tz_convert(NY)
    out.index = idx
    return out


def _filter_date(df: pd.DataFrame, target_date: date) -> pd.DataFrame:
    if df.empty:
        return df
    out = _to_ny_index(df)
    return out[out.index.date == target_date]


def wait_for_snapshot(symbol: str, target_date: date, timeout_sec: int, poll_sec: int):
    deadline = time.time() + timeout_sec
    last_reason = "no_today_rth_bars"
    while True:
        intr = fetch_intraday(symbol)
        intr_today = _filter_date(intr, target_date)
        if not intr_today.empty:
            pts = daily_points(intr_today)
            if pts is not None:
                open_930, close_935, close_today = pts
                return intr_today, float(open_930), float(close_935), float(close_today)
            last_reason = "missing_0930_or_0935_bar"
        else:
            last_reason = "no_today_rth_bars"
        if time.time() >= deadline:
            raise RuntimeError(f"等待行情超时: {last_reason}")
        time.sleep(poll_sec)


def wait_until_0940(poll_sec: int) -> None:
    while pd.Timestamp.now(tz=NY).strftime("%H:%M") < "09:40":
        time.sleep(poll_sec)


def is_late_run(now_hhmm: str) -> bool:
    return now_hhmm > MORNING_CUTOFF


def _daily_features_from_minute_csv(csv_path: str) -> pd.DataFrame:
    src = pd.read_csv(csv_path)
    src = src.rename(columns={c: c.strip().lower().replace("(est)", "").replace(" ", "") for c in src.columns})
    if "time" not in src.columns:
        raise ValueError("minute csv missing time column")
    for c in ["open", "high", "low", "close", "volume"]:
        if c not in src.columns:
            raise ValueError(f"minute csv missing {c} column")

    src["dt"] = pd.to_datetime(src["time"], errors="coerce")
    src = src.dropna(subset=["dt"]).sort_values("dt").reset_index(drop=True)
    src["time_hm"] = src["dt"].dt.strftime("%H:%M")
    src["date"] = pd.to_datetime(src["dt"].dt.date)
    r = src[(src["time_hm"] >= "09:30") & (src["time_hm"] <= "15:59")].copy()

    bar_930 = r[r["time_hm"] == "09:30"].groupby("date").first().reset_index().rename(columns={"open": "open_930"})
    bar_935 = r[r["time_hm"] == "09:35"].groupby("date").first().reset_index().rename(columns={"close": "close_935"})
    last_bar = r.groupby("date").last().reset_index().rename(columns={"close": "close_today"})

    daily = (
        bar_930[["date", "open_930"]]
        .merge(bar_935[["date", "close_935"]], on="date", how="inner")
        .merge(last_bar[["date", "close_today"]], on="date", how="inner")
        .sort_values("date")
        .reset_index(drop=True)
    )
    daily["prev_close"] = daily["close_today"].shift(1)
    daily = daily.dropna(subset=["prev_close"]).copy()
    daily["gap"] = daily["open_930"] / daily["prev_close"] - 1.0
    daily["early_ret"] = daily["close_935"] / daily["open_930"] - 1.0
    return daily[["date", "gap", "early_ret"]]


def refresh_inputs_before_morning(symbol: str, minute_csv_path: str, history_path: str, today: date) -> str | None:
    notes: list[str] = []

    try:
        existing, existing_ts, last_ts = minute_updater.load_existing(Path(minute_csv_path))
        downloaded = minute_updater.fetch_new(symbol, ["8d", "7d", "6d"])
        if not downloaded.empty:
            add = downloaded[downloaded["_ts"] > last_ts].dropna(subset=["_ts"]).copy()
            if not add.empty:
                existing["_ts"] = existing_ts
                merged = pd.concat([existing, add], ignore_index=True)
                merged = merged.dropna(subset=["_ts"])
                merged = merged.sort_values("_ts").drop_duplicates(subset=["_ts"], keep="last")
                merged["Time(EST)"] = minute_updater.format_time_est(merged["_ts"])
                merged = merged[["Time(EST)", "Open", "High", "Low", "Close", "Volume"]]
                merged.to_csv(minute_csv_path, index=False)
                notes.append(f"minute_csv +{len(add)}")
    except Exception as e:
        notes.append(f"minute_csv_refresh_failed: {e}")

    try:
        hist = load_hist(history_path)
        have_dates = {d.date() for d in hist["date"]} if not hist.empty else set()
        daily = _daily_features_from_minute_csv(minute_csv_path)
        daily = daily[daily["date"].dt.date < today].copy()
        appended = 0
        last_appended = None
        for _, row in daily.iterrows():
            d = row["date"].date()
            if d in have_dates:
                continue
            append_hist(history_path, row["date"], float(row["gap"]), float(row["early_ret"]))
            have_dates.add(d)
            appended += 1
            last_appended = d
        if appended > 0:
            if last_appended is not None:
                notes.append(f"history_store +{appended} (last {last_appended})")
            else:
                notes.append(f"history_store +{appended}")
    except Exception as e:
        notes.append(f"history_sync_failed: {e}")

    if not notes:
        return None
    return " | ".join(notes)


def main() -> None:
    ensure_ca_bundle()

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--state", default="data/paper_state.json")
    ap.add_argument("--notify-state", default="data/live_notify_state.json")
    ap.add_argument("--day-state", default="data/live_day_state.json")
    ap.add_argument("--last-close", default="data/last_close.json")
    ap.add_argument("--minute-csv", default="data/QQQ_2Year_Final.csv")
    ap.add_argument("--timeout-sec", type=int, default=2400)
    ap.add_argument("--poll-sec", type=int, default=10)
    args = ap.parse_args()

    cfg = json.load(open(args.config, "r", encoding="utf-8"))
    validate_config(cfg)
    symbol = cfg["symbol"]
    hist_path = cfg["history_path"]
    mode = cfg["trade"]["mode"]
    initial_cash = float(cfg["trade"]["paper"]["initial_cash"])
    cost_bps = float(cfg["trade"]["cost_bps"])
    lookback = int(cfg["strategy"]["lookback"])
    q_gap = float(cfg["strategy"]["q_gap"])
    q_early = float(cfg["strategy"]["q_early"])
    gap_buf = float(cfg["strategy"]["gap_buffer_bp"])
    early_buf = float(cfg["strategy"]["early_buffer_bp"])
    tg = cfg.get("telegram", {"enabled": False})
    runtime_cfg = cfg.get("runtime", {})
    ignore_late_guard = bool(runtime_cfg.get("ignore_late_guard", False))

    sent = load_notify_state(args.notify_state)
    day_state = load_day_state(args.day_state)

    now = pd.Timestamp.now(tz=NY)
    date_key = now.date().isoformat()
    now_hhmm = now.strftime("%H:%M")

    pre_refresh_note = refresh_inputs_before_morning(
        symbol=symbol,
        minute_csv_path=args.minute_csv,
        history_path=hist_path,
        today=now.date(),
    )

    st = load_state(args.state, initial_cash) if mode == "paper" else None
    local_pc, local_src = load_last_close(args.last_close, symbol, now.date())
    fallback_pc = prev_close(symbol)
    mark_price = float(local_pc) if local_pc is not None else (float(fallback_pc) if fallback_pc is not None else 0.0)
    day_start_equity = resolve_day_start_equity(day_state, date_key, st, initial_cash, mark_price)

    trading_day, closed_reason = is_nyse_trading_day(now.date())
    if not trading_day:
        skip_code = "SKIP_MARKET_CLOSED"
        skip_reason = "今天非交易日，跳过开盘评估与09:40执行。"
        if closed_reason == "weekend":
            skip_reason = "今天周末休市，跳过开盘评估与09:40执行。"
        elif closed_reason == "nyse_holiday":
            skip_reason = "今天美股休市，跳过开盘评估与09:40执行。"

        if date_key not in sent["morning"]:
            msg = build_open_msg(
                symbol=symbol,
                date_key=date_key,
                conclusion=f"市场状态: 休市 | 动作={skip_code}",
                account=account_line(st, mark_price, initial_cash, day_start_equity),
                signal_line="信号: 非交易日不计算 gap/early/risk_day",
                price_line="价格: N/A（非交易日）",
                note=skip_reason,
            )
            sent_ok = safe_tg_send(tg, msg)
            print(msg, flush=True)
            if sent_ok:
                mark_sent(args.notify_state, sent, "morning", date_key)
            else:
                print(f"[WARN] morning notification not marked sent for {date_key}")
        if date_key not in sent["exit"]:
            mark_sent(args.notify_state, sent, "exit", date_key)

        day_state.update(
            {
                "date": date_key,
                "symbol": symbol,
                "market_open": False,
                "day_start_equity": float(day_start_equity),
                "risk_day": False,
                "prev_close": float(mark_price),
                "prev_close_source": local_src if local_pc is not None else "fallback_or_na",
                "late_run": False,
                "trade_blocked": True,
                "intraday_action_code": skip_code,
                "intraday_action_desc": skip_reason,
                "intraday_exec_price": None,
                "intraday_exec_price_source": None,
                "position_after_0940": position_state(st)[0],
                "close_plan_code": "NONE",
                "close_plan_desc": "非交易日，无收盘动作",
                "updated_at": pd.Timestamp.now(tz=NY).isoformat(),
            }
        )
        save_day_state(args.day_state, day_state)
        return

    try:
        _, open_930, close_935, _ = wait_for_snapshot(symbol, now.date(), args.timeout_sec, args.poll_sec)
    except RuntimeError as e:
        skip_code = "SKIP_MARKET_CLOSED_OR_NO_DATA"
        skip_reason = f"未拿到当日09:30/09:35分钟数据（{e}），跳过开盘评估与09:40执行。"
        if date_key not in sent["morning"]:
            msg = build_open_msg(
                symbol=symbol,
                date_key=date_key,
                conclusion=f"市场状态: 休市/缺数据 | 动作={skip_code}",
                account=account_line(st, mark_price, initial_cash, day_start_equity),
                signal_line="信号: 今日不计算风险日",
                price_line="价格: N/A（无当日分钟数据）",
                note=skip_reason,
            )
            sent_ok = safe_tg_send(tg, msg)
            print(msg, flush=True)
            if sent_ok:
                mark_sent(args.notify_state, sent, "morning", date_key)
            else:
                print(f"[WARN] morning notification not marked sent for {date_key}")
        if date_key not in sent["exit"]:
            mark_sent(args.notify_state, sent, "exit", date_key)

        day_state.update(
            {
                "date": date_key,
                "symbol": symbol,
                "market_open": False,
                "day_start_equity": float(day_start_equity),
                "risk_day": False,
                "prev_close": float(mark_price),
                "prev_close_source": local_src if local_pc is not None else "fallback_or_na",
                "late_run": False,
                "trade_blocked": True,
                "intraday_action_code": skip_code,
                "intraday_action_desc": skip_reason,
                "intraday_exec_price": None,
                "intraday_exec_price_source": None,
                "position_after_0940": position_state(st)[0],
                "close_plan_code": "NONE",
                "close_plan_desc": "无当日分钟数据，跳过",
                "updated_at": pd.Timestamp.now(tz=NY).isoformat(),
            }
        )
        save_day_state(args.day_state, day_state)
        return

    now = pd.Timestamp.now(tz=NY)
    date_key = now.date().isoformat()
    now_hhmm = now.strftime("%H:%M")

    local_pc, local_src = load_last_close(args.last_close, symbol, now.date())
    fallback_pc = prev_close(symbol)
    if local_pc is not None:
        pc = float(local_pc)
        pc_src = local_src
    elif fallback_pc is not None:
        pc = float(fallback_pc)
        pc_src = f"fallback_yfinance({local_src})"
    else:
        raise RuntimeError("无法获取 prev_close（本地缺失且 yfinance 也失败）。")

    trade_blocked = local_pc is None
    late_run_detected = is_late_run(now_hhmm)
    late_run = late_run_detected and (not ignore_late_guard)

    hist = load_hist(hist_path)
    hist = hist[hist["date"].dt.date < now.date()]
    thr_g, thr_e = thresholds(hist, lookback=lookback, q_gap=q_gap, q_early=q_early)
    gap = open_930 / pc - 1.0
    early = close_935 / open_930 - 1.0
    risk = risk_day(gap, early, thr_g, thr_e, gap_buffer_bp=gap_buf, early_buffer_bp=early_buf)
    signal_line = signal_compare_line(gap, early, thr_g, thr_e, gap_buf, early_buf)

    day_start_equity = resolve_day_start_equity(day_state, date_key, st, initial_cash, pc)

    if trade_blocked:
        intraday_code = "NO_TRADE_MISSING_PREV_CLOSE"
        intraday_desc = "缺少可靠昨收，不执行09:40交易"
        close_plan_code = "NONE"
        close_plan_desc = "保持当前仓位，不触发补执行"
    elif late_run:
        intraday_code = "LATE_NO_TRADE"
        intraday_desc = "错过09:40窗口，不补执行"
        close_plan_code = "NONE"
        close_plan_desc = "继续当前仓位，不补执行"
    elif risk:
        intraday_code = "SELL_TO_FLAT"
        intraday_desc = "09:40 卖出到空仓"
        close_plan_code = "BUYBACK_OR_STAY_FLAT"
        close_plan_desc = "15:59 判断买回或空仓隔夜"
    else:
        intraday_code = "HOLD"
        intraday_desc = "09:40 不卖出，继续持有"
        close_plan_code = "NONE"
        close_plan_desc = "继续持仓隔夜（无需买回）"

    if date_key not in sent["morning"]:
        note = None
        if st is not None and (not trade_blocked) and (not late_run) and st.shares == 0:
            bought = buy_max(st, open_930, cost_bps)
            save_state(args.state, st)
            if bought > 0:
                note = f"开盘补仓: 买入 {bought} 股 @ {open_930:.2f}"
            else:
                note = "开盘补仓失败（现金不足买1股）"

        if intraday_code == "SELL_TO_FLAT":
            conclusion = "风险日=是 -> 09:40卖出"
        elif intraday_code == "HOLD":
            conclusion = "风险日=否 -> 09:40不卖（继续持有）"
        elif intraday_code == "LATE_NO_TRADE":
            conclusion = f"风险日={'是' if risk else '否'} -> LATE_NO_TRADE（错过窗口不补执行）"
        elif intraday_code == "NO_TRADE_MISSING_PREV_CLOSE":
            conclusion = f"风险日={'是' if risk else '否'} -> NO_TRADE_MISSING_PREV_CLOSE"
        else:
            conclusion = f"风险日={'是' if risk else '否'} -> {intraday_code}"

        src_tag = "本地昨收" if pc_src.startswith("local_last_close") else "回退昨收"
        reason_parts = [f"昨收={pc:.2f}（{src_tag}）"]
        if pre_refresh_note:
            reason_parts.append(f"数据预更新: {pre_refresh_note}")
        if trade_blocked:
            reason_parts.append("缺少可靠昨收，今日仅提示不交易")
        if late_run:
            reason_parts.append(f"晚于{MORNING_CUTOFF}ET，仅提示不交易")
        if note is not None:
            reason_parts.append(note)

        msg = build_open_msg(
            symbol=symbol,
            date_key=date_key,
            conclusion=conclusion,
            account=account_line(st, close_935, initial_cash, day_start_equity),
            signal_line=signal_line,
            price_line=f"价格: 昨收 {pc:.2f} | 09:30 {open_930:.2f} | 09:35 {close_935:.2f}",
            note=" | ".join(reason_parts),
        )
        sent_ok = safe_tg_send(tg, msg)
        print(msg, flush=True)
        if sent_ok:
            mark_sent(args.notify_state, sent, "morning", date_key)
        else:
            print(f"[WARN] morning notification not marked sent for {date_key}")

    day_state.update(
        {
            "date": date_key,
            "symbol": symbol,
            "market_open": True,
            "risk_day": bool(risk),
            "prev_close": float(pc),
            "prev_close_source": pc_src,
            "late_run": bool(late_run),
            "late_run_detected": bool(late_run_detected),
            "ignore_late_guard": bool(ignore_late_guard),
            "trade_blocked": bool(trade_blocked),
            "open_930": float(open_930),
            "close_935": float(close_935),
            "gap": float(gap),
            "early_ret": float(early),
            "thr_gap": float(thr_g),
            "thr_early": float(thr_e),
            "gap_buffer_bp": float(gap_buf),
            "early_buffer_bp": float(early_buf),
            "day_start_equity": float(day_start_equity),
            "intraday_action_code": intraday_code,
            "intraday_action_desc": intraday_desc,
            "intraday_exec_price": None,
            "intraday_exec_price_source": None,
            "position_after_0940": position_state(st)[0],
            "close_plan_code": close_plan_code,
            "close_plan_desc": close_plan_desc,
            "updated_at": pd.Timestamp.now(tz=NY).isoformat(),
        }
    )
    save_day_state(args.day_state, day_state)

    if date_key in sent["exit"]:
        return

    if trade_blocked or late_run:
        reason_line = (
            "缺少可靠昨收，按规则不交易"
            if trade_blocked
            else f"当前运行时间 {now_hhmm}ET，已晚于{MORNING_CUTOFF}ET，按规则不补交易"
        )
        phase = f"LATE {now_hhmm}ET" if late_run else "09:40"
        msg = build_exec_msg(
            symbol=symbol,
            date_key=date_key,
            phase=phase,
            action_line=f"无（{intraday_code}）",
            account=account_line(st, close_935, initial_cash, day_start_equity),
            exec_price_line="—（未下单）",
            reason_line=reason_line,
            note=f"信号参考: {signal_line}",
        )
        sent_ok = safe_tg_send(tg, msg)
        print(msg, flush=True)
        if sent_ok:
            mark_sent(args.notify_state, sent, "exit", date_key)
        else:
            print(f"[WARN] exit notification not marked sent for {date_key}")
        return

    wait_until_0940(args.poll_sec)
    try:
        intr2, _, _, _ = wait_for_snapshot(symbol, now.date(), 600, args.poll_sec)
    except RuntimeError as e:
        msg = build_exec_msg(
            symbol=symbol,
            date_key=date_key,
            phase="09:40",
            action_line="无（NO_DATA）",
            account=account_line(st, close_935, initial_cash, day_start_equity),
            exec_price_line="—（未下单）",
            reason_line=f"09:40附近无分钟数据，按规则不交易：{e}",
            note=f"信号参考: {signal_line}",
        )
        sent_ok = safe_tg_send(tg, msg)
        print(msg, flush=True)
        if sent_ok:
            mark_sent(args.notify_state, sent, "exit", date_key)
        else:
            print(f"[WARN] exit notification not marked sent for {date_key}")
        return

    bar_0940 = nearest_bar(intr2, "09:40", max_diff_min=3)
    if bar_0940 is not None:
        px = float(bar_0940["close"])
        ts = pd.to_datetime(bar_0940.name)
        if ts.tzinfo is None:
            ts = ts.tz_localize(NY)
        else:
            ts = ts.tz_convert(NY)
        px_src = ts.strftime("%H:%M")
    else:
        px = float(intr2["close"].iloc[-1])
        px_src = "latest"

    action_line = "无"
    exec_price_line = "—（未下单）"
    if risk:
        if st is not None and st.shares > 0:
            sold = sell_all(st, px, cost_bps)
            save_state(args.state, st)
            intraday_code = "SELL_TO_FLAT"
            intraday_desc = "已在09:40卖出"
            action_line = f"卖出 {sold} 股（{intraday_code}）"
            exec_price_line = f"{px:.2f}（{px_src}）"
            action_reason = "今日风险日，09:40按规则先卖出降风险"
        else:
            intraday_code = "NO_POSITION"
            intraday_desc = "无仓可卖"
            action_line = f"无（{intraday_code}）"
            action_reason = "今日风险日，但当前无仓位，无需卖出"
        close_plan_code = "BUYBACK_OR_STAY_FLAT"
        close_plan_desc = "15:59 根据尾盘强弱决定"
    else:
        intraday_code = "HOLD"
        intraday_desc = "09:40 不卖出，继续持有"
        close_plan_code = "NONE"
        close_plan_desc = "继续持仓隔夜（无需买回）"
        action_line = f"无（{intraday_code}）"
        action_reason = "今日非风险日，09:40不清仓"

    msg = build_exec_msg(
        symbol=symbol,
        date_key=date_key,
        phase="09:40",
        action_line=action_line,
        account=account_line(st, px, initial_cash, day_start_equity),
        exec_price_line=exec_price_line,
        reason_line=f"risk_day={'是' if risk else '否'} | {action_reason}",
        note=f"信号参考: {signal_line}",
    )
    sent_ok = safe_tg_send(tg, msg)
    print(msg, flush=True)
    if sent_ok:
        mark_sent(args.notify_state, sent, "exit", date_key)
    else:
        print(f"[WARN] exit notification not marked sent for {date_key}")

    day_state.update(
        {
            "intraday_action_code": intraday_code,
            "intraday_action_desc": intraday_desc,
            "intraday_exec_price": float(px),
            "intraday_exec_price_source": px_src,
            "position_after_0940": position_state(st)[0],
            "updated_at": pd.Timestamp.now(tz=NY).isoformat(),
        }
    )
    save_day_state(args.day_state, day_state)


if __name__ == "__main__":
    main()
