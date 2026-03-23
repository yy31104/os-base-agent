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
from os_base_agent.paper import buy_max, equity, load as load_state, save as save_state
from os_base_agent.runtime import (
    ensure_ca_bundle,
    load_day_state,
    load_last_close,
    load_notify_state,
    mark_sent,
    pct_text,
    safe_tg_send,
    save_day_state,
    save_last_close,
    validate_config,
)
from os_base_agent.strategy import buyback, risk_day, thresholds
from os_base_agent.tz import nearest_bar

NY = ZoneInfo("America/New_York")
CLOSE_EXEC_AT = "15:59"
CLOSE_EXEC_LATEST = "16:00"
CLOSE_FINALIZE_AT = "16:05"


def position_state(st) -> tuple[str, str]:
    if st is None:
        return "UNKNOWN", "未知"
    if st.shares > 0:
        return "HOLD", f"持有 {st.shares} 股"
    return "FLAT", "空仓"


def account_line(st, mark_price: float, initial_cash: float, day_start_equity: float) -> str:
    if st is None:
        return "净值 N/A"
    eq = equity(st, mark_price)
    cum_ret = eq / initial_cash - 1.0
    today_ret = 0.0 if day_start_equity <= 0 else eq / day_start_equity - 1.0
    return f"净值 ${eq:,.2f} (累计{pct_text(cum_ret)} | 今日{pct_text(today_ret)})"


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
    return (
        f"信号: gap {gap*100:+.3f}% {gap_cmp} {gap_gate*100:+.3f}% | "
        f"early {early*100:+.3f}% {early_cmp} {early_gate*100:+.3f}%"
    )


def build_close_msg(
    symbol: str,
    date_key: str,
    final_line: str,
    trigger_line: str,
    close_line: str,
    net_line: str,
    record_line: str,
    note: str | None = None,
) -> str:
    lines = [
        f"【OS_base 收盘】{symbol} {date_key}",
        f"最终: {final_line}",
        f"触发: {trigger_line}",
        close_line,
        net_line,
        record_line,
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


def wait_until(hhmm: str, poll_sec: int) -> None:
    while pd.Timestamp.now(tz=NY).strftime("%H:%M") < hhmm:
        time.sleep(poll_sec)


def main() -> None:
    ensure_ca_bundle()

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--state", default="data/paper_state.json")
    ap.add_argument("--notify-state", default="data/live_notify_state.json")
    ap.add_argument("--day-state", default="data/live_day_state.json")
    ap.add_argument("--last-close", default="data/last_close.json")
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
    now0 = pd.Timestamp.now(tz=NY)
    date_key0 = now0.date().isoformat()
    if date_key0 in sent["close"]:
        return

    day_state = load_day_state(args.day_state)
    st = load_state(args.state, initial_cash) if mode == "paper" else None

    local_pc0, local_src0 = load_last_close(args.last_close, symbol, now0.date())
    fallback_pc0 = prev_close(symbol)
    mark_price0 = float(local_pc0) if local_pc0 is not None else (float(fallback_pc0) if fallback_pc0 is not None else 0.0)
    day_start_equity0 = resolve_day_start_equity(day_state, date_key0, st, initial_cash, mark_price0)

    trading_day, closed_reason = is_nyse_trading_day(now0.date())
    if not trading_day:
        pos_code, pos_desc = position_state(st)
        final_line = f"隔夜 {pos_code}（{pos_desc}）"
        trigger_line = "非交易日 -> 跳过收盘策略"
        note = "非交易日不做收盘执行/复核。"
        if closed_reason == "weekend":
            trigger_line = "周末休市 -> 跳过收盘策略"
            note = "周末不做收盘执行/复核。"
        elif closed_reason == "nyse_holiday":
            trigger_line = "美股休市 -> 跳过收盘策略"
            note = "美股休市日不做收盘执行/复核。"

        msg = build_close_msg(
            symbol=symbol,
            date_key=date_key0,
            final_line=final_line,
            trigger_line=trigger_line,
            close_line="收盘: N/A（非交易日）",
            net_line=account_line(st, mark_price0, initial_cash, day_start_equity0),
            record_line="记录: 非交易日，不写入 history_store",
            note=note,
        )
        sent_ok = safe_tg_send(tg, msg)
        print(msg, flush=True)
        if sent_ok:
            mark_sent(args.notify_state, sent, "close", date_key0)
        else:
            print(f"[WARN] close notification not marked sent for {date_key0}")

        day_state.update(
            {
                "date": date_key0,
                "symbol": symbol,
                "market_open": False,
                "day_start_equity": float(day_start_equity0),
                "close_action_code": "SKIP_MARKET_CLOSED",
                "close_action_desc": trigger_line,
                "overnight_position": pos_code,
                "updated_at": pd.Timestamp.now(tz=NY).isoformat(),
            }
        )
        save_day_state(args.day_state, day_state)
        return

    wait_until(CLOSE_EXEC_AT, args.poll_sec)
    try:
        intr_exec, open_930, close_935, _ = wait_for_snapshot(symbol, now0.date(), args.timeout_sec, args.poll_sec)
    except RuntimeError as e:
        pos_code, pos_desc = position_state(st)
        msg = build_close_msg(
            symbol=symbol,
            date_key=date_key0,
            final_line=f"隔夜 {pos_code}（{pos_desc}）",
            trigger_line="休市/无当日分钟数据 -> 跳过收盘策略",
            close_line="收盘: N/A（无当日数据）",
            net_line=account_line(st, mark_price0, initial_cash, day_start_equity0),
            record_line="记录: 未写入 history_store",
            note=str(e),
        )
        sent_ok = safe_tg_send(tg, msg)
        print(msg, flush=True)
        if sent_ok:
            mark_sent(args.notify_state, sent, "close", date_key0)
        else:
            print(f"[WARN] close notification not marked sent for {date_key0}")

        day_state.update(
            {
                "date": date_key0,
                "symbol": symbol,
                "market_open": False,
                "day_start_equity": float(day_start_equity0),
                "close_action_code": "SKIP_MARKET_CLOSED_OR_NO_DATA",
                "close_action_desc": str(e),
                "overnight_position": pos_code,
                "updated_at": pd.Timestamp.now(tz=NY).isoformat(),
            }
        )
        save_day_state(args.day_state, day_state)
        return

    now_exec = pd.Timestamp.now(tz=NY)
    date_key = now_exec.date().isoformat()
    now_exec_hhmm = now_exec.strftime("%H:%M")
    late_run_detected = now_exec_hhmm > CLOSE_EXEC_LATEST
    late_run = late_run_detected and (not ignore_late_guard)

    local_pc, local_src = load_last_close(args.last_close, symbol, now_exec.date())
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

    hist = load_hist(hist_path)
    hist = hist[hist["date"].dt.date < now_exec.date()]
    thr_g, thr_e = thresholds(hist, lookback=lookback, q_gap=q_gap, q_early=q_early)
    gap = open_930 / pc - 1.0
    early = close_935 / open_930 - 1.0
    risk = risk_day(gap, early, thr_g, thr_e, gap_buffer_bp=gap_buf, early_buffer_bp=early_buf)
    if day_state.get("date") == date_key and "risk_day" in day_state:
        risk = bool(day_state["risk_day"])
        thr_g = float(day_state.get("thr_gap", thr_g))
        thr_e = float(day_state.get("thr_early", thr_e))
        gap_buf = float(day_state.get("gap_buffer_bp", gap_buf))
        early_buf = float(day_state.get("early_buffer_bp", early_buf))

    signal_line = signal_compare_line(gap, early, thr_g, thr_e, gap_buf, early_buf)
    day_start_equity = resolve_day_start_equity(day_state, date_key, st, initial_cash, pc)

    bar_1559 = nearest_bar(intr_exec, CLOSE_EXEC_AT, max_diff_min=2)
    if bar_1559 is not None:
        exec_px = float(bar_1559["close"])
        ts = pd.to_datetime(bar_1559.name)
        if ts.tzinfo is None:
            ts = ts.tz_localize(NY)
        else:
            ts = ts.tz_convert(NY)
        exec_src = ts.strftime("%H:%M")
    else:
        exec_px = float(intr_exec["close"].iloc[-1])
        exec_src = "latest"

    cc_ret_exec = exec_px / pc - 1.0
    bb = buyback(cc_ret_exec) if risk else False

    close_code = "NONE"
    close_desc = "继续当前仓位"
    if late_run:
        close_code = "LATE_NO_TRADE"
        close_desc = "错过收盘前窗口，不补执行"
    elif trade_blocked:
        close_code = "NO_TRADE_MISSING_PREV_CLOSE"
        close_desc = "缺少可靠昨收，不执行买回"
    elif not risk:
        close_code = "NONE"
        close_desc = "继续持仓隔夜（无需买回）"
    else:
        if st is not None and st.shares == 0:
            if bb:
                bought = buy_max(st, exec_px, cost_bps)
                save_state(args.state, st)
                if bought > 0:
                    close_code = "BUYBACK"
                    close_desc = "已买回，恢复隔夜持仓"
                else:
                    close_code = "BUYBACK_SKIPPED"
                    close_desc = "触发买回但未成交"
            else:
                close_code = "STAY_FLAT"
                close_desc = "保持空仓隔夜"
        elif st is not None and st.shares > 0:
            close_code = "NONE"
            close_desc = "已持仓，继续持仓隔夜"

    wait_until(CLOSE_FINALIZE_AT, args.poll_sec)
    _, _, _, final_close = wait_for_snapshot(symbol, now_exec.date(), 1200, args.poll_sec)
    cc_ret_final = final_close / pc - 1.0

    save_last_close(args.last_close, symbol, date_key, final_close)
    append_hist(hist_path, pd.Timestamp(now_exec.date()), gap, early)

    pos_code, pos_desc = position_state(st)
    if close_code in {"NONE", "BUYBACK"} and pos_code == "HOLD":
        final_line = f"隔夜 HOLD（{pos_desc}）"
    elif pos_code == "FLAT":
        final_line = "隔夜 FLAT（空仓）"
    else:
        final_line = f"隔夜 {pos_code}（{pos_desc}）"

    if close_code == "NONE" and not risk:
        trigger_line = "非风险日 -> 无需买回（本来就持仓）"
    elif close_code == "BUYBACK":
        trigger_line = "风险日且尾盘不弱 -> 收盘前买回"
    elif close_code == "STAY_FLAT":
        trigger_line = "风险日且尾盘偏弱 -> 不买回，空仓隔夜"
    elif close_code == "LATE_NO_TRADE":
        trigger_line = "错过收盘前窗口 -> 不补交易，仅复核"
    elif close_code == "NO_TRADE_MISSING_PREV_CLOSE":
        trigger_line = "昨收不可用 -> 不执行买回"
    else:
        trigger_line = close_desc

    note = None
    if late_run and not ignore_late_guard:
        note = f"晚于{CLOSE_EXEC_LATEST}ET不补交易，仅复核。"
    if late_run_detected and ignore_late_guard:
        note = f"晚运行({now_exec_hhmm}ET)但已按原规则执行（ignore_late_guard=true）。"

    msg = build_close_msg(
        symbol=symbol,
        date_key=date_key,
        final_line=final_line,
        trigger_line=trigger_line,
        close_line=f"收盘: {final_close:.2f} | cc_ret {cc_ret_final*100:+.2f}%",
        net_line=f"{account_line(st, final_close, initial_cash, day_start_equity)} | prev_close {pc:.2f}",
        record_line=f"记录: 已写入 history_store(gap/early) | {signal_line}",
        note=note,
    )
    sent_ok = safe_tg_send(tg, msg)
    print(msg, flush=True)
    if sent_ok:
        mark_sent(args.notify_state, sent, "close", date_key)
    else:
        print(f"[WARN] close notification not marked sent for {date_key}")

    day_state.update(
        {
            "date": date_key,
            "symbol": symbol,
            "market_open": True,
            "risk_day": bool(risk),
            "day_start_equity": float(day_start_equity),
            "prev_close": float(pc),
            "prev_close_source": pc_src,
            "trade_blocked": bool(trade_blocked),
            "late_run_close": bool(late_run),
            "late_run_close_detected": bool(late_run_detected),
            "ignore_late_guard": bool(ignore_late_guard),
            "open_930": float(open_930),
            "close_935": float(close_935),
            "close_exec_price": float(exec_px),
            "close_exec_price_source": exec_src,
            "close_today": float(final_close),
            "gap": float(gap),
            "early_ret": float(early),
            "thr_gap": float(thr_g),
            "thr_early": float(thr_e),
            "gap_buffer_bp": float(gap_buf),
            "early_buffer_bp": float(early_buf),
            "cc_ret_exec": float(cc_ret_exec),
            "cc_ret": float(cc_ret_final),
            "close_action_code": close_code,
            "close_action_desc": close_desc,
            "overnight_position": pos_code,
            "updated_at": pd.Timestamp.now(tz=NY).isoformat(),
        }
    )
    save_day_state(args.day_state, day_state)


if __name__ == "__main__":
    main()
