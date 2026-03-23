from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from zoneinfo import ZoneInfo

import certifi
import pandas as pd

from os_base_agent.history import append as append_hist
from os_base_agent.history import load as load_hist
from os_base_agent.live import daily_points, fetch_intraday, prev_close
from os_base_agent.notify_telegram import send as tg_send
from os_base_agent.paper import buy_max, equity, load as load_state, save as save_state, sell_all
from os_base_agent.strategy import buyback, risk_day, thresholds

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


def pct_text(x: float) -> str:
    return f"{x * 100:+.2f}%"


def load_notify_state(path: str) -> dict[str, set[str]]:
    state: dict[str, set[str]] = {"morning": set(), "exit": set(), "close": set()}
    if not os.path.exists(path):
        return state
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        for k in state:
            vals = raw.get(k, [])
            if isinstance(vals, list):
                state[k] = {str(x) for x in vals}
    except Exception:
        return state
    return state


def save_notify_state(path: str, state: dict[str, set[str]]) -> None:
    for k in state:
        state[k] = set(sorted(state[k])[-90:])
    out = {k: sorted(list(v)) for k, v in state.items()}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)


def mark_sent(path: str, state: dict[str, set[str]], key: str, date_key: str) -> None:
    state[key].add(date_key)
    save_notify_state(path, state)


def position_state(st) -> tuple[str, str]:
    if st is None:
        return "UNKNOWN", "unknown"
    if st.shares > 0:
        return "HOLD", f"holding {st.shares} shares"
    return "FLAT", "flat"


def paper_lines(st, mark_price: float, initial_cash: float, day_start_equity: float) -> list[str]:
    if st is None:
        return ["Account mode: non-paper mode"]
    eq = equity(st, mark_price)
    cum_ret = eq / initial_cash - 1.0
    today_ret = 0.0 if day_start_equity <= 0 else eq / day_start_equity - 1.0
    return [
        f"Account equity: ${eq:,.2f}",
        f"Cumulative return: {pct_text(cum_ret)}",
        f"Today return: {pct_text(today_ret)}",
        f"Cash/Position: ${st.cash:,.2f} / {st.shares} shares",
    ]


def risk_reason_lines(
    gap: float,
    early: float,
    thr_g: float,
    thr_e: float,
    gap_buf_bp: float,
    early_buf_bp: float,
    risk: bool,
) -> list[str]:
    gap_gate = thr_g + gap_buf_bp * 1e-4
    early_gate = thr_e - early_buf_bp * 1e-4
    gap_hit = gap >= gap_gate
    early_hit = early <= early_gate
    if risk:
        triggers: list[str] = []
        if gap_hit:
            triggers.append("high opening gap")
        if early_hit:
            triggers.append("weak first 5-minute move")
        why = ", ".join(triggers) if triggers else "a risk threshold was triggered"
        lead = f"Today is a risk day because {why}."
    else:
        lead = "Today is not a risk day because neither open nor first 5-minute move crossed the risk line."
    detail = (
        f"Reference: gap={gap*100:.3f}% (line {gap_gate*100:.3f}%), "
        f"early_ret={early*100:.3f}% (line {early_gate*100:.3f}%)"
    )
    return [lead, detail]


def build_msg(
    title: str,
    symbol: str,
    date_key: str,
    decision_lines: list[str],
    profit_lines: list[str],
    action_lines: list[str],
    reason_lines: list[str],
) -> str:
    lines = [f"[OS_base {title}] {symbol} {date_key}", "", "Status summary:"]
    lines.extend(decision_lines)
    lines.append("")
    lines.append("Performance:")
    lines.extend(profit_lines)
    lines.append("")
    lines.append("Execution & changes:")
    lines.extend(action_lines)
    lines.append("")
    lines.append("Reason (plain English):")
    lines.extend(reason_lines)
    return "\n".join(lines)


def safe_tg_send(tg_cfg: dict, msg: str) -> None:
    if not tg_cfg.get("enabled"):
        return
    try:
        tg_send(tg_cfg["bot_token"], tg_cfg["chat_id"], msg)
    except Exception as e:
        print(f"[WARN] telegram send failed: {e}")


def main():
    ensure_ca_bundle()

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--state", default="data/paper_state.json")
    ap.add_argument("--notify-state", default="data/live_notify_state.json")
    args = ap.parse_args()
    cfg = json.load(open(args.config, "r", encoding="utf-8"))

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
    sent = load_notify_state(args.notify_state)

    day_ref = None
    day_start_equity = initial_cash
    day_intraday_code = "PENDING"
    day_intraday_desc = "pending 09:40 execution"
    day_close_code = "PENDING"
    day_close_desc = "pending 16:05 close decision"

    while True:
        now = pd.Timestamp.now(tz=NY)
        date = now.date()
        date_key = date.isoformat()
        hhmm = now.strftime("%H:%M")

        intr = fetch_intraday(symbol)
        pc = prev_close(symbol)
        if intr.empty or pc is None:
            time.sleep(20)
            continue

        pts = daily_points(intr)
        if pts is None:
            time.sleep(20)
            continue
        open_930, close_935, close_today = pts
        gap = open_930 / pc - 1.0
        early = close_935 / open_930 - 1.0

        hist = load_hist(hist_path)
        hist = hist[hist["date"].dt.date < date]
        thr_g, thr_e = thresholds(hist, lookback=lookback, q_gap=q_gap, q_early=q_early)
        risk = risk_day(gap, early, thr_g, thr_e, gap_buffer_bp=gap_buf, early_buffer_bp=early_buf)
        base_reasons = risk_reason_lines(gap, early, thr_g, thr_e, gap_buf, early_buf, risk)

        st = None
        if mode == "paper":
            st = load_state(args.state, initial_cash)
            if day_ref != date:
                day_ref = date
                day_start_equity = equity(st, pc)
                day_intraday_code = "PENDING"
                day_intraday_desc = "pending 09:40 execution"
                day_close_code = "PENDING"
                day_close_desc = "pending 16:05 close decision"

        if hhmm >= "09:36" and date_key not in sent["morning"]:
            action_lines = ["Strategy config change: none (using config.json)"]
            open_reason = "Current mode is non-paper, so no open build-in today."
            bought = 0
            if st is not None and st.shares == 0:
                bought = buy_max(st, open_930, cost_bps)
                save_state(args.state, st)
                if bought > 0:
                    action_lines.append(f"Open action: bought {bought} shares @ {open_930:.2f}")
                    open_reason = "Started flat, so paper rule builds position at open."
                else:
                    action_lines.append("Open action: no fill (insufficient cash or invalid price)")
                    open_reason = "Build-in was planned, but cash was insufficient for 1 share."
            elif st is not None:
                action_lines.append(f"Open action: none (already holding {st.shares} shares)")
                open_reason = "Already holding position, so no duplicate open order."

            if risk:
                day_intraday_code = "SELL_TO_FLAT"
                day_intraday_desc = "sell to flat at 09:40"
                day_close_code = "BUYBACK_OR_STAY_FLAT"
                day_close_desc = "re-evaluate buyback vs flat at 16:05"
            else:
                day_intraday_code = "HOLD"
                day_intraday_desc = "no sell at 09:40, keep holding"
                day_close_code = "NONE"
                day_close_desc = "non-risk day, no buyback needed"

            pos_code, pos_desc = position_state(st)
            decision_lines = [
                f"Risk day: {'YES' if risk else 'NO'}",
                f"Intraday action (plan): {day_intraday_code} ({day_intraday_desc})",
                f"Close action (plan): {day_close_code} ({day_close_desc})",
                f"Current position: {pos_code} ({pos_desc})",
            ]
            reason_lines = base_reasons + [open_reason]
            msg = build_msg(
                "09:36 Open assessment",
                symbol,
                date_key,
                decision_lines,
                paper_lines(st, close_935, initial_cash, day_start_equity),
                action_lines,
                reason_lines,
            )
            safe_tg_send(tg, msg)
            print(msg, flush=True)
            mark_sent(args.notify_state, sent, "morning", date_key)

        if hhmm >= "09:40" and date_key not in sent["exit"]:
            px = float(intr["close"].iloc[-1])
            action_lines = ["Strategy config change: none (using config.json)"]
            action_reason = "Current mode is non-paper, so no 09:40 execution."
            if risk:
                if st is not None and st.shares > 0:
                    sold = sell_all(st, px, cost_bps)
                    save_state(args.state, st)
                    day_intraday_code = "SELL_TO_FLAT"
                    day_intraday_desc = "sold at 09:40"
                    action_lines.append(f"09:40 action: sold {sold} shares @ {px:.2f}")
                    action_reason = "Today is a risk day, so we sold at 09:40 to reduce risk."
                else:
                    day_intraday_code = "NO_POSITION"
                    day_intraday_desc = "no position to sell"
                    action_lines.append(f"09:40 action: no order (no position), observed {px:.2f}")
                    action_reason = "Today is a risk day, but there was no position to sell."
                day_close_code = "BUYBACK_OR_STAY_FLAT"
                day_close_desc = "decide by close strength at 16:05"
            else:
                day_intraday_code = "HOLD"
                day_intraday_desc = "no sell at 09:40, keep holding"
                day_close_code = "NONE"
                day_close_desc = "non-risk day, no buyback needed"
                action_lines.append(f"09:40 action: no sell, keep holding, observed {px:.2f}")
                action_reason = "Today is not a risk day, so we do not liquidate at 09:40."

            pos_code, pos_desc = position_state(st)
            decision_lines = [
                f"Risk day: {'YES' if risk else 'NO'}",
                f"Intraday action (executed): {day_intraday_code} ({day_intraday_desc})",
                f"Close action (plan): {day_close_code} ({day_close_desc})",
                f"Current position: {pos_code} ({pos_desc})",
            ]
            reason_lines = base_reasons + [action_reason]
            msg = build_msg(
                "09:40 Execution report",
                symbol,
                date_key,
                decision_lines,
                paper_lines(st, px, initial_cash, day_start_equity),
                action_lines,
                reason_lines,
            )
            safe_tg_send(tg, msg)
            print(msg, flush=True)
            mark_sent(args.notify_state, sent, "exit", date_key)

        if hhmm >= "16:05" and date_key not in sent["close"]:
            cc_ret = close_today / pc - 1.0
            bb = buyback(cc_ret) if risk else False
            action_lines = ["Strategy config change: none (using config.json)"]

            close_reason = "Current mode is non-paper, so no close execution."
            if not risk:
                day_close_code = "NONE"
                day_close_desc = "non-risk day, no buyback needed"
                action_lines.append("Close action: no order (non-risk day)")
                close_reason = "Non-risk day: daytime position was already held, so no buyback is needed."
            else:
                if st is not None and st.shares == 0:
                    if bb:
                        bought = buy_max(st, close_today, cost_bps)
                        save_state(args.state, st)
                        if bought > 0:
                            day_close_code = "BUYBACK"
                            day_close_desc = "buy back at close, restore overnight hold"
                            action_lines.append(f"Close action: bought back {bought} shares @ {close_today:.2f}")
                            close_reason = "Risk day but close is not weak (cc_ret>=0), so buy back for overnight hold."
                        else:
                            day_close_code = "BUYBACK_SKIPPED"
                            day_close_desc = "buyback triggered but not filled"
                            action_lines.append("Close action: buyback triggered but not filled (insufficient cash for 1 share)")
                            close_reason = "Buyback condition passed, but cash was insufficient for 1 share."
                    else:
                        day_close_code = "STAY_FLAT"
                        day_close_desc = "stay flat overnight"
                        action_lines.append("Close action: no buyback, stay flat")
                        close_reason = "Risk day with weak close (cc_ret<0), so stay flat overnight by rule."
                elif st is not None and st.shares > 0:
                    day_close_code = "NONE"
                    day_close_desc = "already holding, no buyback needed"
                    action_lines.append("Close action: no order (already holding)")
                    close_reason = "Already holding a position, so no additional buyback is needed."
                else:
                    day_close_code = "NONE"
                    day_close_desc = "no executable action"
                    action_lines.append("Close action: none")

            pos_code, pos_desc = position_state(st)
            decision_lines = [
                f"Risk day: {'YES' if risk else 'NO'}",
                f"Intraday action: {day_intraday_code} ({day_intraday_desc})",
                f"Close action: {day_close_code} ({day_close_desc})",
                f"Overnight position: {pos_code} ({pos_desc})",
            ]
            reason_lines = base_reasons + [
                f"Close vs prev_close: {cc_ret*100:.3f}%",
                close_reason,
            ]
            if risk:
                reason_lines.append(
                    f"Buyback check: {'cc_ret>=0, buyback allowed' if bb else 'cc_ret<0, no buyback'}"
                )

            msg = build_msg(
                "16:05 Close decision",
                symbol,
                date_key,
                decision_lines,
                paper_lines(st, close_today, initial_cash, day_start_equity),
                action_lines,
                reason_lines,
            )
            safe_tg_send(tg, msg)
            print(msg, flush=True)
            mark_sent(args.notify_state, sent, "close", date_key)

            append_hist(hist_path, pd.Timestamp(date), gap, early)

        time.sleep(20)


if __name__ == "__main__":
    main()
