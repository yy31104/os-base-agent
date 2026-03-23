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
        return "UNKNOWN", "未知"
    if st.shares > 0:
        return "HOLD", f"持有 {st.shares} 股"
    return "FLAT", "空仓"


def paper_lines(st, mark_price: float, initial_cash: float, day_start_equity: float) -> list[str]:
    if st is None:
        return ["账户模式: 非 paper 模式"]
    eq = equity(st, mark_price)
    cum_ret = eq / initial_cash - 1.0
    today_ret = 0.0 if day_start_equity <= 0 else eq / day_start_equity - 1.0
    return [
        f"账户净值: ${eq:,.2f}",
        f"累计收益: {pct_text(cum_ret)}",
        f"今日贡献: {pct_text(today_ret)}",
        f"现金/持仓: ${st.cash:,.2f} / {st.shares} 股",
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
            triggers.append("开盘跳空偏高")
        if early_hit:
            triggers.append("前5分钟走弱")
        why = "、".join(triggers) if triggers else "触发风控线"
        lead = f"今天是风险日，因为{why}。"
    else:
        lead = "今天不是风险日，因为开盘和前5分钟走势都没有越过风控线。"
    detail = f"参考值: gap={gap*100:.3f}% (线 {gap_gate*100:.3f}%), early_ret={early*100:.3f}% (线 {early_gate*100:.3f}%)"
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
    lines = [f"【OS_base {title}】{symbol} {date_key}", "", "状态摘要:"]
    lines.extend(decision_lines)
    lines.append("")
    lines.append("收益:")
    lines.extend(profit_lines)
    lines.append("")
    lines.append("执行与改动:")
    lines.extend(action_lines)
    lines.append("")
    lines.append("原因(通俗):")
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
    day_intraday_desc = "待 09:40 执行"
    day_close_code = "PENDING"
    day_close_desc = "待 16:05 判断"

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
                day_intraday_desc = "待 09:40 执行"
                day_close_code = "PENDING"
                day_close_desc = "待 16:05 判断"

        if hhmm >= "09:36" and date_key not in sent["morning"]:
            action_lines = ["策略参数改动: 无（沿用 config.json）"]
            open_reason = "当前不是 paper 模式，今天不做建仓。"
            bought = 0
            if st is not None and st.shares == 0:
                bought = buy_max(st, open_930, cost_bps)
                save_state(args.state, st)
                if bought > 0:
                    action_lines.append(f"开盘动作: 买入 {bought} 股 @ {open_930:.2f}")
                    open_reason = "启动时是空仓，所以按 paper 规则先尽量建仓。"
                else:
                    action_lines.append("开盘动作: 未成交（现金不足或价格异常）")
                    open_reason = "虽然计划建仓，但可用现金不足以买入 1 股。"
            elif st is not None:
                action_lines.append(f"开盘动作: 无（已有持仓 {st.shares} 股）")
                open_reason = "已经有仓位，为避免重复下单，开盘不再买入。"

            if risk:
                day_intraday_code = "SELL_TO_FLAT"
                day_intraday_desc = "09:40 卖出到空仓"
                day_close_code = "BUYBACK_OR_STAY_FLAT"
                day_close_desc = "16:05 再判断买回还是空仓"
            else:
                day_intraday_code = "HOLD"
                day_intraday_desc = "09:40 不卖出，继续持有"
                day_close_code = "NONE"
                day_close_desc = "非风险日，无需买回"

            pos_code, pos_desc = position_state(st)
            decision_lines = [
                f"风险日: {'是' if risk else '否'}",
                f"盘中动作(计划): {day_intraday_code}（{day_intraday_desc}）",
                f"收盘动作(计划): {day_close_code}（{day_close_desc}）",
                f"当前仓位: {pos_code}（{pos_desc}）",
            ]
            reason_lines = base_reasons + [open_reason]
            msg = build_msg(
                "09:36 开盘评估",
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
            action_lines = ["策略参数改动: 无（沿用 config.json）"]
            action_reason = "当前不是 paper 模式，09:40 不执行交易。"
            if risk:
                if st is not None and st.shares > 0:
                    sold = sell_all(st, px, cost_bps)
                    save_state(args.state, st)
                    day_intraday_code = "SELL_TO_FLAT"
                    day_intraday_desc = "已在 09:40 卖出"
                    action_lines.append(f"09:40 动作: 卖出 {sold} 股 @ {px:.2f}")
                    action_reason = "今天是风险日，所以 09:40 先卖出降风险。"
                else:
                    day_intraday_code = "NO_POSITION"
                    day_intraday_desc = "无仓可卖"
                    action_lines.append(f"09:40 动作: 无下单（当前无持仓），观察价 {px:.2f}")
                    action_reason = "今天是风险日，但你当时没有仓位，所以无需卖出。"
                day_close_code = "BUYBACK_OR_STAY_FLAT"
                day_close_desc = "16:05 根据收盘强弱决定"
            else:
                day_intraday_code = "HOLD"
                day_intraday_desc = "09:40 不卖出，继续持有"
                day_close_code = "NONE"
                day_close_desc = "非风险日，无需买回"
                action_lines.append(f"09:40 动作: 不卖出，继续持有，观察价 {px:.2f}")
                action_reason = "今天不是风险日，所以 09:40 不清仓。"

            pos_code, pos_desc = position_state(st)
            decision_lines = [
                f"风险日: {'是' if risk else '否'}",
                f"盘中动作(执行): {day_intraday_code}（{day_intraday_desc}）",
                f"收盘动作(预案): {day_close_code}（{day_close_desc}）",
                f"当前仓位: {pos_code}（{pos_desc}）",
            ]
            reason_lines = base_reasons + [action_reason]
            msg = build_msg(
                "09:40 执行回报",
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
            action_lines = ["策略参数改动: 无（沿用 config.json）"]

            close_reason = "当前不是 paper 模式，收盘不执行交易。"
            if not risk:
                day_close_code = "NONE"
                day_close_desc = "非风险日，无需买回"
                action_lines.append("收盘动作: 无下单（非风险日）")
                close_reason = "今天非风险日，白天本来就持仓，所以收盘无需买回。"
            else:
                if st is not None and st.shares == 0:
                    if bb:
                        bought = buy_max(st, close_today, cost_bps)
                        save_state(args.state, st)
                        if bought > 0:
                            day_close_code = "BUYBACK"
                            day_close_desc = "收盘买回，恢复隔夜持仓"
                            action_lines.append(f"收盘动作: 买回 {bought} 股 @ {close_today:.2f}")
                            close_reason = "今天是风险日，但收盘不弱（cc_ret>=0），按规则买回隔夜。"
                        else:
                            day_close_code = "BUYBACK_SKIPPED"
                            day_close_desc = "触发买回但未成交"
                            action_lines.append("收盘动作: 触发买回但未成交（现金不足买 1 股）")
                            close_reason = "已满足买回条件，但现金不足以买入 1 股。"
                    else:
                        day_close_code = "STAY_FLAT"
                        day_close_desc = "保持空仓过夜"
                        action_lines.append("收盘动作: 不买回，保持空仓")
                        close_reason = "今天是风险日且收盘偏弱（cc_ret<0），按规则空仓过夜。"
                elif st is not None and st.shares > 0:
                    day_close_code = "NONE"
                    day_close_desc = "当前已持仓，无需买回"
                    action_lines.append("收盘动作: 无下单（已持仓）")
                    close_reason = "当前已经持仓，所以不需要再买回。"
                else:
                    day_close_code = "NONE"
                    day_close_desc = "无可执行动作"
                    action_lines.append("收盘动作: 无")

            pos_code, pos_desc = position_state(st)
            decision_lines = [
                f"风险日: {'是' if risk else '否'}",
                f"盘中动作: {day_intraday_code}（{day_intraday_desc}）",
                f"收盘动作: {day_close_code}（{day_close_desc}）",
                f"隔夜持仓: {pos_code}（{pos_desc}）",
            ]
            reason_lines = base_reasons + [
                f"收盘相对昨收: {cc_ret*100:.3f}%",
                close_reason,
            ]
            if risk:
                reason_lines.append(
                    f"买回判定: {'cc_ret>=0，允许买回' if bb else 'cc_ret<0，不买回'}"
                )

            msg = build_msg(
                "16:05 收盘决策",
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
