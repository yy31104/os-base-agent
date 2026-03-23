from __future__ import annotations

import json
import os
import shutil
from datetime import date
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import certifi
import pandas as pd

from os_base_agent.notify_telegram import send as tg_send

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


def _atomic_json_write(path: str, payload: dict[str, Any]) -> None:
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    tmp = path_obj.with_suffix(path_obj.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path_obj)


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


def save_notify_state(path: str, state: dict[str, set[str]], keep_days: int = 90) -> None:
    for k in state:
        state[k] = set(sorted(state[k])[-keep_days:])
    out = {k: sorted(list(v)) for k, v in state.items()}
    _atomic_json_write(path, out)


def mark_sent(path: str, state: dict[str, set[str]], key: str, date_key: str) -> None:
    state[key].add(date_key)
    save_notify_state(path, state)


def load_day_state(path: str) -> dict[str, Any]:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_day_state(path: str, payload: dict[str, Any]) -> None:
    _atomic_json_write(path, payload)


def load_last_close(path: str, symbol: str, today: date) -> tuple[float | None, str]:
    if not os.path.exists(path):
        return None, "missing_last_close_file"
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
    except Exception:
        return None, "invalid_last_close_file"

    if str(obj.get("symbol", "")).upper() != symbol.upper():
        return None, "last_close_symbol_mismatch"
    if "close" not in obj:
        return None, "last_close_missing_close"
    try:
        d = pd.to_datetime(str(obj.get("date"))).date()
    except Exception:
        return None, "last_close_invalid_date"
    if d >= today:
        return None, f"last_close_not_previous_day({d})"
    try:
        close_val = float(obj["close"])
    except Exception:
        return None, "last_close_invalid_close"
    return close_val, f"local_last_close({d})"


def save_last_close(path: str, symbol: str, date_key: str, close_today: float) -> None:
    payload = {
        "symbol": symbol,
        "date": date_key,
        "close": float(close_today),
        "updated_at": pd.Timestamp.now(tz=NY).isoformat(),
    }
    _atomic_json_write(path, payload)


def safe_tg_send(tg_cfg: dict[str, Any], msg: str) -> bool:
    if not tg_cfg.get("enabled"):
        return True
    token = str(tg_cfg.get("bot_token", "")).strip()
    chat_id = str(tg_cfg.get("chat_id", "")).strip()
    if not token or not chat_id:
        print("[WARN] telegram enabled but bot_token/chat_id is missing")
        return False
    try:
        tg_send(token, chat_id, msg)
        return True
    except Exception as e:
        print(f"[WARN] telegram send failed: {e}")
        return False


def _require_dict(cfg: dict[str, Any], key: str) -> dict[str, Any]:
    value = cfg.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"config.{key} must be an object")
    return value


def _require_number(obj: dict[str, Any], key: str, path: str) -> float:
    value = obj.get(key)
    if not isinstance(value, (int, float)):
        raise ValueError(f"{path}.{key} must be numeric")
    return float(value)


def validate_config(cfg: dict[str, Any]) -> None:
    symbol = cfg.get("symbol")
    if not isinstance(symbol, str) or not symbol.strip():
        raise ValueError("config.symbol must be a non-empty string")
    history_path = cfg.get("history_path")
    if not isinstance(history_path, str) or not history_path.strip():
        raise ValueError("config.history_path must be a non-empty string")

    strategy = _require_dict(cfg, "strategy")
    _require_number(strategy, "lookback", "config.strategy")
    _require_number(strategy, "q_gap", "config.strategy")
    _require_number(strategy, "q_early", "config.strategy")
    _require_number(strategy, "gap_buffer_bp", "config.strategy")
    _require_number(strategy, "early_buffer_bp", "config.strategy")

    trade = _require_dict(cfg, "trade")
    mode = str(trade.get("mode", "")).strip().lower()
    if mode not in {"paper", "signal_only"}:
        raise ValueError("config.trade.mode must be 'paper' or 'signal_only'")
    _require_number(trade, "cost_bps", "config.trade")

    paper = _require_dict(trade, "paper")
    initial_cash = _require_number(paper, "initial_cash", "config.trade.paper")
    if initial_cash <= 0:
        raise ValueError("config.trade.paper.initial_cash must be > 0")

    telegram = cfg.get("telegram", {})
    if telegram is not None and not isinstance(telegram, dict):
        raise ValueError("config.telegram must be an object")
    if isinstance(telegram, dict) and telegram.get("enabled"):
        token = str(telegram.get("bot_token", "")).strip()
        chat_id = str(telegram.get("chat_id", "")).strip()
        if not token or token == "YOUR_BOT_TOKEN":
            raise ValueError("config.telegram.bot_token is required when telegram.enabled=true")
        if not chat_id or chat_id == "YOUR_CHAT_ID":
            raise ValueError("config.telegram.chat_id is required when telegram.enabled=true")
