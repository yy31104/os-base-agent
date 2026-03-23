from __future__ import annotations

import pytest

from os_base_agent.runtime import validate_config


def _base_config() -> dict:
    return {
        "symbol": "QQQ",
        "history_path": "data/history_store.csv",
        "strategy": {
            "lookback": 252,
            "q_gap": 0.8,
            "q_early": 0.3,
            "gap_buffer_bp": 2.0,
            "early_buffer_bp": 2.0,
        },
        "trade": {
            "mode": "paper",
            "cost_bps": 2.0,
            "paper": {"initial_cash": 10000.0},
        },
        "telegram": {"enabled": False, "bot_token": "YOUR_BOT_TOKEN", "chat_id": "YOUR_CHAT_ID"},
        "runtime": {"ignore_late_guard": False},
    }


def test_validate_config_accepts_valid_payload() -> None:
    validate_config(_base_config())


def test_validate_config_rejects_missing_symbol() -> None:
    cfg = _base_config()
    cfg["symbol"] = ""
    with pytest.raises(ValueError):
        validate_config(cfg)


def test_validate_config_rejects_placeholder_when_telegram_enabled() -> None:
    cfg = _base_config()
    cfg["telegram"]["enabled"] = True
    with pytest.raises(ValueError):
        validate_config(cfg)
