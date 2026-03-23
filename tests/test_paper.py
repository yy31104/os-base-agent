from __future__ import annotations

from os_base_agent.paper import State, buy_max, equity, load, save, sell_all


def test_buy_max_uses_all_cash_with_fees() -> None:
    st = State(cash=1000.0, shares=0)
    bought = buy_max(st, price=100.0, cost_bps=10.0)
    assert bought == 9
    assert st.shares == 9
    assert abs(st.cash - 99.1) < 1e-9


def test_sell_all_resets_position_and_adds_cash() -> None:
    st = State(cash=100.0, shares=10)
    sold = sell_all(st, price=50.0, cost_bps=20.0)
    assert sold == 10
    assert st.shares == 0
    assert abs(st.cash - 599.0) < 1e-9


def test_equity() -> None:
    st = State(cash=25.0, shares=4)
    assert equity(st, 10.0) == 65.0


def test_load_and_save_roundtrip(tmp_path) -> None:
    state_path = tmp_path / "paper_state.json"
    st = State(cash=123.45, shares=7)
    save(str(state_path), st)
    loaded = load(str(state_path), initial_cash=0.0)
    assert loaded.cash == st.cash
    assert loaded.shares == st.shares
