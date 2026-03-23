from __future__ import annotations
import os, json, math
from dataclasses import dataclass, asdict

@dataclass
class State:
    cash: float
    shares: int = 0

def load(path: str, initial_cash: float) -> State:
    if not os.path.exists(path):
        return State(cash=float(initial_cash), shares=0)
    with open(path,"r",encoding="utf-8") as f:
        return State(**json.load(f))

def save(path: str, st: State) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path,"w",encoding="utf-8") as f:
        json.dump(asdict(st), f, ensure_ascii=False, indent=2)

def equity(st: State, price: float) -> float:
    return float(st.cash + st.shares*price)

def buy_max(st: State, price: float, cost_bps: float) -> int:
    if price <= 0: return 0
    fee_rate = cost_bps*1e-4
    max_sh = int(math.floor(st.cash/(price*(1+fee_rate))))
    if max_sh<=0: return 0
    notional = max_sh*price
    fee = fee_rate*notional
    st.cash -= (notional+fee)
    st.shares += max_sh
    return max_sh

def sell_all(st: State, price: float, cost_bps: float) -> int:
    if st.shares<=0: return 0
    fee_rate = cost_bps*1e-4
    notional = st.shares*price
    fee = fee_rate*notional
    sold = st.shares
    st.cash += (notional-fee)
    st.shares = 0
    return sold
