"""Simulated order execution with mandate-based authorization.

Teaching demo, not a trading system. Every order goes through one
gateway: proposal -> authorization against a mandate (policy as data)
-> execution only when the verdict is "allow". Every decision is
recorded. The shape follows ideas described in Saifuro's public docs
(mandate, authorization verdict with a decision code, idempotency keys,
fail-closed behavior) with an original minimal implementation. Nothing
here connects to a broker, holds credentials, or moves real funds.

Limits, stated plainly: prices are fixed constants; verdicts are plain
data, not signed tokens; the decision record is an in-memory list, not a
durable audit log; all state resets when the process restarts, so the
session and velocity caps protect one run only; there is no concurrency
control; and this module is standalone -- the AI question-answering path
cannot invoke it.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

# Fixed synthetic prices for the simulation. Not market data.
SIMULATED_PRICES: dict[str, float] = {
    "DEMO": 50.0,
    "ALT-DEMO": 12.5,
}

# Decision codes. "allow" is the only code that changes state.
ALLOW = "allow"
DENY_INVALID = "deny_invalid_request"
DENY_SYMBOL = "deny_symbol_not_allowed"
DENY_PER_ORDER = "deny_per_order_limit"
DENY_SESSION = "deny_session_limit"
DENY_VELOCITY = "deny_velocity"
DENY_CASH = "deny_insufficient_cash"
DENY_POSITION = "deny_insufficient_position"
DENY_CONFLICT = "deny_idempotency_conflict"


@dataclass(frozen=True)
class Mandate:
    """Policy as data. Behavior changes by editing these values,
    not the executor code."""

    max_order_notional: float
    max_session_notional: float
    allowed_symbols: tuple[str, ...]
    max_orders: int


DEFAULT_MANDATE = Mandate(
    max_order_notional=100.0,
    max_session_notional=150.0,
    allowed_symbols=("DEMO",),
    max_orders=5,
)


@dataclass(frozen=True)
class Order:
    symbol: str
    side: str  # "buy" or "sell"
    quantity: float
    idempotency_key: str


@dataclass(frozen=True)
class Verdict:
    code: str
    reason: str
    notional: float


@dataclass
class Portfolio:
    cash: float
    positions: dict[str, float] = field(default_factory=dict)
    executed_notional: float = 0.0
    decisions: list[dict] = field(default_factory=list)
    _by_key: dict[str, dict] = field(default_factory=dict)

    @property
    def executed_orders(self) -> int:
        return sum(1 for d in self.decisions if d["verdict"] == ALLOW)


def authorize(order: Order, mandate: Mandate, portfolio: Portfolio) -> Verdict:
    """Evaluate an order against the mandate. Pure check, no state change.

    Fail-closed: anything that cannot be evaluated is denied.
    """
    if order.side not in ("buy", "sell"):
        return Verdict(DENY_INVALID, f"side must be buy or sell, got {order.side!r}", 0.0)
    if (
        isinstance(order.quantity, bool)
        or not isinstance(order.quantity, (int, float))
        or not math.isfinite(order.quantity)
        or order.quantity <= 0
    ):
        return Verdict(DENY_INVALID, "quantity must be a positive finite number", 0.0)
    if order.symbol not in mandate.allowed_symbols:
        return Verdict(
            DENY_SYMBOL, f"{order.symbol!r} is not in the mandate's allowed symbols", 0.0
        )
    price = SIMULATED_PRICES.get(order.symbol)
    if price is None:
        return Verdict(DENY_INVALID, f"no simulated price for {order.symbol!r}", 0.0)
    notional = order.quantity * price
    if notional > mandate.max_order_notional:
        return Verdict(
            DENY_PER_ORDER,
            f"notional {notional:.2f} exceeds per-order limit {mandate.max_order_notional:.2f}",
            notional,
        )
    if portfolio.executed_notional + notional > mandate.max_session_notional:
        return Verdict(
            DENY_SESSION,
            f"session notional would reach {portfolio.executed_notional + notional:.2f}, "
            f"over limit {mandate.max_session_notional:.2f}",
            notional,
        )
    if portfolio.executed_orders >= mandate.max_orders:
        return Verdict(
            DENY_VELOCITY,
            f"already executed {portfolio.executed_orders} orders, limit {mandate.max_orders}",
            notional,
        )
    if order.side == "buy" and portfolio.cash < notional:
        return Verdict(
            DENY_CASH,
            f"cash {portfolio.cash:.2f} below notional {notional:.2f}",
            notional,
        )
    if order.side == "sell" and portfolio.positions.get(order.symbol, 0.0) < order.quantity:
        return Verdict(DENY_POSITION, "cannot sell more than the held position", notional)
    return Verdict(ALLOW, "within mandate", notional)


def place_simulated_order(
    portfolio: Portfolio, order: Order, mandate: Mandate = DEFAULT_MANDATE
) -> dict:
    """Authorize, then act only on allow. Returns the decision record.

    A repeated idempotency key with the same order returns the recorded
    decision without executing again. A repeated key with a different
    order is rejected as a conflict and recorded; the original decision
    stays the one stored under the key.
    """
    prior = portfolio._by_key.get(order.idempotency_key)
    if prior is not None:
        if (
            prior["symbol"] == order.symbol
            and prior["side"] == order.side
            and prior["quantity"] == order.quantity
        ):
            return {**prior, "replayed": True}
        conflict = {
            "seq": len(portfolio.decisions) + 1,
            "idempotency_key": order.idempotency_key,
            "symbol": order.symbol,
            "side": order.side,
            "quantity": order.quantity,
            "notional": 0.0,
            "verdict": DENY_CONFLICT,
            "reason": "idempotency key already used for a different order",
        }
        portfolio.decisions.append(conflict)
        return conflict
    verdict = authorize(order, mandate, portfolio)
    if verdict.code == ALLOW:
        if order.side == "buy":
            portfolio.cash -= verdict.notional
            portfolio.positions[order.symbol] = (
                portfolio.positions.get(order.symbol, 0.0) + order.quantity
            )
        else:
            portfolio.cash += verdict.notional
            portfolio.positions[order.symbol] -= order.quantity
        portfolio.executed_notional += verdict.notional
    record = {
        "seq": len(portfolio.decisions) + 1,
        "idempotency_key": order.idempotency_key,
        "symbol": order.symbol,
        "side": order.side,
        "quantity": order.quantity,
        "notional": round(verdict.notional, 2),
        "verdict": verdict.code,
        "reason": verdict.reason,
    }
    portfolio.decisions.append(record)
    portfolio._by_key[order.idempotency_key] = record
    return record


class Gateway:
    """The single intended entry point: proposal -> authorization -> execution.

    Every submitted order passes through authorize() before any state can
    change; there is no other intended execution path. (Python cannot stop
    code that bypasses the gateway and edits the portfolio directly -- the
    point is that the intended path always enforces.)
    """

    def __init__(self, mandate: Mandate = DEFAULT_MANDATE, cash: float = 1000.0,
                 portfolio: Portfolio | None = None):
        self.mandate = mandate
        self.portfolio = portfolio if portfolio is not None else Portfolio(cash=cash)

    def submit(self, order: Order) -> dict:
        return place_simulated_order(self.portfolio, order, self.mandate)


DEMO_ORDERS: tuple[Order, ...] = (
    Order("DEMO", "buy", 4, "ord-001"),      # 200.00 notional: over per-order limit
    Order("DEMO", "buy", 4, "ord-001"),      # retry, same key + same order: replayed
    Order("DEMO", "buy", 2, "ord-002"),      # 100.00: allowed
    Order("DEMO", "buy", 3, "ord-002"),      # same key, different order: conflict
    Order("DEMO", "buy", 2, "ord-003"),      # split attempt: session cap catches it
    Order("ALT-DEMO", "buy", 1, "ord-004"),  # symbol outside the mandate
    Order("DEMO", "sell", 1, "ord-005"),     # allowed sell
)


def run_demo(mandate: Mandate = DEFAULT_MANDATE, cash: float = 1000.0):
    gateway = Gateway(mandate=mandate, cash=cash)
    records = [gateway.submit(order) for order in DEMO_ORDERS]
    return gateway.portfolio, records


def main() -> None:
    portfolio, records = run_demo()
    m = DEFAULT_MANDATE
    print("Simulated trading demo (no broker, no real funds; in-memory state for one run)")
    print(
        f"Mandate: per-order <= {m.max_order_notional:.2f}, "
        f"session <= {m.max_session_notional:.2f}, "
        f"symbols {list(m.allowed_symbols)}, max {m.max_orders} orders"
    )
    print("Start: cash 1000.00, no positions\n")
    for r in records:
        note = " (replay of stored decision, nothing re-executed)" if r.get("replayed") else ""
        print(
            f"#{r['seq']} {r['side']} {r['quantity']} {r['symbol']} "
            f"-> {r['verdict']}: {r['reason']}{note}"
        )
    print(
        f"\nEnd: cash {portfolio.cash:.2f}, positions {portfolio.positions}, "
        f"executed {portfolio.executed_orders} orders"
    )


if __name__ == "__main__":
    main()


def portfolio_to_dict(portfolio: Portfolio) -> dict:
    return {
        "cash": portfolio.cash,
        "positions": portfolio.positions,
        "executed_notional": portfolio.executed_notional,
        "decisions": portfolio.decisions,
    }


def portfolio_from_dict(data: dict) -> Portfolio:
    decisions = list(data.get("decisions", []))
    portfolio = Portfolio(
        cash=float(data["cash"]),
        positions={k: float(v) for k, v in data.get("positions", {}).items()},
        executed_notional=float(data.get("executed_notional", 0.0)),
        decisions=decisions,
    )
    portfolio._by_key = {d["idempotency_key"]: d for d in decisions if "idempotency_key" in d}
    return portfolio


def load_portfolio(path) -> Portfolio:
    """Load persisted state, or a fresh 1000-cash portfolio when absent."""
    try:
        return portfolio_from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
    except FileNotFoundError:
        return Portfolio(cash=1000.0)


def save_portfolio(portfolio: Portfolio, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(portfolio_to_dict(portfolio), indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)
