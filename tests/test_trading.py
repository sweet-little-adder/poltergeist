"""Behavior tests for the simulated trading demo.

These tests assert state transitions: an allowed order changes the
portfolio, a denied order changes nothing, and a replayed idempotency
key never executes twice.
"""

import dataclasses

from poltergeist.trading import (
    ALLOW,
    DENY_CASH,
    DENY_CONFLICT,
    DENY_INVALID,
    DENY_PER_ORDER,
    DENY_POSITION,
    DENY_SESSION,
    DENY_SYMBOL,
    DENY_VELOCITY,
    Mandate,
    Order,
    Portfolio,
    place_simulated_order,
)


def buy(symbol="DEMO", qty=2, key="k1"):
    return Order(symbol=symbol, side="buy", quantity=qty, idempotency_key=key)


def test_buy_within_mandate_executes():
    p = Portfolio(cash=1000.0)
    rec = place_simulated_order(p, buy(qty=2))
    assert rec["verdict"] == ALLOW
    assert rec["notional"] == 100.0
    assert p.cash == 900.0
    assert p.positions == {"DEMO": 2}
    assert len(p.decisions) == 1


def test_over_order_limit_denied_without_state_change():
    p = Portfolio(cash=1000.0)
    rec = place_simulated_order(p, buy(qty=4))  # 200 > 100 per-order cap
    assert rec["verdict"] == DENY_PER_ORDER
    assert p.cash == 1000.0
    assert p.positions == {}
    assert p.executed_notional == 0.0
    assert len(p.decisions) == 1  # denial is still recorded


def test_session_limit_catches_split_orders():
    p = Portfolio(cash=1000.0)
    first = place_simulated_order(p, buy(qty=2, key="k1"))  # 100, allowed
    second = place_simulated_order(p, buy(qty=2, key="k2"))  # would reach 200 > 150
    assert first["verdict"] == ALLOW
    assert second["verdict"] == DENY_SESSION
    assert p.cash == 900.0
    assert p.positions == {"DEMO": 2}
    assert p.executed_orders == 1


def test_symbol_outside_mandate_denied():
    p = Portfolio(cash=1000.0)
    rec = place_simulated_order(p, buy(symbol="ALT-DEMO", qty=1))
    assert rec["verdict"] == DENY_SYMBOL
    assert p.positions == {}


def test_mandate_is_data_not_code():
    mandate = dataclasses.replace(Mandate(100.0, 150.0, ("ALT-DEMO",), 5))
    p = Portfolio(cash=1000.0)
    assert place_simulated_order(p, buy(symbol="DEMO", qty=1), mandate)["verdict"] == DENY_SYMBOL
    rec = place_simulated_order(p, buy(symbol="ALT-DEMO", qty=1, key="k2"), mandate)
    assert rec["verdict"] == ALLOW
    assert p.positions == {"ALT-DEMO": 1}


def test_sell_requires_position_then_executes():
    p = Portfolio(cash=1000.0)
    denied = place_simulated_order(p, Order("DEMO", "sell", 1, "s1"))
    assert denied["verdict"] == DENY_POSITION
    place_simulated_order(p, buy(qty=2, key="b1"))
    ok = place_simulated_order(p, Order("DEMO", "sell", 1, "s2"))
    assert ok["verdict"] == ALLOW
    assert p.cash == 950.0
    assert p.positions == {"DEMO": 1}


def test_invalid_quantities_denied():
    for bad in (0, -1, float("nan"), float("inf"), "2", True):
        p = Portfolio(cash=1000.0)
        rec = place_simulated_order(p, buy(qty=bad))
        assert rec["verdict"] == DENY_INVALID, bad
        assert p.positions == {}


def test_velocity_limit_denies_after_max_orders():
    mandate = Mandate(100.0, 1000.0, ("DEMO",), max_orders=2)
    p = Portfolio(cash=1000.0)
    assert place_simulated_order(p, buy(qty=1, key="v1"), mandate)["verdict"] == ALLOW
    assert place_simulated_order(p, buy(qty=1, key="v2"), mandate)["verdict"] == ALLOW
    third = place_simulated_order(p, buy(qty=1, key="v3"), mandate)
    assert third["verdict"] == DENY_VELOCITY
    assert p.executed_orders == 2


def test_idempotent_replay_never_executes_twice():
    p = Portfolio(cash=1000.0)
    first = place_simulated_order(p, buy(qty=2, key="same"))
    second = place_simulated_order(p, buy(qty=2, key="same"))
    assert first["verdict"] == ALLOW
    assert second["verdict"] == ALLOW
    assert second["replayed"] is True
    assert len(p.decisions) == 1
    assert p.cash == 900.0  # charged once


def test_missing_price_fails_closed():
    mandate = Mandate(100.0, 150.0, ("NO-PRICE",), 5)
    p = Portfolio(cash=1000.0)
    rec = place_simulated_order(p, buy(symbol="NO-PRICE", qty=1), mandate)
    assert rec["verdict"] == DENY_INVALID
    assert p.positions == {}


def test_insufficient_cash_denied():
    p = Portfolio(cash=80.0)
    rec = place_simulated_order(p, buy(qty=2))  # 100 within per-order cap, cash short
    assert rec["verdict"] == DENY_CASH
    assert p.cash == 80.0


def test_same_key_different_order_rejected_as_conflict():
    p = Portfolio(cash=1000.0)
    first = place_simulated_order(p, buy(qty=2, key="dup"))
    conflict = place_simulated_order(p, buy(qty=3, key="dup"))
    assert first["verdict"] == ALLOW
    assert conflict["verdict"] == DENY_CONFLICT
    assert p.cash == 900.0  # only the original executed
    assert p.positions == {"DEMO": 2}
    assert len(p.decisions) == 2  # both the allow and the conflict are recorded
    # the stored key still maps to the original decision
    replay = place_simulated_order(p, buy(qty=2, key="dup"))
    assert replay["replayed"] is True
    assert replay["quantity"] == 2


def test_gateway_is_the_enforcing_entry_point():
    from poltergeist.trading import Gateway

    g = Gateway()
    denied = g.submit(buy(qty=4))
    assert denied["verdict"] == DENY_PER_ORDER
    assert g.portfolio.cash == 1000.0
    allowed = g.submit(buy(qty=2, key="g2"))
    assert allowed["verdict"] == ALLOW
    assert g.portfolio.cash == 900.0
