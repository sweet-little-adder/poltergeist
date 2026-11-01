"""Session and validation behavior of the web trading endpoint."""

from poltergeist.webtrade import _STORE, handle_request


def setup_function():
    _STORE.clear()


def _order(key="k1", qty=2, session=None, symbol="DEMO", side="buy"):
    payload = {
        "action": "order",
        "side": side,
        "symbol": symbol,
        "quantity": qty,
        "idempotency_key": key,
    }
    if session:
        payload["session"] = session
    return payload


def test_allow_then_session_cap_in_one_session():
    status, body = handle_request(_order("k1", 2))
    assert status == 200
    assert body["record"]["verdict"] == "allow"
    sid = body["session"]
    assert sid
    status, body = handle_request(_order("k2", 2, sid))
    assert status == 200
    assert body["record"]["verdict"] == "deny_session_limit"


def test_replay_same_key_does_not_reexecute():
    _, body = handle_request(_order("k1", 2))
    sid = body["session"]
    _, body2 = handle_request(_order("k1", 2, sid))
    assert body2["record"].get("replayed") is True
    assert body2["record"]["seq"] == body["record"]["seq"]
    assert body2["portfolio"]["cash"] == body["portfolio"]["cash"]


def test_conflict_same_key_different_order():
    _, body = handle_request(_order("k1", 2))
    sid = body["session"]
    _, body2 = handle_request(_order("k1", 3, sid))
    assert body2["record"]["verdict"] == "deny_idempotency_conflict"
    assert body2["portfolio"]["cash"] == body["portfolio"]["cash"]


def test_per_order_limit():
    _, body = handle_request(_order("k1", 4))
    assert body["record"]["verdict"] == "deny_per_order_limit"


def test_symbol_not_allowed():
    _, body = handle_request(_order("k9", 1, symbol="ALT-DEMO"))
    assert body["record"]["verdict"] == "deny_symbol_not_allowed"


def test_sessions_are_isolated():
    _, a = handle_request(_order("k1", 2))
    _, b = handle_request(_order("k1", 2))
    assert a["session"] != b["session"]
    assert b["record"]["verdict"] == "allow"
    assert b["portfolio"]["cash"] == 900.0


def test_reset_returns_fresh_session():
    _, body = handle_request(_order("k1", 2))
    sid = body["session"]
    status, body2 = handle_request({"action": "reset", "session": sid})
    assert status == 200
    assert body2["session"] != sid
    assert body2["portfolio"]["cash"] == 1000.0
    assert body2["portfolio"]["decisions"] == []


def test_unknown_session_gets_fresh_state():
    _, a = handle_request(_order("k1", 2))
    _, b = handle_request(_order("k2", 2, session=a["session"] + "xx"))
    assert b["record"]["verdict"] == "allow"


def test_invalid_payloads_rejected():
    assert handle_request("nope")[0] == 400
    assert handle_request({"action": "nope"})[0] == 400
    assert handle_request({"action": "order"})[0] == 400
    bad_qty = _order("k1", True)
    assert handle_request(bad_qty)[0] == 400
    no_key = _order("k1", 2)
    del no_key["idempotency_key"]
    assert handle_request(no_key)[0] == 400
