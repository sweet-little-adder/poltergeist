"""HTTP-facing simulated trading sessions over the mandate gateway.

Teaching demo, not production order safety. State is per-visitor session
held in this process's memory: on a serverless platform it can reset on a
cold start, and a visitor can always mint a fresh session - the same as
re-running the CLI without --portfolio. The mandate is still enforced
server-side on every order: a tampered client cannot exceed caps within
its session, and there is no broker, no real funds, and nothing durable
to corrupt.
"""

from __future__ import annotations

import math
import secrets
import time
from collections import OrderedDict

from poltergeist.trading import (
    DEFAULT_MANDATE,
    SIMULATED_PRICES,
    Gateway,
    Order,
    portfolio_to_dict,
)

MAX_SESSIONS = 1000
MAX_BODY_BYTES = 4096
MAX_QUANTITY = 10**9
SESSION_IDLE_SECONDS = 3600

# Module-level session store: session_id -> (gateway, last_touch). On a
# serverless platform this lives only as long as the warm instance.
_STORE: "OrderedDict[str, tuple[Gateway, float]]" = OrderedDict()

MANDATE_VIEW = {
    "max_order_notional": DEFAULT_MANDATE.max_order_notional,
    "max_session_notional": DEFAULT_MANDATE.max_session_notional,
    "allowed_symbols": list(DEFAULT_MANDATE.allowed_symbols),
    "max_orders": DEFAULT_MANDATE.max_orders,
    "simulated_prices": dict(SIMULATED_PRICES),
}


def _valid_session_id(value) -> bool:
    return (
        isinstance(value, str)
        and 8 <= len(value) <= 64
        and all(c.isalnum() or c in "-_" for c in value)
    )


def _evict(now: float) -> None:
    for sid in [s for s, (_, t) in _STORE.items() if now - t > SESSION_IDLE_SECONDS]:
        _STORE.pop(sid, None)
    while len(_STORE) > MAX_SESSIONS:
        _STORE.popitem(last=False)


def _get_gateway(session_id) -> tuple[str, Gateway]:
    now = time.time()
    _evict(now)
    if _valid_session_id(session_id) and session_id in _STORE:
        gateway, _ = _STORE.pop(session_id)
        _STORE[session_id] = (gateway, now)
        return session_id, gateway
    sid = secrets.token_urlsafe(16)
    gateway = Gateway()
    _STORE[sid] = (gateway, now)
    return sid, gateway


def _state_body(sid: str, gateway: Gateway, record=None, note=None) -> dict:
    body = {
        "session": sid,
        "mandate": MANDATE_VIEW,
        "portfolio": portfolio_to_dict(gateway.portfolio),
    }
    if record is not None:
        body["record"] = record
    if note is not None:
        body["note"] = note
    return body


def handle_request(payload) -> tuple[int, dict]:
    """Validate and route one request. Returns (status, body).

    Validation rejects malformed input with 400; well-formed orders are
    decided by the mandate gateway (which is the enforcement point).
    """
    if not isinstance(payload, dict):
        return 400, {"error": "JSON object expected"}
    action = payload.get("action")
    session_id = payload.get("session")
    if action == "reset":
        sid, gateway = _get_gateway(None)
        return 200, _state_body(sid, gateway, note="session reset: fresh portfolio")
    if action != "order":
        return 400, {"error": "action must be 'order' or 'reset'"}
    symbol = payload.get("symbol")
    side = payload.get("side")
    quantity = payload.get("quantity")
    key = payload.get("idempotency_key")
    if not isinstance(symbol, str) or not isinstance(side, str):
        return 400, {"error": "symbol and side must be strings"}
    if (
        isinstance(quantity, bool)
        or not isinstance(quantity, (int, float))
        or not math.isfinite(quantity)
        or abs(quantity) > MAX_QUANTITY
    ):
        return 400, {"error": "quantity must be a finite number"}
    if not isinstance(key, str) or not 1 <= len(key) <= 128:
        return 400, {"error": "idempotency_key must be a string of 1-128 characters"}
    sid, gateway = _get_gateway(session_id)
    record = gateway.submit(
        Order(symbol=symbol, side=side, quantity=quantity, idempotency_key=key)
    )
    return 200, _state_body(sid, gateway, record=record)
