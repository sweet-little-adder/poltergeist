"""Deterministic comparison tool over recorded run metrics only."""
from __future__ import annotations

from .store import RunStore


def compare_runs(store: RunStore, left: str, right: str, metric: str = "return_pct") -> dict:
    """Typed tool: compare_runs(left, right, metric) for selected numeric metrics."""
    if metric not in ("return_pct", "pnl_cash", "max_drawdown_pct", "sharpe", "fills_count"):
        raise ValueError("unsupported metric")
    a, b = store.get_run(left), store.get_run(right)
    av, bv = a["metrics"].get(metric), b["metrics"].get(metric)
    if av is None or bv is None:
        raise ValueError("metric missing")
    return {"left": left, "right": right, "metric": metric,
            "left_value": av, "right_value": bv, "difference": round(av - bv, 8),
            "sources": [a["source"], b["source"]]}
