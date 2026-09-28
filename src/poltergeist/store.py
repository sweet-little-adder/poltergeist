"""Strict, read-only adapter for Crucibo's data/runs/bt_*/run_manifest.json bundles.

Only allowlisted, bounded scalar fields enter prompts or rendered answers. The
original manifest can contain paths or user-supplied text: never treat it as an
instruction or return it whole to the model.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

RUN_ID = re.compile(r"^bt_[A-Za-z0-9_-]{1,100}$")
FIELDS = ("return_pct", "pnl_cash", "max_drawdown_pct", "sharpe", "fills_count")


@dataclass(frozen=True)
class Run:
    run_id: str
    symbol: str
    strategy: str
    metrics: dict[str, int | float | None]
    source: str

    def as_dict(self) -> dict:
        return {"run_id": self.run_id, "symbol": self.symbol,
                "strategy": self.strategy, "metrics": self.metrics, "source": self.source}


class RunStore:
    def __init__(self, data_root: Path):
        self.root = data_root.resolve()
        self.runs = self.root / "runs"

    def list_runs(self) -> list[dict]:
        """Typed tool: list_backtest_runs() - only Crucibo bt_ bundles, max 50."""
        if not self.runs.is_dir():
            return []
        return [r.as_dict() for p in sorted(self.runs.iterdir())
                if p.is_dir() and RUN_ID.fullmatch(p.name)
                and (r := self._load(p)) is not None][:50]

    def get_run(self, run_id: str) -> dict:
        """Typed tool: get_backtest_run(run_id) - exact ID only, never a path."""
        if not RUN_ID.fullmatch(run_id):
            raise ValueError("invalid run ID")
        run = self._load(self.runs / run_id)
        if run is None:
            raise ValueError("run not found or invalid")
        return run.as_dict()

    def _load(self, path: Path) -> Run | None:
        # Reject symlinks, traversal, and oversized/invalid manifests.
        if path.is_symlink() or not path.is_dir() or not RUN_ID.fullmatch(path.name):
            return None
        mf = path / "run_manifest.json"
        if mf.is_symlink() or not mf.is_file() or mf.stat().st_size > 100_000:
            return None
        if not mf.resolve().is_relative_to(self.root):
            return None
        try:
            data = json.loads(mf.read_text(encoding="utf-8"))
            if not isinstance(data, dict) or data.get("cmd") != "backtest":
                return None
            if data.get("run_id") != path.name:
                return None
            # Metrics may be inline (Crucibo schema 2) or adjacent metrics.json.
            raw = data.get("metrics")
            mp = path / "metrics.json"
            if raw is None and mp.is_file() and not mp.is_symlink() and mp.stat().st_size <= 100_000:
                raw = json.loads(mp.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                raw = {}
            metrics = {k: v for k in FIELDS if (v := raw.get(k)) is None or
                       (type(v) in (int, float) and math.isfinite(v) and abs(v) < 1e15)}
            # Never echo free-form manifest fields or prompt-like strings.
            symbol = data.get("symbol", "unknown")
            strategy = data.get("strategy", "unknown")
            symbol = symbol if isinstance(symbol, str) and re.fullmatch(r"[A-Za-z0-9._-]{1,24}", symbol) else "unknown"
            strategy = strategy if isinstance(strategy, str) and re.fullmatch(r"[A-Za-z0-9_-]{1,40}", strategy) else "unknown"
            return Run(path.name, symbol, strategy, metrics,
                       f"runs/{path.name}/run_manifest.json")
        except (OSError, ValueError, TypeError, UnicodeError):
            return None
