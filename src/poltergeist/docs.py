"""Allowlisted, read-only local notes. No arbitrary filesystem access."""
from __future__ import annotations

from pathlib import Path

ALLOWED_NOTES = frozenset({"README.md"})


def read_note(notes_root: Path, note: str) -> dict:
    """Typed tool: read_note(note) within a fixed directory and allowlist."""
    if note not in ALLOWED_NOTES:
        raise ValueError("unknown note")
    root = notes_root.resolve()
    path = root / note
    if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root):
        raise ValueError("note not found")
    if path.stat().st_size > 16_384:
        raise ValueError("note too large")
    text = path.read_text(encoding="utf-8")
    return {"note": note, "text": text, "source": f"docs/{note}"}
