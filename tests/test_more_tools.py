from pathlib import Path
import pytest
from poltergeist.comparison import compare_runs
from poltergeist.docs import read_note
from poltergeist.store import RunStore

BASE = Path(__file__).resolve().parents[1]


def test_compare_runs_is_cited():
    result = compare_runs(RunStore(BASE / "examples/data"), "bt_DEMO_mean_reversion_001", "bt_DEMO_mean_reversion_002")
    assert result["difference"] == 0.37
    assert len(result["sources"]) == 2
    assert all(s.endswith("/run_manifest.json") for s in result["sources"])


def test_compare_rejects_unknown_metric():
    with pytest.raises(ValueError):
        compare_runs(RunStore(BASE / "examples/data"), "bt_DEMO_mean_reversion_001", "bt_DEMO_mean_reversion_002", "predicted_profit")


def test_read_note_is_bounded(tmp_path):
    root = BASE / "examples/docs"
    note = read_note(root, "README.md")
    assert "synthetic" in note["text"] and note["source"] == "docs/README.md"
    for path in ("../../.env", "/etc/passwd", "bad.md"):
        with pytest.raises(ValueError):
            read_note(root, path)
    (tmp_path / "README.md").symlink_to(root / "README.md")
    with pytest.raises(ValueError):
        read_note(tmp_path, "README.md")


def test_agent_routes_note_and_comparison():
    from poltergeist.agent import answer
    store = RunStore(BASE / "examples/data")
    a = answer("Read the research note", store)
    assert a.tool_calls[0]["name"] == "read_note" and "[source: docs/README.md]" in a.answer
    b = answer("Compare bt_DEMO_mean_reversion_001 and bt_DEMO_mean_reversion_002", store)
    assert b.tool_calls[0]["name"] == "compare_runs" and "difference" in b.answer and "[sources:" in b.answer
