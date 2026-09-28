import json
from pathlib import Path
from poltergeist.agent import answer
from poltergeist.store import RunStore

BASE = Path(__file__).resolve().parents[1]


def test_offline_eval_suite():
    cases = json.loads((BASE / "tests" / "eval_cases.json").read_text())
    for case in cases:
        output = answer(case["question"], RunStore(BASE / "examples" / "data")).answer
        for text in case["required"]:
            assert text.lower() in output.lower(), (case["question"], text, output)
        for text in case["forbidden"]:
            assert text.lower() not in output.lower(), (case["question"], text, output)
