import argparse
import os
from pathlib import Path
from .agent import answer
from .store import RunStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only Crucibo backtest research")
    parser.add_argument("question")
    parser.add_argument("--data-root", type=Path, default=Path("examples/data"))
    parser.add_argument("--trace", type=Path, default=Path("traces/runs.jsonl"))
    parser.add_argument("--model", help="OpenAI-compatible model for tool planning; otherwise offline")
    args = parser.parse_args()
    trace = answer(args.question, RunStore(args.data_root), model=args.model,
                   api_key=os.environ.get("OPENAI_API_KEY"), trace_path=args.trace)
    print(trace.answer)
    print(f"\ntrace: {args.trace} ({trace.latency_ms} ms; prompt={trace.prompt_tokens}, completion={trace.completion_tokens}, estimated USD={trace.estimated_cost_usd})")


if __name__ == "__main__":
    main()
