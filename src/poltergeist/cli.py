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
    parser.add_argument("--enable-trading", action="store_true",
                        help="Allow SIMULATED order proposals through the mandate gateway; default stays read-only")
    parser.add_argument("--portfolio", type=Path,
                        help="Persist simulated portfolio state to this JSON file across runs")
    args = parser.parse_args()
    trace = answer(args.question, RunStore(args.data_root), model=args.model,
                   api_key=os.environ.get("OPENAI_API_KEY"), trace_path=args.trace,
                   enable_trading=args.enable_trading, portfolio_path=args.portfolio)
    print(trace.answer)
    print(f"\ntrace: {args.trace} ({trace.latency_ms} ms; prompt={trace.prompt_tokens}, completion={trace.completion_tokens}, estimated USD={trace.estimated_cost_usd})")


if __name__ == "__main__":
    main()
