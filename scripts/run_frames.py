#!/usr/bin/env python3
"""FRAMES benchmark runner — standalone script.

Evaluates RLM against google/frames-benchmark (824 multi-hop questions).
Published baselines: 0.40 (no retrieval), 0.66 (multi-step), 0.73 (oracle).

Prerequisites:
    pip install datasets

Usage:
    python scripts/run_frames.py --limit 5              # quick smoke test
    python scripts/run_frames.py --limit 50             # meaningful run (~$1.50)
    python scripts/run_frames.py --limit 50 --resume bench_results_*.json
    python scripts/run_frames.py --mode retrieval --search tavily
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Ensure src/ is importable when running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rich.console import Console  # noqa: E402

from rlm_research.benchmark import run_frames_benchmark  # noqa: E402
from rlm_research.config import load_config  # noqa: E402

console = Console()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run FRAMES benchmark against RLM system",
    )
    p.add_argument(
        "-n", "--limit", type=int, default=20,
        help="Max questions to evaluate (default: 20)",
    )
    p.add_argument(
        "-m", "--mode", choices=["oracle", "retrieval", "closed"], default="oracle",
        help="Source mode (default: oracle)",
    )
    p.add_argument(
        "-o", "--output", type=Path, default=None,
        help="Output JSON path (default: bench_results_<timestamp>.json)",
    )
    p.add_argument(
        "-r", "--resume", type=Path, default=None,
        help="Resume from partial results file",
    )
    p.add_argument("--provider", help="LLM provider override")
    p.add_argument("--root-model", help="Root model override")
    p.add_argument("--sub-model", help="Sub model override")
    p.add_argument("--search", help="Search provider (for retrieval mode)")
    p.add_argument(
        "--max-depth", type=int, default=2,
        help="Max recursion depth for sub_lm calls (default: 2)",
    )
    p.add_argument(
        "--temperature", type=float, default=None,
        help="LLM temperature (default: provider default; use 0 for deterministic runs)",
    )
    p.add_argument(
        "-c", "--config", type=Path, default=None,
        help="Config file path",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


async def main() -> None:
    args = parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if args.verbose:
        fh = logging.FileHandler("debug.log", mode="w")
        fh.setLevel(logging.DEBUG)
        handlers.append(fh)
    logging.basicConfig(
        level=log_level,
        format="%(levelname)s %(name)s: %(message)s",
        handlers=handlers,
    )

    # Check datasets dependency
    try:
        import datasets  # noqa: F401
    except ImportError:
        console.print(
            "[red]Error:[/red] 'datasets' package required.\n"
            "Install with: pip install datasets"
        )
        sys.exit(1)

    # Build config overrides
    overrides: dict = {}
    if args.provider or args.root_model or args.sub_model:
        overrides["llm"] = {}
        if args.provider:
            overrides["llm"]["provider"] = args.provider
        if args.root_model:
            overrides["llm"]["root_model"] = args.root_model
        if args.sub_model:
            overrides["llm"]["sub_model"] = args.sub_model
    if args.search:
        overrides.setdefault("search", {})["provider"] = args.search
    if args.max_depth is not None:
        overrides.setdefault("engine", {})["max_recursion_depth"] = args.max_depth
    if args.temperature is not None:
        overrides.setdefault("llm", {})["temperature"] = args.temperature

    cfg = load_config(config_path=args.config, overrides=overrides or None)

    if not cfg.llm.api_key:
        console.print(
            "[red]Error:[/red] No API key configured.\n"
            "Set RLM_API_KEY or create ~/.rlm-research.yaml"
        )
        sys.exit(1)

    console.print(
        f"\n[bold]FRAMES Benchmark[/bold] — {args.mode} mode, "
        f"limit={args.limit}"
    )
    console.print(
        f"  Root: {cfg.llm.root_model}  Sub: {cfg.llm.sub_model}\n"
    )

    # Progress callback
    correct_count = 0
    done_count = 0

    def on_progress(current: int, total: int, result: dict) -> None:
        nonlocal correct_count, done_count
        done_count = current
        if result.get("correct"):
            correct_count += 1
        acc = correct_count / done_count if done_count else 0
        mark = "[green]✓[/green]" if result.get("correct") else "[red]✗[/red]"
        console.print(
            f"  {mark} [{current}/{total}] "
            f"acc={acc:.1%}  "
            f"{result.get('question', '')[:60]}"
        )

    summary = await run_frames_benchmark(
        config=cfg,
        limit=args.limit,
        mode=args.mode,
        output_path=args.output,
        resume_path=args.resume,
        on_progress=on_progress,
    )

    # Final summary
    agg = summary["aggregate"]
    console.print(
        f"\n[bold]Results:[/bold] {agg['accuracy']:.1%} accuracy "
        f"({agg.get('correct', 0)}/{agg.get('total', 0)})"
    )
    console.print(
        f"  Avg turns: {agg.get('avg_turns', 0)}  "
        f"Avg time: {agg.get('avg_duration_seconds', 0):.0f}s"
    )

    if agg.get("by_reasoning_type"):
        console.print("\n  [bold]By reasoning type:[/bold]")
        for rt, stats in sorted(agg["by_reasoning_type"].items()):
            n = f"{stats['correct']}/{stats['count']}"
            console.print(f"    {rt}: {stats['accuracy']:.1%} ({n})")

    console.print(f"\n  Saved to: {summary['output_path']}\n")


if __name__ == "__main__":
    asyncio.run(main())
