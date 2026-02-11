"""FRAMES benchmark runner — evaluate RLM against google/frames-benchmark."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rlm_research.config import Config
from rlm_research.judge import judge_answer
from rlm_research.llm import LLMClient
from rlm_research.loaders import load_sources
from rlm_research.search import create_search_provider

log = logging.getLogger(__name__)

DATASET_ID = "google/frames-benchmark"


def _load_dataset(limit: int | None = None) -> list[dict]:
    """Load FRAMES benchmark from HuggingFace.

    Returns list of dicts with keys: Prompt, Answer, wiki_links, Reasoning_Type.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "The 'datasets' package is required for benchmarking.\n"
            "Install it with: pip install 'rlm-research-mcp[bench]'"
        )

    ds = load_dataset(DATASET_ID, split="test")
    items = [dict(row) for row in ds]

    if limit is not None:
        items = items[:limit]

    return items


def _load_existing_results(path: Path) -> dict:
    """Load partial results file for resuming."""
    if not path.exists():
        return {}
    with open(path) as f:
        data = json.load(f)
    # Index by question text for stable resume across different --limit values
    return {r["question"]: r for r in data.get("results", [])}


def _save_results(path: Path, meta: dict, aggregate: dict, results: list[dict]) -> None:
    """Save results JSON atomically."""
    data = {"meta": meta, "aggregate": aggregate, "results": results}
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    tmp.rename(path)


def _compute_aggregate(results: list[dict]) -> dict:
    """Compute aggregate stats from individual results."""
    if not results:
        return {"accuracy": 0.0, "by_reasoning_type": {}, "avg_turns": 0, "avg_duration_seconds": 0}

    correct = sum(1 for r in results if r.get("correct"))
    total = len(results)

    # Group by reasoning type
    by_type: dict[str, dict] = {}
    for r in results:
        rt = r.get("reasoning_type", "unknown")
        if rt not in by_type:
            by_type[rt] = {"correct": 0, "count": 0}
        by_type[rt]["count"] += 1
        if r.get("correct"):
            by_type[rt]["correct"] += 1

    for rt, stats in by_type.items():
        stats["accuracy"] = round(stats["correct"] / stats["count"], 3) if stats["count"] else 0.0

    # Averages from stats (only for completed results with stats)
    completed = [r for r in results if r.get("stats")]
    n = len(completed) or 1
    avg_turns = sum(r["stats"].get("turns", 0) for r in completed) / n
    avg_duration = sum(
        r["stats"].get("duration_seconds", 0) for r in completed
    ) / n
    total_input = sum(r["stats"].get("tokens_input", 0) for r in completed)
    total_output = sum(r["stats"].get("tokens_output", 0) for r in completed)

    return {
        "accuracy": round(correct / total, 3) if total else 0.0,
        "correct": correct,
        "total": total,
        "by_reasoning_type": by_type,
        "avg_turns": round(avg_turns, 1),
        "avg_duration_seconds": round(avg_duration, 1),
        "total_tokens_input": total_input,
        "total_tokens_output": total_output,
    }


async def _prepare_sources(item: dict, mode: str) -> list[str]:
    """Build source URIs for a single benchmark question.

    Closed mode: no sources, model uses only its own knowledge.
    Oracle mode: use wiki_links as HTTPS sources.
    Retrieval mode: use web:// to enable live search.
    """
    if mode == "closed":
        return []
    if mode == "retrieval":
        return ["web://"]

    # Oracle mode: load from wiki_links
    wiki_links = item.get("wiki_links", [])
    if isinstance(wiki_links, str):
        # Dataset stores as stringified Python list: "['url1', 'url2']"
        # Replace single quotes with double quotes for JSON parsing
        try:
            wiki_links = json.loads(wiki_links.replace("'", '"'))
        except (json.JSONDecodeError, TypeError):
            wiki_links = [wiki_links] if wiki_links else []

    if not wiki_links:
        log.warning(
            "Question %s has no wiki_links, falling back to web://",
            item.get("Prompt", "")[:50],
        )
        return ["web://"]

    return [url for url in wiki_links if isinstance(url, str) and url.startswith("http")]


async def run_frames_benchmark(
    config: Config,
    limit: int | None = None,
    mode: str = "oracle",
    output_path: Path | None = None,
    resume_path: Path | None = None,
    on_progress: Any = None,
) -> dict:
    """Run FRAMES benchmark evaluation.

    Args:
        config: RLM configuration
        limit: Max questions to evaluate (None = all 824)
        mode: "oracle" (use wiki_links) or "retrieval" (web search)
        output_path: Where to save results JSON
        resume_path: Path to partial results to resume from
        on_progress: Optional callback(current, total, result_dict)
    """
    from rlm_research.engine import run_rlm

    # Load dataset
    items = _load_dataset(limit)
    total = len(items)

    # Resume support
    existing: dict[str, dict] = {}
    if resume_path:
        existing = _load_existing_results(resume_path)
        log.info("Resuming: found %d existing results", len(existing))

    # Output path
    if output_path is None:
        ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        output_path = Path(f"bench_results_{ts}.json")

    # Meta
    meta = {
        "timestamp": datetime.now(UTC).isoformat(),
        "root_model": config.llm.root_model,
        "sub_model": config.llm.sub_model,
        "provider": config.llm.provider,
        "mode": mode,
        "total_questions": total,
        "dataset": DATASET_ID,
    }

    # Create LLM client (shared across all questions — cumulative tracking)
    llm = LLMClient(config)

    # Collect results
    results: list[dict] = list(existing.values())

    for idx, item in enumerate(items):
        question = item.get("Prompt", "")

        # Skip if already done (resume — matched by question text)
        if question in existing:
            if on_progress:
                on_progress(idx + 1, total, existing[question])
            continue
        gold = item.get("Answer", "")
        reasoning_type = item.get("reasoning_types", "unknown")

        log.info("[%d/%d] %s", idx + 1, total, question[:80])

        result_entry: dict = {
            "id": idx,
            "question": question,
            "gold_answer": gold,
            "reasoning_type": reasoning_type,
        }

        try:
            # Prepare sources
            source_uris = await _prepare_sources(item, mode)
            loaded = await load_sources(source_uris)
            web_enabled = loaded.pop("_web_enabled", False)

            # Search provider (for retrieval mode)
            search_provider = None
            if web_enabled:
                search_provider = create_search_provider(config.search)

            # Snapshot token counts before this question
            tokens_before_in = llm.cumulative_usage.input_tokens
            tokens_before_out = llm.cumulative_usage.output_tokens

            # Run RLM
            engine_result = await run_rlm(
                query=question,
                sources=loaded,
                config=config,
                llm=llm,
                search_provider=search_provider,
            )

            predicted = engine_result.content

            # Judge
            verdict = await judge_answer(
                predicted=predicted,
                gold=gold,
                question=question,
                llm=llm,
            )

            # Per-question token delta (includes both RLM + judge)
            tokens_in = llm.cumulative_usage.input_tokens - tokens_before_in
            tokens_out = llm.cumulative_usage.output_tokens - tokens_before_out

            result_entry.update({
                "predicted_answer": predicted[:5000],  # Cap for JSON size
                "correct": verdict["correct"],
                "judge_reasoning": verdict["reasoning"],
                "stats": {
                    **engine_result.stats,
                    "tokens_input": tokens_in,
                    "tokens_output": tokens_out,
                },
            })

        except Exception as exc:
            log.error("Error on question %d: %s", idx, exc, exc_info=True)
            result_entry.update({
                "predicted_answer": "",
                "correct": False,
                "judge_reasoning": f"Error: {exc}",
                "stats": {},
                "error": str(exc),
            })

        results.append(result_entry)

        # Save intermediate (resumable)
        aggregate = _compute_aggregate(results)
        _save_results(output_path, meta, aggregate, results)

        if on_progress:
            on_progress(idx + 1, total, result_entry)

    # Final aggregate
    aggregate = _compute_aggregate(results)
    _save_results(output_path, meta, aggregate, results)

    return {"meta": meta, "aggregate": aggregate, "output_path": str(output_path)}
