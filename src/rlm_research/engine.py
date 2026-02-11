"""RLM engine — code-execute-observe loop with recursive sub_lm calls."""

from __future__ import annotations

import ast
import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from rlm_research.config import Config
from rlm_research.llm import LLMClient
from rlm_research.repl import REPL
from rlm_research.search import SearchProvider, fetch_url

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are an RLM (Recursive Language Model) research agent. You analyze data by writing Python code.

## How you work
- Data is pre-loaded as Python variables in your environment (see "Available data" below)
- Write Python code blocks to inspect, search, slice, and transform the data
- Use `sub_lm(prompt)` to delegate sub-questions to a fresh LLM instance
- Use `llm_batch([prompts])` for parallel sub-queries
- Use `web_search(query, n=5)` if web search is available
- Use `print()` to output intermediate results you want to observe

## Available functions
- `sub_lm(prompt: str) -> str` — recursive LLM call (fresh instance with its own loop)
- `llm_batch(prompts: list[str]) -> list[str]` — parallel sub_lm calls
- `web_search(query: str, n: int = 5) -> list[dict]` — web search (if enabled)
- `fetch_url(url: str) -> str` — fetch and extract text from URL
- Standard Python: re, json, math, collections, itertools, etc.

## Rules
1. Write ONE Python code block per message (```python ... ```)
2. After each execution, you'll see the output — then write the next code block
3. Set `answer["content"] = "your findings"` with your analysis
4. Set `answer["ready"] = True` when you're done
5. Be systematic: explore data structure first, then analyze
6. Cite sources in your findings: [source: variable_name] or [source: url]

## Important
- Variables persist between code blocks (same REPL session)
- Modules re, json, math, collections, itertools, functools, textwrap, difflib, statistics, string, datetime are pre-loaded — use them directly without importing
- Keep code blocks focused — one logical step per block
- NEVER embed raw document content into strings — reference variables directly
- Pass data to sub_lm via f-strings: `sub_lm(f"Analyze: {doc_0[:500]}")`
- NEVER put markdown backtick fences (```) inside Python strings
"""

# Regex patterns for code block extraction
_FENCE_OPEN = re.compile(r"```python\s*\n")
_FENCE_CLOSE = re.compile(r"```")

# Stuck detection: if answer hasn't changed in this many turns, inject hint
STUCK_THRESHOLD = 5
STUCK_FORCE_STOP = 8


@dataclass
class ProgressEvent:
    stage: str  # loading | decomposing | analyzing | synthesizing | done
    message: str
    depth: int
    turn: int
    stats: dict[str, Any] = field(default_factory=dict)


@dataclass
class EngineResult:
    content: str
    stats: dict[str, Any]
    call_tree: list[dict[str, Any]]


def extract_code_block(response: str) -> str | None:
    """Extract Python code block from LLM response.

    LLMs often embed markdown backtick fences inside generated Python code
    (e.g., in string literals passed to sub_lm). A naive non-greedy regex
    stops at the first ```, truncating the code. Instead, we find all possible
    closing fences and try from the last one backwards, validating with AST.
    """
    start = _FENCE_OPEN.search(response)
    if not start:
        return None

    remaining = response[start.end():]
    fence_positions = [m.start() for m in _FENCE_CLOSE.finditer(remaining)]

    if not fence_positions:
        return None

    # Try from last fence to first — first valid parse wins
    for pos in reversed(fence_positions):
        candidate = remaining[:pos].strip()
        if not candidate:
            continue
        try:
            ast.parse(candidate)
            return candidate
        except SyntaxError:
            continue

    # Nothing parsed — return shortest candidate so REPL reports the actual error
    return remaining[:fence_positions[0]].strip() or None


async def run_rlm(
    query: str,
    sources: dict[str, Any],
    config: Config,
    llm: LLMClient,
    search_provider: SearchProvider | None = None,
    depth: int = 0,
    on_progress: Callable[[ProgressEvent], None] | None = None,
) -> EngineResult:
    """Run the RLM code-execute-observe loop.

    This is the core of the system. Works identically for root and sub-LM calls —
    the only difference is the model (reasoning vs cheap) selected by depth.
    """
    start_time = time.time()
    call_tree: list[dict[str, Any]] = []
    sub_lm_count = 0

    # --- Async-to-sync bridge ---
    # REPL runs code synchronously (via Python exec) but sub_lm/search are async.
    # We use run_coroutine_threadsafe to schedule async work on the running loop.
    _loop = asyncio.get_running_loop()

    def _run_async(coro):
        """Run an async coroutine from synchronous REPL context."""
        future = asyncio.run_coroutine_threadsafe(coro, _loop)
        return future.result()

    # --- Wire up sub_lm as a recursive call back into run_rlm ---
    def _make_sub_lm():
        def sub_lm(prompt: str) -> str:
            nonlocal sub_lm_count
            if depth >= config.engine.max_recursion_depth:
                return f"(max recursion depth {config.engine.max_recursion_depth} reached)"
            if sub_lm_count >= config.engine.max_sub_lm_calls:
                return f"(max sub_lm calls {config.engine.max_sub_lm_calls} reached)"

            sub_lm_count += 1
            sub_start = time.time()
            result = _run_async(run_rlm(
                query=prompt, sources=sources, config=config, llm=llm,
                search_provider=search_provider, depth=depth + 1,
                on_progress=on_progress,
            ))
            call_tree.append({
                "depth": depth + 1,
                "prompt": prompt[:200],
                "result": result.content[:200],
                "duration": time.time() - sub_start,
            })
            return result.content
        return sub_lm

    def _make_llm_batch():
        def llm_batch(prompts: list[str]) -> list[str]:
            async def _run_all():
                tasks = [
                    run_rlm(
                        query=p, sources=sources, config=config, llm=llm,
                        search_provider=search_provider, depth=depth + 1,
                        on_progress=on_progress,
                    )
                    for p in prompts
                ]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                return [
                    r.content if isinstance(r, EngineResult) else f"(error: {r})"
                    for r in results
                ]
            return _run_async(_run_all())
        return llm_batch

    # --- Wire up search functions ---
    sync_search_fn = None
    sync_fetch_fn = None
    if search_provider:
        def sync_search(query: str, n: int = 5) -> list[dict]:
            async def _search():
                results = await search_provider.search(query, n)
                return [{"title": r.title, "url": r.url, "snippet": r.snippet} for r in results]
            return _run_async(_search())
        sync_search_fn = sync_search

        def sync_fetch(url: str) -> str:
            return _run_async(fetch_url(url))
        sync_fetch_fn = sync_fetch

    # --- Create REPL with wired functions ---
    repl = REPL(
        sources=sources,
        sub_lm_fn=_make_sub_lm(),
        llm_batch_fn=_make_llm_batch(),
        search_fn=sync_search_fn,
        fetch_url_fn=sync_fetch_fn,
    )

    # --- Build initial messages ---
    sources_summary = repl.sources_summary()
    messages: list[dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Query: {query}\n\n{sources_summary}"},
    ]

    # --- Main loop: code → execute → observe ---
    last_answer_content = None
    stuck_count = 0
    turn = 0

    if on_progress:
        on_progress(ProgressEvent(
            stage="analyzing", message=f"Starting analysis (depth {depth})",
            depth=depth, turn=0,
        ))

    for turn in range(config.engine.max_turns):
        # 1. LLM generates response
        response = await llm.generate(messages, depth=depth)
        code = extract_code_block(response.content)

        if not code:
            # LLM didn't produce code — might be reasoning or final answer
            # Check if it set answer in a previous turn
            if repl.get("answer", {}).get("ready"):
                break
            # Append response and ask for code
            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": "Please write a Python code block to continue your analysis."})
            continue

        # 2. Execute code in REPL
        exec_result = repl.execute(code, timeout=config.engine.timeout_per_exec)

        # 3. Append to conversation
        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": f"Execution result:\n{exec_result}"})

        # 4. Check completion
        answer = repl.get("answer", {})
        if answer.get("ready"):
            if on_progress:
                on_progress(ProgressEvent(
                    stage="done", message="Analysis complete",
                    depth=depth, turn=turn + 1,
                ))
            break

        # 5. Stuck detection
        current_content = answer.get("content")
        if current_content == last_answer_content:
            stuck_count += 1
        else:
            stuck_count = 0
            last_answer_content = current_content

        if stuck_count >= STUCK_FORCE_STOP:
            log.warning("Force stopping at depth %d — stuck for %d turns", depth, stuck_count)
            break
        elif stuck_count >= STUCK_THRESHOLD:
            messages.append({"role": "user", "content": (
                "Hint: You seem stuck. Try a different approach, or if you have enough "
                "information, set answer['content'] and answer['ready'] = True."
            )})

        # 6. Progress
        if on_progress and turn % 3 == 0:
            on_progress(ProgressEvent(
                stage="analyzing",
                message=f"Turn {turn + 1}/{config.engine.max_turns} (depth {depth})",
                depth=depth, turn=turn + 1,
                stats={
                    "llm_calls_total": llm.cumulative_usage.total,
                    "sub_lm_calls": sub_lm_count,
                    "elapsed": time.time() - start_time,
                },
            ))

    # --- Collect result ---
    answer = repl.get("answer", {})
    content = answer.get("content", "Analysis incomplete — budget exhausted.")
    elapsed = time.time() - start_time

    return EngineResult(
        content=content,
        stats={
            "duration_seconds": round(elapsed, 1),
            "turns": turn + 1,
            "depth": depth,
            "sub_lm_calls": sub_lm_count,
            "tokens_input": llm.cumulative_usage.input_tokens,
            "tokens_output": llm.cumulative_usage.output_tokens,
        },
        call_tree=call_tree,
    )
