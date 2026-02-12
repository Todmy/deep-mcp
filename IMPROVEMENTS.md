# Improvement Ideas

Potential improvements based on analysis of the codebase. Each originated as a patch concept during development. None are applied — they require validation against FRAMES benchmark before adoption.

**Current baseline:** ~40-45% accuracy on FRAMES (closed mode, 20 questions, max-depth 1).

---

## 1. Source metadata tracking

**Files:** `loaders.py`, `benchmark.py`

Track which REPL variable came from which source URL/file. Store as `_source_meta` mapping (e.g., `doc_0 → https://...`). Also adds "closed" benchmark mode (no sources — model uses own knowledge).

**Status:** Partially implemented (closed mode works, metadata tracking reverted due to accuracy regression).

**Risk:** Increases prompt size, which degraded benchmark scores in testing.

---

## 2. Separate root/sub system prompts

**Files:** `engine.py`

Give different behavioral instructions to root (depth=0) vs sub-agents (depth>0). Root gets orchestrator guidance: when to delegate, cost awareness (~30s per sub_lm call), variable storage patterns. Sub gets focused analyst role: answer the specific question directly.

**Status:** Tested and reverted. The RLM paper uses a single prompt for both roles — difference comes from model selection, not instructions. Benchmark showed no improvement.

**Risk:** Medium. Directly changes LLM behavior. "Finish promptly" instruction in sub prompt caused premature termination in testing (Q5 regression: 10→4 turns, correct→wrong).

---

## 3. Synthesis turn on budget exhaustion

**Files:** `engine.py`

When the turn budget runs out without `answer["ready"] = True`, inject one final message asking the LLM to synthesize all partial findings into a coherent answer.

**Status:** Implemented in current code. Dead code when budgets are unlimited (model always finishes before sys.maxsize turns). Becomes active with `--max-turns` flag.

**Risk:** Low. Only triggers on failure path. Adds one extra LLM call.

---

## 4. Persistent HTTP clients

**Files:** `search.py`, `server.py`, `cli.py`

Replace per-request `httpx.AsyncClient()` with persistent instances. Reuse TCP/TLS connections across search API calls within a session. Add proper cleanup via `close()` methods.

**Status:** Not applied.

**Risk:** Low. Same observable behavior, just connection pooling. Minor: global `_fetch_client` singleton lacks explicit cleanup in long-running server contexts.

---

## 5. Rate-limit concurrent sub_lm calls

**Files:** `engine.py`

Add `asyncio.Semaphore(5)` shared across all recursive depths. Prevents API rate limiting (429s) when `llm_batch` fires many parallel requests.

**Status:** Not applied.

**Risk:** Low. Only throttles concurrency. Hardcoded limit of 5 may be suboptimal for some providers. Could increase wall-clock time for large batches.

---

## 6. Thread-safe stdout capture

**Files:** `repl.py`

Replace global `sys.stdout` redirection with per-execution `print()` override. Fixes race condition where parallel `llm_batch` executions clobber each other's output capture. Adds 5000-char output truncation.

**Status:** Not applied.

**Risk:** Low. Strictly better for parallel use. Only `print()` is captured — direct `sys.stdout.write()` in LLM-generated code would be missed (acceptable trade-off).

---

## 7. Sub-LM result caching

**Files:** `engine.py`

In-memory cache keyed by `MD5("{depth}:{prompt}")`. Same sub-question at same depth returns cached result, skipping the full recursive `run_rlm` call.

**Status:** Not applied. Depends on idea #5 (uses shared concurrency parameter).

**Risk:** Low. Cache miss path unchanged. No TTL or invalidation — acceptable for single-session use. Conservative keying (includes depth).

---

## 8. Remove 404 from retryable errors

**Files:** `llm.py`

HTTP 404 means the endpoint doesn't exist — retrying is pointless. Remove from retry set, keep only transient errors (429, 500, 502, 503, 504).

**Status:** Not applied.

**Risk:** Low. Single line change, clearly correct.

---

## 9. Source previews in initial prompt

**Files:** `repl.py`

Include first 2000 chars of each source in the initial `sources_summary()`. Lets the LLM answer simple questions directly from previews without writing code to inspect data first.

**Status:** Not applied. Depends on idea #1 (source metadata).

**Risk:** Medium. Significantly increases initial prompt size. For queries with many sources, could consume substantial context window. Trade-off: fewer turns vs larger prompts.

---

## Dependency graph

```
#1 (source metadata) ← #9 (source previews)
#5 (rate limiting)   ← #7 (sub-LM caching)
```

All others are independent.

## Recommended application order

Low risk, high confidence (apply first):
1. **#8** — Remove 404 retry (trivial bug fix)
2. **#6** — Thread-safe stdout (fixes real race condition)
3. **#4** — Persistent HTTP clients (free performance)
4. **#5** — Rate-limit concurrency (reliability)

Medium risk, needs benchmark validation:
5. **#3** — Synthesis turn (already in code, needs budget limits to activate)
6. **#7** — Sub-LM caching (depends on #5)

Higher risk, requires careful testing:
7. **#1 + #9** — Source metadata + previews (increased prompt size may hurt)
8. **#2** — Separate prompts (tested and reverted — needs different approach)
