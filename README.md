# RLM Research (Learning Project)

An educational reimplementation of [Recursive Language Models](https://arxiv.org/abs/2512.24601) (RLM) — the inference paradigm where LLMs write and run code to analyze data, recursively delegating sub-tasks to fresh LLM instances.

> **This is not a novel tool.** It is a learning project that attempts to reproduce the RLM approach described in the paper by Alex L. Zhang, Tim Kraska, and Omar Khattab (MIT CSAIL / Stanford). For production use, see the [official implementation](https://github.com/alexzhang13/rlm) by the paper authors (~2.4k stars, MIT license, actively maintained).

## What is RLM?

Traditional LLMs process input as a flat token sequence. RLM changes this: instead of feeding data into the context window, it stores data as Python variables in a REPL environment. The LLM then writes code to inspect, slice, and transform the data, calling `sub_lm()` to recursively delegate sub-questions to fresh model instances.

This allows handling inputs far exceeding context window limits (10M+ tokens) through systematic decomposition.

## How this project differs from the reference implementations

| | [alexzhang13/rlm](https://github.com/alexzhang13/rlm) | [ysz/recursive-llm](https://github.com/ysz/recursive-llm) | This project |
|---|---|---|---|
| Purpose | Reference implementation | Alternative implementation | Learning exercise |
| Sandbox | Docker, Modal, local | RestrictedPython | Local only |
| LLM providers | OpenAI, Anthropic, OpenRouter, Portkey | 100+ via LiteLLM | OpenRouter (OpenAI-compatible) |
| Interface | Python API + visualization | Python API | CLI + MCP server |
| Maturity | Production-capable | Stable | Experimental |

## Known limitations

- No sandboxed code execution — not suitable for untrusted inputs
- FRAMES benchmark accuracy (~40-45% closed mode) is below the paper's reported results
- Budget tuning (max turns, sub_lm call limits) significantly affects quality
- Stuck detection heuristics need further work
- Single provider support (OpenAI-compatible API only)

## Project structure

```
src/rlm_research/
  engine.py    — core code-observe loop with recursive sub_lm calls
  repl.py      — Python REPL with variable injection
  llm.py       — LLM client (OpenAI-compatible API)
  config.py    — YAML + env vars + CLI overrides configuration
  loaders.py   — file/URL/web source loading
  search.py    — web search providers (Tavily, Brave, SearxNG)
  cli.py       — typer CLI interface
  server.py    — MCP server (deep_research tool)
  benchmark.py — FRAMES benchmark runner
  judge.py     — LLM-as-judge for answer evaluation
  report.py    — markdown report generation
```

## Quick start

```bash
pip install -e .
export RLM_API_KEY=sk-...  # OpenRouter API key

# CLI
rlm-research run "analyze this document" --source doc.pdf

# MCP server
rlm-research-mcp
```

## Benchmark

```bash
pip install datasets
python scripts/run_frames.py --mode closed --limit 20 --max-depth 1 --temperature 0 -v
```

## References

- [Recursive Language Models](https://arxiv.org/abs/2512.24601) — original paper
- [alexzhang13/rlm](https://github.com/alexzhang13/rlm) — official implementation by paper authors
- [ysz/recursive-llm](https://github.com/ysz/recursive-llm) — alternative implementation
- [FRAMES benchmark](https://huggingface.co/datasets/google/frames-benchmark) — evaluation dataset

## License

MIT
