"""MCP server entry point — exposes deep_research and rlm_setup tools."""

from __future__ import annotations

import logging

from mcp.server.fastmcp import FastMCP

from rlm_research.config import load_config
from rlm_research.engine import run_rlm
from rlm_research.llm import LLMClient
from rlm_research.loaders import load_sources
from rlm_research.report import generate_report
from rlm_research.search import create_search_provider

log = logging.getLogger(__name__)

mcp = FastMCP(
    name="rlm-research",
    instructions=(
        "Deep research over documents, code, and web using RLM recursive decomposition. "
        "Use the deep_research tool for analysis tasks that need systematic exploration."
    ),
)


@mcp.tool()
async def deep_research(
    query: str,
    sources: list[str],
    config: dict | None = None,
) -> str:
    """Deep research over documents, code, and web using RLM recursive decomposition.

    Handles inputs of any size (10M+ tokens) by decomposing analysis recursively.

    Args:
        query: Research question or analysis request.
        sources: List of sources — file:// for local files/dirs,
                 https:// for URLs, web:// to enable web search.
        config: Optional overrides (e.g. {"engine": {"max_recursion_depth": 2}}).
    """
    # Load config with optional overrides from tool params
    cfg = load_config(overrides=config)

    if not cfg.llm.api_key:
        return (
            "Error: No API key configured. Set RLM_API_KEY environment variable "
            "or create ~/.rlm-research.yaml with llm.api_key."
        )

    # Load sources into REPL variables
    loaded = await load_sources(sources)
    web_enabled = loaded.pop("_web_enabled", False)

    # Create LLM client and search provider
    llm = LLMClient(cfg)
    search_provider = None
    if web_enabled:
        search_provider = create_search_provider(cfg.search)

    # Run RLM engine
    result = await run_rlm(
        query=query,
        sources=loaded,
        config=cfg,
        llm=llm,
        search_provider=search_provider,
    )

    # Generate report
    report = generate_report(
        query=query,
        content=result.content,
        stats=result.stats,
        sources=sources,
        call_tree=result.call_tree,
    )

    return report


@mcp.tool()
async def rlm_setup() -> str:
    """Re-run the RLM Research configuration setup.

    Checks current config and reports what needs to be configured.
    """
    cfg = load_config()

    status_lines = ["# RLM Research Configuration Status", ""]

    # LLM
    if cfg.llm.api_key:
        status_lines.append(f"- LLM: {cfg.llm.provider} (configured)")
        status_lines.append(f"  - Root model: {cfg.llm.root_model}")
        status_lines.append(f"  - Sub model: {cfg.llm.sub_model}")
    else:
        status_lines.append("- LLM: NOT CONFIGURED")
        status_lines.append("  - Set RLM_API_KEY env var or add api_key to ~/.rlm-research.yaml")

    # Search
    search = create_search_provider(cfg.search)
    if search:
        status_lines.append(f"- Search: {cfg.search.provider} (configured)")
    else:
        status_lines.append("- Search: not configured (optional)")

    # Engine
    status_lines.extend([
        "",
        "## Engine Settings",
        f"- Max recursion depth: {cfg.engine.max_recursion_depth}",
        f"- Max turns: {cfg.engine.max_turns or 'unlimited'}",
        f"- Max sub-LM calls: {cfg.engine.max_sub_lm_calls or 'unlimited'}",
        f"- Timeout per execution: {cfg.engine.timeout_per_exec}s",
    ])

    return "\n".join(status_lines)


def main() -> None:
    """Entry point for MCP server — runs with stdio transport."""
    logging.basicConfig(level=logging.INFO)
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
