"""Report generator — markdown output with findings, citations, stats."""

from __future__ import annotations

from typing import Any


def generate_report(
    query: str,
    content: str,
    stats: dict[str, Any],
    sources: list[str],
    call_tree: list[dict[str, Any]] | None = None,
) -> str:
    """Generate a markdown research report from engine results."""
    sections = [
        f"# Research: {query}",
        "",
        "## Findings",
        "",
        content,
        "",
        _format_sources(sources),
        "",
        _format_stats(stats),
    ]

    if call_tree:
        sections.extend(["", _format_call_tree(call_tree)])

    return "\n".join(sections)


def _format_sources(sources: list[str]) -> str:
    if not sources:
        return ""
    lines = ["## Sources"]
    for i, source in enumerate(sources, 1):
        lines.append(f"{i}. {source}")
    return "\n".join(lines)


def _format_stats(stats: dict[str, Any]) -> str:
    lines = ["## Research Stats"]
    if "duration_seconds" in stats:
        duration = stats["duration_seconds"]
        if duration >= 60:
            lines.append(f"- Duration: {int(duration // 60)}m {int(duration % 60)}s")
        else:
            lines.append(f"- Duration: {duration}s")

    if "turns" in stats:
        lines.append(f"- Turns: {stats['turns']}")

    if "sub_lm_calls" in stats:
        lines.append(f"- Sub-LM calls: {stats['sub_lm_calls']}")

    if "tokens_input" in stats or "tokens_output" in stats:
        inp = stats.get("tokens_input", 0)
        out = stats.get("tokens_output", 0)
        lines.append(f"- Tokens: {_format_tokens(inp)} input, {_format_tokens(out)} output")

    if "depth" in stats:
        lines.append(f"- Max depth reached: {stats['depth']}")

    return "\n".join(lines)


def _format_call_tree(call_tree: list[dict[str, Any]]) -> str:
    lines = ["## Call Tree"]
    for entry in call_tree:
        indent = "  " * entry.get("depth", 0)
        prompt = entry.get("prompt", "")[:80]
        duration = entry.get("duration", 0)
        lines.append(f"{indent}- [{duration:.1f}s] {prompt}")
    return "\n".join(lines)


def _format_tokens(n: int) -> str:
    if n >= 1000:
        return f"{n / 1000:.0f}K"
    return str(n)
