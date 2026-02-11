"""CLI interface — typer app for rlm-research command."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.live import Live
from rich.spinner import Spinner

from rlm_research.config import load_config
from rlm_research.engine import ProgressEvent, run_rlm
from rlm_research.llm import LLMClient
from rlm_research.loaders import load_sources
from rlm_research.report import generate_report
from rlm_research.search import close_search_provider, create_search_provider

app = typer.Typer(
    name="rlm-research",
    help="Deep research over documents, code, and web using RLM recursive decomposition.",
    no_args_is_help=True,
)
console = Console()


@app.command()
def run(
    query: Annotated[str, typer.Argument(help="Research question or analysis request")],
    source: Annotated[
        list[str], typer.Option("--source", "-s", help="Source: file path, URL, or web://")
    ] = [],
    depth: Annotated[
        Optional[int], typer.Option("--depth", "-d", help="Max recursion depth (1-10)")
    ] = None,
    provider: Annotated[
        Optional[str], typer.Option("--provider", "-p", help="LLM provider")
    ] = None,
    root_model: Annotated[
        Optional[str], typer.Option("--root-model", help="Root LM model name")
    ] = None,
    sub_model: Annotated[
        Optional[str], typer.Option("--sub-model", help="Sub-LM model name")
    ] = None,
    search: Annotated[
        Optional[str], typer.Option("--search", help="Search provider (tavily/brave/searxng)")
    ] = None,
    config_file: Annotated[
        Optional[Path], typer.Option("--config", "-c", help="Config file path")
    ] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Verbose output")] = False,
) -> None:
    """Run deep research on given sources."""
    if verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.WARNING)

    # Build overrides from CLI args
    overrides: dict = {}
    if provider or root_model or sub_model:
        overrides["llm"] = {}
        if provider:
            overrides["llm"]["provider"] = provider
        if root_model:
            overrides["llm"]["root_model"] = root_model
        if sub_model:
            overrides["llm"]["sub_model"] = sub_model
    if depth:
        overrides.setdefault("engine", {})["max_recursion_depth"] = depth
    if search:
        overrides.setdefault("search", {})["provider"] = search

    cfg = load_config(config_path=config_file, overrides=overrides or None)

    if not cfg.llm.api_key:
        console.print(
            "[red]Error:[/red] No API key configured.\n"
            "Set RLM_API_KEY environment variable or create ~/.rlm-research.yaml"
        )
        raise typer.Exit(1)

    # Resolve sources: plain paths → file:// URIs
    resolved_sources = []
    for s in source:
        if s.startswith(("file://", "http://", "https://", "web://")):
            resolved_sources.append(s)
        else:
            path = Path(s).resolve()
            if path.exists():
                resolved_sources.append(f"file://{path}")
            else:
                console.print(f"[yellow]Warning:[/yellow] Source not found: {s}")

    if not resolved_sources:
        console.print("[red]Error:[/red] No valid sources provided. Use --source to specify.")
        raise typer.Exit(1)

    # Run async research
    asyncio.run(_run_research(query, resolved_sources, cfg))


async def _run_research(query: str, sources: list[str], cfg) -> None:
    """Async entry point for research execution."""
    # Load sources
    with console.status("Loading sources..."):
        loaded = await load_sources(sources)
        web_enabled = loaded.pop("_web_enabled", False)

    # Create clients
    llm = LLMClient(cfg)
    search_provider = None
    if web_enabled:
        search_provider = create_search_provider(cfg.search)

    # Progress display
    spinner_text = "Analyzing..."

    def on_progress(event: ProgressEvent) -> None:
        nonlocal spinner_text
        spinner_text = f"[{event.stage}] {event.message}"
        if event.stats.get("elapsed"):
            elapsed = event.stats["elapsed"]
            spinner_text += f" ({elapsed:.0f}s)"

    # Run engine with live progress
    try:
        with Live(Spinner("dots", text=spinner_text), console=console, refresh_per_second=4):
            result = await run_rlm(
                query=query,
                sources=loaded,
                config=cfg,
                llm=llm,
                search_provider=search_provider,
                on_progress=on_progress,
            )
    finally:
        await close_search_provider(search_provider)

    # Generate and print report
    report = generate_report(
        query=query,
        content=result.content,
        stats=result.stats,
        sources=sources,
        call_tree=result.call_tree,
    )
    console.print()
    console.print(report)


@app.command()
def setup(
    config_file: Annotated[
        Optional[Path], typer.Option("--config", "-c", help="Config file path")
    ] = None,
) -> None:
    """Show current configuration status."""
    cfg = load_config(config_path=config_file)

    console.print("\n[bold]RLM Research Configuration[/bold]\n")

    # LLM
    if cfg.llm.api_key:
        console.print(f"  LLM: [green]{cfg.llm.provider}[/green]")
        console.print(f"    Root model: {cfg.llm.root_model}")
        console.print(f"    Sub model:  {cfg.llm.sub_model}")
    else:
        console.print("  LLM: [red]NOT CONFIGURED[/red]")
        console.print("    Set RLM_API_KEY env var or add api_key to ~/.rlm-research.yaml")

    # Search
    search = create_search_provider(cfg.search)
    if search:
        console.print(f"  Search: [green]{cfg.search.provider}[/green]")
    else:
        console.print("  Search: [dim]not configured (optional)[/dim]")

    # Engine
    console.print(f"\n  Max depth: {cfg.engine.max_recursion_depth}")
    console.print(f"  Max turns: {cfg.engine.max_turns}")
    console.print(f"  Timeout:   {cfg.engine.timeout_per_exec}s")
    console.print()


if __name__ == "__main__":
    app()
