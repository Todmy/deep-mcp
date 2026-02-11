"""Source loaders — file://, https://, directory trees → REPL variables."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import httpx

log = logging.getLogger(__name__)

# Max file size to load fully into memory (10MB)
MAX_FILE_SIZE = 10 * 1024 * 1024

# Max text to keep per URL source (20K chars ≈ 5K tokens).
# Wikipedia and similar sites front-load key facts, so truncating
# keeps the most useful content while preventing context explosion.
MAX_URL_TEXT = 20_000


async def load_source(source: str) -> tuple[str, Any]:
    """Load a single source into a (var_name, value) pair for REPL namespace.

    Supported schemes:
      file:///path/to/file.ext  → text content
      file:///path/to/dir/      → {"_tree": str, "_files": dict, "_stats": str}
      https://example.com       → extracted text
    """
    if source.startswith("file://"):
        path = Path(source.removeprefix("file://"))
        return _load_file_source(path)
    elif source.startswith("https://") or source.startswith("http://"):
        return await _load_url_source(source)
    elif source == "web://":
        # web:// is a marker for enabling web search, not a loadable source
        return ("_web_enabled", True)
    else:
        # Try as local path
        path = Path(source)
        if path.exists():
            return _load_file_source(path)
        raise ValueError(f"Unknown source format: {source}")


async def load_sources(sources: list[str]) -> dict[str, Any]:
    """Load all sources in parallel, returning a dict for REPL namespace injection."""
    import asyncio

    tasks = [load_source(source) for source in sources]
    loaded = await asyncio.gather(*tasks, return_exceptions=True)

    result: dict[str, Any] = {}
    for i, item in enumerate(loaded):
        if isinstance(item, Exception):
            log.error("Failed to load source %s: %s", sources[i], item)
            result[f"doc_{i}_error"] = f"Failed to load: {item}"
            continue

        name, value = item
        if name == "_web_enabled":
            result[name] = value
            continue
        if isinstance(value, dict) and "_files" in value:
            result[f"doc_{i}_tree"] = value["_tree"]
            result[f"doc_{i}_files"] = value["_files"]
            result[f"doc_{i}_stats"] = value["_stats"]
        else:
            result[f"doc_{i}"] = value
    return result


def _load_file_source(path: Path) -> tuple[str, Any]:
    """Load a local file or directory."""
    if path.is_dir():
        return (path.name, _load_directory(path))
    return (path.stem, _load_single_file(path))


def _load_single_file(path: Path) -> str:
    """Load a single file as text. PDF/DOCX use optional dependencies."""
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        return _load_pdf(path)
    elif suffix == ".docx":
        return _load_docx(path)
    else:
        # Text-based files: md, txt, csv, json, py, etc.
        if path.stat().st_size > MAX_FILE_SIZE:
            log.warning("File %s exceeds %dMB, loading first %dMB", path, MAX_FILE_SIZE // 1024 // 1024, MAX_FILE_SIZE // 1024 // 1024)
            return path.read_text(errors="replace")[:MAX_FILE_SIZE]
        return path.read_text(errors="replace")


def _load_directory(path: Path) -> dict[str, Any]:
    """Load directory as tree + file contents dict."""
    tree_lines = []
    files: dict[str, str] = {}
    total_lines = 0

    for file_path in sorted(path.rglob("*")):
        if file_path.is_dir():
            continue
        # Skip hidden files, __pycache__, node_modules, .git
        parts = file_path.relative_to(path).parts
        if any(p.startswith(".") or p == "__pycache__" or p == "node_modules" for p in parts):
            continue

        rel = str(file_path.relative_to(path))
        tree_lines.append(rel)

        try:
            content = file_path.read_text(errors="replace")
            files[rel] = content
            total_lines += content.count("\n") + 1
        except Exception as e:
            files[rel] = f"(error reading: {e})"

    return {
        "_tree": "\n".join(tree_lines),
        "_files": files,
        "_stats": f"{len(files)} files, {total_lines} lines total",
    }


def _load_pdf(path: Path) -> str:
    """Load PDF using pymupdf (optional dependency)."""
    try:
        import pymupdf  # noqa: F811
    except ImportError:
        raise ImportError("PDF support requires pymupdf: pip install rlm-research-mcp[pdf]")

    doc = pymupdf.open(str(path))
    pages = []
    for page in doc:
        text = page.get_text()
        if text.strip():
            pages.append(f"--- Page {page.number + 1} ---\n{text}")
    doc.close()
    return "\n\n".join(pages)


def _load_docx(path: Path) -> str:
    """Load DOCX using python-docx (optional dependency)."""
    try:
        import docx  # noqa: F811
    except ImportError:
        raise ImportError("DOCX support requires python-docx: pip install rlm-research-mcp[docx]")

    doc = docx.Document(str(path))
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n\n".join(paragraphs)


async def _load_url_source(url: str) -> tuple[str, str]:
    """Fetch URL and extract text content."""
    headers = {"User-Agent": "rlm-research/0.1 (research bot; +https://github.com)"}
    async with httpx.AsyncClient(timeout=30, follow_redirects=True, headers=headers) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        html = resp.text

    # Try trafilatura for content extraction, fall back to raw text
    try:
        import trafilatura
        extracted = trafilatura.extract(html)
        if extracted:
            return ("url", extracted[:MAX_URL_TEXT])
    except ImportError:
        pass

    # Fallback: strip tags naively
    import re
    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"\s+", " ", text).strip()
    return ("url", text[:MAX_URL_TEXT])


def generate_sources_summary(loaded: dict[str, Any]) -> str:
    """Generate a human-readable summary of loaded sources for Root LM."""
    lines = ["Available data:"]
    for key, value in loaded.items():
        if key.startswith("_"):
            continue
        if key.endswith("_error"):
            lines.append(f"- {key}: LOAD ERROR — {value}")
        elif isinstance(value, str):
            lines.append(f"- {key}: text ({len(value)} chars)")
        elif isinstance(value, dict):
            lines.append(f"- {key}: dict ({len(value)} entries)")
    return "\n".join(lines)
