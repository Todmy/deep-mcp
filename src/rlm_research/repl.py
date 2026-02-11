"""Python REPL sandbox — restricted namespace, subprocess execution, timeout."""

from __future__ import annotations

import ast
import io
import logging
import signal
import traceback
from typing import Any, Callable

log = logging.getLogger(__name__)

# Modules allowed inside REPL namespace
ALLOWED_MODULES = {
    "re", "json", "math", "collections", "itertools", "functools",
    "textwrap", "difflib", "statistics", "string", "datetime",
}

# Builtins blocked for security — no file writes, no system access
BLOCKED_BUILTINS = {
    "exec", "eval", "compile", "open",
    "breakpoint", "exit", "quit", "input",
}


def _make_safe_import(allowed: set[str]):
    """Create a __import__ that only allows whitelisted modules."""
    real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__

    def safe_import(name, *args, **kwargs):
        root = name.split(".")[0]
        if root not in allowed:
            raise ImportError(f"Import blocked: '{root}' is not in allowed modules")
        return real_import(name, *args, **kwargs)

    return safe_import


def _make_safe_builtins() -> dict[str, Any]:
    """Create a builtins dict with dangerous functions removed and safe __import__."""
    import builtins
    safe = {k: v for k, v in vars(builtins).items() if k not in BLOCKED_BUILTINS}
    safe["__import__"] = _make_safe_import(ALLOWED_MODULES)
    return safe


def _validate_code(code: str) -> str | None:
    """Static check for forbidden patterns. Returns error message or None if OK."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return f"SyntaxError: {e}"

    for node in ast.walk(tree):
        # Block: import os, import sys, import subprocess, etc.
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name.split(".")[0] for alias in node.names]
            elif node.module:
                names = [node.module.split(".")[0]]
            for name in names:
                if name not in ALLOWED_MODULES:
                    return f"Import blocked: '{name}' is not in allowed modules"
    return None


class REPL:
    """Restricted Python execution environment for RLM code-execute-observe loop.

    Sources are pre-loaded as namespace variables. The LLM writes code
    that manipulates these variables, calls sub_lm() for recursion,
    and sets answer["ready"] = True when done.
    """

    def __init__(
        self,
        sources: dict[str, Any] | None = None,
        sub_lm_fn: Callable | None = None,
        llm_batch_fn: Callable | None = None,
        search_fn: Callable | None = None,
        fetch_url_fn: Callable | None = None,
        progress_fn: Callable | None = None,
    ) -> None:
        self.namespace: dict[str, Any] = {
            "__builtins__": _make_safe_builtins(),
            "answer": {},
        }

        # Load allowed stdlib modules
        import re, json, math, collections, itertools, functools  # noqa: E401
        import textwrap, difflib, statistics, string, datetime  # noqa: E401
        self.namespace.update({
            "re": re, "json": json, "math": math,
            "collections": collections, "itertools": itertools,
            "functools": functools, "textwrap": textwrap,
            "difflib": difflib, "statistics": statistics,
            "string": string, "datetime": datetime,
        })

        # Load sources as variables
        if sources:
            self.namespace.update(sources)

        # RLM primitives
        if sub_lm_fn:
            self.namespace["sub_lm"] = sub_lm_fn
        if llm_batch_fn:
            self.namespace["llm_batch"] = llm_batch_fn

        # Convenience functions
        if search_fn:
            self.namespace["web_search"] = search_fn
        if fetch_url_fn:
            self.namespace["fetch_url"] = fetch_url_fn
        if progress_fn:
            self.namespace["report_progress"] = progress_fn

    def get(self, key: str, default: Any = None) -> Any:
        return self.namespace.get(key, default)

    def sources_summary(self) -> str:
        """Generate summary of available data for Root LM prompt.

        Includes source URL and first 2K chars preview for string sources,
        so the model can read data without writing code first.
        """
        source_meta = self.namespace.get("_source_meta", {})
        preview_limit = 2000

        lines = ["Available data:"]
        for key, value in self.namespace.items():
            if key.startswith("_") or key in (
                "answer", "sub_lm", "llm_batch", "web_search",
                "fetch_url", "report_progress", "__builtins__",
                "re", "json", "math", "collections", "itertools",
                "functools", "textwrap", "difflib", "statistics",
                "string", "datetime",
            ):
                continue

            # Source URL metadata
            url_hint = ""
            if key in source_meta:
                url_hint = f" [source: {source_meta[key]}]"

            if isinstance(value, str):
                lines.append(f"- {key}: string ({len(value)} chars){url_hint}")
                # Include preview so model can read without code
                preview = value[:preview_limit]
                if len(value) > preview_limit:
                    preview += f"\n... ({len(value) - preview_limit} more chars in {key})"
                lines.append(f"  Content preview:\n{preview}")
            elif isinstance(value, dict):
                lines.append(f"- {key}: dict ({len(value)} keys){url_hint}")
            elif isinstance(value, list):
                lines.append(f"- {key}: list ({len(value)} items){url_hint}")
            else:
                lines.append(f"- {key}: {type(value).__name__}{url_hint}")

        if "web_search" in self.namespace:
            lines.append("\nTools: web_search(query, n=5): web search enabled")
        if "sub_lm" in self.namespace:
            lines.append("Tools: sub_lm(prompt): recursive LLM call (~30s per call)")
        if "llm_batch" in self.namespace:
            lines.append("Tools: llm_batch([prompts]): parallel sub_lm calls")
        return "\n".join(lines)

    def execute(self, code: str, timeout: int = 300) -> str:  # noqa: S102
        """Execute code in restricted namespace. Returns captured output or error.

        NOTE: This uses Python's exec() builtin intentionally — the REPL is
        the core of RLM's code-execute-observe loop. Security is enforced via:
        1. AST validation (_validate_code) blocks dangerous imports
        2. Restricted builtins (no open, no eval/exec in namespace)
        3. Safe __import__ only allows whitelisted modules
        4. SIGALRM timeout (main thread) or caller-managed timeout (worker thread)
        This is MVP-level sandboxing. Production requires Docker isolation.

        Output capture uses a per-execution print() override in the namespace
        instead of global sys.stdout redirection, making it thread-safe for
        parallel llm_batch executions.
        """
        import threading

        # Static validation
        error = _validate_code(code)
        if error:
            return f"VALIDATION ERROR: {error}"

        # Per-execution output buffer — no global sys.stdout mutation
        captured = io.StringIO()
        real_print = print  # save reference to builtin

        def _captured_print(*args, **kwargs):
            kwargs.setdefault("file", captured)
            real_print(*args, **kwargs)

        # Inject captured print into namespace for this execution
        self.namespace["print"] = _captured_print

        is_main = threading.current_thread() is threading.main_thread()

        result_parts = []
        try:
            if is_main:
                # SIGALRM timeout (only works in main thread)
                def _timeout_handler(_signum, _frame):
                    raise TimeoutError(f"Execution timed out after {timeout}s")

                old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
                signal.alarm(timeout)
                try:
                    # Python exec() builtin — executes LLM-generated code in sandbox
                    exec(code, self.namespace)  # noqa: S102
                finally:
                    signal.alarm(0)
                    signal.signal(signal.SIGALRM, old_handler)
            else:
                # Worker thread — no SIGALRM, timeout managed by caller
                exec(code, self.namespace)  # noqa: S102

            output_val = captured.getvalue()
            if output_val:
                result_parts.append(output_val)

        except TimeoutError as e:
            result_parts.append(f"TIMEOUT: {e}")
        except Exception:
            result_parts.append(f"ERROR:\n{traceback.format_exc()}")

        output = "\n".join(result_parts) if result_parts else "(no output)"
        # Cap output to prevent context explosion in conversation history
        max_output = 5000
        if len(output) > max_output:
            output = output[:max_output] + f"\n... (truncated, {len(output) - max_output} chars omitted)"
        return output
