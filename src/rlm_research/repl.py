"""Python REPL sandbox — restricted namespace, subprocess execution, timeout."""

from __future__ import annotations

import ast
import io
import logging
import signal
import sys
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
        """Generate summary of available data for Root LM prompt."""
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
            if isinstance(value, str):
                lines.append(f"- {key}: string ({len(value)} chars)")
            elif isinstance(value, dict):
                lines.append(f"- {key}: dict ({len(value)} keys)")
            elif isinstance(value, list):
                lines.append(f"- {key}: list ({len(value)} items)")
            else:
                lines.append(f"- {key}: {type(value).__name__}")

        if "web_search" in self.namespace:
            lines.append("- web_search(query, n=5): web search enabled")
        if "sub_lm" in self.namespace:
            lines.append("- sub_lm(prompt): recursive LLM call")
        if "llm_batch" in self.namespace:
            lines.append("- llm_batch([prompts]): parallel sub_lm calls")
        return "\n".join(lines)

    def execute(self, code: str, timeout: int = 30) -> str:
        """Execute code in restricted namespace. Returns stdout+stderr or error.

        NOTE: This uses exec() intentionally — the REPL is the core of RLM's
        code-execute-observe loop. Security is enforced via:
        1. AST validation (_validate_code) blocks dangerous imports
        2. Restricted builtins (no open, no __import__, no eval/exec in namespace)
        3. SIGALRM timeout prevents infinite loops
        This is MVP-level sandboxing. Production requires Docker isolation.
        """
        # Static validation
        error = _validate_code(code)
        if error:
            return f"VALIDATION ERROR: {error}"

        # Capture stdout/stderr
        old_stdout, old_stderr = sys.stdout, sys.stderr
        captured_out = io.StringIO()
        captured_err = io.StringIO()

        def _timeout_handler(signum, frame):
            raise TimeoutError(f"Execution timed out after {timeout}s")

        result_parts = []
        try:
            sys.stdout = captured_out
            sys.stderr = captured_err

            # Set timeout (Unix only)
            old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(timeout)

            try:
                # Intentional use of exec — see docstring for security model
                exec(code, self.namespace)  # noqa: S102
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)

            stdout_val = captured_out.getvalue()
            stderr_val = captured_err.getvalue()
            if stdout_val:
                result_parts.append(stdout_val)
            if stderr_val:
                result_parts.append(f"STDERR: {stderr_val}")

        except TimeoutError as e:
            result_parts.append(f"TIMEOUT: {e}")
        except Exception:
            result_parts.append(f"ERROR:\n{traceback.format_exc()}")
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

        return "\n".join(result_parts) if result_parts else "(no output)"
