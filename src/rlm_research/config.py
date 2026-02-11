"""Configuration — YAML file + env vars + CLI overrides, three-layer merge."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

DEFAULT_CONFIG_PATH = Path.home() / ".rlm-research.yaml"

PROVIDERS = {
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "root_models": ["o3-mini", "o3", "o4-mini"],
        "sub_models": ["gpt-4o-mini", "gpt-4o"],
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com",
        "root_models": ["deepseek-reasoner"],
        "sub_models": ["deepseek-chat"],
    },
    "glm": {
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "root_models": ["glm-4-long"],
        "sub_models": ["glm-4-flash"],
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "root_models": ["deepseek/deepseek-r1-0528:free"],
        "sub_models": ["openai/gpt-oss-120b:free"],
    },
}


@dataclass
class LLMConfig:
    provider: str = "openrouter"
    root_model: str = "deepseek/deepseek-r1-0528:free"
    sub_model: str = "openai/gpt-oss-120b:free"
    api_key: str = ""
    base_url: str = ""

    def effective_base_url(self) -> str:
        if self.base_url:
            return self.base_url
        return PROVIDERS.get(self.provider, {}).get("base_url", "https://api.openai.com/v1")


@dataclass
class SearchConfig:
    provider: str = "tavily"
    api_key: str = ""
    base_url: str = "http://localhost:8080"
    max_results_per_query: int = 10


@dataclass
class EngineConfig:
    max_recursion_depth: int = 3
    max_turns: int = 30
    max_sub_lm_calls: int = 50
    timeout_per_exec: int = 30


@dataclass
class Config:
    llm: LLMConfig = field(default_factory=LLMConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    engine: EngineConfig = field(default_factory=EngineConfig)


def _resolve_env_var(value: str) -> str:
    """Resolve ${ENV_VAR} references in string values."""
    if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
        env_name = value[2:-1]
        return os.environ.get(env_name, "")
    return value


def _merge_section(target, data: dict) -> None:
    """Merge a flat dict into a dataclass, resolving env var references."""
    for key, value in data.items():
        if hasattr(target, key):
            resolved = _resolve_env_var(value) if isinstance(value, str) else value
            setattr(target, key, resolved)


def _apply_env_overrides(config: Config) -> None:
    """Apply RLM_* environment variable overrides."""
    env_map = {
        "RLM_PROVIDER": ("llm", "provider"),
        "RLM_API_KEY": ("llm", "api_key"),
        "RLM_ROOT_MODEL": ("llm", "root_model"),
        "RLM_SUB_MODEL": ("llm", "sub_model"),
        "RLM_BASE_URL": ("llm", "base_url"),
        "RLM_SEARCH_PROVIDER": ("search", "provider"),
        "RLM_SEARCH_API_KEY": ("search", "api_key"),
        "RLM_MAX_DEPTH": ("engine", "max_recursion_depth"),
        "RLM_MAX_TURNS": ("engine", "max_turns"),
    }
    # Also check provider-specific key vars as fallback for api_key
    if not os.environ.get("RLM_API_KEY"):
        for key_var in ("OPENROUTER_API_KEY", "OPENAI_API_KEY", "DEEPSEEK_API_KEY"):
            val = os.environ.get(key_var)
            if val:
                env_map[key_var] = ("llm", "api_key")
                break

    for env_name, (section, attr) in env_map.items():
        value = os.environ.get(env_name)
        if value is not None:
            target = getattr(config, section)
            current = getattr(target, attr)
            # Cast to int if the target field is int
            if isinstance(current, int):
                value = int(value)
            setattr(target, attr, value)


def load_config(
    config_path: Path | None = None,
    overrides: dict | None = None,
) -> Config:
    """Load config: YAML file → env vars → overrides. Each layer wins over previous."""
    config = Config()

    # Layer 1: YAML file
    path = config_path or DEFAULT_CONFIG_PATH
    if path.exists():
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        if "llm" in raw:
            _merge_section(config.llm, raw["llm"])
        if "search" in raw:
            _merge_section(config.search, raw["search"])
        if "engine" in raw:
            _merge_section(config.engine, raw["engine"])

    # Layer 2: Environment variables
    _apply_env_overrides(config)

    # Layer 3: Programmatic overrides (from CLI args or MCP tool params)
    if overrides:
        for section_name in ("llm", "search", "engine"):
            if section_name in overrides:
                _merge_section(getattr(config, section_name), overrides[section_name])

    return config
