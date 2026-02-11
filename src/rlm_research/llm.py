"""LLM client — unified OpenAI-compatible interface for all providers."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field

from openai import AsyncOpenAI

from rlm_research.config import Config

log = logging.getLogger(__name__)

MAX_RETRIES = 3
RETRY_BASE_DELAY = 1.0


@dataclass
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class LLMResponse:
    content: str
    usage: TokenUsage = field(default_factory=TokenUsage)


class LLMClient:
    """Async OpenAI-compatible client. One SDK, different base_url per provider."""

    def __init__(self, config: Config) -> None:
        self._config = config
        self._client = AsyncOpenAI(
            api_key=config.llm.api_key,
            base_url=config.llm.effective_base_url(),
        )
        self.cumulative_usage = TokenUsage()

    def select_model(self, depth: int) -> str:
        """Root (depth 0) uses reasoning model, deeper levels use cheaper model."""
        if depth == 0:
            return self._config.llm.root_model
        return self._config.llm.sub_model

    async def generate(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        depth: int = 0,
    ) -> LLMResponse:
        """Generate a completion with exponential backoff retry on rate limits."""
        target_model = model or self.select_model(depth)

        for attempt in range(MAX_RETRIES):
            try:
                response = await self._client.chat.completions.create(
                    model=target_model,
                    messages=messages,
                )
                choice = response.choices[0]
                usage = TokenUsage()
                if response.usage:
                    usage = TokenUsage(
                        input_tokens=response.usage.prompt_tokens,
                        output_tokens=response.usage.completion_tokens,
                    )
                self.cumulative_usage.input_tokens += usage.input_tokens
                self.cumulative_usage.output_tokens += usage.output_tokens

                return LLMResponse(content=choice.message.content or "", usage=usage)

            except Exception as exc:
                is_rate_limit = getattr(exc, "status_code", None) == 429
                if is_rate_limit and attempt < MAX_RETRIES - 1:
                    delay = RETRY_BASE_DELAY * (2**attempt)
                    log.warning("Rate limited, retrying in %.1fs (attempt %d)", delay, attempt + 1)
                    await asyncio.sleep(delay)
                    continue
                raise

        raise RuntimeError("Unreachable: retry loop exhausted without return or raise")
