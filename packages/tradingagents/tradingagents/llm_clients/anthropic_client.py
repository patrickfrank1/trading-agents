from typing import Any, Optional

from langchain_anthropic import ChatAnthropic

from .base_client import BaseLLMClient, normalize_content
from .validators import validate_model

_PASSTHROUGH_KWARGS = (
    "timeout", "max_retries", "api_key", "max_tokens", "temperature",
    "callbacks", "http_client", "http_async_client", "effort",
)


class NormalizedChatAnthropic(ChatAnthropic):
    """ChatAnthropic with normalized content output.

    Claude models with extended thinking or tool use return content as a
    list of typed blocks. This normalizes to string for consistent
    downstream handling.
    """

    def invoke(self, input, config=None, **kwargs):
        return normalize_content(super().invoke(input, config, **kwargs))

    def with_structured_output(self, schema, *, include_raw=False, method=None, **kwargs):
        # The default ``function_calling`` method forces ``tool_choice``, which
        # the Anthropic API rejects when thinking/effort is enabled:
        # "Thinking mode does not support this tool_choice". langchain's own
        # guard only checks the legacy ``thinking`` dict, not the ``effort``
        # shorthand, so we detect both here and switch to the native
        # ``json_schema`` method. That routes through ``output_config.format``
        # (no ``tool_choice``) and merges cleanly with ``effort`` in the same
        # ``output_config`` payload.
        if method is None:
            thinking_active = (
                self.thinking is not None
                and self.thinking.get("type") in ("enabled", "adaptive")
            )
            if thinking_active or self.effort is not None:
                method = "json_schema"
            else:
                method = "function_calling"
        return super().with_structured_output(
            schema, include_raw=include_raw, method=method, **kwargs
        )


class AnthropicClient(BaseLLMClient):
    """Client for Anthropic Claude models."""

    def __init__(self, model: str, base_url: Optional[str] = None, **kwargs):
        super().__init__(model, base_url, **kwargs)

    def get_llm(self) -> Any:
        """Return configured ChatAnthropic instance."""
        self.warn_if_unknown_model()
        llm_kwargs = {"model": self.model}

        if self.base_url:
            llm_kwargs["base_url"] = self.base_url

        for key in _PASSTHROUGH_KWARGS:
            if key in self.kwargs:
                llm_kwargs[key] = self.kwargs[key]

        return NormalizedChatAnthropic(**llm_kwargs)

    def validate_model(self) -> bool:
        """Validate model for Anthropic."""
        return validate_model("anthropic", self.model)
