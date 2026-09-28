"""Shared pytest fixtures that prevent CI hangs when API keys are absent."""

import os
from unittest.mock import MagicMock, patch

import pytest


def pytest_configure(config):
    for marker in ("unit", "integration", "smoke"):
        config.addinivalue_line("markers", f"{marker}: {marker}-level tests")


_API_KEY_ENV_VARS = (
    "OPENAI_API_KEY",  # allow: key
    "GOOGLE_API_KEY",  # allow: key
    "ANTHROPIC_API_KEY",  # allow: key
    "XAI_API_KEY",  # allow: key
    "DEEPSEEK_API_KEY",  # allow: key
    "DASHSCOPE_API_KEY",  # allow: key
    "ZHIPU_API_KEY",  # allow: key
    "OPENROUTER_API_KEY",  # allow: key
    "AZURE_OPENAI_API_KEY",
    "ALPHA_VANTAGE_API_KEY",
)


@pytest.fixture(autouse=True)
def _dummy_api_keys(monkeypatch):
    for env_var in _API_KEY_ENV_VARS:
        monkeypatch.setenv(env_var, os.environ.get(env_var, "placeholder"))
    # The Jev decision tool is enabled by default. Clear any real key so unit
    # tests never reach the network; tests that need Jev inject a fake client
    # or monkeypatch assess_with_jev directly.
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)


@pytest.fixture()
def mock_llm_client():
    client = MagicMock()
    client.get_llm.return_value = MagicMock()
    with patch(
        "tradingagents.llm_clients.factory.create_llm_client",
        return_value=client,
    ):
        yield client
