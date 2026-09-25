"""Tests for the shared tool error-handling infrastructure.

Covers:
- ``safe_tool`` decorator catches exceptions, logs, records to collector,
  and returns a clean error string.
- ``ToolErrorCollector`` record/drain/snapshot/clear lifecycle.
- ``@tool`` + ``@safe_tool`` stacking preserves the tool schema (name,
  description, args schema) so the LLM can still call it.
- Every ``@tool`` function in every tool module has ``@safe_tool`` applied.
"""

import logging
from unittest.mock import MagicMock

import pytest
from langchain_core.tools import tool

from tradingagents.agents.utils.tool_errors import (
    ToolErrorCollector,
    safe_tool,
    tool_error_collector,
)


@pytest.mark.unit
class TestSafeToolDecorator:
    def setup_method(self):
        tool_error_collector.clear()

    def test_catches_exception_and_returns_error_string(self):
        @tool
        @safe_tool
        def boom(x: str) -> str:
            """Docstring."""
            raise ValueError("kaboom")

        result = boom.invoke({"x": "test"})
        assert "Tool Error" in result
        assert "boom" in result
        assert "kaboom" in result

    def test_records_to_collector(self):
        @tool
        @safe_tool
        def fail(x: str) -> str:
            """Doc."""
            raise RuntimeError("nope")

        fail.invoke({"x": "a"})
        errors = tool_error_collector.snapshot()
        assert len(errors) == 1
        assert errors[0]["tool"] == "fail"
        assert "nope" in errors[0]["error"]
        assert "traceback" in errors[0]["error"].lower() or "traceback" in errors[0]["traceback"].lower()

    def test_preserves_tool_name_and_description(self):
        @tool
        @safe_tool
        def my_great_tool(ticker: str) -> str:
            """A great tool."""
            return "ok"

        assert my_great_tool.name == "my_great_tool"
        assert "great tool" in my_great_tool.description.lower()

    def test_preserves_args_schema(self):
        @tool
        @safe_tool
        def schema_tool(x: str, y: int = 5) -> str:
            """Schema test."""
            return f"{x}{y}"

        schema = schema_tool.args_schema
        assert "x" in schema.model_fields
        assert "y" in schema.model_fields

    def test_successful_call_unaffected(self):
        @tool
        @safe_tool
        def good(x: str) -> str:
            """Good tool."""
            return f"result-{x}"

        result = good.invoke({"x": "abc"})
        assert result == "result-abc"
        assert tool_error_collector.count == 0

    def test_logs_at_warning_level(self, caplog):
        @tool
        @safe_tool
        def logged_fail(x: str) -> str:
            """Doc."""
            raise ValueError("logged")

        with caplog.at_level(logging.WARNING, logger="tradingagents.tools"):
            logged_fail.invoke({"x": "t"})
        assert any("logged_fail" in r.message and "logged" in r.message for r in caplog.records)


@pytest.mark.unit
class TestToolErrorCollector:
    def test_record_drain_clear(self):
        c = ToolErrorCollector()
        assert c.count == 0
        assert c.drain() == []

        c.record("tool_a", ("arg1",), {"k": "v"}, ValueError("err1"))
        c.record("tool_b", (), {}, RuntimeError("err2"))
        assert c.count == 2

        drained = c.drain()
        assert len(drained) == 2
        assert drained[0]["tool"] == "tool_a"
        assert drained[1]["tool"] == "tool_b"
        assert c.count == 0  # drained

    def test_snapshot_does_not_clear(self):
        c = ToolErrorCollector()
        c.record("t", (), {}, ValueError("e"))
        snap = c.snapshot()
        assert len(snap) == 1
        assert c.count == 1  # still there

    def test_clear(self):
        c = ToolErrorCollector()
        c.record("t", (), {}, ValueError("e"))
        c.clear()
        assert c.count == 0
        assert c.drain() == []

    def test_record_includes_traceback(self):
        c = ToolErrorCollector()
        try:
            raise ValueError("with traceback")
        except ValueError as e:
            c.record("t", (), {}, e)
        errors = c.drain()
        assert "Traceback" in errors[0]["traceback"]

    def test_record_truncates_long_args(self):
        c = ToolErrorCollector()
        long_arg = "x" * 1000
        c.record("t", (long_arg,), {}, ValueError("e"))
        errors = c.drain()
        assert len(errors[0]["args"]) <= 510  # 500 + ellipsis


@pytest.mark.unit
class TestAllToolsHaveSafeTool:
    """Verify every @tool in every tool module has @safe_tool applied."""

    def _check_module(self, module_path, expected_min_tools):
        import importlib
        from langchain_core.tools import BaseTool

        mod = importlib.import_module(module_path)
        tool_names = [
            name for name, val in vars(mod).items()
            if isinstance(val, BaseTool) and not name.startswith("_")
        ]
        assert len(tool_names) >= expected_min_tools, (
            f"{module_path}: expected >= {expected_min_tools} tools, found {len(tool_names)}: {tool_names}"
        )

        # Verify each tool's underlying function is wrapped by safe_tool
        import ast, inspect
        source = inspect.getsource(mod)
        tree = ast.parse(source)
        unwrapped = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                decorators = [ast.dump(d) for d in node.decorator_list]
                has_tool = any("tool" in d and "safe_tool" not in d for d in decorators)
                has_safe = any("safe_tool" in d for d in decorators)
                if has_tool and not has_safe:
                    unwrapped.append(node.name)
        # Factory-built tools use tool(safe_tool(...)) — check source
        assert not unwrapped, f"{module_path}: missing @safe_tool on: {unwrapped}"
        assert "safe_tool" in source, f"{module_path}: no safe_tool usage at all"

    def test_filing_signals_tools(self):
        self._check_module("tradingagents.agents.utils.filing_signals_tools", 24)

    def test_business_data_tools(self):
        self._check_module("tradingagents.agents.utils.business_data_tools", 9)

    def test_equity_intel_tools(self):
        self._check_module("tradingagents.agents.utils.equity_intel_tools", 8)

    def test_fundamental_data_tools(self):
        self._check_module("tradingagents.agents.utils.fundamental_data_tools", 14)

    def test_news_data_tools(self):
        self._check_module("tradingagents.agents.utils.news_data_tools", 3)

    def test_macro_data_tools(self):
        self._check_module("tradingagents.agents.utils.macro_data_tools", 8)

    def test_core_stock_tools(self):
        self._check_module("tradingagents.agents.utils.core_stock_tools", 1)

    def test_technical_indicators_tools(self):
        self._check_module("tradingagents.agents.utils.technical_indicators_tools", 1)

    def test_options_greeks_tools(self):
        self._check_module("tradingagents.agents.utils.options_greeks_tools", 1)


@pytest.mark.unit
class TestNoOuterExceptInTools:
    """Verify no tool function has a bare ``except Exception as e:`` catch-all
    that would swallow errors before @safe_tool can log them."""

    def _check_no_outer_except(self, module_path):
        """Flag try/except Exception handlers that *return* an error string
        (the swallowing pattern @safe_tool replaces). Inner defensive
        catches like ``except Exception: pass`` are fine."""
        import importlib, inspect, ast
        mod = importlib.import_module(module_path)
        source = inspect.getsource(mod)
        tree = ast.parse(source)

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                decorators = [ast.dump(d) for d in node.decorator_list]
                if not any("tool" in d for d in decorators):
                    continue
                for stmt in node.body:
                    if isinstance(stmt, ast.Try):
                        for handler in stmt.handlers:
                            if (handler.type and isinstance(handler.type, ast.Name)
                                    and handler.type.id == "Exception"):
                                # Check if handler returns (swallows error)
                                has_return = any(
                                    isinstance(hs, ast.Return) for hs in handler.body
                                )
                                if has_return:
                                    pytest.fail(
                                        f"{module_path}:{node.name} has outer try/except Exception "
                                        f"with return — @safe_tool should be the catch"
                                    )

    def test_equity_intel_no_outer_except(self):
        self._check_no_outer_except("tradingagents.agents.utils.equity_intel_tools")

    def test_macro_data_no_outer_except(self):
        self._check_no_outer_except("tradingagents.agents.utils.macro_data_tools")

    def test_options_greeks_no_outer_except(self):
        self._check_no_outer_except("tradingagents.agents.utils.options_greeks_tools")

    def test_filing_signals_no_outer_except(self):
        self._check_no_outer_except("tradingagents.agents.utils.filing_signals_tools")
