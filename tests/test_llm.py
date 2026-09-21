"""Tests for LLM backend selection and the offline mock.

The mock substitutes at the chat-model boundary, so these assert the interface the graph
actually consumes: plain generation, bind_tools, and with_structured_output.
"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from pydantic import BaseModel

from invoice_agents.config import Settings
from invoice_agents.llm import (
    _HANDLERS,
    LLMConfigurationError,
    MockChatModel,
    MockHandlerNotRegistered,
    Purpose,
    get_llm,
    register_mock_handler,
)


class Extracted(BaseModel):
    vendor: str
    total: float


@pytest.fixture(autouse=True)
def isolated_registry():
    """Keep handler registrations from leaking between tests."""
    saved = dict(_HANDLERS)
    _HANDLERS.clear()
    yield
    _HANDLERS.clear()
    _HANDLERS.update(saved)


@pytest.fixture
def mock_settings():
    return Settings(llm_mode="mock", xai_api_key=None)


# --------------------------------------------------------------------------
# Backend selection
# --------------------------------------------------------------------------


def test_mock_mode_needs_no_api_key(mock_settings):
    assert isinstance(get_llm(Purpose.EXTRACTOR, mock_settings), MockChatModel)


def test_grok_mode_without_key_fails_with_an_actionable_message():
    settings = Settings(llm_mode="grok", xai_api_key=None)
    with pytest.raises(LLMConfigurationError, match="XAI_API_KEY"):
        get_llm(Purpose.EXTRACTOR, settings)


def test_each_purpose_gets_its_own_handler(mock_settings):
    assert get_llm(Purpose.EXTRACTOR, mock_settings).purpose == "extractor"
    assert get_llm(Purpose.APPROVER, mock_settings).purpose == "approver"


# --------------------------------------------------------------------------
# Missing handlers fail loudly
# --------------------------------------------------------------------------


def test_unregistered_handler_raises_rather_than_stubbing(mock_settings):
    """A silent stub would look like a passing pipeline producing fabricated data."""
    llm = get_llm(Purpose.VALIDATOR, mock_settings)
    with pytest.raises(MockHandlerNotRegistered, match="validator"):
        llm.invoke("anything")


def test_missing_handler_error_names_the_fix(mock_settings):
    llm = get_llm(Purpose.APPROVER, mock_settings)
    with pytest.raises(MockHandlerNotRegistered) as excinfo:
        llm.invoke("anything")
    assert "register_mock_handler" in str(excinfo.value)
    assert "LLM_MODE=grok" in str(excinfo.value)


# --------------------------------------------------------------------------
# The three capabilities the graph relies on
# --------------------------------------------------------------------------


def test_plain_generation(mock_settings):
    register_mock_handler(
        Purpose.APPROVER, lambda **_: AIMessage(content="approved: within policy")
    )
    reply = get_llm(Purpose.APPROVER, mock_settings).invoke("Decide.")
    assert "approved" in reply.content


def test_structured_output_validates_against_the_schema(mock_settings):
    register_mock_handler(
        Purpose.EXTRACTOR, lambda **_: {"vendor": "Widgets Inc.", "total": 5000.0}
    )
    llm = get_llm(Purpose.EXTRACTOR, mock_settings)

    result = llm.with_structured_output(Extracted).invoke("Extract this invoice.")

    assert isinstance(result, Extracted)
    assert result.vendor == "Widgets Inc."
    assert result.total == 5000.0


def test_structured_output_rejects_a_malformed_payload(mock_settings):
    """Validation must behave identically offline, so schema bugs surface in mock runs."""
    register_mock_handler(Purpose.EXTRACTOR, lambda **_: {"vendor": "Widgets Inc."})
    llm = get_llm(Purpose.EXTRACTOR, mock_settings)

    with pytest.raises(Exception, match="total"):
        llm.with_structured_output(Extracted).invoke("Extract this invoice.")


def test_handler_receives_the_prompt_and_schema(mock_settings):
    seen: dict = {}

    def handler(messages, schema, tools):
        seen["text"] = messages[-1].content
        seen["schema"] = schema
        return {"vendor": "Acme", "total": 1.0}

    register_mock_handler(Purpose.EXTRACTOR, handler)
    get_llm(Purpose.EXTRACTOR, mock_settings).with_structured_output(Extracted).invoke(
        "Vendor: Acme"
    )

    assert seen["text"] == "Vendor: Acme"
    assert seen["schema"] is Extracted


def test_bind_tools_passes_tool_schemas_through(mock_settings):
    """The validator agent is a tool-using adjudicator, so the mock must carry tools."""

    @tool
    def lookup_item_tool(name: str) -> str:
        """Look up an item in the inventory catalog."""
        return "ok"

    seen: dict = {}

    def handler(messages, schema, tools):
        seen["tools"] = tools
        return AIMessage(content="done")

    register_mock_handler(Purpose.VALIDATOR, handler)
    llm = get_llm(Purpose.VALIDATOR, mock_settings).bind_tools([lookup_item_tool])
    llm.invoke("Check WidgetA.")

    assert seen["tools"] is not None
    assert seen["tools"][0]["function"]["name"] == "lookup_item_tool"
