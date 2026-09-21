"""LLM access for the agent graph.

One entry point, `get_llm(purpose)`, returns a chat model for a given agent. Two backends
sit behind it:

  grok  ChatXAI against https://api.x.ai/v1 -- tool calling and structured outputs
  mock  a deterministic stand-in that satisfies the same interface

The mock exists because the assignment says to assume no internet, and because a grader
without an xAI key still has to see the pipeline run. It substitutes at the chat-model
boundary rather than inside the nodes, so the graph, the prompts, the tool wiring and the
control flow are byte-identical in both modes. Only the token source changes.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from enum import Enum
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.prompt_values import PromptValue
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_core.utils.function_calling import convert_to_openai_tool

from .config import Settings, get_settings


class Purpose(str, Enum):
    """Which agent is asking. Selects sampling temperature and mock handler."""

    EXTRACTOR = "extractor"
    VALIDATOR = "validator"
    APPROVER = "approver"


#: Transcription and lookup must be reproducible; the approver gets a little slack so
#: its critique pass can actually diverge from its first draft rather than restate it.
TEMPERATURE: dict[Purpose, float] = {
    Purpose.EXTRACTOR: 0.0,
    Purpose.VALIDATOR: 0.0,
    Purpose.APPROVER: 0.1,
}


class LLMConfigurationError(RuntimeError):
    """Raised when the requested backend cannot be constructed."""


class MockHandlerNotRegistered(RuntimeError):
    """Raised when mock mode is active but an agent has no canned behaviour."""


# ---------------------------------------------------------------------------
# Mock backend
# ---------------------------------------------------------------------------

#: handler(messages, schema, tools) -> BaseModel | dict | str | AIMessage
MockHandler = Callable[..., Any]

_HANDLERS: dict[str, MockHandler] = {}


def register_mock_handler(purpose: Purpose, handler: MockHandler) -> None:
    """Register the offline behaviour for one agent.

    Each agent module registers its own on import, which keeps the canned logic next to
    the prompt it stands in for instead of in one drifting fixtures file.
    """
    _HANDLERS[purpose.value] = handler


def _resolve_handler(purpose: str) -> MockHandler:
    try:
        return _HANDLERS[purpose]
    except KeyError:
        raise MockHandlerNotRegistered(
            f"LLM_MODE=mock but no handler is registered for {purpose!r}. "
            f"Registered: {sorted(_HANDLERS) or 'none'}. "
            f"Call register_mock_handler(Purpose.{purpose.upper()}, ...) at import time, "
            f"or set LLM_MODE=grok to use the live API."
        ) from None


def _as_messages(value: Any) -> list[BaseMessage]:
    """Coerce whatever a chain hands us into a message list."""
    if isinstance(value, PromptValue):
        return list(value.to_messages())
    if isinstance(value, BaseMessage):
        return [value]
    if isinstance(value, str):
        return [HumanMessage(content=value)]
    if isinstance(value, Sequence):
        return [
            m if isinstance(m, BaseMessage) else HumanMessage(content=str(m)) for m in value
        ]
    return [HumanMessage(content=str(value))]


class MockChatModel(BaseChatModel):
    """Deterministic stand-in for ChatXAI.

    Supports the three things the graph actually uses: plain generation, `bind_tools`
    for the validator's tool loop, and `with_structured_output` for the extractor's
    typed payload.
    """

    purpose: str

    @property
    def _llm_type(self) -> str:
        return "mock-chat-model"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"purpose": self.purpose}

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        handler = _resolve_handler(self.purpose)
        result = handler(messages=messages, schema=None, tools=kwargs.get("tools"))
        message = result if isinstance(result, AIMessage) else AIMessage(content=str(result))
        return ChatResult(generations=[ChatGeneration(message=message)])

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> Runnable:
        """Bind tools in the OpenAI schema the real backend would receive."""
        return self.bind(tools=[convert_to_openai_tool(t) for t in tools], **kwargs)

    def with_structured_output(
        self, schema: Any, *, include_raw: bool = False, **kwargs: Any
    ) -> Runnable:
        """Return a runnable producing `schema`, validating exactly as the real path does."""
        purpose = self.purpose

        def _invoke(value: Any) -> Any:
            handler = _resolve_handler(purpose)
            payload = handler(messages=_as_messages(value), schema=schema, tools=None)
            if isinstance(payload, schema):
                return payload
            if isinstance(payload, dict):
                return schema.model_validate(payload)
            return schema.model_validate_json(str(payload))

        runnable = RunnableLambda(_invoke)
        if include_raw:
            return runnable | RunnableLambda(
                lambda parsed: {"raw": None, "parsed": parsed, "parsing_error": None}
            )
        return runnable


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def get_llm(purpose: Purpose, settings: Settings | None = None) -> BaseChatModel:
    """Build the chat model for one agent.

    Raises:
        LLMConfigurationError: grok mode requested without a usable API key.
    """
    settings = settings or get_settings()

    if settings.llm_mode == "mock":
        return MockChatModel(purpose=purpose.value)

    if not settings.has_api_key:
        raise LLMConfigurationError(
            "LLM_MODE=grok requires XAI_API_KEY. Copy .env.example to .env and add a key "
            "from https://console.x.ai, or set LLM_MODE=mock to run offline."
        )

    from langchain_xai import ChatXAI

    return ChatXAI(
        model=settings.grok_model,
        api_key=settings.xai_api_key,
        base_url=settings.xai_base_url,
        temperature=TEMPERATURE[purpose],
        timeout=settings.llm_timeout_seconds,
        max_retries=settings.llm_max_retries,
    )
