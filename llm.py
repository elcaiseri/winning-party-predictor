"""
LLM interface for the winning-party prediction test.

The real client wraps the OpenAI Chat Completions API with tool calling
and cumulative token/cost tracking. Set OPENAI_API_KEY and it is used
automatically.

  llm = LLM()
  resp = llm.complete(messages, tools=TOOL_SCHEMAS)
  resp = llm.complete(messages, response_format=SCHEMA)   # structured output
  llm.total_tokens(); llm.total_cost()

`complete` returns an LLMResponse with `.text`, `.tool_calls` (when the
model wants to call a tool), and token usage. Pass `response_format` to
get a structured/JSON-schema-constrained response back in `.text`.

If OPENAI_API_KEY is NOT set, a small deterministic MockLLM runs so the
harness still executes offline (plumbing only; real signal comes from
OpenAI).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# Load OPENAI_API_KEY (and any other vars) from a local .env if present.
load_dotenv(Path(__file__).resolve().parent / ".env")


@dataclass
class ToolCall:
    id: str
    name: str
    input: dict[str, Any]


@dataclass
class LLMResponse:
    text: str
    stop_reason: str = "stop"
    tool_calls: list[ToolCall] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0


# Price per 1M tokens (input, output).
PRICING = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1": (2.00, 8.00),
}
DEFAULT_MODEL = "gpt-4.1"

# Fixed decoding params so runs are reproducible.
SEED = 20240517
TEMPERATURE = 0.0


class _UsageMixin:
    _input_tokens: int
    _output_tokens: int
    _model: str

    def total_tokens(self) -> int:
        return self._input_tokens + self._output_tokens

    def total_cost(self) -> float:
        pin, pout = PRICING.get(self._model, PRICING[DEFAULT_MODEL])
        return (self._input_tokens * pin + self._output_tokens * pout) / 1_000_000


class OpenAILLM(_UsageMixin):
    def __init__(self, model: str = DEFAULT_MODEL) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("Real-mode LLM requires the `openai` package.") from exc
        self._client = OpenAI()
        self._model = model
        self._input_tokens = 0
        self._output_tokens = 0

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        response_format: dict[str, Any] | None = None,
        tool_choice: str | None = None,
    ) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": TEMPERATURE,
            "seed": SEED,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice or "auto"
        if response_format:
            kwargs["response_format"] = response_format
        resp = self._client.chat.completions.create(**kwargs)
        choice = resp.choices[0]
        msg = choice.message
        tool_calls = [
            ToolCall(
                id=tc.id,
                name=tc.function.name,
                input=json.loads(tc.function.arguments or "{}"),
            )
            for tc in (msg.tool_calls or [])
        ]
        usage = resp.usage
        if usage:
            self._input_tokens += usage.prompt_tokens
            self._output_tokens += usage.completion_tokens
        return LLMResponse(
            text=msg.content or "",
            stop_reason=choice.finish_reason or "stop",
            tool_calls=tool_calls,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
        )


# ---------------------------------------------------------------------------
# Offline smoke-test fallback. Deterministic, not intelligent.
# ---------------------------------------------------------------------------


class MockLLM(_UsageMixin):
    def __init__(self, model: str = DEFAULT_MODEL) -> None:
        self._model = model
        self._input_tokens = 0
        self._output_tokens = 0

    def _charge(self, messages, out_text) -> tuple[int, int]:
        inp = (
            sum(
                len(m.get("content", "") or "")
                for m in messages
                if isinstance(m.get("content"), str)
            )
            // 4
        )
        out = max(1, len(out_text) // 4)
        self._input_tokens += inp
        self._output_tokens += out
        return inp, out

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        response_format: dict[str, Any] | None = None,
        tool_choice: str | None = None,
    ) -> LLMResponse:
        blob = " ".join(
            m.get("content", "") for m in messages if isinstance(m.get("content"), str)
        )
        already_searched = "winning_party" in blob or any(
            m.get("role") == "tool" for m in messages
        )
        # First turn with tools offered: ask to call the first available tool.
        if tools and not already_searched:
            tool_name = tools[0]["function"]["name"]
            inp, out = self._charge(messages, "")
            return LLMResponse(
                text="",
                stop_reason="tool_calls",
                tool_calls=[ToolCall("call_1", tool_name, {"query": blob[:200]})],
                input_tokens=inp,
                output_tokens=out,
            )
        # Answer turn. Structured or plain, always a deterministic guess.
        if response_format:
            text = json.dumps(
                {
                    "winning_party": "petitioner",
                    "reasoning": "(mock) similar precedents favor the petitioner.",
                }
            )
        else:
            text = "Based on the precedent, the petitioner is likely to prevail."
        inp, out = self._charge(messages, text)
        return LLMResponse(text=text, input_tokens=inp, output_tokens=out)


def LLM(mock: bool | None = None, model: str = DEFAULT_MODEL):
    if mock is None:
        mock = not bool(os.environ.get("OPENAI_API_KEY"))
    return MockLLM(model) if mock else OpenAILLM(model)
