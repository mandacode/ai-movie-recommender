"""LLM chat agent over the movie tools (OpenAI function calling).

The agent is a thin orchestration loop: it hands the user's message + the tool
schemas to the model, executes whatever tools the model asks for, feeds the
results back, and repeats until the model produces a final answer.

Design rule enforced via the system prompt: the model may only recommend movies
returned by the tools — never from its own memory. The tools are the ground
truth; the LLM is the language layer on top.
"""
from __future__ import annotations

import json
import os
from typing import Any

from .tools import TOOL_SCHEMAS, MovieTools

DEFAULT_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
MAX_TOOL_ROUNDS = 5  # safety cap on the tool-calling loop

SYSTEM_PROMPT = (
    "You are a movie recommendation assistant for a MovieLens-based system.\n"
    "Rules:\n"
    "- Only recommend movies returned by the tools. Never invent titles, years "
    "or ratings from your own knowledge.\n"
    "- When the user mentions a movie, call `search_movies` first to resolve it "
    "to a movieId, then `similar_movies` for recommendations.\n"
    "- For an existing userId, use `recommend_for_user`.\n"
    "- When the user describes a vibe/plot instead of naming a movie (e.g. 'a "
    "heist with a twist'), use `search_by_description` (semantic search).\n"
    "- Keep answers concise, list a handful of picks, and briefly say why each "
    "fits (genre overlap, shared fans). Answer in the user's language."
)


def _openai_client():
    """Create an OpenAI client, raising a clear error if the key is missing."""
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Export it (or add it to Streamlit "
            "secrets) before using the chat."
        )
    from openai import OpenAI  # imported lazily so the rest of the app runs without openai

    return OpenAI()


class MovieChatAgent:
    """Stateless-per-call agent wrapping the movie tools with an LLM."""

    def __init__(self, tools: MovieTools, model: str = DEFAULT_MODEL):
        self.tools = tools
        self.model = model

    def reply(self, history: list[dict[str, str]]) -> str:
        """Run the tool-calling loop for a chat history and return the answer.

        Args:
            history: list of ``{"role": "user"|"assistant", "content": str}``.
        """
        client = _openai_client()
        messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}, *history]

        for _ in range(MAX_TOOL_ROUNDS):
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=TOOL_SCHEMAS,
            )
            msg = response.choices[0].message

            if not msg.tool_calls:
                return msg.content or ""

            # Record the assistant's tool request, then answer each tool call.
            messages.append(msg.model_dump(exclude_none=True))
            for call in msg.tool_calls:
                try:
                    args = json.loads(call.function.arguments or "{}")
                    result = self.tools.call(call.function.name, args)
                except Exception as exc:  # surface tool errors to the model, don't crash
                    result = {"error": str(exc)}
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": json.dumps(result, ensure_ascii=False),
                })

        return "Sorry — I couldn't complete that request within the tool limit."
