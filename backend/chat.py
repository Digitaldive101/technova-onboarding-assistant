"""Local Ollama chat-model adapter."""

from __future__ import annotations

import logging
from functools import lru_cache

from langchain_core.messages import BaseMessage
from langchain_ollama import ChatOllama

from backend.config import get_settings


logger = logging.getLogger(__name__)


class OllamaUnavailableError(RuntimeError):
    """Raised when the configured local Ollama model cannot respond."""


@lru_cache
def get_chat_model() -> ChatOllama:
    """Create and cache the configured deterministic local Ollama client."""
    settings = get_settings()
    return ChatOllama(
        model=settings.ollama_model,
        base_url=settings.ollama_base_url,
        temperature=settings.llm_temperature,
    )


def generate_answer(messages: list[BaseMessage]) -> str:
    """Send a prepared RAG prompt to Ollama and return plain response text."""
    settings = get_settings()
    logger.info("Requesting answer from local Ollama model: %s", settings.ollama_model)
    try:
        response = get_chat_model().invoke(messages)
    except Exception as exc:
        logger.exception("Ollama request failed")
        raise OllamaUnavailableError(
            "The local language model is unavailable. Start Ollama and ensure "
            f"the '{settings.ollama_model}' model has been pulled."
        ) from exc

    content = response.content
    if isinstance(content, str):
        answer = content.strip()
    else:
        answer = "".join(
            item.get("text", "") if isinstance(item, dict) else str(item)
            for item in content
        ).strip()

    if not answer:
        logger.error("Ollama returned an empty response")
        raise OllamaUnavailableError("The local language model returned an empty response.")
    return answer

