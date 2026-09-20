"""Hosted LLM adapter for grounded RAG answers."""

from __future__ import annotations

import logging
from typing import Any

import requests
from langchain_core.messages import BaseMessage

from backend.config import get_settings


logger = logging.getLogger(__name__)


class LLMUnavailableError(RuntimeError):
    """Raised when the configured hosted language model cannot respond."""


def _openrouter_error_message(response: requests.Response) -> str:
    """Return a safe, user-actionable description of an API failure."""
    status = response.status_code
    if status in {401, 403}:
        return "OpenRouter rejected the API key. Check Streamlit Cloud Secrets and reboot the app."
    if status == 429:
        return "OpenRouter's free-tier request limit was reached. Please try again later."
    if status == 404:
        return "The configured OpenRouter model is unavailable. Choose another free model in Streamlit Secrets."

    try:
        detail = response.json().get("error", {}).get("message", "")
    except ValueError:
        detail = ""
    logger.warning("OpenRouter returned HTTP %s: %s", status, detail or response.text[:300])
    return f"OpenRouter could not complete the request (HTTP {status}). Please try again later."


def _message_payload(message: BaseMessage) -> dict[str, str]:
    """Convert a LangChain message to OpenRouter's chat-completions format."""
    role_map = {"human": "user", "ai": "assistant", "system": "system"}
    content = message.content
    if not isinstance(content, str):
        content = "".join(
            item.get("text", "") if isinstance(item, dict) else str(item)
            for item in content
        )
    return {"role": role_map.get(message.type, "user"), "content": content}


def generate_answer(messages: list[BaseMessage]) -> str:
    """Generate an answer through OpenRouter's OpenAI-compatible API."""
    settings = get_settings()
    if settings.llm_provider.lower() != "openrouter":
        raise LLMUnavailableError(f"Unsupported LLM provider: {settings.llm_provider}")
    if not settings.openrouter_api_key:
        raise LLMUnavailableError(
            "OpenRouter is not configured. Add OPENROUTER_API_KEY to Streamlit Secrets."
        )

    logger.info("Requesting answer from OpenRouter model: %s", settings.openrouter_model)
    try:
        response = requests.post(
            f"{settings.openrouter_base_url.rstrip('/')}/chat/completions",
            headers={
                "Authorization": f"Bearer {settings.openrouter_api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": settings.openrouter_model,
                "messages": [_message_payload(message) for message in messages],
                "temperature": settings.llm_temperature,
            },
            timeout=settings.llm_request_timeout_seconds,
        )
        if not response.ok:
            raise LLMUnavailableError(_openrouter_error_message(response))
        payload: dict[str, Any] = response.json()
        answer = str(payload["choices"][0]["message"]["content"]).strip()
    except LLMUnavailableError:
        raise
    except (requests.RequestException, KeyError, IndexError, TypeError, ValueError) as exc:
        logger.exception("OpenRouter request failed")
        raise LLMUnavailableError("The hosted language model is unavailable. Please try again.") from exc

    if not answer:
        raise LLMUnavailableError("The hosted language model returned an empty response.")
    return answer
