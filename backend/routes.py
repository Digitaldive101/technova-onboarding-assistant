"""FastAPI route definitions and public API schemas."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from backend.chat import LLMUnavailableError
from backend.rag import ConversationTurn, RAGResponse, answer_question
from backend.utils import get_knowledge_base_status
from backend.vector_store import VectorStoreError


logger = logging.getLogger(__name__)
router = APIRouter(tags=["chat"])


class ConversationMessage(BaseModel):
    """A bounded prior chat message used only for follow-up context."""

    role: str = Field(..., pattern="^(user|assistant)$")
    content: str = Field(..., min_length=1, max_length=2_000)


class ChatRequest(BaseModel):
    """Employee question accepted by the chat endpoint."""

    question: str = Field(..., min_length=1, max_length=2_000, examples=["How does payroll work?"])
    history: list[ConversationMessage] = Field(default_factory=list, max_length=6)


class SourceResponse(BaseModel):
    """Public citation structure returned with an answer."""

    document: str
    page: int


class ChatResponse(BaseModel):
    """Grounded chat result returned to the frontend."""

    answer: str
    sources: list[SourceResponse]


class KnowledgeBaseStatusResponse(BaseModel):
    """Read-only index readiness details for the employee dashboard."""

    document_count: int
    indexed_chunk_count: int
    is_ready: bool


def _to_chat_response(result: RAGResponse) -> ChatResponse:
    """Convert internal RAG models into the stable public response model."""
    return ChatResponse(
        answer=result.answer,
        sources=[SourceResponse(document=item.document, page=item.page) for item in result.sources],
    )


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """Answer one employee question using only indexed company documents."""
    question = request.question.strip()
    if not question:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Question must not be empty.",
        )

    logger.info("Received employee question: %s", question)
    try:
        history = [ConversationTurn(role=item.role, content=item.content) for item in request.history]
        return _to_chat_response(answer_question(question, history))
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    except VectorStoreError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
    except LLMUnavailableError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected chat-processing failure")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="The assistant could not process the question.",
        ) from exc


@router.get("/knowledge-base/status", response_model=KnowledgeBaseStatusResponse)
def knowledge_base_status() -> KnowledgeBaseStatusResponse:
    """Return safe readiness information without exposing policy documents."""
    snapshot = get_knowledge_base_status()
    return KnowledgeBaseStatusResponse(
        document_count=snapshot.document_count,
        indexed_chunk_count=snapshot.indexed_chunk_count,
        is_ready=snapshot.is_ready,
    )
