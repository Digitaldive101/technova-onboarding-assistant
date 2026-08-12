"""Grounded retrieval-augmented generation pipeline."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from time import perf_counter

from langchain_core.documents import Document

from backend.chat import generate_answer
from backend.config import get_settings
from backend.prompt import NO_ANSWER_MESSAGE, get_rag_prompt
from backend.vector_store import VectorStoreError, collection_document_count, get_vector_store


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SourceCitation:
    """A source page used to ground an assistant answer."""

    document: str
    page: int


@dataclass(frozen=True)
class RAGResponse:
    """The generated answer and its supporting company-document pages."""

    answer: str
    sources: list[SourceCitation]


@dataclass(frozen=True)
class ConversationTurn:
    """A prior chat turn used only to resolve follow-up references."""

    role: str
    content: str


def _format_context(documents: list[Document]) -> str:
    """Render retrieved chunks with source labels for the language model."""
    sections: list[str] = []
    for document in documents:
        filename = document.metadata.get("document", "Unknown document")
        page = document.metadata.get("page", "Unknown page")
        sections.append(f"[Source: {filename}, page {page}]\n{document.page_content}")
    return "\n\n---\n\n".join(sections)


def _citations(documents: list[Document]) -> list[SourceCitation]:
    """Create stable, duplicate-free citations from retrieved chunks."""
    citations: list[SourceCitation] = []
    seen: set[tuple[str, int]] = set()
    for document in documents:
        filename = str(document.metadata.get("document", "Unknown document"))
        page = int(document.metadata.get("page", 0))
        key = (filename, page)
        if page > 0 and key not in seen:
            seen.add(key)
            citations.append(SourceCitation(document=filename, page=page))
    return citations


def _format_chat_history(history: list[ConversationTurn]) -> str:
    """Render a bounded conversation history without treating it as evidence."""
    if not history:
        return "No prior conversation."
    return "\n".join(f"{turn.role.title()}: {turn.content}" for turn in history[-6:])


def answer_question(
    question: str,
    history: list[ConversationTurn] | None = None,
) -> RAGResponse:
    """Answer a question strictly from indexed company-policy content."""
    cleaned_question = question.strip()
    if not cleaned_question:
        raise ValueError("A question is required.")

    settings = get_settings()
    vector_store = get_vector_store()
    if collection_document_count(vector_store) == 0:
        raise VectorStoreError("No document embeddings exist. Run the ingestion script first.")

    retrieval_started = perf_counter()
    try:
        matches = vector_store.similarity_search_with_relevance_scores(
            cleaned_question,
            k=settings.retrieval_k,
        )
    except Exception as exc:
        logger.exception("Similarity search failed")
        raise VectorStoreError("The vector database could not retrieve company documents.") from exc

    retrieved_documents = [
        document
        for document, score in matches
        if score >= settings.similarity_score_threshold
    ]
    logger.info(
        "Retrieved %s relevant chunks in %.3f seconds",
        len(retrieved_documents),
        perf_counter() - retrieval_started,
    )
    if not retrieved_documents:
        return RAGResponse(answer=NO_ANSWER_MESSAGE, sources=[])

    prompt_messages = get_rag_prompt().format_messages(
        context=_format_context(retrieved_documents),
        chat_history=_format_chat_history(history or []),
        question=cleaned_question,
    )
    response_started = perf_counter()
    answer = generate_answer(prompt_messages)
    logger.info("Generated local LLM response in %.3f seconds", perf_counter() - response_started)
    return RAGResponse(answer=answer, sources=_citations(retrieved_documents))
