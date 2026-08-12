"""Operational helpers shared by backend components."""

from __future__ import annotations

from dataclasses import dataclass

from backend.pdf_loader import DocumentLoadError, find_company_pdfs
from backend.vector_store import VectorStoreError, collection_document_count


@dataclass(frozen=True)
class KnowledgeBaseStatus:
    """A concise snapshot of the fixed document corpus and its index."""

    document_count: int
    indexed_chunk_count: int
    is_ready: bool


def get_knowledge_base_status() -> KnowledgeBaseStatus:
    """Return corpus and index readiness without loading PDF content."""
    try:
        document_count = len(find_company_pdfs())
    except DocumentLoadError:
        document_count = 0

    try:
        indexed_chunk_count = collection_document_count()
    except VectorStoreError:
        indexed_chunk_count = 0

    return KnowledgeBaseStatus(
        document_count=document_count,
        indexed_chunk_count=indexed_chunk_count,
        is_ready=document_count > 0 and indexed_chunk_count > 0,
    )

