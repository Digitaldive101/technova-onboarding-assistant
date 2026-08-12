"""Persistent ChromaDB access for the fixed onboarding knowledge base."""

from __future__ import annotations

import logging
from collections.abc import Sequence

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document

from backend.config import get_settings
from backend.embeddings import get_embeddings


logger = logging.getLogger(__name__)


class VectorStoreError(RuntimeError):
    """Raised when the local Chroma vector database cannot be used."""


def get_vector_store() -> Chroma:
    """Open the configured persistent Chroma collection."""
    settings = get_settings()
    try:
        settings.chroma_directory.mkdir(parents=True, exist_ok=True)
        return Chroma(
            collection_name=settings.collection_name,
            persist_directory=str(settings.chroma_directory),
            embedding_function=get_embeddings(),
            # Chroma returns L2 distance for normalized embeddings (0 is best,
            # 2 is worst). Convert it to the 0-1 relevance scale used by RAG.
            relevance_score_fn=lambda distance: max(0.0, 1.0 - (distance / 2.0)),
        )
    except Exception as exc:
        logger.exception("Unable to open ChromaDB at %s", settings.chroma_directory)
        raise VectorStoreError("The local vector database is unavailable.") from exc


def collection_document_count(vector_store: Chroma | None = None) -> int:
    """Return the number of indexed chunks in the active collection."""
    try:
        if vector_store is not None:
            return int(vector_store._collection.count())

        # Counting does not require an embedding model. This keeps health and
        # sidebar status checks responsive when outbound network access is off.
        settings = get_settings()
        settings.chroma_directory.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(settings.chroma_directory))
        collection = client.get_or_create_collection(settings.collection_name)
        return int(collection.count())
    except Exception as exc:
        logger.exception("Unable to read ChromaDB collection count")
        raise VectorStoreError("Could not read the vector database collection.") from exc


def rebuild_vector_store(documents: Sequence[Document]) -> int:
    """Replace all indexed chunks with ``documents`` and return their count."""
    if not documents:
        raise VectorStoreError("Cannot build a vector database from zero document chunks.")

    settings = get_settings()
    try:
        store = get_vector_store()
        store.delete_collection()
        logger.info("Cleared existing Chroma collection: %s", settings.collection_name)

        replacement_store = get_vector_store()
        replacement_store.add_documents(list(documents))
        indexed_count = collection_document_count(replacement_store)
        logger.info("Stored %s chunks in ChromaDB", indexed_count)
        return indexed_count
    except VectorStoreError:
        raise
    except Exception as exc:
        logger.exception("Unable to rebuild the ChromaDB collection")
        raise VectorStoreError("The vector database could not be rebuilt.") from exc
