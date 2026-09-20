"""Embedding-model factory used by ingestion and retrieval."""

from __future__ import annotations

import logging
from functools import lru_cache

from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings

from backend.config import get_settings


logger = logging.getLogger(__name__)


@lru_cache
def get_embeddings() -> Embeddings:
    """Create and cache the configured local sentence-transformer embedder.

    The model is loaded only when the vector store is first used, avoiding a
    slow application import and ensuring ingestion and serving use identical
    embedding settings.
    """
    settings = get_settings()
    logger.info("Loading embedding model: %s", settings.embedding_model)

    try:
        return HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            model_kwargs={
                "device": "cpu",
                "local_files_only": settings.embedding_local_files_only,
            },
            encode_kwargs={"normalize_embeddings": True},
        )
    except Exception as exc:
        logger.exception("Unable to initialize the embedding model")
        raise RuntimeError(
            "The embedding model could not be initialized. Ensure its "
            "dependencies are installed and the model is available locally."
        ) from exc
