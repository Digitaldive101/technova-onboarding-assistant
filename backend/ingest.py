"""Build the persistent ChromaDB index from fixed company PDF documents.

Run from the project directory with: ``python backend/ingest.py``.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend.config import get_settings
from backend.pdf_loader import DocumentLoadError, load_company_documents
from backend.vector_store import VectorStoreError, rebuild_vector_store


def configure_logging() -> None:
    """Configure concise logging for command-line ingestion runs."""
    settings = get_settings()
    logging.basicConfig(
        level=settings.log_level.upper(),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def ingest_documents() -> int:
    """Load, split, embed, and persist all approved company PDFs."""
    logger = logging.getLogger(__name__)
    settings = get_settings()
    logger.info("Starting document ingestion from %s", settings.data_directory)

    pages = load_company_documents()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        length_function=len,
    )
    chunks = splitter.split_documents(pages)
    if not chunks:
        raise DocumentLoadError("Document loading succeeded, but no text chunks were produced.")

    logger.info("Split %s pages into %s chunks", len(pages), len(chunks))
    indexed_count = rebuild_vector_store(chunks)
    logger.info("Ingestion complete: %s chunks indexed", indexed_count)
    return indexed_count


def main() -> None:
    """Run ingestion and provide a non-zero exit code for operational failures."""
    configure_logging()
    try:
        count = ingest_documents()
    except (DocumentLoadError, VectorStoreError, RuntimeError) as exc:
        logging.getLogger(__name__).error("Ingestion failed: %s", exc)
        raise SystemExit(1) from exc
    print(f"Successfully indexed {count} document chunks.")


if __name__ == "__main__":
    main()

