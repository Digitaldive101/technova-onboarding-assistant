"""Safe loading of fixed company-policy PDF documents."""

from __future__ import annotations

import logging
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

from backend.config import get_settings


logger = logging.getLogger(__name__)


class DocumentLoadError(RuntimeError):
    """Raised when no readable company policy documents are available."""


def find_company_pdfs(directory: Path | None = None) -> list[Path]:
    """Return the sorted list of PDF files in the fixed knowledge directory."""
    data_directory = directory or get_settings().data_directory
    if not data_directory.exists():
        raise DocumentLoadError(f"Company document directory does not exist: {data_directory}")
    if not data_directory.is_dir():
        raise DocumentLoadError(f"Company document path is not a directory: {data_directory}")

    pdf_paths = sorted(path for path in data_directory.iterdir() if path.suffix.lower() == ".pdf")
    if not pdf_paths:
        raise DocumentLoadError(f"No PDF documents were found in: {data_directory}")
    return pdf_paths


def load_company_documents(directory: Path | None = None) -> list[Document]:
    """Load all readable policy PDFs and attach citation-friendly metadata.

    Corrupt or password-protected files are logged and skipped so a single bad
    document cannot prevent the rest of the fixed knowledge base from indexing.
    """
    documents: list[Document] = []
    failed_files: list[str] = []

    for pdf_path in find_company_pdfs(directory):
        try:
            pages = PyPDFLoader(str(pdf_path)).load()
            for page in pages:
                page.metadata["document"] = pdf_path.name
                page.metadata["page"] = int(page.metadata.get("page", 0)) + 1
                page.metadata["source"] = str(pdf_path)
            documents.extend(pages)
            logger.info("Loaded %s page(s) from %s", len(pages), pdf_path.name)
        except Exception:
            failed_files.append(pdf_path.name)
            logger.exception("Unable to load PDF: %s", pdf_path.name)

    if not documents:
        detail = f" Unreadable files: {', '.join(failed_files)}." if failed_files else ""
        raise DocumentLoadError(f"No readable company PDF documents were loaded.{detail}")

    if failed_files:
        logger.warning("Skipped unreadable PDFs: %s", ", ".join(failed_files))
    logger.info("Loaded %s page document(s) in total", len(documents))
    return documents

