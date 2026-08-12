"""FastAPI application entry point for the onboarding assistant."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.config import get_settings
from backend.routes import router


def configure_logging() -> None:
    """Configure application-wide structured console logging."""
    settings = get_settings()
    logging.basicConfig(
        level=settings.log_level.upper(),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Log deterministic application startup and shutdown events."""
    configure_logging()
    settings = get_settings()
    logging.getLogger(__name__).info(
        "Starting %s in %s mode", settings.app_name, settings.app_environment
    )
    yield
    logging.getLogger(__name__).info("Shutting down %s", settings.app_name)


settings = get_settings()
app = FastAPI(
    title=settings.app_name,
    version="1.0.0",
    description="A local, document-grounded employee onboarding assistant.",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
app.include_router(router)


@app.get("/", tags=["health"])
def health_check() -> dict[str, str]:
    """Return a minimal service-health payload."""
    return {"status": "running"}

