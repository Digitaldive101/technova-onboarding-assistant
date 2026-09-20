"""Centralized, environment-driven settings for the onboarding assistant."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_DIR = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    """Application settings loaded from environment variables and ``.env``."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_DIR / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "AI Employee Onboarding Assistant"
    app_environment: str = "development"
    api_host: str = "0.0.0.0"
    api_port: int = Field(default=8000, ge=1, le=65535)
    log_level: str = "INFO"

    data_directory: Path = PROJECT_DIR / "data"
    chroma_directory: Path = PROJECT_DIR / "chroma_db"
    collection_name: str = "employee_onboarding_documents"

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_local_files_only: bool = False
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"
    llm_provider: str = "openrouter"
    openrouter_api_key: str | None = None
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_model: str = "liquid/lfm-2.5-2.6b:free"
    llm_request_timeout_seconds: int = Field(default=90, ge=10, le=300)
    llm_temperature: float = Field(default=0.0, ge=0.0, le=1.0)

    chunk_size: int = Field(default=1000, gt=0)
    chunk_overlap: int = Field(default=200, ge=0)
    retrieval_k: int = Field(default=5, gt=0)
    similarity_score_threshold: float = Field(default=0.10, ge=0.0, le=1.0)

    @field_validator("data_directory", "chroma_directory", mode="before")
    @classmethod
    def resolve_path(cls, value: str | Path) -> Path:
        """Resolve relative storage paths from the project directory."""
        path = Path(value)
        return path if path.is_absolute() else PROJECT_DIR / path

    @field_validator("chunk_overlap")
    @classmethod
    def validate_chunk_overlap(cls, value: int, info: object) -> int:
        """Ensure overlap is smaller than a chunk."""
        chunk_size = getattr(info, "data", {}).get("chunk_size")
        if chunk_size is not None and value >= chunk_size:
            raise ValueError("CHUNK_OVERLAP must be less than CHUNK_SIZE")
        return value


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide validated application configuration."""
    return Settings()
