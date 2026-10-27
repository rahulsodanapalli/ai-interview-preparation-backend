from functools import lru_cache
import json
from typing import Any, List, Optional, Union
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration settings loaded from environment variables."""

    APP_NAME: str = "AI Interview Preparation Platform"
    APP_ENV: str = "development"
    API_V1_PREFIX: str = "/api/v1"

    # CORS Configuration for local frontend integration
    CORS_ORIGINS: Union[str, List[str]] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]

    # Ollama Service Settings
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "qwen2.5:1.5b"
    OLLAMA_EMBEDDING_MODEL: str = "nomic-embed-text"
    OLLAMA_TIMEOUT_SECONDS: float = 60.0

    # Database Configuration (Neon PostgreSQL)
    DATABASE_URL: Optional[str] = None

    # Document Upload & Ingestion Settings
    MAX_UPLOAD_SIZE_BYTES: int = 10 * 1024 * 1024  # 10 MB limit
    MAX_PDF_PAGES: int = 100  # Reasonable limit to protect against DoS

    # Text Chunking Settings (RAG Phase 3)
    # Approximation: 1 token ≈ 4 characters in standard English text.
    # Target: ~500 tokens ≈ 2,000 characters; ~10% overlap ≈ 200 characters (~50 tokens).
    CHUNK_SIZE: int = 2000
    CHUNK_OVERLAP: int = 200

    # Pinecone Vector Database Settings (RAG Phase 4)
    PINECONE_API_KEY: Optional[str] = None
    PINECONE_INDEX_NAME: str = "ai-interview-prep-v2"
    PINECONE_NAMESPACE: str = "default"
    PINECONE_TOP_K: int = 4
    PINECONE_DIMENSION: int = 768
    PINECONE_METRIC: str = "cosine"

    # Semantic Retrieval & Context Grounding Settings (RAG Phase 5)
    # Minimum cosine similarity threshold for retrieved chunks using nomic-embed-text.
    # Excerpts scoring below this threshold are filtered out as irrelevant/noise.
    RAG_MIN_RELEVANCE_SCORE: float = 0.55

    @field_validator("CORS_ORIGINS")
    @classmethod
    def assemble_cors_origins(cls, v: Any) -> List[str]:
        if isinstance(v, str):
            trimmed = v.strip()
            if trimmed.startswith("[") and trimmed.endswith("]"):
                try:
                    return [str(item).strip() for item in json.loads(trimmed) if str(item).strip()]
                except Exception:
                    pass
            return [item.strip() for item in trimmed.split(",") if item.strip()]
        elif isinstance(v, (list, tuple, set)):
            return [str(item).strip() for item in v if str(item).strip()]
        return ["http://localhost:5173", "http://127.0.0.1:5173"]

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """Return a cached instance of the application settings."""
    return Settings()


settings = get_settings()
