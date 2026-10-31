"""Ollama Embedding Service for RAG vector generation.

Generates dense vector embeddings using local Ollama embedding models (e.g., nomic-embed-text)
with strict dimension consistency validation, robust error handling, and safe logging.
"""

import logging
import math
from typing import Any, Dict, List, Optional
import httpx

from app.core.config import settings

logger = logging.getLogger(__name__)


# ============================================================================
# Embedding Exceptions
# ============================================================================


class EmbeddingServiceError(Exception):
    """Base exception for embedding generation failures."""

    pass


class EmbeddingConnectionError(EmbeddingServiceError):
    """Raised when the Ollama embedding service cannot be reached."""

    pass


class EmbeddingTimeoutError(EmbeddingServiceError):
    """Raised when an embedding request to Ollama times out."""

    pass


class EmbeddingModelNotFoundError(EmbeddingServiceError):
    """Raised when the requested embedding model is not installed or found in Ollama."""

    pass


class EmbeddingResponseError(EmbeddingServiceError):
    """Raised when Ollama returns an unexpected HTTP error or malformed response structure."""

    pass


class EmbeddingDimensionMismatchError(EmbeddingServiceError):
    """Raised when returned embedding vectors have inconsistent dimensions."""

    pass


# ============================================================================
# Embedding Service Implementation
# ============================================================================


class EmbeddingService:
    """Service wrapper for generating numeric vector embeddings using Ollama.

    Attributes:
        base_url: Ollama base HTTP URL (e.g. http://localhost:11434).
        model: Default embedding model name (e.g. nomic-embed-text).
        timeout: HTTP request timeout in seconds.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[float] = None,
    ):
        self.base_url = (base_url or settings.OLLAMA_BASE_URL).rstrip("/")
        self.model = model or settings.OLLAMA_EMBEDDING_MODEL
        self.timeout = timeout or settings.OLLAMA_TIMEOUT_SECONDS

    async def check_health(self) -> Dict[str, Any]:
        """Checks if Ollama is reachable and verifies if the configured embedding model is installed.

        Returns:
            Dict containing connection status, configured embedding model, and availability.
        """
        url = f"{self.base_url}/api/tags"
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(url)
                response.raise_for_status()
                data = response.json()
        except (httpx.ConnectError, httpx.NetworkError) as exc:
            logger.error("Failed to connect to Ollama embedding service at %s: %s", url, exc)
            raise EmbeddingConnectionError(
                "Unable to connect to Ollama service. Please ensure Ollama is running."
            ) from exc
        except (httpx.ConnectTimeout, httpx.ReadTimeout, httpx.WriteTimeout) as exc:
            logger.error("Timeout during Ollama embedding health check at %s: %s", url, exc)
            raise EmbeddingTimeoutError("Connection to Ollama service timed out.") from exc
        except httpx.HTTPStatusError as exc:
            logger.error("Ollama /api/tags returned HTTP status %s", exc.response.status_code)
            raise EmbeddingResponseError(
                f"Ollama returned HTTP {exc.response.status_code} while checking models."
            ) from exc
        except Exception as exc:
            logger.error("Unexpected error during Ollama embedding health check: %s", exc)
            raise EmbeddingResponseError("Received malformed response from Ollama server.") from exc

        raw_models = data.get("models", [])
        available_models: List[str] = [
            m.get("name") for m in raw_models if isinstance(m, dict) and m.get("name")
        ]

        configured = self.model
        model_available = False
        for m in available_models:
            if m == configured or m.startswith(f"{configured}:") or configured.startswith(f"{m}:"):
                model_available = True
                break

        return {
            "ollama_connected": True,
            "configured_embedding_model": self.model,
            "model_available": model_available,
            "available_models": available_models,
        }

    @staticmethod
    def _validate_vector(vector: Any, expected_dim: Optional[int] = None) -> List[float]:
        """Validates that a vector is a non-empty list of finite floating-point numbers.

        Args:
            vector: Raw vector returned from model response.
            expected_dim: Optional expected dimension for consistency verification.

        Returns:
            Validated list of floats.

        Raises:
            EmbeddingResponseError: If vector is not a list, is empty, or contains non-numeric/NaN values.
            EmbeddingDimensionMismatchError: If vector dimension does not match expected_dim.
        """
        if not isinstance(vector, list) or len(vector) == 0:
            raise EmbeddingResponseError("Ollama returned an empty or non-list embedding vector.")

        validated: List[float] = []
        for val in vector:
            if not isinstance(val, (int, float)) or isinstance(val, bool) or math.isnan(val) or math.isinf(val):
                raise EmbeddingResponseError("Embedding vector contains non-numeric, NaN, or infinite values.")
            validated.append(float(val))

        dim = len(validated)
        if expected_dim is not None and dim != expected_dim:
            raise EmbeddingDimensionMismatchError(
                f"Embedding vector dimension mismatch: expected {expected_dim}, but got {dim}."
            )

        return validated

    async def _post_embed(
        self,
        endpoint: str,
        payload: Dict[str, Any],
    ) -> httpx.Response:
        """Sends an HTTP POST request to an Ollama embedding endpoint with standardized error mapping."""
        url = f"{self.base_url}{endpoint}"
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(url, json=payload)
        except (httpx.ConnectError, httpx.NetworkError) as exc:
            logger.error("Failed to connect to Ollama embedding endpoint at %s: %s", url, exc)
            raise EmbeddingConnectionError(
                "Unable to connect to Ollama embedding service. Please ensure Ollama is running."
            ) from exc
        except (httpx.ConnectTimeout, httpx.ReadTimeout, httpx.WriteTimeout) as exc:
            logger.error("Timeout during Ollama embedding request to %s: %s", url, exc)
            raise EmbeddingTimeoutError("Embedding request to Ollama timed out.") from exc
        except Exception as exc:
            logger.error("Unexpected network error while contacting Ollama at %s: %s", url, exc)
            raise EmbeddingResponseError(f"Failed to communicate with Ollama service: {exc}") from exc

        # Handle HTTP error statuses
        if response.status_code == 404:
            try:
                error_body = response.json()
                error_msg = str(error_body.get("error", "")).lower()
            except Exception:
                error_msg = response.text.lower()

            if "not found" in error_msg and "model" in error_msg:
                target_model = payload.get("model", self.model)
                logger.error("Embedding model '%s' not found on Ollama server.", target_model)
                raise EmbeddingModelNotFoundError(
                    f"Embedding model '{target_model}' not found on Ollama server. "
                    "Ensure the model is pulled using 'ollama pull nomic-embed-text'."
                )
            # Route 404 (e.g. /api/embed not found on older Ollama versions)
            return response

        if response.is_error:
            logger.error("Ollama %s returned HTTP status %s", endpoint, response.status_code)
            raise EmbeddingResponseError(
                f"Ollama returned HTTP error {response.status_code} during embedding generation."
            )

        return response

    async def embed_chunks(
        self,
        texts: List[str],
        model: Optional[str] = None,
        batch_size: int = 32,
    ) -> List[List[float]]:
        """Generates embedding vectors for a list of text chunks.

        Note on Privacy: Full text contents are strictly omitted from logs. Only chunk counts
        and character summaries are logged.

        Args:
            texts: List of text chunk strings to embed.
            model: Optional model override. Defaults to self.model.
            batch_size: Maximum number of chunks to send per Ollama HTTP request.

        Returns:
            List of embedding vectors (each vector is a list of floats) with consistent dimensions.

        Raises:
            EmbeddingConnectionError: If connection cannot be established.
            EmbeddingTimeoutError: If the request times out.
            EmbeddingModelNotFoundError: If the model is not installed.
            EmbeddingResponseError: If the response is malformed or invalid.
            EmbeddingDimensionMismatchError: If vectors have inconsistent dimensions.
        """
        if not texts:
            return []

        target_model = model or self.model
        all_embeddings: List[List[float]] = []
        expected_dimension: Optional[int] = None

        logger.info(
            "Generating embeddings for %d text chunks using model '%s' (batch_size=%d).",
            len(texts),
            target_model,
            batch_size,
        )

        for batch_idx in range(0, len(texts), batch_size):
            batch = texts[batch_idx : batch_idx + batch_size]
            batch_vectors = await self._embed_batch_with_fallback(batch, target_model)

            for vec in batch_vectors:
                # If we haven't established expected dimension yet, learn from first vector
                if expected_dimension is None:
                    validated_vec = self._validate_vector(vec)
                    expected_dimension = len(validated_vec)
                else:
                    validated_vec = self._validate_vector(vec, expected_dim=expected_dimension)

                all_embeddings.append(validated_vec)

        logger.info(
            "Successfully generated %d embedding vectors of dimension %d.",
            len(all_embeddings),
            expected_dimension or 0,
        )
        return all_embeddings

    async def _embed_batch_with_fallback(
        self,
        batch: List[str],
        model: str,
    ) -> List[List[float]]:
        """Attempts to embed a batch using /api/embed, falling back to legacy /api/embeddings if /api/embed is unavailable."""
        # 1. Primary endpoint: /api/embed (Ollama 0.1.44+)
        payload = {
            "model": model,
            "input": batch,
        }
        response = await self._post_embed("/api/embed", payload)

        if response.status_code == 200:
            try:
                data = response.json()
            except Exception as exc:
                raise EmbeddingResponseError("Ollama /api/embed returned non-JSON response.") from exc

            raw_embeddings = data.get("embeddings")
            if not isinstance(raw_embeddings, list):
                raise EmbeddingResponseError(
                    "Ollama /api/embed response missing 'embeddings' list."
                )

            if len(raw_embeddings) != len(batch):
                raise EmbeddingResponseError(
                    f"Ollama returned {len(raw_embeddings)} embeddings for {len(batch)} input chunks."
                )

            return raw_embeddings

        # 2. If /api/embed returned 404 (endpoint not supported in older Ollama versions), fall back to /api/embeddings
        if response.status_code == 404:
            logger.info("Ollama /api/embed not found; attempting legacy /api/embeddings endpoint.")
            fallback_vectors: List[List[float]] = []
            for text_item in batch:
                legacy_payload = {"model": model, "prompt": text_item}
                legacy_resp = await self._post_embed("/api/embeddings", legacy_payload)
                if legacy_resp.status_code != 200:
                    raise EmbeddingResponseError(
                        f"Legacy Ollama /api/embeddings returned HTTP {legacy_resp.status_code}."
                    )
                try:
                    legacy_data = legacy_resp.json()
                except Exception as exc:
                    raise EmbeddingResponseError("Legacy Ollama /api/embeddings returned non-JSON response.") from exc

                vec = legacy_data.get("embedding")
                if not isinstance(vec, list):
                    raise EmbeddingResponseError("Legacy Ollama response missing 'embedding' list.")
                fallback_vectors.append(vec)

            return fallback_vectors

        # Any other unexpected non-200 code
        raise EmbeddingResponseError(
            f"Ollama /api/embed returned unexpected HTTP status {response.status_code}."
        )

    async def embed_text(
        self,
        text: str,
        model: Optional[str] = None,
    ) -> List[float]:
        """Generates an embedding vector for a single text string.

        Args:
            text: Text string to embed.
            model: Optional model override. Defaults to self.model.

        Returns:
            List of floats representing the embedding vector.

        Raises:
            EmbeddingResponseError: If text is empty or response is malformed.
        """
        if not text or not text.strip():
            raise EmbeddingResponseError("Cannot generate embedding for empty or whitespace-only text.")

        vectors = await self.embed_chunks([text], model=model)
        if not vectors:
            raise EmbeddingResponseError("Ollama failed to return an embedding for the provided text.")
        return vectors[0]


def get_embedding_service() -> EmbeddingService:
    """FastAPI dependency provider for EmbeddingService."""
    return EmbeddingService()
