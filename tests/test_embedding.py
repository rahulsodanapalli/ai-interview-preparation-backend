"""Unit tests for EmbeddingService (RAG Phase 3).

Validates vector generation, response validation, dimension consistency checks,
timeout and connection failure handling, model-not-found detection, legacy fallback,
and privacy-safe logging using isolated mocks without external network calls.
"""

import logging
from unittest.mock import AsyncMock, patch
import httpx
import pytest

from app.services.embedding_service import (
    EmbeddingConnectionError,
    EmbeddingDimensionMismatchError,
    EmbeddingModelNotFoundError,
    EmbeddingResponseError,
    EmbeddingService,
    EmbeddingTimeoutError,
    get_embedding_service,
)

# Standard mock 768-dimensional vector matching nomic-embed-text
MOCK_DIM = 768
MOCK_VECTOR_A = [0.01 * (i % 10) for i in range(MOCK_DIM)]
MOCK_VECTOR_B = [0.02 * (i % 10) for i in range(MOCK_DIM)]


@pytest.fixture
def embedding_service() -> EmbeddingService:
    """Fixture providing an EmbeddingService instance pointing to a test base URL."""
    return EmbeddingService(
        base_url="http://test-ollama:11434",
        model="nomic-embed-text",
        timeout=5.0,
    )


# ============================================================================
# Successful Embedding Generation Tests
# ============================================================================


@pytest.mark.anyio
async def test_embed_single_text_success(embedding_service: EmbeddingService):
    """Test generating an embedding for a single text chunk returns a valid 768-dim float vector."""
    mock_resp = httpx.Response(
        status_code=200,
        json={"model": "nomic-embed-text", "embeddings": [MOCK_VECTOR_A]},
        request=httpx.Request("POST", "http://test-ollama:11434/api/embed"),
    )

    with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock, return_value=mock_resp):
        vector = await embedding_service.embed_text("Sample candidate technical experience.")

    assert isinstance(vector, list)
    assert len(vector) == MOCK_DIM
    assert all(isinstance(val, float) for val in vector)


@pytest.mark.anyio
async def test_embed_chunks_batch_success(embedding_service: EmbeddingService):
    """Test generating embeddings for a batch of text chunks returns consistent vectors."""
    mock_resp = httpx.Response(
        status_code=200,
        json={"model": "nomic-embed-text", "embeddings": [MOCK_VECTOR_A, MOCK_VECTOR_B]},
        request=httpx.Request("POST", "http://test-ollama:11434/api/embed"),
    )

    with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock, return_value=mock_resp):
        vectors = await embedding_service.embed_chunks(
            ["Chunk one text.", "Chunk two text."],
            batch_size=10,
        )

    assert len(vectors) == 2
    assert len(vectors[0]) == MOCK_DIM
    assert len(vectors[1]) == MOCK_DIM


@pytest.mark.anyio
async def test_embed_chunks_empty_list_returns_immediately(embedding_service: EmbeddingService):
    """Test that embedding an empty list of chunks returns [] without making any HTTP request."""
    with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock) as mock_post:
        result = await embedding_service.embed_chunks([])

    assert result == []
    mock_post.assert_not_called()


@pytest.mark.anyio
async def test_embed_text_empty_rejected(embedding_service: EmbeddingService):
    """Test that empty or whitespace-only text raises EmbeddingResponseError."""
    with pytest.raises(EmbeddingResponseError, match="Cannot generate embedding for empty"):
        await embedding_service.embed_text("")

    with pytest.raises(EmbeddingResponseError, match="Cannot generate embedding for empty"):
        await embedding_service.embed_text("   \n\t   ")


# ============================================================================
# Vector Validation & Consistency Tests
# ============================================================================


@pytest.mark.anyio
async def test_embed_inconsistent_dimensions_raises_error(embedding_service: EmbeddingService):
    """Test that inconsistent vector dimensions in a batch raises EmbeddingDimensionMismatchError."""
    vector_512 = [0.1] * 512
    mock_resp = httpx.Response(
        status_code=200,
        json={"model": "nomic-embed-text", "embeddings": [MOCK_VECTOR_A, vector_512]},
        request=httpx.Request("POST", "http://test-ollama:11434/api/embed"),
    )

    with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock, return_value=mock_resp):
        with pytest.raises(EmbeddingDimensionMismatchError, match="dimension mismatch"):
            await embedding_service.embed_chunks(["Chunk A", "Chunk B"])


@pytest.mark.anyio
async def test_embed_non_numeric_elements_raises_error(embedding_service: EmbeddingService):
    """Test that non-numeric, NaN, boolean, or infinite elements raise EmbeddingResponseError."""
    # Test JSON-serializable non-numeric types through HTTP mock
    invalid_vectors = [
        ["invalid_string", 0.2, 0.3],
        [None, 0.2, 0.3],
        [True, 0.2, 0.3],
    ]

    for inv_vec in invalid_vectors:
        mock_resp = httpx.Response(
            status_code=200,
            json={"model": "nomic-embed-text", "embeddings": [inv_vec]},
            request=httpx.Request("POST", "http://test-ollama:11434/api/embed"),
        )
        with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock, return_value=mock_resp):
            with pytest.raises(EmbeddingResponseError, match="non-numeric"):
                await embedding_service.embed_chunks(["Chunk test"])

    # Test NaN and Infinity directly via _validate_vector
    with pytest.raises(EmbeddingResponseError, match="non-numeric"):
        EmbeddingService._validate_vector([float("nan"), 0.2, 0.3])

    with pytest.raises(EmbeddingResponseError, match="non-numeric"):
        EmbeddingService._validate_vector([float("inf"), 0.2, 0.3])


@pytest.mark.anyio
async def test_embed_empty_vector_raises_error(embedding_service: EmbeddingService):
    """Test that an empty vector in embeddings list raises EmbeddingResponseError."""
    mock_resp = httpx.Response(
        status_code=200,
        json={"model": "nomic-embed-text", "embeddings": [[]]},
        request=httpx.Request("POST", "http://test-ollama:11434/api/embed"),
    )

    with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock, return_value=mock_resp):
        with pytest.raises(EmbeddingResponseError, match="empty or non-list"):
            await embedding_service.embed_chunks(["Chunk test"])


@pytest.mark.anyio
async def test_embed_missing_embeddings_key(embedding_service: EmbeddingService):
    """Test that response missing the 'embeddings' key raises EmbeddingResponseError."""
    mock_resp = httpx.Response(
        status_code=200,
        json={"model": "nomic-embed-text"},
        request=httpx.Request("POST", "http://test-ollama:11434/api/embed"),
    )

    with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock, return_value=mock_resp):
        with pytest.raises(EmbeddingResponseError, match="missing 'embeddings' list"):
            await embedding_service.embed_chunks(["Chunk test"])


@pytest.mark.anyio
async def test_embed_count_mismatch_raises_error(embedding_service: EmbeddingService):
    """Test that a mismatch between input chunk count and returned vector count raises EmbeddingResponseError."""
    mock_resp = httpx.Response(
        status_code=200,
        json={"model": "nomic-embed-text", "embeddings": [MOCK_VECTOR_A]},  # Only 1 vector for 2 inputs
        request=httpx.Request("POST", "http://test-ollama:11434/api/embed"),
    )

    with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock, return_value=mock_resp):
        with pytest.raises(EmbeddingResponseError, match="returned 1 embeddings for 2 input chunks"):
            await embedding_service.embed_chunks(["Chunk 1", "Chunk 2"])


# ============================================================================
# Failure & Network Error Handling Tests
# ============================================================================


@pytest.mark.anyio
async def test_embed_model_not_found(embedding_service: EmbeddingService):
    """Test that Ollama 404 with 'model not found' raises EmbeddingModelNotFoundError."""
    mock_resp = httpx.Response(
        status_code=404,
        json={"error": "model 'nomic-embed-text' not found, try pulling it first"},
        request=httpx.Request("POST", "http://test-ollama:11434/api/embed"),
    )

    with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock, return_value=mock_resp):
        with pytest.raises(EmbeddingModelNotFoundError, match="Embedding model 'nomic-embed-text' not found"):
            await embedding_service.embed_chunks(["Chunk text"])


@pytest.mark.anyio
async def test_embed_connection_failure(embedding_service: EmbeddingService):
    """Test that network connection refusal raises EmbeddingConnectionError."""
    with patch.object(
        httpx.AsyncClient,
        "post",
        new_callable=AsyncMock,
        side_effect=httpx.ConnectError("Connection refused"),
    ):
        with pytest.raises(EmbeddingConnectionError, match="Unable to connect to Ollama"):
            await embedding_service.embed_chunks(["Chunk text"])


@pytest.mark.anyio
async def test_embed_timeout_failure(embedding_service: EmbeddingService):
    """Test that request timeout raises EmbeddingTimeoutError."""
    with patch.object(
        httpx.AsyncClient,
        "post",
        new_callable=AsyncMock,
        side_effect=httpx.ReadTimeout("Read timed out"),
    ):
        with pytest.raises(EmbeddingTimeoutError, match="timed out"):
            await embedding_service.embed_chunks(["Chunk text"])


@pytest.mark.anyio
async def test_embed_server_500_error(embedding_service: EmbeddingService):
    """Test that HTTP 500 server error raises EmbeddingResponseError."""
    mock_resp = httpx.Response(
        status_code=500,
        text="Internal Server Error",
        request=httpx.Request("POST", "http://test-ollama:11434/api/embed"),
    )

    with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock, return_value=mock_resp):
        with pytest.raises(EmbeddingResponseError, match="HTTP error 500"):
            await embedding_service.embed_chunks(["Chunk text"])


# ============================================================================
# Legacy Endpoint Fallback Tests
# ============================================================================


@pytest.mark.anyio
async def test_embed_legacy_fallback_when_api_embed_404(embedding_service: EmbeddingService):
    """Test fallback from /api/embed to legacy /api/embeddings if /api/embed is unsupported (404)."""
    resp_404 = httpx.Response(
        status_code=404,
        text="404 page not found",
        request=httpx.Request("POST", "http://test-ollama:11434/api/embed"),
    )
    resp_legacy = httpx.Response(
        status_code=200,
        json={"embedding": MOCK_VECTOR_A},
        request=httpx.Request("POST", "http://test-ollama:11434/api/embeddings"),
    )

    # First call to /api/embed returns 404, second call to /api/embeddings returns 200
    with patch.object(
        httpx.AsyncClient,
        "post",
        new_callable=AsyncMock,
        side_effect=[resp_404, resp_legacy],
    ):
        vectors = await embedding_service.embed_chunks(["Chunk text"])

    assert len(vectors) == 1
    assert len(vectors[0]) == MOCK_DIM


# ============================================================================
# Health Check Tests
# ============================================================================


@pytest.mark.anyio
async def test_check_health_success(embedding_service: EmbeddingService):
    """Test Ollama embedding health check returns model availability."""
    mock_resp = httpx.Response(
        status_code=200,
        json={"models": [{"name": "nomic-embed-text:latest"}, {"name": "qwen2.5:1.5b"}]},
        request=httpx.Request("GET", "http://test-ollama:11434/api/tags"),
    )

    with patch.object(httpx.AsyncClient, "get", new_callable=AsyncMock, return_value=mock_resp):
        health = await embedding_service.check_health()

    assert health["ollama_connected"] is True
    assert health["model_available"] is True
    assert health["configured_embedding_model"] == "nomic-embed-text"


@pytest.mark.anyio
async def test_check_health_model_missing(embedding_service: EmbeddingService):
    """Test Ollama embedding health check reports model_available=False when model is absent."""
    mock_resp = httpx.Response(
        status_code=200,
        json={"models": [{"name": "llama3:latest"}]},
        request=httpx.Request("GET", "http://test-ollama:11434/api/tags"),
    )

    with patch.object(httpx.AsyncClient, "get", new_callable=AsyncMock, return_value=mock_resp):
        health = await embedding_service.check_health()

    assert health["ollama_connected"] is True
    assert health["model_available"] is False


@pytest.mark.anyio
async def test_check_health_connection_failure(embedding_service: EmbeddingService):
    """Test Ollama embedding health check raises EmbeddingConnectionError when unreachable."""
    with patch.object(
        httpx.AsyncClient,
        "get",
        new_callable=AsyncMock,
        side_effect=httpx.ConnectError("Connection refused"),
    ):
        with pytest.raises(EmbeddingConnectionError):
            await embedding_service.check_health()


# ============================================================================
# Privacy & Security Logging Tests
# ============================================================================


@pytest.mark.anyio
async def test_logging_privacy_never_logs_sensitive_text(
    embedding_service: EmbeddingService,
    caplog: pytest.LogCaptureFixture,
):
    """Verify that complete resume text and sensitive content are never written to application logs."""
    sensitive_resume = "SECRET_CANDIDATE_RESUME_IDENTIFIER_SSN_999-88-7777"
    mock_resp = httpx.Response(
        status_code=200,
        json={"model": "nomic-embed-text", "embeddings": [MOCK_VECTOR_A]},
        request=httpx.Request("POST", "http://test-ollama:11434/api/embed"),
    )

    with caplog.at_level(logging.DEBUG):
        with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock, return_value=mock_resp):
            await embedding_service.embed_text(sensitive_resume)

    # Check all captured log output
    full_log_output = " ".join(record.message for record in caplog.records)
    assert sensitive_resume not in full_log_output
    assert "SECRET_CANDIDATE" not in full_log_output


def test_get_embedding_service_provider():
    """Test FastAPI dependency provider returns a configured EmbeddingService instance."""
    service = get_embedding_service()
    assert isinstance(service, EmbeddingService)
    assert service.model == "nomic-embed-text"
    assert service.timeout > 0
