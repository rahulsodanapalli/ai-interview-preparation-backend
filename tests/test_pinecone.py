"""Unit tests for PineconeService and Pinecone PDF upload integration (RAG Standard Dense Vectors).

Validates:
- Standard dense vector index verification and dimension compatibility.
- Document upserts via index.upsert() with (id, embedding, metadata) tuples.
- Numeric embeddings are stored strictly in vector values and NEVER in metadata.
- Semantic retrieval via index.query() with document isolation filter {"document_id": {"$eq": ...}}.
- Document deletion via index.delete() with metadata filter.
- Dimension checks (strictly 768-dim), configuration errors, timeouts, service failures, and upload pipeline integration.
All tests run with isolated mocks without requiring a live Pinecone account or Ollama server.
"""

from unittest.mock import MagicMock
from fastapi.testclient import TestClient
import pytest

from app.core.config import settings
from app.main import app
from app.services.chunking_service import TextChunk
from app.services.embedding_service import EmbeddingConnectionError, get_embedding_service
from app.services.pinecone_service import (
    PineconeConfigurationError,
    PineconeConnectionError,
    PineconeDimensionMismatchError,
    PineconeIndexNotFoundError,
    PineconeResponseError,
    PineconeService,
    PineconeTimeoutError,
    get_pinecone_service,
)
from tests.test_documents import create_test_pdf

client = TestClient(app)

MOCK_DIM = 768
MOCK_VECTOR_A = [0.01 * (i % 10) for i in range(MOCK_DIM)]
MOCK_VECTOR_B = [0.02 * (i % 10) for i in range(MOCK_DIM)]


@pytest.fixture
def mock_pinecone_service() -> PineconeService:
    """Fixture providing a configured PineconeService instance with mocked client and standard vector index."""
    service = PineconeService(
        api_key="mock-test-key-12345",
        index_name="ai-interview-prep-v2",
        namespace="default",
        top_k=4,
        dimension=MOCK_DIM,
        metric="cosine",
    )
    mock_client = MagicMock()
    mock_index = MagicMock()

    service._client = mock_client
    service._index = mock_index
    return service


# ============================================================================
# Configuration & Client Initialization Tests
# ============================================================================


def test_pinecone_missing_api_key_raises_configuration_error(monkeypatch):
    """Test that attempting operations without PINECONE_API_KEY raises PineconeConfigurationError."""
    monkeypatch.setattr(settings, "PINECONE_API_KEY", "")

    service = PineconeService(api_key=None)
    with pytest.raises(PineconeConfigurationError, match="API key is not configured"):
        service.get_client()

    empty_service = PineconeService(api_key="   ")
    with pytest.raises(PineconeConfigurationError, match="API key is not configured"):
        empty_service.get_client()


def test_pinecone_client_initialization_failure():
    """Test that client initialization failures raise PineconeConnectionError."""
    service = PineconeService(api_key="valid-looking-key")
    with pytest.raises(Exception):
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("app.services.pinecone_service.Pinecone", MagicMock(side_effect=Exception("Connection failed")))
            service.get_client()


# ============================================================================
# Index Verification Tests (Standard Dense-Vector Index)
# ============================================================================


@pytest.mark.anyio
async def test_pinecone_verify_index_success(mock_pinecone_service: PineconeService):
    """Test verify_index succeeds when standard index exists and has matching dimension and metric."""
    mock_pinecone_service._client.has_index.return_value = True
    mock_pinecone_service._client.describe_index.return_value = {
        "dimension": MOCK_DIM,
        "metric": "cosine",
    }

    result = await mock_pinecone_service.verify_index()
    assert result["index_name"] == "ai-interview-prep-v2"
    assert result["dimension"] == MOCK_DIM
    assert result["metric"] == "cosine"
    assert result["status"] == "ready"


@pytest.mark.anyio
async def test_pinecone_verify_index_not_found(mock_pinecone_service: PineconeService):
    """Test verify_index raises PineconeIndexNotFoundError when index does not exist."""
    mock_pinecone_service._client.has_index.return_value = False

    with pytest.raises(PineconeIndexNotFoundError, match="does not exist"):
        await mock_pinecone_service.verify_index()


@pytest.mark.anyio
async def test_pinecone_verify_index_dimension_mismatch(mock_pinecone_service: PineconeService):
    """Test verify_index raises PineconeDimensionMismatchError when index dimension differs."""
    mock_pinecone_service._client.has_index.return_value = True
    mock_pinecone_service._client.describe_index.return_value = {
        "dimension": 1024,  # e.g. 1024-dim index instead of 768
        "metric": "cosine",
    }

    with pytest.raises(PineconeDimensionMismatchError, match="has dimension 1024"):
        await mock_pinecone_service.verify_index()


@pytest.mark.anyio
async def test_pinecone_verify_index_rejects_document_schema(mock_pinecone_service: PineconeService):
    """Test verify_index raises PineconeDimensionMismatchError when index has a document schema rather than standard vector index."""
    mock_pinecone_service._client.has_index.return_value = True
    mock_pinecone_service._client.describe_index.return_value = {
        "dimension": MOCK_DIM,
        "metric": "cosine",
        "schema": {
            "fields": {
                "chunk_text": {"type": "string"},
            }
        },
    }

    with pytest.raises(PineconeDimensionMismatchError, match="configured as a document index with schema fields"):
        await mock_pinecone_service.verify_index()


# ============================================================================
# Vector Upsert & Stable ID Tests (Standard Vector Index)
# ============================================================================


@pytest.mark.anyio
async def test_pinecone_upsert_document_chunks_standard_vectors(mock_pinecone_service: PineconeService):
    """Test upserting chunks passes (id, embedding, metadata) tuples with embedding strictly in values, NEVER in metadata."""
    chunks = [
        TextChunk(index=0, text="First chunk text.", character_count=17, token_estimate=4),
        TextChunk(index=1, text="Second chunk text.", character_count=18, token_estimate=5),
    ]
    vectors = [MOCK_VECTOR_A, MOCK_VECTOR_B]

    count = await mock_pinecone_service.upsert_document_chunks(
        document_id="doc_test123",
        chunks=chunks,
        vectors=vectors,
        filename="resume.pdf",
    )

    assert count == 2
    mock_index = mock_pinecone_service._index
    mock_index.upsert.assert_called_once()
    _, kwargs = mock_index.upsert.call_args
    assert kwargs["namespace"] == "default"
    records = kwargs["vectors"]
    assert len(records) == 2

    # Verify first vector record structure: (id, values, metadata)
    rec_0 = records[0]
    assert isinstance(rec_0, tuple)
    rec_0_id, rec_0_vec, rec_0_meta = rec_0

    # Deterministic chunk ID
    assert rec_0_id == "doc_test123#chunk_0"
    # Embedding is vector values
    assert rec_0_vec == MOCK_VECTOR_A
    assert len(rec_0_vec) == 768

    # Metadata validation: supported fields ONLY, embedding NEVER in metadata
    assert "embedding" not in rec_0_meta
    assert "values" not in rec_0_meta
    assert rec_0_meta["document_id"] == "doc_test123"
    assert rec_0_meta["chunk_index"] == 0
    assert rec_0_meta["filename"] == "resume.pdf"
    assert rec_0_meta["text"] == "First chunk text."
    assert rec_0_meta["character_count"] == 17
    assert rec_0_meta["token_estimate"] == 4

    # Verify second vector record
    rec_1 = records[1]
    rec_1_id, rec_1_vec, rec_1_meta = rec_1
    assert rec_1_id == "doc_test123#chunk_1"
    assert rec_1_vec == MOCK_VECTOR_B
    assert "embedding" not in rec_1_meta
    assert rec_1_meta["chunk_index"] == 1


@pytest.mark.anyio
async def test_pinecone_upsert_empty_chunks(mock_pinecone_service: PineconeService):
    """Test that upserting empty chunks list returns 0 without calling index.upsert."""
    count = await mock_pinecone_service.upsert_document_chunks(
        document_id="doc_test",
        chunks=[],
        vectors=[],
        filename="empty.pdf",
    )
    assert count == 0
    mock_pinecone_service._index.upsert.assert_not_called()


@pytest.mark.anyio
async def test_pinecone_upsert_count_mismatch_raises_error(mock_pinecone_service: PineconeService):
    """Test that mismatch between chunk count and vector count raises PineconeResponseError."""
    chunks = [TextChunk(index=0, text="A", character_count=1, token_estimate=1)]
    vectors = [MOCK_VECTOR_A, MOCK_VECTOR_B]

    with pytest.raises(PineconeResponseError, match="Mismatch between chunks count"):
        await mock_pinecone_service.upsert_document_chunks("doc_1", chunks, vectors, "file.pdf")


@pytest.mark.anyio
async def test_pinecone_upsert_dimension_mismatch_raises_error(mock_pinecone_service: PineconeService):
    """Test that vector with dimension != 768 raises PineconeDimensionMismatchError."""
    chunks = [TextChunk(index=0, text="A", character_count=1, token_estimate=1)]
    vector_512 = [0.1] * 512

    with pytest.raises(PineconeDimensionMismatchError, match="Vector at index 0 has dimension 512"):
        await mock_pinecone_service.upsert_document_chunks("doc_1", chunks, [vector_512], "file.pdf")


@pytest.mark.anyio
async def test_pinecone_upsert_api_failure(mock_pinecone_service: PineconeService):
    """Test that Pinecone API errors during upsert raise PineconeResponseError."""
    mock_pinecone_service._index.upsert.side_effect = Exception("Pinecone network timeout")
    chunks = [TextChunk(index=0, text="A", character_count=1, token_estimate=1)]

    with pytest.raises(PineconeResponseError, match="Pinecone upsert failed"):
        await mock_pinecone_service.upsert_document_chunks("doc_1", chunks, [MOCK_VECTOR_A], "file.pdf")


# ============================================================================
# Query & Document Isolation Tests (Standard Index Query)
# ============================================================================


@pytest.mark.anyio
async def test_pinecone_query_similar_chunks_success(mock_pinecone_service: PineconeService):
    """Test similarity search via standard index.query returns structured matches with text and scores."""
    mock_match = MagicMock()
    mock_match.id = "doc_1#chunk_0"
    mock_match.score = 0.92
    mock_match.metadata = {
        "document_id": "doc_1",
        "chunk_index": 0,
        "filename": "resume.pdf",
        "text": "Extensive experience in FastAPI microservices.",
        "character_count": 46,
        "token_estimate": 11,
    }
    mock_result = MagicMock()
    mock_result.matches = [mock_match]
    mock_pinecone_service._index.query.return_value = mock_result

    results = await mock_pinecone_service.query_similar_chunks(
        query_vector=MOCK_VECTOR_A,
        document_id="doc_1",
        top_k=2,
    )

    assert len(results) == 1
    assert results[0]["id"] == "doc_1#chunk_0"
    assert results[0]["score"] == 0.92
    assert results[0]["document_id"] == "doc_1"
    assert "FastAPI" in results[0]["text"]
    assert results[0]["metadata"]["filename"] == "resume.pdf"

    mock_pinecone_service._index.query.assert_called_once_with(
        vector=MOCK_VECTOR_A,
        top_k=2,
        include_metadata=True,
        namespace="default",
        filter={"document_id": {"$eq": "doc_1"}},
    )


@pytest.mark.anyio
async def test_pinecone_query_enforces_document_isolation(mock_pinecone_service: PineconeService):
    """Test that passing document_id strictly applies metadata filter to prevent cross-document leakage."""
    mock_result = MagicMock()
    mock_result.matches = []
    mock_pinecone_service._index.query.return_value = mock_result

    await mock_pinecone_service.query_similar_chunks(
        query_vector=MOCK_VECTOR_A,
        document_id="isolated_doc_999",
        top_k=3,
    )

    mock_index = mock_pinecone_service._index
    mock_index.query.assert_called_once()
    _, kwargs = mock_index.query.call_args

    assert kwargs["filter"] == {"document_id": {"$eq": "isolated_doc_999"}}
    assert kwargs["top_k"] == 3
    assert kwargs["include_metadata"] is True
    assert kwargs["vector"] == MOCK_VECTOR_A


@pytest.mark.anyio
async def test_pinecone_query_dimension_mismatch(mock_pinecone_service: PineconeService):
    """Test that query vector with dimension != 768 raises PineconeDimensionMismatchError."""
    with pytest.raises(PineconeDimensionMismatchError, match="Query vector dimension 256 does not match"):
        await mock_pinecone_service.query_similar_chunks(query_vector=[0.1] * 256)


@pytest.mark.anyio
async def test_pinecone_query_failure(mock_pinecone_service: PineconeService):
    """Test that query API failures raise PineconeResponseError."""
    mock_pinecone_service._index.query.side_effect = Exception("Query timed out")
    with pytest.raises(PineconeResponseError, match="Pinecone query failed"):
        await mock_pinecone_service.query_similar_chunks(query_vector=MOCK_VECTOR_A, document_id="doc_1")


# ============================================================================
# Document Deletion Tests (Standard Index Delete)
# ============================================================================


@pytest.mark.anyio
async def test_pinecone_delete_document_success(mock_pinecone_service: PineconeService):
    """Test delete_document calls index.delete with document_id metadata filter."""
    result = await mock_pinecone_service.delete_document("doc_to_delete")

    assert result is True
    mock_pinecone_service._index.delete.assert_called_once_with(
        filter={"document_id": {"$eq": "doc_to_delete"}},
        namespace="default",
    )


@pytest.mark.anyio
async def test_pinecone_delete_document_empty_id_rejected(mock_pinecone_service: PineconeService):
    """Test delete_document with empty ID raises PineconeResponseError."""
    with pytest.raises(PineconeResponseError, match="must not be empty"):
        await mock_pinecone_service.delete_document("   ")


@pytest.mark.anyio
async def test_pinecone_delete_document_failure(mock_pinecone_service: PineconeService):
    """Test delete_document API failure raises PineconeResponseError."""
    mock_pinecone_service._index.delete.side_effect = Exception("Index unreachable")
    with pytest.raises(PineconeResponseError, match="Pinecone delete failed"):
        await mock_pinecone_service.delete_document("doc_error")


# ============================================================================
# PDF Upload Route + Pinecone Integration Tests
# ============================================================================


def test_upload_pdf_integration_stores_in_pinecone():
    """Test that POST /api/v1/documents/upload extracts, chunks, embeds, and stores in Pinecone."""
    pdf_bytes = create_test_pdf("Senior Distributed Systems Engineer", pages=2)
    response = client.post(
        "/api/v1/documents/upload",
        files={"file": ("candidate_profile.pdf", pdf_bytes, "application/pdf")},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["filename"] == "candidate_profile.pdf"
    assert data["page_count"] == 2
    assert data["character_count"] > 0
    assert data["document_id"] is not None
    assert data["document_id"].startswith("doc_")
    assert data["chunk_count"] > 0
    assert data["vector_count"] > 0
    assert "stored" in data["message"]
    assert "Pinecone" in data["message"]


def test_upload_pdf_pinecone_service_unavailable():
    """Test graceful HTTP 503 error when Pinecone is unreachable."""
    class MockUnreachablePinecone(PineconeService):
        async def upsert_document_chunks(self, *args, **kwargs):
            raise PineconeConnectionError("Unable to reach Pinecone cloud endpoint.")

    app.dependency_overrides[get_pinecone_service] = lambda: MockUnreachablePinecone()
    try:
        pdf_bytes = create_test_pdf("Staff Engineer", pages=1)
        response = client.post(
            "/api/v1/documents/upload",
            files={"file": ("staff_resume.pdf", pdf_bytes, "application/pdf")},
        )
        assert response.status_code == 503
        assert "Pinecone vector storage service is currently unavailable" in response.json()["detail"]
    finally:
        app.dependency_overrides.pop(get_pinecone_service, None)


def test_upload_pdf_pinecone_missing_api_key():
    """Test graceful HTTP 500 error when Pinecone API key is unconfigured."""
    class MockUnconfiguredPinecone(PineconeService):
        async def upsert_document_chunks(self, *args, **kwargs):
            raise PineconeConfigurationError("Pinecone API key is not configured.")

    app.dependency_overrides[get_pinecone_service] = lambda: MockUnconfiguredPinecone()
    try:
        pdf_bytes = create_test_pdf("Staff Engineer", pages=1)
        response = client.post(
            "/api/v1/documents/upload",
            files={"file": ("staff_resume.pdf", pdf_bytes, "application/pdf")},
        )
        assert response.status_code == 500
        assert "API key is not configured" in response.json()["detail"]
    finally:
        app.dependency_overrides.pop(get_pinecone_service, None)


def test_upload_pdf_embedding_service_failure():
    """Test graceful HTTP 503 error when Ollama embedding service fails during upload."""
    class MockEmbeddingServiceFail:
        async def embed_chunks(self, *args, **kwargs):
            raise EmbeddingConnectionError("Connection to Ollama timed out.")

    app.dependency_overrides[get_embedding_service] = lambda: MockEmbeddingServiceFail()
    try:
        pdf_bytes = create_test_pdf("Staff Engineer", pages=1)
        response = client.post(
            "/api/v1/documents/upload",
            files={"file": ("staff_resume.pdf", pdf_bytes, "application/pdf")},
        )
        assert response.status_code == 503
        assert "Embedding service is currently unavailable" in response.json()["detail"]
    finally:
        app.dependency_overrides.pop(get_embedding_service, None)


def test_upload_pdf_dimension_mismatch_failure():
    """Test graceful HTTP 500 error when Pinecone vector dimension mismatch occurs."""
    class MockDimMismatchPinecone(PineconeService):
        async def upsert_document_chunks(self, *args, **kwargs):
            raise PineconeDimensionMismatchError("Vector dimension mismatch: expected 768, got 1536.")

    app.dependency_overrides[get_pinecone_service] = lambda: MockDimMismatchPinecone()
    try:
        pdf_bytes = create_test_pdf("Staff Engineer", pages=1)
        response = client.post(
            "/api/v1/documents/upload",
            files={"file": ("staff_resume.pdf", pdf_bytes, "application/pdf")},
        )
        assert response.status_code == 500
        assert "dimension mismatch" in response.json()["detail"]
    finally:
        app.dependency_overrides.pop(get_pinecone_service, None)


@pytest.mark.anyio
async def test_pinecone_query_normalizes_chunk_id(mock_pinecone_service: PineconeService):
    """Test that query_similar_chunks normalizes chunk ID to base document ID in metadata filter."""
    mock_results = MagicMock()
    mock_match = MagicMock()
    mock_match.id = "doc_test123#chunk_0"
    mock_match.score = 0.95
    mock_match.metadata = {"document_id": "doc_test123", "chunk_index": 0, "text": "Test chunk text"}
    mock_results.matches = [mock_match]
    mock_pinecone_service._index.query.return_value = mock_results

    results = await mock_pinecone_service.query_similar_chunks(
        query_vector=[0.1] * 768,
        document_id="doc_test123#chunk_0",
    )

    assert len(results) == 1
    mock_pinecone_service._index.query.assert_called_once_with(
        vector=[0.1] * 768,
        top_k=4,
        include_metadata=True,
        namespace="default",
        filter={"document_id": {"$eq": "doc_test123"}},
    )


@pytest.mark.anyio
async def test_pinecone_delete_document_normalizes_chunk_id(mock_pinecone_service: PineconeService):
    """Test that delete_document normalizes chunk ID to base document ID in deletion filter."""
    success = await mock_pinecone_service.delete_document("doc_test123#chunk_0")

    assert success is True
    mock_pinecone_service._index.delete.assert_called_once_with(
        filter={"document_id": {"$eq": "doc_test123"}},
        namespace="default",
    )

