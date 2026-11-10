"""Unit and integration test suite for Advanced RAG capabilities:
- Hybrid Search (BM25 lexical search + Reciprocal Rank Fusion)
- Cross-Encoder / Heuristic Reranker
- RAG Evaluation (Faithfulness, Context Relevance, Question Quality)
- Evaluation API endpoint contracts (/api/v1/interview/evaluate-rag)
"""

import json
from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient
import pytest

from app.main import app
from app.schemas.evaluation import (
    EvaluationRating,
    RAGEvaluationRequest,
    RAGEvaluationResponse,
)
from app.services.embedding_service import EmbeddingService
from app.services.hybrid_search_service import (
    BM25Okapi,
    HybridSearchService,
    reciprocal_rank_fusion,
    tokenize,
)
from app.services.ollama_service import OllamaService
from app.services.pinecone_service import PineconeService
from app.services.rag_eval_service import RAGEvaluationService
from app.services.reranker_service import RerankerService
from app.services.retrieval_service import RetrievedChunk, RetrievalService

client = TestClient(app)

MOCK_VECTOR = [0.01 * (i % 10) for i in range(768)]


# ============================================================================
# BM25 & Lexical Tokenization Tests
# ============================================================================


def test_tokenize_basic_and_stopwords():
    """Verify tokenizer lowercases, extracts terms, and removes standard stopwords."""
    text = "The FastAPI backend uses Redis and PostgreSQL for caching and storage."
    tokens = tokenize(text, remove_stopwords=True)
    assert "the" not in tokens
    assert "and" not in tokens
    assert "for" not in tokens
    assert "fastapi" in tokens
    assert "backend" in tokens
    assert "redis" in tokens
    assert "postgresql" in tokens


def test_tokenize_empty():
    """Verify tokenizer handles empty or whitespace text gracefully."""
    assert tokenize("") == []
    assert tokenize("   ") == []


def test_bm25_scoring():
    """Verify BM25 gives higher score to documents containing query terms."""
    corpus = [
        tokenize("Architected high-throughput message queues using Redis Pub/Sub."),
        tokenize("Developed responsive user interfaces with React and Tailwind CSS."),
        tokenize("Configured PostgreSQL connection pooling and SQLAlchemy async sessions."),
    ]
    bm25 = BM25Okapi(corpus)

    # Query for Redis message queues
    query_tokens = tokenize("Redis message queues")
    scores = bm25.get_scores(query_tokens)

    assert len(scores) == 3
    # First document should have highest BM25 score
    assert scores[0] > scores[1]
    assert scores[0] > scores[2]
    assert scores[0] > 0.0


def test_bm25_empty_query_or_corpus():
    """Verify BM25 handles empty queries and empty corpus without raising exceptions."""
    bm25 = BM25Okapi([tokenize("Some document text.")])
    assert bm25.get_scores([]) == [0.0]

    empty_bm25 = BM25Okapi([])
    assert empty_bm25.get_scores(["python"]) == []


# ============================================================================
# Reciprocal Rank Fusion (RRF) Tests
# ============================================================================


def test_reciprocal_rank_fusion_logic():
    """Verify RRF calculates combined reciprocal rank scores across rankings."""
    # List 1: chunk_A is rank 1, chunk_B is rank 2
    # List 2: chunk_B is rank 1, chunk_A is rank 2
    list_dense = ["chunk_A", "chunk_B", "chunk_C"]
    list_lexical = ["chunk_B", "chunk_A", "chunk_D"]

    rrf = reciprocal_rank_fusion([list_dense, list_lexical], k=60)

    # chunk_A and chunk_B appear in top 2 of both lists and should have highest scores
    assert rrf["chunk_A"] > rrf["chunk_C"]
    assert rrf["chunk_B"] > rrf["chunk_D"]
    # chunk_A and chunk_B have symmetric ranks (1, 2) and (2, 1) with equal weights
    assert pytest.approx(rrf["chunk_A"], rel=1e-4) == rrf["chunk_B"]


def test_reciprocal_rank_fusion_weighted():
    """Verify custom weights adjust the ranking preference in RRF."""
    list_dense = ["chunk_dense_first", "chunk_other"]
    list_lexical = ["chunk_lexical_first", "chunk_other"]

    # Give 80% weight to dense, 20% to lexical
    rrf = reciprocal_rank_fusion([list_dense, list_lexical], weights=[0.8, 0.2], k=60)
    assert rrf["chunk_dense_first"] > rrf["chunk_lexical_first"]


# ============================================================================
# Hybrid Search Service Tests
# ============================================================================


def test_hybrid_search_service_fuses_dense_and_bm25():
    """Verify HybridSearchService re-scores candidate matches with BM25 keyword matching."""
    dense_matches = [
        {
            "id": "doc#chunk_1",
            "score": 0.85,
            "text": "General software engineering principles and design patterns.",
            "metadata": {"chunk_index": 1},
        },
        {
            "id": "doc#chunk_2",
            "score": 0.80,
            "text": "Implemented distributed locking using Redis Redlock algorithm.",
            "metadata": {"chunk_index": 2},
        },
    ]

    service = HybridSearchService(enabled=True, alpha=0.5)
    fused = service.fuse_matches(
        dense_matches=dense_matches,
        query_text="Redis Redlock distributed locking",
    )

    assert len(fused) == 2
    # doc#chunk_2 has exact matching terms for "Redis Redlock distributed locking", so it should be promoted to #1
    assert fused[0]["id"] == "doc#chunk_2"
    assert "dense_score" in fused[0]["metadata"]
    assert "bm25_score" in fused[0]["metadata"]
    assert "rrf_score" in fused[0]["metadata"]


def test_hybrid_search_service_disabled_passthrough():
    """Verify HybridSearchService returns unmodified dense matches when disabled."""
    dense_matches = [
        {"id": "doc#1", "score": 0.90, "text": "First"},
        {"id": "doc#2", "score": 0.80, "text": "Second"},
    ]
    service = HybridSearchService(enabled=False)
    fused = service.fuse_matches(dense_matches=dense_matches, query_text="any query")
    assert fused == dense_matches


# ============================================================================
# Reranker Service Tests
# ============================================================================


@pytest.mark.anyio
async def test_reranker_heuristic_reordering():
    """Verify heuristic reranker boosts chunks with high term coverage and exact phrases."""
    candidates = [
        {
            "id": "doc#1",
            "score": 0.75,
            "text": "General cloud infrastructure and containerization topics.",
            "metadata": {},
        },
        {
            "id": "doc#2",
            "score": 0.70,
            "text": "Engineered asynchronous event workers using FastAPI and Celery with Redis broker.",
            "metadata": {},
        },
    ]

    reranker = RerankerService(enabled=True, reranker_type="heuristic")
    reranked = await reranker.rerank_matches(
        query="FastAPI and Celery with Redis",
        candidates=candidates,
    )

    assert len(reranked) == 2
    # doc#2 contains the exact key phrases and should be reranked to top
    assert reranked[0]["id"] == "doc#2"
    assert reranked[0]["metadata"]["rerank_score"] > reranked[1]["metadata"]["rerank_score"]


@pytest.mark.anyio
async def test_reranker_ollama_scoring():
    """Verify Ollama LLM reranker parses JSON ratings and sorts candidate chunks."""
    candidates = [
        {"id": "doc#1", "score": 0.70, "text": "Generic web development."},
        {"id": "doc#2", "score": 0.65, "text": "Deep dive into PostgreSQL table partitioning."},
    ]

    mock_ollama = AsyncMock(spec=OllamaService)
    mock_ollama.generate_chat_completion.return_value = json.dumps({
        "scores": [
            {"index": 0, "score": 3.0},
            {"index": 1, "score": 9.5},
        ]
    })

    reranker = RerankerService(enabled=True, reranker_type="ollama", ollama_service=mock_ollama)
    reranked = await reranker.rerank_matches(query="PostgreSQL table partitioning", candidates=candidates)

    assert len(reranked) == 2
    assert reranked[0]["id"] == "doc#2"
    assert reranked[0]["score"] > reranked[1]["score"]


# ============================================================================
# Full Advanced RAG Retrieval Integration Tests
# ============================================================================


@pytest.mark.anyio
async def test_retrieval_service_full_advanced_rag_pipeline():
    """Verify complete Advanced RAG pipeline: Pinecone -> Hybrid BM25 Fusion -> Reranker."""
    mock_embed = AsyncMock(spec=EmbeddingService)
    mock_embed.embed_text.return_value = MOCK_VECTOR

    mock_pinecone = AsyncMock(spec=PineconeService)
    mock_pinecone.query_similar_chunks.return_value = [
        {
            "id": "doc_user#chunk_1",
            "score": 0.88,
            "chunk_index": 1,
            "filename": "cv.pdf",
            "text": "Maintained various internal administrative web portals and tools.",
            "metadata": {"document_id": "doc_user", "chunk_index": 1},
        },
        {
            "id": "doc_user#chunk_2",
            "score": 0.82,
            "chunk_index": 2,
            "filename": "cv.pdf",
            "text": "Led architecture of zero-downtime database migration using Alembic and PostgreSQL.",
            "metadata": {"document_id": "doc_user", "chunk_index": 2},
        },
    ]

    hybrid_service = HybridSearchService(enabled=True, alpha=0.5)
    reranker_service = RerankerService(enabled=True, reranker_type="heuristic")

    service = RetrievalService(
        embedding_service=mock_embed,
        pinecone_service=mock_pinecone,
        hybrid_search_service=hybrid_service,
        reranker_service=reranker_service,
        min_score=0.50,
    )

    chunks = await service.retrieve_chunks(
        query_text="Alembic PostgreSQL zero-downtime database migration",
        document_id="doc_user",
        top_k=2,
    )

    assert len(chunks) == 2
    # chunk_2 should be ranked first due to exact keyword and phrase matches
    assert chunks[0].chunk_id == "doc_user#chunk_2"
    assert "Alembic and PostgreSQL" in chunks[0].text
    assert "dense_score" in chunks[0].metadata
    assert "bm25_score" in chunks[0].metadata
    assert "rerank_score" in chunks[0].metadata


# ============================================================================
# RAG Evaluation Tests
# ============================================================================


@pytest.mark.anyio
async def test_rag_evaluation_service_llm_judge_success():
    """Verify RAGEvaluationService calculates Faithfulness, Context Relevance, and Question Relevance."""
    mock_ollama = AsyncMock(spec=OllamaService)
    mock_ollama.generate_chat_completion.return_value = json.dumps({
        "faithfulness_score": 0.95,
        "faithfulness_reasoning": "The question references Redis and Celery which are directly mentioned in the context.",
        "context_relevance_score": 0.90,
        "context_relevance_reasoning": "Context chunks describe async queue architecture.",
        "question_relevance_score": 0.88,
        "question_relevance_reasoning": "Well suited for Senior Backend Engineer.",
        "summary": "Excellent grounded technical interview question.",
    })

    eval_service = RAGEvaluationService(ollama_service=mock_ollama)
    req = RAGEvaluationRequest(
        query="FastAPI Redis queue",
        question="How did you handle task retries and worker concurrency in Celery with Redis?",
        expected_answer_points=["Idempotency keys", "Dead letter queue"],
        context_chunks=["Engineered asynchronous background task queues using Celery and Redis."],
        role="Senior Backend Engineer",
        difficulty="Hard",
    )

    result = await eval_service.evaluate_rag_output(req)

    assert isinstance(result, RAGEvaluationResponse)
    assert result.faithfulness.score == 0.95
    assert result.faithfulness.passed is True
    assert result.context_relevance.score == 0.90
    assert result.question_relevance.score == 0.88
    assert result.overall_score >= 0.85
    assert result.rating == EvaluationRating.EXCELLENT


@pytest.mark.anyio
async def test_rag_evaluation_service_fallback_on_llm_failure():
    """Verify RAGEvaluationService falls back gracefully to heuristic assessment if Ollama fails."""
    mock_ollama = AsyncMock(spec=OllamaService)
    mock_ollama.generate_chat_completion.side_effect = Exception("Ollama connection error")

    eval_service = RAGEvaluationService(ollama_service=mock_ollama)
    req = RAGEvaluationRequest(
        query="PostgreSQL table partitioning",
        question="How did you partition large tables in PostgreSQL?",
        expected_answer_points=["Range partitioning", "Indexes"],
        context_chunks=["Implemented PostgreSQL declarative range table partitioning for audit logs."],
        role="Database Engineer",
        difficulty="Medium",
    )

    result = await eval_service.evaluate_rag_output(req)

    assert isinstance(result, RAGEvaluationResponse)
    assert result.overall_score > 0.0
    assert "Heuristic" in result.faithfulness.reasoning


# ============================================================================
# API Endpoint Integration Tests
# ============================================================================


def test_api_evaluate_rag_endpoint_success():
    """Test POST /api/v1/interview/evaluate-rag endpoint."""
    with patch("app.services.rag_eval_service.RAGEvaluationService.evaluate_rag_output") as mock_eval:
        mock_eval.return_value = RAGEvaluationResponse(
            overall_score=0.92,
            rating=EvaluationRating.EXCELLENT,
            faithfulness={
                "metric_name": "Faithfulness (Groundedness)",
                "score": 0.95,
                "reasoning": "Grounded in resume.",
                "passed": True,
            },
            context_relevance={
                "metric_name": "Context Relevance",
                "score": 0.90,
                "reasoning": "Relevant context.",
                "passed": True,
            },
            question_relevance={
                "metric_name": "Question Relevance",
                "score": 0.90,
                "reasoning": "Matches role.",
                "passed": True,
            },
            summary="High quality grounded question.",
        )

        payload = {
            "query": "FastAPI microservices",
            "question": "How did you design service discovery in FastAPI?",
            "expected_answer_points": ["Consul or Eureka", "Health check probes"],
            "context_chunks": ["Designed and deployed FastAPI microservices with Docker."],
            "role": "Backend Engineer",
            "difficulty": "Medium",
        }

        resp = client.post("/api/v1/interview/evaluate-rag", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["overall_score"] == 0.92
        assert data["rating"] == "Excellent"
        assert data["faithfulness"]["score"] == 0.95


def test_api_evaluate_rag_endpoint_validation_error():
    """Test POST /api/v1/interview/evaluate-rag returns 422 for missing required fields."""
    resp = client.post("/api/v1/interview/evaluate-rag", json={"query": "test"})
    assert resp.status_code == 422

