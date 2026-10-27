"""Unit tests for TextChunkingService (RAG Phase 3).

Validates whitespace normalization, paragraph boundary preservation,
chunk sizing, overlap correctness, edge case handling, and token estimation.
"""

import pytest

from app.services.chunking_service import (
    TextChunk,
    TextChunkingService,
    get_chunking_service,
)


@pytest.fixture
def chunking_service() -> TextChunkingService:
    """Fixture providing a default TextChunkingService instance."""
    return TextChunkingService(chunk_size=2000, chunk_overlap=200)


# ============================================================================
# Whitespace Normalization Tests
# ============================================================================


def test_normalize_whitespace_basic(chunking_service: TextChunkingService):
    """Test that CRLF line endings, tabs, and excess intra-line spaces are cleaned."""
    raw = "Hello   world!\r\n\tThis is a test line with   extra   spaces.  "
    normalized = chunking_service.normalize_whitespace(raw)
    assert normalized == "Hello world!\nThis is a test line with extra spaces."


def test_normalize_whitespace_paragraphs(chunking_service: TextChunkingService):
    """Test that paragraph breaks (\n\n) are preserved while 3+ consecutive newlines are collapsed."""
    raw = "Paragraph One.\n\n\n\n\nParagraph Two.\n\nParagraph Three."
    normalized = chunking_service.normalize_whitespace(raw)
    assert normalized == "Paragraph One.\n\nParagraph Two.\n\nParagraph Three."


def test_normalize_whitespace_special_unicode(chunking_service: TextChunkingService):
    """Test that non-breaking spaces (\u00a0), zero-width spaces (\u200b), and form feeds (\x0c) are handled."""
    raw = "Word\u00a0with\u00a0nbsp.\x0cNew\u200bpage."
    normalized = chunking_service.normalize_whitespace(raw)
    assert "Word with nbsp." in normalized
    assert "Newpage." in normalized


def test_normalize_whitespace_empty_and_none(chunking_service: TextChunkingService):
    """Test that empty string, whitespace-only, and None safely return empty string."""
    assert chunking_service.normalize_whitespace("") == ""
    assert chunking_service.normalize_whitespace("   \n\t\r\n   ") == ""
    assert chunking_service.normalize_whitespace(None) == ""


# ============================================================================
# Token Estimation Tests
# ============================================================================


def test_estimate_tokens():
    """Test standard 1 token ≈ 4 characters token estimation."""
    assert TextChunkingService.estimate_tokens("") == 0
    assert TextChunkingService.estimate_tokens(None) == 0
    # 4 characters ≈ 1 token
    assert TextChunkingService.estimate_tokens("ABCD") == 1
    # 2000 characters ≈ 500 tokens
    text_2000 = "x" * 2000
    assert TextChunkingService.estimate_tokens(text_2000) == 500
    # Single character still yields at least 1 token
    assert TextChunkingService.estimate_tokens("A") == 1


# ============================================================================
# Chunk Boundary & Sizing Tests
# ============================================================================


def test_chunk_empty_and_whitespace(chunking_service: TextChunkingService):
    """Test that empty string, None, and whitespace-only text return an empty chunk list."""
    assert chunking_service.chunk_text("") == []
    assert chunking_service.chunk_text(None) == []
    assert chunking_service.chunk_text("   \n\n\t   ") == []


def test_chunk_short_input(chunking_service: TextChunkingService):
    """Test that text shorter than chunk_size returns a single chunk containing the full normalized text."""
    short_text = "Experienced Senior Python Engineer with expertise in FastAPI and distributed systems."
    chunks = chunking_service.chunk_text(short_text, chunk_size=500, chunk_overlap=50)
    assert len(chunks) == 1
    assert chunks[0] == short_text


def test_chunk_paragraph_boundaries():
    """Test that splitting prefers natural paragraph boundaries (\n\n) over mid-paragraph cuts."""
    p1 = "Paragraph 1: " + ("First paragraph content. " * 3).strip()
    p2 = "Paragraph 2: " + ("Second paragraph content. " * 3).strip()
    p3 = "Paragraph 3: " + ("Third paragraph content. " * 3).strip()
    full_text = f"{p1}\n\n{p2}\n\n{p3}"

    service = TextChunkingService(chunk_size=130, chunk_overlap=20)
    chunks = service.chunk_text(full_text)

    assert len(chunks) >= 3
    # Verify chunks start and end cleanly with paragraph content
    assert chunks[0].startswith("Paragraph 1:")
    assert any("Paragraph 2:" in c for c in chunks)
    assert chunks[-1].endswith(p3[-20:])


def test_chunk_sentence_and_word_boundaries():
    """Test that long paragraphs without blank lines split at sentence or word boundaries."""
    long_single_paragraph = (
        "Software engineering requires deep analytical thinking. "
        "Architecting distributed microservices demands careful consideration of latency and throughput. "
        "PostgreSQL and vector embeddings enable high-precision semantic search across technical documents. "
        "Continuous integration ensures system stability over time."
    )
    service = TextChunkingService(chunk_size=120, chunk_overlap=25)
    chunks = service.chunk_text(long_single_paragraph)

    assert len(chunks) > 1
    # Verify none of the chunks start or end with broken words
    for chunk in chunks:
        assert not chunk.startswith(" ")
        assert not chunk.endswith(" ")
        assert len(chunk) > 0


# ============================================================================
# Overlap Tests
# ============================================================================


def test_chunk_overlap_correctness():
    """Test that consecutive chunks share context from the overlap region."""
    text = (
        "Alpha section describes database schemas and primary key choices. "
        "Beta section outlines API design patterns using REST and GraphQL. "
        "Gamma section covers caching strategies with Redis and memory pools. "
        "Delta section explains CI/CD automation pipelines with GitHub Actions."
    )
    service = TextChunkingService(chunk_size=110, chunk_overlap=30)
    chunks = service.chunk_text(text)

    assert len(chunks) >= 3
    # Check that chunk N ends with text that appears near the beginning of chunk N+1
    for i in range(len(chunks) - 1):
        chunk_a = chunks[i]
        chunk_b = chunks[i + 1]
        # Find shared trailing words from chunk_a in chunk_b
        words_a = chunk_a.split()
        last_word = words_a[-1].rstrip(".,;:")
        # The last word or second-to-last word of chunk_a should be present in chunk_b
        assert (last_word in chunk_b) or (words_a[-2].rstrip(".,;:") in chunk_b)


def test_chunk_zero_overlap():
    """Test that zero overlap produces contiguous chunks without repeating words."""
    text = "First block of documentation. Second block of documentation. Third block of documentation."
    service = TextChunkingService(chunk_size=40, chunk_overlap=0)
    chunks = service.chunk_text(text)

    assert len(chunks) > 1
    # Total combined characters in chunks should equal or closely match the original stripped text
    combined = " ".join(chunks)
    assert "First block" in combined
    assert "Third block" in combined


# ============================================================================
# Ordering & Cleanliness Tests
# ============================================================================


def test_chunk_order_preserved():
    """Test that chunks strictly preserve original document order."""
    sections = [f"Section {i:02d} detailed description." for i in range(1, 10)]
    full_text = "\n\n".join(sections)

    service = TextChunkingService(chunk_size=70, chunk_overlap=15)
    chunks = service.chunk_text(full_text)

    last_found_idx = 0
    for chunk in chunks:
        for i in range(1, 10):
            tag = f"Section {i:02d}"
            if tag in chunk:
                assert i >= last_found_idx, f"Order violated: found {tag} after Section {last_found_idx}"
                last_found_idx = i


def test_chunk_no_empty_chunks():
    """Test that no empty or whitespace-only chunks are ever produced."""
    text = "A" * 50 + "\n\n\n\n" + "B" * 50 + "\n\n\n\n" + "C" * 50
    service = TextChunkingService(chunk_size=60, chunk_overlap=10)
    chunks = service.chunk_text(text)

    for chunk in chunks:
        assert len(chunk.strip()) > 0


# ============================================================================
# Validation & Configuration Error Tests
# ============================================================================


@pytest.mark.parametrize(
    "invalid_size, invalid_overlap",
    [
        (0, 0),         # size 0
        (-10, 0),       # negative size
        (100, -5),      # negative overlap
        (100, 100),     # overlap equals size
        (100, 150),     # overlap exceeds size
    ],
)
def test_chunking_invalid_parameters_rejected(invalid_size, invalid_overlap):
    """Test that invalid chunk size or overlap configurations raise ValueError."""
    with pytest.raises(ValueError):
        TextChunkingService(chunk_size=invalid_size, chunk_overlap=invalid_overlap)

    service = TextChunkingService()
    with pytest.raises(ValueError):
        service.chunk_text("Some valid text", chunk_size=invalid_size, chunk_overlap=invalid_overlap)


# ============================================================================
# Rich TextChunk Dataclass Tests
# ============================================================================


def test_create_chunks_metadata():
    """Test that create_chunks produces TextChunk objects with accurate index and token counts."""
    text = "Machine learning models require clean training datasets and rigorous validation."
    service = TextChunkingService(chunk_size=50, chunk_overlap=10)
    chunks = service.create_chunks(text)

    assert len(chunks) > 1
    for idx, c in enumerate(chunks):
        assert isinstance(c, TextChunk)
        assert c.index == idx
        assert c.character_count == len(c.text)
        assert c.token_estimate == TextChunkingService.estimate_tokens(c.text)
        assert c.token_estimate > 0


def test_get_chunking_service_dependency():
    """Test FastAPI dependency provider returns a configured TextChunkingService instance."""
    service = get_chunking_service()
    assert isinstance(service, TextChunkingService)
    assert service.chunk_size > 0
    assert service.chunk_overlap >= 0
