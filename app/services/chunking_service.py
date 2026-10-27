"""Text Chunking Service for RAG pipeline.

Splits document text into semantically cohesive, overlapping chunks suitable
for embedding generation and vector search retrieval.
"""

from dataclasses import dataclass
import logging
import re
from typing import List, Optional
import unicodedata

from app.core.config import settings

logger = logging.getLogger(__name__)

# Character-to-token approximation:
# In standard English prose, 1 token is approximately 4 characters (or ~0.75 words).
# Target: ~500 tokens corresponds to ~2,000 characters.
# Target overlap: ~10% corresponds to ~200 characters (~50 tokens).
CHARS_PER_TOKEN_APPROX: float = 4.0


@dataclass(frozen=True)
class TextChunk:
    """Represents a discrete text chunk with metadata."""

    index: int
    text: str
    character_count: int
    token_estimate: int


class TextChunkingService:
    """Service for normalizing and splitting text into manageable, overlapping chunks.

    Attributes:
        chunk_size: Target maximum character length for each chunk.
        chunk_overlap: Target character overlap between consecutive chunks.
    """

    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ):
        size = chunk_size if chunk_size is not None else settings.CHUNK_SIZE
        overlap = chunk_overlap if chunk_overlap is not None else settings.CHUNK_OVERLAP

        self._validate_parameters(size, overlap)
        self.chunk_size = size
        self.chunk_overlap = overlap

    @staticmethod
    def _validate_parameters(chunk_size: int, chunk_overlap: int) -> None:
        """Validates that chunk size and overlap configurations are mathematically sound."""
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        if chunk_overlap < 0:
            raise ValueError(f"chunk_overlap must be non-negative, got {chunk_overlap}")
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be strictly less than chunk_size ({chunk_size})"
            )

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Estimates the token count of a given text string.

        Approximation note:
            Based on standard English subword tokenization (e.g., SentencePiece, Tiktoken),
            1 token is approximately 4 characters in English prose.
        """
        if not text:
            return 0
        return max(1, round(len(text) / CHARS_PER_TOKEN_APPROX))

    @staticmethod
    def normalize_whitespace(text: Optional[str]) -> str:
        """Safely normalizes whitespace without destroying paragraph boundaries.

        - Converts carriage returns (CRLF, CR) to standard newlines (LF).
        - Replaces non-breaking spaces, form feeds, and zero-width spaces.
        - Trims trailing spaces on each line and condenses multiple horizontal spaces.
        - Preserves paragraph separations by collapsing 3+ consecutive newlines down to 2.
        - Strips document leading and trailing whitespace.
        """
        if not text:
            return ""

        # Normalize unicode representation
        normalized = unicodedata.normalize("NFKC", text)

        # Standardize line endings and special control characters
        normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
        normalized = normalized.replace("\u00a0", " ").replace("\u200b", "").replace("\x0c", "\n")

        # Normalize horizontal spaces per line while preserving line structure
        lines = [re.sub(r"[^\S\n]+", " ", line).strip() for line in normalized.split("\n")]
        normalized = "\n".join(lines)

        # Collapse 3 or more consecutive newlines into 2 (preserving paragraph breaks)
        normalized = re.sub(r"\n{3,}", "\n\n", normalized)

        return normalized.strip()

    def chunk_text(
        self,
        text: Optional[str],
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ) -> List[str]:
        """Splits normalized text into a list of strings with configurable size and overlap.

        Args:
            text: Raw extracted document text.
            chunk_size: Optional override for maximum chunk character length.
            chunk_overlap: Optional override for chunk character overlap.

        Returns:
            List of non-empty text chunk strings in chronological order.
        """
        target_size = chunk_size if chunk_size is not None else self.chunk_size
        target_overlap = chunk_overlap if chunk_overlap is not None else self.chunk_overlap
        self._validate_parameters(target_size, target_overlap)

        clean_text = self.normalize_whitespace(text)
        if not clean_text:
            return []

        text_len = len(clean_text)
        if text_len <= target_size:
            return [clean_text]

        chunks: List[str] = []
        start = 0

        # Hierarchical boundary markers in order of semantic preservation:
        # Paragraphs -> Newlines -> Sentences -> Clauses -> Words
        boundaries = ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " "]

        while start < text_len:
            end = min(start + target_size, text_len)

            if end < text_len:
                best_break = -1
                # Search back within the second half of the current chunk window
                search_start = max(start + (target_size // 2), end - target_size // 2)
                for boundary in boundaries:
                    pos = clean_text.rfind(boundary, search_start, end)
                    if pos != -1:
                        best_break = pos + len(boundary)
                        break
                if best_break != -1:
                    end = best_break

            chunk = clean_text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            if end >= text_len:
                break

            # Calculate start of next chunk with overlap
            next_start = end - target_overlap

            # Snap next_start forward to a word or newline boundary to avoid cutting words
            if target_overlap > 0 and next_start > start and next_start < end:
                space_pos = clean_text.find(" ", next_start, min(next_start + 20, end))
                if space_pos != -1:
                    next_start = space_pos + 1

            # Guarantee strict forward progress to prevent infinite loops
            if next_start <= start:
                next_start = end

            start = next_start

        return chunks

    def create_chunks(
        self,
        text: Optional[str],
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ) -> List[TextChunk]:
        """Splits normalized text and returns structured TextChunk objects with token estimates.

        Args:
            text: Raw extracted document text.
            chunk_size: Optional override for maximum chunk character length.
            chunk_overlap: Optional override for chunk character overlap.

        Returns:
            List of TextChunk instances containing index, text, character count, and token estimate.
        """
        raw_chunks = self.chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        return [
            TextChunk(
                index=idx,
                text=chunk_str,
                character_count=len(chunk_str),
                token_estimate=self.estimate_tokens(chunk_str),
            )
            for idx, chunk_str in enumerate(raw_chunks)
        ]


def get_chunking_service() -> TextChunkingService:
    """FastAPI dependency provider for TextChunkingService."""
    return TextChunkingService()
