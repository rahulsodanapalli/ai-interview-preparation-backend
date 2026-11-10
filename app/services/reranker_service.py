"""Reranker Service for post-retrieval re-ordering of document chunks.

Re-scores retrieved candidate chunks using cross-attention relevance scoring,
improving precision and grounding before prompt construction.
Supports FlashRank (local CPU cross-encoder), Ollama LLM scoring, and heuristic cross-scoring.
"""

import asyncio
import json
import logging
import re
from typing import Any, Dict, List, Optional
from fastapi import Depends

from app.core.config import settings
from app.services.chunking_service import TextChunk
from app.services.ollama_service import OllamaService, get_ollama_service
from app.utils.json_parser import strip_markdown_fences

logger = logging.getLogger(__name__)

# Check for flashrank availability
try:
    from flashrank import Ranker, RerankRequest
    HAS_FLASHRANK = True
except ImportError:
    HAS_FLASHRANK = False
    Ranker = None
    RerankRequest = None


class RerankerService:
    """Orchestrates candidate chunk reranking using local cross-encoder or LLM scoring.

    Attributes:
        enabled: Whether reranking is enabled in configuration.
        reranker_type: Selected reranker ('auto', 'flashrank', 'ollama', 'rrf').
        ollama_service: Optional OllamaService for LLM-based scoring.
    """

    def __init__(
        self,
        enabled: Optional[bool] = None,
        reranker_type: Optional[str] = None,
        ollama_service: Optional[OllamaService] = None,
    ):
        self.enabled = (
            enabled if enabled is not None else getattr(settings, "RAG_ENABLE_RERANKING", True)
        )
        self.reranker_type = (
            reranker_type or getattr(settings, "RAG_RERANKER_TYPE", "auto")
        ).lower()
        self.ollama_service = ollama_service or get_ollama_service()
        self._flashrank_model = None

    def _get_flashrank_model(self) -> Any:
        """Lazily initializes the local FlashRank cross-encoder model."""
        if not HAS_FLASHRANK:
            return None
        if self._flashrank_model is None:
            try:
                # Default tiny cross-encoder (~4MB footprint, extremely fast on CPU)
                self._flashrank_model = Ranker(model_name="ms-marco-TinyBERT-L-2-v2")
            except Exception as exc:
                logger.warning("Failed to initialize FlashRank ranker: %s. Falling back to heuristic reranker.", exc)
                self._flashrank_model = None
        return self._flashrank_model

    @staticmethod
    def _heuristic_cross_score(query: str, text: str, initial_score: float) -> float:
        """Lightweight token co-occurrence and phrase matching cross-scorer."""
        if not query or not text:
            return initial_score

        q_clean = re.sub(r"[^\w\s]", " ", query.lower()).split()
        t_clean = re.sub(r"[^\w\s]", " ", text.lower()).split()
        if not q_clean or not t_clean:
            return initial_score

        t_set = set(t_clean)
        matches = sum(1 for term in q_clean if term in t_set)
        term_coverage = matches / len(q_clean) if q_clean else 0.0

        # Exact multi-word phrase boost
        phrase_boost = 0.0
        if len(q_clean) >= 2:
            two_word_phrases = [" ".join(q_clean[i:i+2]) for i in range(len(q_clean) - 1)]
            t_str = " ".join(t_clean)
            phrase_hits = sum(1 for phrase in two_word_phrases if phrase in t_str)
            phrase_boost = min(0.20, (phrase_hits / len(two_word_phrases)) * 0.20)

        # Combine initial similarity (60%) with lexical coverage & phrase alignment (40%)
        combined = (0.60 * initial_score) + (0.30 * term_coverage) + (0.10 * phrase_boost)
        return round(min(1.0, max(0.0, combined)), 4)

    async def _rerank_with_ollama(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Scores candidate chunk relevance using a fast structured Ollama rating prompt."""
        if not candidates:
            return []

        # Prepare candidate excerpts (max 300 chars each) for fast LLM rating
        chunk_snippets = []
        for i, c in enumerate(candidates):
            text = c.get("text", "")
            snippet = " ".join(text.split())[:300]
            chunk_snippets.append(f"[{i}] {snippet}")

        prompt = (
            f"Query: {query}\n\n"
            "Evaluate each candidate excerpt below and rate its direct technical relevance to the query "
            "on a scale of 0 to 10 (where 10 is directly answering or discussing the query topic).\n"
            "Excerpts:\n"
            + "\n".join(chunk_snippets)
            + "\n\nRespond ONLY with a JSON object: {\"scores\": [{\"index\": int, \"score\": float}]}"
        )

        try:
            raw = await self.ollama_service.generate_chat_completion(
                messages=[{"role": "user", "content": prompt}],
                format="json",
            )
            parsed = json.loads(strip_markdown_fences(raw))
            scores_list = parsed.get("scores", [])
            scores_map = {}
            for item in scores_list:
                if isinstance(item, dict) and "index" in item and "score" in item:
                    scores_map[int(item["index"])] = float(item["score"]) / 10.0

            scored_candidates = []
            for i, c in enumerate(candidates):
                llm_score = scores_map.get(i, float(c.get("score", 0.0)))
                orig_score = float(c.get("score", 0.0))
                blended = round(0.50 * orig_score + 0.50 * llm_score, 4)
                updated = dict(c)
                updated["score"] = blended
                meta = dict(updated.get("metadata", {}))
                meta["rerank_score"] = blended
                updated["metadata"] = meta
                scored_candidates.append(updated)

            scored_candidates.sort(key=lambda x: x["score"], reverse=True)
            return scored_candidates
        except Exception as exc:
            logger.warning("Ollama reranking failed: %s. Using heuristic reranking.", exc)
            return self._rerank_with_heuristics(query, candidates)

    def _rerank_with_flashrank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Reranks candidates using local FlashRank ONNX cross-encoder."""
        ranker = self._get_flashrank_model()
        if not ranker:
            return self._rerank_with_heuristics(query, candidates)

        passages = [
            {"id": i, "text": c.get("text", "")}
            for i, c in enumerate(candidates)
        ]
        rerank_req = RerankRequest(query=query, passages=passages)
        results = ranker.rerank(rerank_req)

        scored_candidates = []
        for res in results:
            idx = res.get("id")
            score = float(res.get("score", 0.0))
            if idx is not None and idx < len(candidates):
                c = dict(candidates[idx])
                c["score"] = round(score, 4)
                meta = dict(c.get("metadata", {}))
                meta["rerank_score"] = round(score, 4)
                c["metadata"] = meta
                scored_candidates.append(c)

        scored_candidates.sort(key=lambda x: x["score"], reverse=True)
        return scored_candidates

    def _rerank_with_heuristics(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Reranks candidates using term-coverage and phrase-alignment heuristics."""
        scored = []
        for c in candidates:
            orig_score = float(c.get("score", 0.0))
            text = c.get("text", "")
            h_score = self._heuristic_cross_score(query, text, orig_score)
            updated = dict(c)
            updated["score"] = h_score
            meta = dict(updated.get("metadata", {}))
            meta["rerank_score"] = h_score
            updated["metadata"] = meta
            scored.append(updated)

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored

    async def rerank_matches(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Reranks candidate match dictionaries and returns top_k most relevant.

        Args:
            query: The search query text.
            candidates: Candidate chunk dictionaries (from Pinecone/Hybrid search).
            top_k: Number of highest-ranked chunks to return.

        Returns:
            Re-ordered list of candidate chunk dictionaries.
        """
        if not candidates or not self.enabled or not query or not query.strip():
            target_k = top_k if top_k is not None and top_k > 0 else len(candidates)
            return candidates[:target_k]

        mode = self.reranker_type
        if mode == "auto":
            if HAS_FLASHRANK:
                mode = "flashrank"
            else:
                mode = "heuristic"

        if mode == "flashrank" and HAS_FLASHRANK:
            reranked = await asyncio.to_thread(self._rerank_with_flashrank, query, candidates)
        elif mode == "ollama" and self.ollama_service:
            reranked = await self._rerank_with_ollama(query, candidates)
        else:
            reranked = self._rerank_with_heuristics(query, candidates)

        target_k = top_k if top_k is not None and top_k > 0 else len(reranked)
        return reranked[:target_k]


def get_reranker_service(
    ollama_service: OllamaService = Depends(get_ollama_service),
) -> RerankerService:
    """FastAPI dependency provider for RerankerService."""
    return RerankerService(ollama_service=ollama_service)

