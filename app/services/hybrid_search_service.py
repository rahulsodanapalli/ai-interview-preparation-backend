"""Hybrid Search Service combining semantic dense vector search with BM25 lexical keyword search.

Uses Reciprocal Rank Fusion (RRF) and Convex Score Combination to merge dense vector matches
from Pinecone with document-scoped BM25 lexical matching across candidate chunk texts.
"""

from collections import Counter
import logging
import math
import re
from typing import Any, Dict, List, Optional, Tuple

from app.core.config import settings

logger = logging.getLogger(__name__)

# Standard English stopwords to filter during lexical tokenization
STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't",
    "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
    "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't",
    "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i",
    "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's",
    "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself",
    "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought",
    "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she",
    "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such",
    "than", "that", "that's", "the", "their", "theirs", "them", "themselves",
    "then", "there", "there's", "these", "they", "they'd", "they'll", "they're",
    "they've", "this", "those", "through", "to", "too", "under", "until", "up",
    "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were",
    "weren't", "what", "what's", "when", "when's", "where", "where's", "which",
    "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would",
    "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
    "yourself", "yourselves",
}


def tokenize(text: str, remove_stopwords: bool = True) -> List[str]:
    """Tokenize text into lowercase alphanumeric terms with optional stopword filtering."""
    if not text:
        return []
    tokens = re.findall(r"\b[a-zA-Z0-9_\-\.\#\+\/]+\b", text.lower())
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    return tokens


class BM25Okapi:
    """Pure-Python, zero-dependency BM25Okapi implementation for lexical search scoring.

    Attributes:
        k1: Term frequency saturation parameter (default: 1.5).
        b: Document length normalization parameter (default: 0.75).
        corpus_size: Number of documents in the corpus.
        avgdl: Average document length in tokens.
    """

    def __init__(
        self,
        corpus: List[List[str]],
        k1: float = 1.5,
        b: float = 0.75,
        epsilon: float = 0.25,
    ):
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
        self.corpus_size = len(corpus)
        self.doc_lengths = [len(doc) for doc in corpus]
        self.avgdl = sum(self.doc_lengths) / self.corpus_size if self.corpus_size > 0 else 0.0

        self.doc_freqs: List[Counter] = []
        self.idf: Dict[str, float] = {}
        self._initialize(corpus)

    def _initialize(self, corpus: List[List[str]]) -> None:
        nd: Counter = Counter()
        for doc in corpus:
            freq = Counter(doc)
            self.doc_freqs.append(freq)
            for term in freq.keys():
                nd[term] += 1

        negative_idfs = []
        idf_sum = 0.0
        for term, freq in nd.items():
            # Standard Lucene/BM25 IDF formula
            idf = math.log((self.corpus_size - freq + 0.5) / (freq + 0.5) + 1.0)
            self.idf[term] = idf
            idf_sum += idf
            if idf < 0:
                negative_idfs.append(term)

        average_idf = idf_sum / len(self.idf) if self.idf else 0.0
        eps = self.epsilon * average_idf
        for term in negative_idfs:
            self.idf[term] = eps

    def get_scores(self, query_tokens: List[str]) -> List[float]:
        """Calculates BM25 relevance scores for all documents in the corpus."""
        scores = [0.0] * self.corpus_size
        if not query_tokens or self.corpus_size == 0 or self.avgdl == 0:
            return scores

        query_counts = Counter(query_tokens)
        for term, q_weight in query_counts.items():
            if term not in self.idf:
                continue
            idf = self.idf[term]
            for i in range(self.corpus_size):
                freq = self.doc_freqs[i].get(term, 0)
                if freq == 0:
                    continue
                num = freq * (self.k1 + 1)
                denom = freq + self.k1 * (1 - self.b + self.b * (self.doc_lengths[i] / self.avgdl))
                scores[i] += idf * (num / denom) * q_weight

        return scores


def reciprocal_rank_fusion(
    ranked_lists: List[List[str]],
    weights: Optional[List[float]] = None,
    k: int = 60,
) -> Dict[str, float]:
    """Computes Reciprocal Rank Fusion (RRF) scores across multiple ranked lists of IDs.

    Formula:
        RRF_Score(d) = sum(w_i / (k + rank_i(d))) for each ranked list i

    Args:
        ranked_lists: List of ranked item ID lists from different retrieval methods.
        weights: Optional importance weights for each list (defaults to 1.0 each).
        k: Smoothing constant to dampen the impact of high-rank differences (default: 60).

    Returns:
        Dictionary mapping item ID to its combined RRF score.
    """
    if not ranked_lists:
        return {}

    num_lists = len(ranked_lists)
    if weights is None or len(weights) != num_lists:
        weights = [1.0] * num_lists

    rrf_scores: Dict[str, float] = {}

    for list_idx, ranked_items in enumerate(ranked_lists):
        w = weights[list_idx]
        for rank, item_id in enumerate(ranked_items, start=1):
            rrf_scores[item_id] = rrf_scores.get(item_id, 0.0) + (w / (k + rank))

    return rrf_scores


class HybridSearchService:
    """Combines dense vector retrieval results with lexical BM25 matching.

    Attributes:
        enabled: Whether hybrid search is enabled (from settings).
        alpha: Weight for dense vector search vs lexical search (0.0 to 1.0).
        k_rrf: Constant factor for Reciprocal Rank Fusion.
    """

    def __init__(
        self,
        enabled: Optional[bool] = None,
        alpha: Optional[float] = None,
        k_rrf: int = 60,
    ):
        self.enabled = (
            enabled if enabled is not None else getattr(settings, "RAG_ENABLE_HYBRID_SEARCH", True)
        )
        self.alpha = (
            alpha if alpha is not None else getattr(settings, "RAG_HYBRID_ALPHA", 0.5)
        )
        self.k_rrf = k_rrf

    def fuse_matches(
        self,
        dense_matches: List[Dict[str, Any]],
        query_text: str,
        top_k: Optional[int] = None,
        alpha: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Fuses dense vector matches with BM25 lexical keyword scores.

        Args:
            dense_matches: List of match dictionaries returned by Pinecone.
            query_text: Raw user/topic query string.
            top_k: Number of fused results to return.
            alpha: Optional override for dense vs lexical weight.

        Returns:
            List of fused match dictionaries re-ordered by combined relevance score.
        """
        if not dense_matches:
            return []

        if not self.enabled or not query_text or not query_text.strip():
            target_k = top_k if top_k is not None and top_k > 0 else len(dense_matches)
            return dense_matches[:target_k]

        effective_alpha = alpha if alpha is not None else self.alpha
        effective_alpha = max(0.0, min(1.0, effective_alpha))

        # 1. Extract texts and build BM25 corpus from dense candidate chunks
        corpus_tokens: List[List[str]] = []
        for match in dense_matches:
            text = match.get("text") or match.get("metadata", {}).get("text", "")
            corpus_tokens.append(tokenize(text))

        # 2. Score candidate chunks against the query using BM25
        query_tokens = tokenize(query_text)
        bm25 = BM25Okapi(corpus_tokens)
        bm25_scores = bm25.get_scores(query_tokens)

        # 3. Create ranked lists for RRF
        # Dense ranking (already sorted descending by Pinecone score)
        dense_ranked_ids = [m.get("id", f"idx_{i}") for i, m in enumerate(dense_matches)]

        # Lexical ranking
        lexical_pairs = [
            (dense_matches[i].get("id", f"idx_{i}"), bm25_scores[i])
            for i in range(len(dense_matches))
        ]
        lexical_pairs.sort(key=lambda x: x[1], reverse=True)
        lexical_ranked_ids = [pair[0] for pair in lexical_pairs]

        # 4. Compute RRF scores with alpha weights
        dense_weight = effective_alpha
        lexical_weight = 1.0 - effective_alpha
        rrf_dict = reciprocal_rank_fusion(
            [dense_ranked_ids, lexical_ranked_ids],
            weights=[dense_weight, lexical_weight],
            k=self.k_rrf,
        )

        # Normalize RRF scores to [0.0, 1.0] scale preserving relative order
        max_rrf = max(rrf_dict.values()) if rrf_dict else 1.0
        min_rrf = min(rrf_dict.values()) if rrf_dict else 0.0
        range_rrf = max_rrf - min_rrf

        # 5. Build enriched match objects
        # Normalize BM25 scores for metadata reporting
        max_bm25 = max(bm25_scores) if bm25_scores and max(bm25_scores) > 0 else 1.0

        matches_by_id = {m.get("id", f"idx_{i}"): (i, m) for i, m in enumerate(dense_matches)}
        fused_matches: List[Dict[str, Any]] = []

        for m_id, rrf_val in rrf_dict.items():
            if m_id not in matches_by_id:
                continue
            orig_idx, orig_match = matches_by_id[m_id]
            dense_score = float(orig_match.get("score", 0.0))
            raw_bm25 = float(bm25_scores[orig_idx])
            norm_bm25 = raw_bm25 / max_bm25 if max_bm25 > 0 else 0.0

            # Convex score combination: weighted blend of original cosine score and normalized BM25
            convex_score = (effective_alpha * dense_score) + ((1.0 - effective_alpha) * norm_bm25)

            # Combined score combines convex score with RRF ordering signal
            if range_rrf > 0:
                normalized_rrf = (rrf_val - min_rrf) / range_rrf
                # Smooth blend: 70% convex semantic/keyword score, 30% rank-fusion score
                final_score = round(0.70 * convex_score + 0.30 * normalized_rrf, 4)
            else:
                final_score = round(convex_score, 4)

            # Shallow copy and update scores
            updated_match = dict(orig_match)
            updated_match["score"] = final_score
            meta = dict(updated_match.get("metadata", {}))
            meta["dense_score"] = round(dense_score, 4)
            meta["bm25_score"] = round(raw_bm25, 4)
            meta["rrf_score"] = round(rrf_val, 6)
            updated_match["metadata"] = meta

            fused_matches.append(updated_match)

        # Sort descending by final combined score
        fused_matches.sort(key=lambda x: x["score"], reverse=True)

        target_k = top_k if top_k is not None and top_k > 0 else len(fused_matches)
        return fused_matches[:target_k]


def get_hybrid_search_service() -> HybridSearchService:
    """FastAPI dependency provider for HybridSearchService."""
    return HybridSearchService()

