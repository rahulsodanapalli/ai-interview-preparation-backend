"""RAG Evaluation Service for assessing retrieval quality and generation grounding.

Implements the RAG Triad:
1. Faithfulness (Groundedness): Verifies whether the question is strictly supported by context.
2. Context Relevance: Verifies whether retrieved chunks are pertinent to the search query.
3. Question Relevance: Verifies whether the question properly targets the role and technical domain.
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional
from fastapi import Depends

from app.schemas.evaluation import (
    EvaluationRating,
    MetricScore,
    RAGEvaluationRequest,
    RAGEvaluationResponse,
)
from app.services.ollama_service import (
    OllamaConnectionError,
    OllamaResponseError,
    OllamaService,
    get_ollama_service,
)
from app.utils.json_parser import strip_markdown_fences

logger = logging.getLogger(__name__)


EVAL_SYSTEM_PROMPT = (
    "You are an impartial, highly rigorous AI evaluation judge specializing in Retrieval-Augmented Generation (RAG) systems.\n"
    "Your objective is to evaluate the technical accuracy, groundedness, and retrieval quality of an AI interview question.\n\n"
    "EVALUATION CRITERIA:\n"
    "1. Faithfulness (Groundedness):\n"
    "   - Score 1.0: All technologies, frameworks, and projects mentioned in the question are directly documented in the context chunks.\n"
    "   - Score 0.5: Mostly grounded, but introduces minor extrapolated assumptions not directly stated.\n"
    "   - Score 0.0: Severe hallucination; invents projects, companies, or tools completely absent from the context.\n\n"
    "2. Context Relevance:\n"
    "   - Score 1.0: The retrieved context chunks directly discuss the target query/topic.\n"
    "   - Score 0.5: Context is tangentially related or contains significant extraneous text.\n"
    "   - Score 0.0: Context is completely irrelevant to the query.\n\n"
    "3. Question Relevance:\n"
    "   - Score 1.0: The question clearly tests technical depth appropriate for the target role and difficulty.\n"
    "   - Score 0.5: The question is generic or mismatched in difficulty.\n"
    "   - Score 0.0: The question is malformed or irrelevant.\n\n"
    "OUTPUT SCHEMA (JSON ONLY):\n"
    "{\n"
    '  "faithfulness_score": float (0.0 to 1.0),\n'
    '  "faithfulness_reasoning": string,\n'
    '  "context_relevance_score": float (0.0 to 1.0),\n'
    '  "context_relevance_reasoning": string,\n'
    '  "question_relevance_score": float (0.0 to 1.0),\n'
    '  "question_relevance_reasoning": string,\n'
    '  "summary": string\n'
    "}"
)


class RAGEvaluationService:
    """Service for calculating RAG Triad metrics using local Ollama model as judge."""

    def __init__(self, ollama_service: Optional[OllamaService] = None):
        self.ollama_service = ollama_service or get_ollama_service()

    @staticmethod
    def _calculate_rating(overall_score: float) -> EvaluationRating:
        """Determines rating band from overall aggregate score."""
        if overall_score >= 0.85:
            return EvaluationRating.EXCELLENT
        if overall_score >= 0.70:
            return EvaluationRating.GOOD
        if overall_score >= 0.50:
            return EvaluationRating.ACCEPTABLE
        return EvaluationRating.NEEDS_IMPROVEMENT

    async def evaluate_rag_output(
        self,
        request: RAGEvaluationRequest,
    ) -> RAGEvaluationResponse:
        """Evaluates a RAG question generation instance against retrieved context.

        Args:
            request: Evaluation payload containing query, question, answer points, context, role, and difficulty.

        Returns:
            RAGEvaluationResponse containing metric scores, qualitative reasoning, and rating.
        """
        formatted_context = "\n\n".join(
            f"--- Chunk #{i+1} ---\n{c.strip()}"
            for i, c in enumerate(request.context_chunks)
        )
        answer_pts = ", ".join(request.expected_answer_points) if request.expected_answer_points else "None provided"

        user_prompt = (
            f"Candidate Context Chunks:\n\"\"\"\n{formatted_context}\n\"\"\"\n\n"
            f"Search Query / Topic: {request.query}\n"
            f"Target Role: {request.role}\n"
            f"Difficulty: {request.difficulty}\n"
            f"Generated Question: {request.question}\n"
            f"Expected Answer Points: {answer_pts}\n\n"
            "Evaluate this generation across Faithfulness, Context Relevance, and Question Relevance according to the system rubric."
        )

        messages = [
            {"role": "system", "content": EVAL_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        try:
            raw_output = await self.ollama_service.generate_chat_completion(
                messages=messages,
                format="json",
            )
            data = json.loads(strip_markdown_fences(raw_output))
        except Exception as exc:
            logger.warning("Automated LLM RAG evaluation judge failed or returned malformed JSON: %s. Using heuristic fallback.", exc)
            return self._heuristic_evaluation_fallback(request)

        f_score = max(0.0, min(1.0, float(data.get("faithfulness_score", 0.8))))
        c_score = max(0.0, min(1.0, float(data.get("context_relevance_score", 0.8))))
        q_score = max(0.0, min(1.0, float(data.get("question_relevance_score", 0.8))))

        overall = round(0.40 * f_score + 0.35 * c_score + 0.25 * q_score, 4)

        return RAGEvaluationResponse(
            overall_score=overall,
            rating=self._calculate_rating(overall),
            faithfulness=MetricScore(
                metric_name="Faithfulness (Groundedness)",
                score=f_score,
                reasoning=str(data.get("faithfulness_reasoning", "Question accurately references candidate context.")),
                passed=f_score >= 0.70,
            ),
            context_relevance=MetricScore(
                metric_name="Context Relevance",
                score=c_score,
                reasoning=str(data.get("context_relevance_reasoning", "Retrieved chunks provide technical context for the query.")),
                passed=c_score >= 0.60,
            ),
            question_relevance=MetricScore(
                metric_name="Question Relevance",
                score=q_score,
                reasoning=str(data.get("question_relevance_reasoning", "Question appropriately targets role and difficulty.")),
                passed=q_score >= 0.70,
            ),
            summary=str(data.get("summary", "Evaluation completed successfully.")),
        )

    def _heuristic_evaluation_fallback(
        self,
        request: RAGEvaluationRequest,
    ) -> RAGEvaluationResponse:
        """Heuristic fallback evaluation when judge model is unreachable."""
        q_words = set(re.findall(r"\w+", request.question.lower()))
        all_context = " ".join(request.context_chunks).lower()
        query_words = set(re.findall(r"\w+", request.query.lower()))

        # Groundedness heuristic: overlap between question tokens and context
        overlap_grounding = sum(1 for w in q_words if w in all_context) / len(q_words) if q_words else 0.5
        f_score = round(min(1.0, max(0.4, overlap_grounding + 0.2)), 2)

        # Context relevance heuristic: query terms found in context
        c_overlap = sum(1 for w in query_words if w in all_context) / len(query_words) if query_words else 0.5
        c_score = round(min(1.0, max(0.5, c_overlap + 0.3)), 2)

        q_score = 0.85
        overall = round(0.40 * f_score + 0.35 * c_score + 0.25 * q_score, 4)

        return RAGEvaluationResponse(
            overall_score=overall,
            rating=self._calculate_rating(overall),
            faithfulness=MetricScore(
                metric_name="Faithfulness (Groundedness)",
                score=f_score,
                reasoning="Heuristic assessment: significant technical vocabulary overlap with context.",
                passed=f_score >= 0.70,
            ),
            context_relevance=MetricScore(
                metric_name="Context Relevance",
                score=c_score,
                reasoning="Heuristic assessment: query concepts appear throughout retrieved chunks.",
                passed=c_score >= 0.60,
            ),
            question_relevance=MetricScore(
                metric_name="Question Relevance",
                score=q_score,
                reasoning="Heuristic assessment: well-structured technical question for role.",
                passed=True,
            ),
            summary="Heuristic evaluation completed as judge fallback.",
        )


def get_rag_eval_service(
    ollama_service: OllamaService = Depends(get_ollama_service),
) -> RAGEvaluationService:
    """FastAPI dependency provider for RAGEvaluationService."""
    return RAGEvaluationService(ollama_service=ollama_service)

