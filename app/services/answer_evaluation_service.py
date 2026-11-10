"""Interview Answer Evaluation Service using local Ollama LLM."""

import json
import logging
from typing import Optional
from fastapi import Depends
from pydantic import ValidationError

from app.schemas.interview import InterviewEvaluationRequest, InterviewEvaluationResponse
from app.services.ollama_service import OllamaResponseError, OllamaService, get_ollama_service
from app.utils.json_parser import strip_markdown_fences

logger = logging.getLogger(__name__)


class EvaluationOutputFormatError(OllamaResponseError):
    """Raised when the AI model output for answer evaluation is malformed or fails schema validation."""


class AnswerEvaluationService:
    """Service responsible for evaluating candidate interview answers against expected criteria."""

    def __init__(self, ollama_service: Optional[OllamaService] = None) -> None:
        self.ollama_service = ollama_service or get_ollama_service()

    async def evaluate_answer(
        self,
        payload: InterviewEvaluationRequest,
    ) -> InterviewEvaluationResponse:
        """Evaluate a candidate's answer and return structured rubric feedback."""
        raw_output = await self.ollama_service.evaluate_interview_answer(
            question=payload.question,
            expected_answer_points=payload.expected_answer_points,
            candidate_answer=payload.candidate_answer,
        )

        cleaned_output = strip_markdown_fences(raw_output)

        try:
            parsed_data = json.loads(cleaned_output)
        except json.JSONDecodeError as exc:
            logger.error("Failed to decode JSON from AI evaluation response: %s", raw_output)
            raise EvaluationOutputFormatError("The AI model returned an invalid JSON response.") from exc

        if not isinstance(parsed_data, dict):
            logger.error("AI evaluation response was not a JSON dictionary: %s", parsed_data)
            raise EvaluationOutputFormatError("The AI model returned an unexpected JSON structure.")

        for list_field in ("correct_points", "missing_points", "strengths", "improvements"):
            if list_field not in parsed_data or parsed_data[list_field] is None:
                parsed_data[list_field] = []

        try:
            return InterviewEvaluationResponse.model_validate(parsed_data)
        except ValidationError as exc:
            logger.error("AI evaluation response failed schema validation: %s | Errors: %s", parsed_data, exc)
            raise EvaluationOutputFormatError(
                "The AI model returned data that does not match the required evaluation format."
            ) from exc


def get_answer_evaluation_service(
    ollama_service: OllamaService = Depends(get_ollama_service),
) -> AnswerEvaluationService:
    """FastAPI dependency provider for AnswerEvaluationService."""
    return AnswerEvaluationService(ollama_service=ollama_service)

