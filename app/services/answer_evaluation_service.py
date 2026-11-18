"""Interview Answer Evaluation Service using local Ollama LLM."""

import logging
from typing import Optional
from fastapi import Depends
from pydantic import ValidationError

from app.schemas.interview import InterviewEvaluationRequest, InterviewEvaluationResponse
from app.services.ollama_service import OllamaResponseError, OllamaService, get_ollama_service
from app.utils.json_parser import parse_llm_json

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

        parsed_data = parse_llm_json(
            raw_output,
            error_cls=EvaluationOutputFormatError,
            decode_error_message="The AI model returned an invalid JSON response.",
            type_error_message="The AI model returned an unexpected JSON structure.",
        )

        for list_field in ("correct_points", "missing_points", "strengths", "improvements"):
            val = parsed_data.get(list_field)
            if val is None:
                parsed_data[list_field] = []
            elif isinstance(val, str):
                parsed_data[list_field] = [val.strip()] if val.strip() else []
            elif isinstance(val, list):
                parsed_data[list_field] = [str(item).strip() for item in val if str(item).strip()]

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

