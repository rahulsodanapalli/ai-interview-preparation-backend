"""Standalone Interview Question Generation Service using local Ollama LLM."""

import json
import logging
from typing import Optional
from fastapi import Depends
from pydantic import ValidationError

from app.schemas.interview import InterviewQuestionRequest, InterviewQuestionResponse
from app.services.ollama_service import OllamaResponseError, OllamaService, get_ollama_service
from app.utils.json_parser import strip_markdown_fences

logger = logging.getLogger(__name__)

INTERVIEW_QUESTION_SYSTEM_PROMPT = (
    "You are an expert technical interviewer conducting a live structured interview. "
    "Your task is to generate exactly ONE high-quality interview question tailored to the candidate's "
    "role, experience level, difficulty, topic, and question type.\n\n"
    "CRITICAL OUTPUT REQUIREMENTS:\n"
    "1. Respond ONLY with a valid JSON object matching the requested schema.\n"
    "2. Do NOT wrap the JSON in markdown code blocks (no ```json or ```).\n"
    "3. Do NOT include any introductory or concluding text, explanations, or notes.\n"
    "4. The JSON object MUST contain the following keys:\n"
    "   - \"question\": string (the tailored interview question)\n"
    "   - \"expected_answer_points\": list of strings (2-4 key technical criteria or concepts a strong answer should cover)\n"
    "   - \"follow_up_question\": string (one direct follow-up question to probe deeper)\n"
    "   - \"difficulty\": string (the requested difficulty level)\n"
    "   - \"topic\": string (the requested topic)\n"
)


class QuestionOutputFormatError(OllamaResponseError):
    """Raised when the AI model output for question generation is malformed or fails schema validation."""


class QuestionGenerationService:
    """Service responsible for generating structured standalone and session interview questions."""

    def __init__(self, ollama_service: Optional[OllamaService] = None) -> None:
        self.ollama_service = ollama_service or get_ollama_service()

    async def generate_question(
        self,
        payload: InterviewQuestionRequest,
    ) -> InterviewQuestionResponse:
        """Generate a tailored interview question with expected answer points and follow-up."""
        user_prompt = (
            f"Role: {payload.role}\n"
            f"Experience Level: {payload.experience_level.value}\n"
            f"Difficulty: {payload.difficulty.value}\n"
            f"Topic: {payload.topic}\n"
            f"Question Type: {payload.question_type.value}\n\n"
            f"Generate a {payload.difficulty.value} {payload.question_type.value} interview question "
            f"on '{payload.topic}' for a {payload.experience_level.value} {payload.role}."
        )

        messages = [
            {"role": "system", "content": INTERVIEW_QUESTION_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        raw_output = await self.ollama_service.generate_chat_completion(
            messages=messages,
            format="json",
        )

        cleaned_output = strip_markdown_fences(raw_output)

        try:
            parsed_data = json.loads(cleaned_output)
        except json.JSONDecodeError as exc:
            logger.error("Failed to decode JSON from AI model response: %s", raw_output)
            raise QuestionOutputFormatError("The AI model returned an invalid JSON response.") from exc

        if not isinstance(parsed_data, dict):
            logger.error("AI model response was not a JSON dictionary: %s", parsed_data)
            raise QuestionOutputFormatError("The AI model returned an unexpected JSON structure.")

        if not parsed_data.get("difficulty"):
            parsed_data["difficulty"] = payload.difficulty.value
        if not parsed_data.get("topic"):
            parsed_data["topic"] = payload.topic

        try:
            return InterviewQuestionResponse.model_validate(parsed_data)
        except ValidationError as exc:
            logger.error("AI model response failed schema validation: %s | Errors: %s", parsed_data, exc)
            raise QuestionOutputFormatError(
                "The AI model returned data that does not match the required question format."
            ) from exc


def get_question_generation_service(
    ollama_service: OllamaService = Depends(get_ollama_service),
) -> QuestionGenerationService:
    """FastAPI dependency provider for QuestionGenerationService."""
    return QuestionGenerationService(ollama_service=ollama_service)

