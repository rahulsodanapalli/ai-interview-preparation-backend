import json
import logging
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import ValidationError
from sqlalchemy.orm import Session

from app.db.session import get_db_optional

from app.schemas.interview import (
    DocumentQuestionRequest,
    DocumentQuestionResponse,
    InterviewEvaluationRequest,
    InterviewEvaluationResponse,
    InterviewQuestionRequest,
    InterviewQuestionResponse,
)
from app.schemas.session import (
    CreateSessionRequest,
    InterviewSessionSummaryResponse,
    SessionDetailResponse,
    SessionNextQuestionResponse,
    SessionStatus,
    SessionSubmitAnswerResponse,
    SubmitAnswerRequest,
)
from app.services.embedding_service import (
    EmbeddingConnectionError,
    EmbeddingTimeoutError,
)
from app.services.ollama_service import (
    OllamaConnectionError,
    OllamaResponseError,
    OllamaService,
    OllamaTimeoutError,
    get_ollama_service,
)
from app.services.pinecone_service import (
    PineconeConfigurationError,
    PineconeConnectionError,
    PineconeDimensionMismatchError,
    PineconeIndexNotFoundError,
    PineconeResponseError,
    PineconeTimeoutError,
)
from app.services.rag_question_service import (
    RAGQuestionService,
    get_rag_question_service,
)
from app.services.retrieval_service import (
    EmptyQueryError,
    InvalidDocumentIdError,
)
from app.services.session_service import (
    SessionNotFoundError,
    SessionService,
    SessionStateError,
    get_session_service,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/interview", tags=["Interview"])

SYSTEM_PROMPT = (
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


# ============================================================================
# Reusable AI Question Generation & Answer Evaluation Logic
# ============================================================================


async def _execute_question_generation(
    payload: InterviewQuestionRequest,
    ollama_service: OllamaService,
) -> InterviewQuestionResponse:
    """Core question generation pipeline reusable across standalone and session endpoints."""
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
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    try:
        raw_output = await ollama_service.generate_chat_completion(
            messages=messages,
            format="json",
        )
    except (OllamaConnectionError, OllamaTimeoutError) as exc:
        logger.error("Ollama service unavailable during interview question generation: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AI inference service is currently unavailable. Please verify Ollama is running.",
        ) from exc
    except OllamaResponseError as exc:
        logger.error("Ollama error during interview question generation: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Failed to generate interview question from the local AI model.",
        ) from exc
    except Exception as exc:
        logger.error("Unexpected error during interview question generation: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while processing the request.",
        ) from exc

    # Clean up markdown code blocks if the LLM wrapped the JSON in ```json ... ```
    cleaned_output = raw_output.strip()
    if cleaned_output.startswith("```"):
        lines = cleaned_output.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned_output = "\n".join(lines).strip()

    # Parse JSON from model output
    try:
        parsed_data = json.loads(cleaned_output)
    except json.JSONDecodeError as exc:
        logger.error("Failed to decode JSON from AI model response: %s", raw_output)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="The AI model returned an invalid JSON response.",
        ) from exc

    if not isinstance(parsed_data, dict):
        logger.error("AI model response was not a JSON dictionary: %s", parsed_data)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="The AI model returned an unexpected JSON structure.",
        )

    # Fallback to request attributes if omitted or empty in model output
    if not parsed_data.get("difficulty"):
        parsed_data["difficulty"] = payload.difficulty.value
    if not parsed_data.get("topic"):
        parsed_data["topic"] = payload.topic

    # Strictly validate against the response schema
    try:
        return InterviewQuestionResponse.model_validate(parsed_data)
    except ValidationError as exc:
        logger.error("AI model response failed schema validation: %s | Errors: %s", parsed_data, exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="The AI model returned data that does not match the required question format.",
        ) from exc


async def _execute_answer_evaluation(
    payload: InterviewEvaluationRequest,
    ollama_service: OllamaService,
) -> InterviewEvaluationResponse:
    """Core answer evaluation pipeline reusable across standalone and session endpoints."""
    try:
        raw_output = await ollama_service.evaluate_interview_answer(
            question=payload.question,
            expected_answer_points=payload.expected_answer_points,
            candidate_answer=payload.candidate_answer,
        )
    except (OllamaConnectionError, OllamaTimeoutError) as exc:
        logger.error("Ollama service unavailable during answer evaluation: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AI inference service is currently unavailable. Please verify Ollama is running.",
        ) from exc
    except OllamaResponseError as exc:
        logger.error("Ollama error during answer evaluation: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Failed to evaluate interview answer from the local AI model.",
        ) from exc
    except Exception as exc:
        logger.error("Unexpected error during answer evaluation: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while processing the request.",
        ) from exc

    # Clean up markdown code blocks if the LLM wrapped the JSON in ```json ... ```
    cleaned_output = raw_output.strip()
    if cleaned_output.startswith("```"):
        lines = cleaned_output.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned_output = "\n".join(lines).strip()

    # Parse JSON from model output
    try:
        parsed_data = json.loads(cleaned_output)
    except json.JSONDecodeError as exc:
        logger.error("Failed to decode JSON from AI evaluation response: %s", raw_output)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="The AI model returned an invalid JSON response.",
        ) from exc

    if not isinstance(parsed_data, dict):
        logger.error("AI evaluation response was not a JSON dictionary: %s", parsed_data)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="The AI model returned an unexpected JSON structure.",
        )

    # Ensure list fields default to empty lists if omitted by LLM
    for list_field in ["correct_points", "missing_points", "strengths", "improvements"]:
        if list_field not in parsed_data or parsed_data[list_field] is None:
            parsed_data[list_field] = []

    # Strictly validate against the response schema
    try:
        return InterviewEvaluationResponse.model_validate(parsed_data)
    except ValidationError as exc:
        logger.error("AI evaluation response failed schema validation: %s | Errors: %s", parsed_data, exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="The AI model returned data that does not match the required evaluation format.",
        ) from exc


# ============================================================================
# Day 2 & Day 3 Standalone Endpoints
# ============================================================================


@router.post(
    "/question",
    response_model=InterviewQuestionResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate Structured Interview Question",
    description=(
        "Generates a tailored interview question with evaluation points and a follow-up question "
        "using the local Qwen 2.5 model via Ollama with guaranteed JSON structure."
    ),
    responses={
        200: {"description": "Structured interview question generated successfully"},
        422: {"description": "Validation error in request payload"},
        502: {"description": "Bad gateway or malformed response from AI model"},
        503: {"description": "Ollama service unavailable or timed out"},
    },
)
async def generate_interview_question(
    payload: InterviewQuestionRequest,
    ollama_service: OllamaService = Depends(get_ollama_service),
) -> InterviewQuestionResponse:
    """Generate a structured interview question with evaluation criteria and follow-up."""
    return await _execute_question_generation(payload, ollama_service)


@router.post(
    "/document-question",
    response_model=DocumentQuestionResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate Document-Grounded Interview Question (RAG)",
    description=(
        "Retrieves the most relevant technical chunks from an uploaded document in Pinecone "
        "and generates a tailored interview question strictly grounded in the candidate's "
        "documented experience and skills."
    ),
    responses={
        200: {
            "description": "Document-grounded question generated successfully",
            "model": DocumentQuestionResponse,
        },
        400: {"description": "Invalid document ID or query parameters"},
        422: {"description": "Validation error in request payload"},
        500: {"description": "Internal model output or vector processing error"},
        502: {"description": "Bad gateway or malformed JSON from Ollama model"},
        503: {"description": "Pinecone or Ollama service is unavailable"},
    },
)
async def generate_document_interview_question(
    payload: DocumentQuestionRequest,
    rag_service: RAGQuestionService = Depends(get_rag_question_service),
) -> DocumentQuestionResponse:
    """Generate a tailored interview question grounded in candidate document chunks from Pinecone."""
    if not payload.document_id or not payload.document_id.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="'document_id' is required in request payload.",
        )

    try:
        return await rag_service.generate_document_question(payload)
    except (InvalidDocumentIdError, EmptyQueryError) as exc:
        logger.warning("Invalid RAG question request parameter: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except (EmbeddingConnectionError, EmbeddingTimeoutError) as exc:
        logger.error("Embedding service unavailable during document question generation: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Embedding service is currently unavailable. Please verify Ollama is running.",
        ) from exc
    except (PineconeConnectionError, PineconeTimeoutError) as exc:
        logger.error("Pinecone service unavailable during document question generation: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Pinecone vector storage service is currently unavailable.",
        ) from exc
    except (OllamaConnectionError, OllamaTimeoutError) as exc:
        logger.error("Ollama inference service unavailable during RAG question generation: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AI inference service is currently unavailable. Please verify Ollama is running.",
        ) from exc
    except (
        PineconeConfigurationError,
        PineconeDimensionMismatchError,
        PineconeIndexNotFoundError,
        PineconeResponseError,
    ) as exc:
        logger.error("Pinecone storage error during document question generation: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Pinecone vector storage error: {exc}",
        ) from exc
    except OllamaResponseError as exc:
        logger.error("Ollama response error during document question generation: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Failed to generate interview question from the local AI model.",
        ) from exc
    except ValidationError as exc:
        logger.error("RAG question output failed schema validation: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="The AI model returned data that does not match the required question format.",
        ) from exc
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Unexpected error during document question generation: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected internal error occurred while generating the document question.",
        ) from exc


@router.post(
    "/evaluate",
    response_model=InterviewEvaluationResponse,
    status_code=status.HTTP_200_OK,
    summary="Evaluate Interview Answer",
    description=(
        "Evaluates a candidate's answer against expected criteria using local Qwen 2.5 "
        "and returns structured feedback, strengths, improvements, and a rubric-aligned score."
    ),
    responses={
        200: {"description": "Interview answer evaluated successfully"},
        422: {"description": "Validation error in request payload"},
        502: {"description": "Bad gateway or malformed response from AI model"},
        503: {"description": "Ollama service unavailable or timed out"},
    },
)
async def evaluate_interview_answer(
    payload: InterviewEvaluationRequest,
    ollama_service: OllamaService = Depends(get_ollama_service),
) -> InterviewEvaluationResponse:
    """Evaluate candidate answer and provide structured feedback and rubric scoring."""
    return await _execute_answer_evaluation(payload, ollama_service)


# ============================================================================
# Day 4 Session Management Endpoints
# ============================================================================


@router.post(
    "/sessions",
    response_model=SessionDetailResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create Interview Session",
    description="Initialize a new interview session with configured role, level, difficulty, and total questions.",
    responses={
        201: {"description": "Session created successfully"},
        422: {"description": "Validation error in session configuration"},
    },
)
async def create_session(
    payload: CreateSessionRequest,
    session_service: SessionService = Depends(get_session_service),
    db: Optional[Session] = Depends(get_db_optional),
) -> SessionDetailResponse:
    """Create a new interview session."""
    return session_service.create_session(payload, db=db)


@router.get(
    "/sessions",
    response_model=List[SessionDetailResponse],
    status_code=status.HTTP_200_OK,
    summary="List Interview Sessions",
    description="Retrieve a paginated list of interview sessions ordered by most recent.",
    responses={
        200: {"description": "List of interview sessions retrieved successfully"},
    },
)
async def list_sessions(
    limit: int = 50,
    skip: int = 0,
    session_service: SessionService = Depends(get_session_service),
    db: Optional[Session] = Depends(get_db_optional),
) -> List[SessionDetailResponse]:
    """Retrieve list of interview sessions with optional pagination."""
    return session_service.list_sessions(db=db, limit=limit, skip=skip)


@router.get(
    "/sessions/{session_id}",
    response_model=SessionDetailResponse,
    status_code=status.HTTP_200_OK,
    summary="Get Interview Session Details",
    description="Retrieve full details, question history, and progress for an interview session.",
    responses={
        200: {"description": "Session details retrieved successfully"},
        404: {"description": "Interview session not found"},
    },
)
async def get_session(
    session_id: str,
    session_service: SessionService = Depends(get_session_service),
    db: Optional[Session] = Depends(get_db_optional),
) -> SessionDetailResponse:
    """Retrieve session details and current progress."""
    try:
        return session_service.get_session(session_id, db=db)
    except SessionNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc


@router.post(
    "/sessions/{session_id}/next-question",
    response_model=SessionNextQuestionResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate Next Interview Question for Session",
    description="Generate and advance to the next interview question using Qwen 2.5 on Ollama.",
    responses={
        200: {"description": "Next question generated successfully"},
        400: {"description": "Invalid session state, question unanswered, or maximum questions reached"},
        404: {"description": "Interview session not found"},
        502: {"description": "Bad gateway from AI model"},
        503: {"description": "Ollama service unavailable or timed out"},
    },
)
async def get_next_question_for_session(
    session_id: str,
    session_service: SessionService = Depends(get_session_service),
    ollama_service: OllamaService = Depends(get_ollama_service),
    db: Optional[Session] = Depends(get_db_optional),
) -> SessionNextQuestionResponse:
    """Generate the next interview question for the session using Qwen 2.5."""
    try:
        session = session_service.validate_can_generate_question(session_id, db=db)
    except SessionNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except SessionStateError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    # Generate next question using session configuration
    question_payload = InterviewQuestionRequest(
        role=session.role,
        experience_level=session.experience_level,
        difficulty=session.difficulty,
        topic=session.topic,
        question_type=session.question_type,
    )

    question_data = await _execute_question_generation(question_payload, ollama_service)
    return session_service.add_question_to_session(session_id, question_data, db=db)


@router.post(
    "/sessions/{session_id}/answers",
    response_model=SessionSubmitAnswerResponse,
    status_code=status.HTTP_200_OK,
    summary="Submit and Evaluate Candidate Answer for Session",
    description="Submit candidate answer for the active session question and evaluate it using Qwen 2.5 on Ollama.",
    responses={
        200: {"description": "Answer evaluated and recorded successfully"},
        400: {"description": "Invalid session state, question already answered, or session completed"},
        404: {"description": "Interview session not found"},
        422: {"description": "Validation error in submitted answer"},
        502: {"description": "Bad gateway from AI model"},
        503: {"description": "Ollama service unavailable or timed out"},
    },
)
async def submit_answer_for_session(
    session_id: str,
    payload: SubmitAnswerRequest,
    session_service: SessionService = Depends(get_session_service),
    ollama_service: OllamaService = Depends(get_ollama_service),
    db: Optional[Session] = Depends(get_db_optional),
) -> SessionSubmitAnswerResponse:
    """Submit candidate answer and evaluate against expected criteria."""
    try:
        target_question = session_service.validate_can_submit_answer(
            session_id=session_id,
            question_number=payload.question_number,
            db=db,
        )
    except SessionNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except SessionStateError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    eval_payload = InterviewEvaluationRequest(
        question=target_question.question,
        expected_answer_points=target_question.expected_answer_points,
        candidate_answer=payload.candidate_answer,
    )

    evaluation = await _execute_answer_evaluation(eval_payload, ollama_service)
    return session_service.record_answer_and_evaluation(
        session_id=session_id,
        question_number=target_question.question_number,
        candidate_answer=payload.candidate_answer,
        evaluation=evaluation,
        db=db,
    )


@router.post(
    "/sessions/{session_id}/complete",
    response_model=InterviewSessionSummaryResponse,
    status_code=status.HTTP_200_OK,
    summary="Complete Interview Session and Generate Summary",
    description="Mark session as completed and calculate overall score, rubric rating, strengths, and improvement areas.",
    responses={
        200: {"description": "Session completed and summary generated successfully"},
        404: {"description": "Interview session not found"},
    },
)
async def complete_session(
    session_id: str,
    session_service: SessionService = Depends(get_session_service),
    db: Optional[Session] = Depends(get_db_optional),
) -> InterviewSessionSummaryResponse:
    """Complete the session and produce final interview summary."""
    try:
        return session_service.complete_session(session_id, db=db)
    except SessionNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc

