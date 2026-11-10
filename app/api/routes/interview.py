"""API routes for interview question generation, answer evaluation, RAG evaluation, and sessions."""

import logging
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.exceptions import (
    handle_interview_ai_errors,
    handle_rag_question_errors,
    handle_session_errors,
)
from app.db.session import get_db_optional
from app.schemas.document import SourceQuestionsRequest, SourceQuestionsResponse
from app.schemas.evaluation import RAGEvaluationRequest, RAGEvaluationResponse
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
    SessionSubmitAnswerResponse,
    SubmitAnswerRequest,
)
from app.services.answer_evaluation_service import (
    AnswerEvaluationService,
    get_answer_evaluation_service,
)
from app.services.question_generation_service import (
    QuestionGenerationService,
    get_question_generation_service,
)
from app.services.rag_eval_service import RAGEvaluationService, get_rag_eval_service
from app.services.rag_question_service import RAGQuestionService, get_rag_question_service
from app.services.session_service import SessionService, get_session_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/interview", tags=["Interview"])


# ============================================================================
# Standalone AI Question Generation & Answer Evaluation Endpoints
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
    question_service: QuestionGenerationService = Depends(get_question_generation_service),
) -> InterviewQuestionResponse:
    """Generate a structured interview question with evaluation criteria and follow-up."""
    with handle_interview_ai_errors(
        operation="interview question generation",
        upstream_error_detail="Failed to generate interview question from the local AI model.",
    ):
        return await question_service.generate_question(payload)


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

    with handle_rag_question_errors(
        unexpected_detail="An unexpected internal error occurred while generating the document question.",
        detailed_service_messages=True,
    ):
        return await rag_service.generate_document_question(payload)


@router.post(
    "/source-questions",
    response_model=SourceQuestionsResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate Multiple Source-Grounded Interview Questions",
    description="Generates multiple structured interview questions grounded in chunks retrieved for a selected source ID.",
    responses={
        200: {"description": "Grounded interview questions generated successfully"},
        400: {"description": "Source not ready or insufficient content"},
        404: {"description": "Source not found"},
        422: {"description": "Missing source_id in payload"},
        502: {"description": "Bad gateway from AI model"},
        503: {"description": "Ollama or Pinecone unavailable"},
    },
)
async def generate_source_interview_questions(
    payload: SourceQuestionsRequest,
    rag_service: RAGQuestionService = Depends(get_rag_question_service),
    db: Optional[Session] = Depends(get_db_optional),
) -> SourceQuestionsResponse:
    """Generate structured interview questions grounded in source content."""
    if not payload.source_id or not payload.source_id.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="'source_id' is required in request payload.",
        )

    with handle_rag_question_errors(
        unexpected_detail="An unexpected internal error occurred while generating interview questions."
    ):
        return await rag_service.generate_source_questions(payload, db=db)


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
    evaluation_service: AnswerEvaluationService = Depends(get_answer_evaluation_service),
) -> InterviewEvaluationResponse:
    """Evaluate candidate answer and provide structured feedback and rubric scoring."""
    with handle_interview_ai_errors(
        operation="answer evaluation",
        upstream_error_detail="Failed to evaluate interview answer from the local AI model.",
    ):
        return await evaluation_service.evaluate_answer(payload)


@router.post(
    "/evaluate-rag",
    response_model=RAGEvaluationResponse,
    status_code=status.HTTP_200_OK,
    summary="Evaluate RAG Question Generation and Context Grounding",
    description=(
        "Evaluates a generated interview question against retrieved context chunks using "
        "the RAG Triad (Faithfulness/Groundedness, Context Relevance, and Question Quality)."
    ),
    responses={
        200: {"description": "RAG evaluation completed successfully"},
        422: {"description": "Validation error in request payload"},
        500: {"description": "Internal error during RAG evaluation"},
    },
)
async def evaluate_rag_output(
    payload: RAGEvaluationRequest,
    eval_service: RAGEvaluationService = Depends(get_rag_eval_service),
) -> RAGEvaluationResponse:
    """Assess RAG generation accuracy, groundedness, and retrieval quality."""
    try:
        return await eval_service.evaluate_rag_output(payload)
    except Exception as exc:
        logger.error("Error evaluating RAG output: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred during RAG evaluation.",
        ) from exc


# ============================================================================
# Interview Session Management Endpoints
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
    responses={200: {"description": "List of interview sessions retrieved successfully"}},
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
    with handle_session_errors():
        return session_service.get_session(session_id, db=db)


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
    question_service: QuestionGenerationService = Depends(get_question_generation_service),
    db: Optional[Session] = Depends(get_db_optional),
) -> SessionNextQuestionResponse:
    """Generate the next interview question for the session using Qwen 2.5."""
    with handle_session_errors():
        session = session_service.validate_can_generate_question(session_id, db=db)

    question_payload = InterviewQuestionRequest(
        role=session.role,
        experience_level=session.experience_level,
        difficulty=session.difficulty,
        topic=session.topic,
        question_type=session.question_type,
    )

    with handle_interview_ai_errors(
        operation="interview question generation",
        upstream_error_detail="Failed to generate interview question from the local AI model.",
    ):
        question_data = await question_service.generate_question(question_payload)

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
    evaluation_service: AnswerEvaluationService = Depends(get_answer_evaluation_service),
    db: Optional[Session] = Depends(get_db_optional),
) -> SessionSubmitAnswerResponse:
    """Submit candidate answer and evaluate against expected criteria."""
    with handle_session_errors():
        target_question = session_service.validate_can_submit_answer(
            session_id=session_id,
            question_number=payload.question_number,
            db=db,
        )

    eval_payload = InterviewEvaluationRequest(
        question=target_question.question,
        expected_answer_points=target_question.expected_answer_points,
        candidate_answer=payload.candidate_answer,
    )

    with handle_interview_ai_errors(
        operation="answer evaluation",
        upstream_error_detail="Failed to evaluate interview answer from the local AI model.",
    ):
        evaluation = await evaluation_service.evaluate_answer(eval_payload)

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
    with handle_session_errors():
        return session_service.complete_session(session_id, db=db)
