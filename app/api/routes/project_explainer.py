"""API routes for GitHub Project Explainer analysis, progress tracking, RAG Q&A, suggestions, and interview prep."""

from typing import List, Literal, Optional
from fastapi import APIRouter, BackgroundTasks, Depends, Query, Response, status
from sqlalchemy.orm import Session

from app.api.dependencies.auth import require_active_user
from app.core.exceptions import handle_project_explainer_errors
from app.db.session import get_db_optional
from app.models.auth import User
from app.schemas.project_explainer import (
    ProjectAnalysisDetailResponse,
    ProjectAnalysisStartResponse,
    ProjectAnalysisStatus,
    ProjectAnalysisStatusResponse,
    ProjectAnalyzeRequest,
    ProjectExplainQueryRequest,
    ProjectExplainQueryResponse,
    ProjectInterviewQuestionsResponse,
    ProjectSuggestionsResponse,
    parse_github_repo_url,
)
from app.services.project_explainer_service import (
    ProjectExplainerService,
    get_project_explainer_service,
)

router = APIRouter(prefix="/project-explainer", tags=["Project Explainer"])


@router.post(
    "/analyze",
    response_model=ProjectAnalysisStartResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Start GitHub Project Analysis",
    description=(
        "Validates the GitHub repository URL and accessibility via GitHub MCP, "
        "ingests important source and configuration files into Pinecone RAG, and generates "
        "a comprehensive 9-section project explanation using Qwen."
    ),
    responses={
        200: {"description": "Existing completed analysis reused or synchronous analysis completed"},
        202: {"description": "Project analysis queued for asynchronous background processing"},
        400: {"description": "Invalid repository URL or empty repository"},
        403: {"description": "Repository access denied or missing permissions"},
        404: {"description": "Repository not found"},
        429: {"description": "GitHub API rate limit exceeded"},
        502: {"description": "AI model returned malformed analysis JSON"},
        503: {"description": "GitHub MCP, Ollama, or database unavailable"},
        504: {"description": "Upstream service timed out"},
    },
)
async def start_project_analysis(
    payload: ProjectAnalyzeRequest,
    background_tasks: BackgroundTasks,
    response: Response,
    service: ProjectExplainerService = Depends(get_project_explainer_service),
    db: Optional[Session] = Depends(get_db_optional),
    current_user: User = Depends(require_active_user),
) -> ProjectAnalysisStartResponse:
    """Validate a GitHub repository and initiate (or reuse) Project Explainer analysis."""
    with handle_project_explainer_errors():
        start_resp, metadata = await service.prepare_analysis(
            payload,
            db=db,
            user_id=current_user.id,
        )
        if start_resp.reused_existing:
            response.status_code = status.HTTP_200_OK
            return start_resp

        owner, repo, canonical_url = parse_github_repo_url(payload.repo_url)
        if payload.async_mode:
            background_tasks.add_task(
                service.execute_analysis_background,
                analysis_id=start_resp.analysis_id,
                owner=owner,
                repo=repo,
                repo_url=canonical_url,
                ref=payload.ref,
                source_id=start_resp.source_id,
                user_id=current_user.id,
                metadata=metadata,
                db_bind_source=db,
            )
            return start_resp

        await service.execute_analysis(
            analysis_id=start_resp.analysis_id,
            owner=owner,
            repo=repo,
            repo_url=canonical_url,
            ref=payload.ref,
            source_id=start_resp.source_id,
            user_id=current_user.id,
            metadata=metadata,
            db=db,
        )
        response.status_code = status.HTTP_200_OK
        return ProjectAnalysisStartResponse(
            analysis_id=start_resp.analysis_id,
            repository=start_resp.repository,
            repo_url=canonical_url,
            source_id=start_resp.source_id,
            status=ProjectAnalysisStatus.COMPLETED,
            progress_percent=100,
            current_step="Analysis completed",
            reused_existing=False,
            message="Project analysis completed successfully.",
        )


@router.get(
    "",
    response_model=List[ProjectAnalysisStatusResponse],
    status_code=status.HTTP_200_OK,
    summary="List User Project Analyses",
    description="Retrieve recent Project Explainer analyses owned by the authenticated user.",
)
async def list_project_analyses(
    limit: int = Query(20, ge=1, le=100),
    skip: int = Query(0, ge=0),
    service: ProjectExplainerService = Depends(get_project_explainer_service),
    db: Optional[Session] = Depends(get_db_optional),
    current_user: User = Depends(require_active_user),
) -> List[ProjectAnalysisStatusResponse]:
    """List recent project analyses for the authenticated user."""
    with handle_project_explainer_errors():
        return service.list_analyses(db=db, user_id=current_user.id, limit=limit, skip=skip)


@router.get(
    "/{analysis_id}/status",
    response_model=ProjectAnalysisStatusResponse,
    status_code=status.HTTP_200_OK,
    summary="Get Project Analysis Status",
    description="Poll lifecycle status, percentage progress, ingestion metrics, and errors for an analysis ID.",
)
async def get_project_analysis_status(
    analysis_id: str,
    service: ProjectExplainerService = Depends(get_project_explainer_service),
    db: Optional[Session] = Depends(get_db_optional),
    current_user: User = Depends(require_active_user),
) -> ProjectAnalysisStatusResponse:
    """Retrieve status and progress for a Project Explainer analysis job."""
    with handle_project_explainer_errors():
        return service.get_analysis_status(analysis_id, db=db, user_id=current_user.id)


@router.get(
    "/{analysis_id}",
    response_model=ProjectAnalysisDetailResponse,
    status_code=status.HTTP_200_OK,
    summary="Get Completed Project Analysis",
    description="Retrieve full repository metadata, ingestion statistics, and the 9-section project explanation.",
)
async def get_project_analysis_detail(
    analysis_id: str,
    service: ProjectExplainerService = Depends(get_project_explainer_service),
    db: Optional[Session] = Depends(get_db_optional),
    current_user: User = Depends(require_active_user),
) -> ProjectAnalysisDetailResponse:
    """Retrieve detailed project analysis results."""
    with handle_project_explainer_errors():
        return service.get_analysis_detail(analysis_id, db=db, user_id=current_user.id)


@router.post(
    "/{analysis_id}/explain",
    response_model=ProjectExplainQueryResponse,
    status_code=status.HTTP_200_OK,
    summary="Ask Repository-Specific Explanation Question",
    description=(
        "Retrieves code chunks strictly isolated to the analyzed repository from Pinecone "
        "and generates an evidence-cited explanation with Qwen."
    ),
)
async def explain_project_question(
    analysis_id: str,
    payload: ProjectExplainQueryRequest,
    service: ProjectExplainerService = Depends(get_project_explainer_service),
    db: Optional[Session] = Depends(get_db_optional),
    current_user: User = Depends(require_active_user),
) -> ProjectExplainQueryResponse:
    """Answer a repository-specific explanation query using isolated Pinecone RAG."""
    with handle_project_explainer_errors():
        return await service.explain_repository_question(
            analysis_id,
            payload,
            db=db,
            user_id=current_user.id,
        )


@router.get(
    "/{analysis_id}/suggestions",
    response_model=ProjectSuggestionsResponse,
    status_code=status.HTTP_200_OK,
    summary="Get Improvement & Feature Suggestions",
    description="Retrieve categorized improvement suggestions and new feature ideas for a completed analysis.",
)
async def get_project_suggestions(
    analysis_id: str,
    service: ProjectExplainerService = Depends(get_project_explainer_service),
    db: Optional[Session] = Depends(get_db_optional),
    current_user: User = Depends(require_active_user),
) -> ProjectSuggestionsResponse:
    """Retrieve improvement suggestions and new feature ideas."""
    with handle_project_explainer_errors():
        return service.get_suggestions(analysis_id, db=db, user_id=current_user.id)


@router.get(
    "/{analysis_id}/interview-questions",
    response_model=ProjectInterviewQuestionsResponse,
    status_code=status.HTTP_200_OK,
    summary="Get Repository-Grounded Interview Questions",
    description="Retrieve beginner, intermediate, advanced, and architecture interview questions for the repository.",
)
async def get_project_interview_questions(
    analysis_id: str,
    level: Optional[Literal["beginner", "intermediate", "advanced", "architecture_and_scenario"]] = Query(
        None,
        description="Optional difficulty/category filter.",
    ),
    service: ProjectExplainerService = Depends(get_project_explainer_service),
    db: Optional[Session] = Depends(get_db_optional),
    current_user: User = Depends(require_active_user),
) -> ProjectInterviewQuestionsResponse:
    """Retrieve repository-grounded interview preparation questions."""
    with handle_project_explainer_errors():
        return service.get_interview_questions(
            analysis_id,
            level=level,
            db=db,
            user_id=current_user.id,
        )

