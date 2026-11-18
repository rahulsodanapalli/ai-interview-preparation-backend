"""API v1 router aggregation."""

from fastapi import APIRouter

from app.api.routes import auth, documents, github, health, interview, project_explainer, sources

api_v1_router = APIRouter()
api_v1_router.include_router(health.router)
api_v1_router.include_router(auth.router)
api_v1_router.include_router(interview.router)
api_v1_router.include_router(documents.router)
api_v1_router.include_router(github.router)
api_v1_router.include_router(project_explainer.router)
api_v1_router.include_router(sources.router)

