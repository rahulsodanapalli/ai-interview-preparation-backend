"""API v1 router aggregation."""

from fastapi import APIRouter

from app.api.routes import documents, github, health, interview, sources

api_v1_router = APIRouter()
api_v1_router.include_router(health.router)
api_v1_router.include_router(interview.router)
api_v1_router.include_router(documents.router)
api_v1_router.include_router(github.router)
api_v1_router.include_router(sources.router)

