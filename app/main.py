from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import health, interview
from app.core.config import settings

app = FastAPI(
    title=settings.APP_NAME,
    description=(
        "Backend API for the AI Interview Preparation Platform powered by FastAPI "
        "and local Qwen 2.5 on Ollama."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Enable CORS for frontend integration (React + TypeScript planned for upcoming days)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Open for local dev; can be tightened in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API v1 Routers
api_v1_prefix = settings.API_V1_PREFIX
app.include_router(health.router, prefix=api_v1_prefix)
app.include_router(interview.router, prefix=api_v1_prefix)


@app.get(
    "/",
    tags=["Root"],
    summary="API Root",
    description="Returns a welcome message and links to API documentation.",
)
async def root():
    """Root endpoint returning welcome message and status."""
    return {
        "message": "Welcome to the AI Interview Preparation Platform API",
        "status": "online",
        "docs": "/docs",
        "health": f"{api_v1_prefix}/health",
        "ollama_health": f"{api_v1_prefix}/health/ollama",
    }
