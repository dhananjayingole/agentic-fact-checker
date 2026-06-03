"""
Health check endpoint.
"""
from fastapi import APIRouter, Request
from src.models.schemas import HealthResponse

router = APIRouter(tags=["Health"])


@router.get("/", summary="API Info")
async def root():
    return {
        "name": "Agentic Fact Checker API",
        "version": "1.0.0",
        "description": "AI-powered fact verification using multi-source search + LLM analysis",
        "endpoints": {
            "POST /verify/": "Verify a single claim",
            "POST /verify/batch": "Verify multiple claims",
            "POST /search/": "Search web without verdict",
            "POST /extract/claims": "Extract claims from text",
            "GET /health": "Service health status",
            "GET /docs": "Swagger UI documentation",
        },
        "free_resources": ["Groq LLM", "DuckDuckGo", "Wikipedia", "Neo4j AuraDB"],
    }


@router.get("/health", response_model=HealthResponse, summary="Health Check")
async def health(request: Request):
    agent = getattr(request.app.state, "agent", None)
    services = agent.status() if agent else {"status": "initializing"}
    return HealthResponse(
        status="ok",
        version="1.0.0",
        services=services,
    )