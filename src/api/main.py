"""
FastAPI Application - Main entry point for the Agentic Fact Checker API.
"""
import os
from contextlib import asynccontextmanager
from loguru import logger
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from src.agents.fact_checker_agent import FactCheckerAgent
from src.api.routes import verify, search, extract, health


# ── Rate Limiter ──────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address)
RATE_LIMIT = os.getenv("RATE_LIMIT_PER_MINUTE", "20")


# ── Lifespan: startup / shutdown ──────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Starting Agentic Fact Checker API...")
    app.state.agent = FactCheckerAgent()
    logger.info("✅ Agent ready")
    yield
    # Shutdown
    logger.info("Shutting down...")
    if hasattr(app.state.agent, "neo4j_service"):
        app.state.agent.neo4j_service.close()


# ── App Factory ───────────────────────────────────────────────
def create_app() -> FastAPI:
    app = FastAPI(
        title="Agentic Fact Checker API",
        description="""
## 🔍 AI-Powered Fact Verification System

Verify any factual claim automatically using:
- **Multi-source search**: DuckDuckGo + Wikipedia
- **Evidence analysis**: Relevance scoring + stance detection
- **LLM judgment**: Groq (llama3-8b) for fast, accurate verdicts
- **Knowledge graph**: Neo4j for claim history (optional)

### Quick Start
```bash
curl -X POST http://localhost:8000/verify/ \\
  -H "Content-Type: application/json" \\
  -d '{"claim": "Humans only use 10% of their brain"}'
```

### All resources are FREE — no paid API keys required!
        """,
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # ── Middleware ─────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Rate limiting ──────────────────────────────────────────
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    # ── Global exception handler ───────────────────────────────
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error(f"Unhandled exception: {exc}")
        return JSONResponse(
            status_code=500,
            content={"detail": "An internal error occurred. Please try again."},
        )

    # ── Routers ────────────────────────────────────────────────
    app.include_router(health.router)
    app.include_router(verify.router)
    app.include_router(search.router)
    app.include_router(extract.router)

    return app


app = create_app()