"""
Verify endpoints - Core claim verification API.
"""
import time
from fastapi import APIRouter, HTTPException, Depends, Request
from loguru import logger

from src.models.schemas import VerifyRequest, VerifyResponse, BatchVerifyRequest, BatchVerifyResponse

router = APIRouter(prefix="/verify", tags=["Verification"])


def get_agent(request: Request):
    """Dependency to get the shared agent instance from app state."""
    return request.app.state.agent


@router.post(
    "/",
    response_model=VerifyResponse,
    summary="Verify a single claim",
    description="""
    Verify a factual claim using multi-source web search and LLM analysis.
    
    **How it works:**
    1. Searches DuckDuckGo + Wikipedia for relevant evidence
    2. Analyzes each source for relevance and stance
    3. Uses Groq LLM to produce a verdict with reasoning
    4. Returns verdict, confidence score, and full evidence list
    
    **Verdict meanings:**
    - `TRUE`: Evidence supports the claim (confidence ≥ 7.0)
    - `FALSE`: Evidence contradicts the claim (confidence ≤ 3.0)
    - `INCONCLUSIVE`: Mixed or ambiguous evidence
    - `UNVERIFIABLE`: No relevant evidence found
    """
)
async def verify_claim(
    body: VerifyRequest,
    agent=Depends(get_agent)
) -> VerifyResponse:
    try:
        result = await agent.verify(body.claim, max_sources=body.max_sources)
        return result
    except Exception as e:
        logger.error(f"Verify error: {e}")
        raise HTTPException(status_code=500, detail=f"Verification failed: {str(e)}")


@router.post(
    "/batch",
    response_model=BatchVerifyResponse,
    summary="Verify multiple claims at once",
    description="Verify up to 10 claims concurrently. Uses reduced max_sources per claim to keep latency manageable."
)
async def verify_batch(
    body: BatchVerifyRequest,
    agent=Depends(get_agent)
) -> BatchVerifyResponse:
    if len(body.claims) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 claims per batch request")
    try:
        start = time.time()
        results = await agent.verify_batch(body.claims, max_sources=body.max_sources)
        total_time = round(time.time() - start, 2)
        return BatchVerifyResponse(
            results=results,
            total_claims=len(results),
            processing_time_seconds=total_time,
        )
    except Exception as e:
        logger.error(f"Batch verify error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch verification failed: {str(e)}")