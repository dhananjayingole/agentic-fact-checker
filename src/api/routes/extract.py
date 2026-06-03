"""
Extract claims endpoint - Pull factual claims from any text block.
"""
from fastapi import APIRouter, HTTPException, Depends, Request
from loguru import logger

from src.models.schemas import ExtractClaimsRequest, ExtractClaimsResponse

router = APIRouter(prefix="/extract", tags=["Extraction"])


def get_agent(request: Request):
    return request.app.state.agent


@router.post(
    "/claims",
    response_model=ExtractClaimsResponse,
    summary="Extract factual claims from text",
    description="""
    Given any block of text (article, social media post, etc.), 
    extract individual verifiable factual claims using the LLM.
    Each extracted claim can then be sent to /verify/ individually.
    """
)
async def extract_claims(body: ExtractClaimsRequest, agent=Depends(get_agent)) -> ExtractClaimsResponse:
    try:
        claims = await agent.groq_service.extract_claims(body.text)
        return ExtractClaimsResponse(
            original_text=body.text[:200] + "..." if len(body.text) > 200 else body.text,
            claims=claims,
            total_claims=len(claims),
        )
    except Exception as e:
        logger.error(f"Claim extraction error: {e}")
        raise HTTPException(status_code=500, detail=f"Extraction failed: {str(e)}")