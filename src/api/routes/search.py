"""
Search endpoints - Direct web search without fact-checking.
"""
from fastapi import APIRouter, HTTPException, Depends, Request
from loguru import logger

from src.models.schemas import SearchRequest, SearchResponse

router = APIRouter(prefix="/search", tags=["Search"])


def get_agent(request: Request):
    return request.app.state.agent


@router.post(
    "/",
    response_model=SearchResponse,
    summary="Search the web",
    description="Search DuckDuckGo + Wikipedia and return raw results (no verdict)."
)
async def search(body: SearchRequest, agent=Depends(get_agent)) -> SearchResponse:
    try:
        results = await agent.search_tool.search_all(body.query, max_results=body.max_results)
        return SearchResponse(
            query=body.query,
            results=results,
            total_results=len(results),
        )
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")