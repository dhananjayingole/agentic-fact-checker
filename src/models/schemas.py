"""
Pydantic schemas for request/response validation.
"""
from typing import Optional, List
from pydantic import BaseModel, Field, validator
from enum import Enum


class Verdict(str, Enum):
    TRUE = "TRUE"
    FALSE = "FALSE"
    INCONCLUSIVE = "INCONCLUSIVE"
    UNVERIFIABLE = "UNVERIFIABLE"


class Stance(str, Enum):
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    NEUTRAL = "neutral"


class SourceCredibility(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# ─── Request Models ──────────────────────────────────────────

class VerifyRequest(BaseModel):
    claim: str = Field(..., min_length=5, max_length=1000, description="The claim to verify")
    max_sources: int = Field(default=5, ge=1, le=10, description="Max sources to search")

    @validator("claim")
    def clean_claim(cls, v):
        return v.strip()

    class Config:
        json_schema_extra = {
            "example": {
                "claim": "The Great Wall of China is visible from space",
                "max_sources": 5
            }
        }


class BatchVerifyRequest(BaseModel):
    claims: List[str] = Field(..., min_items=1, max_items=10)
    max_sources: int = Field(default=3, ge=1, le=10)

    @validator("claims", each_item=True)
    def clean_claims(cls, v):
        return v.strip()


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=500)
    max_results: int = Field(default=5, ge=1, le=10)


class ExtractClaimsRequest(BaseModel):
    text: str = Field(..., min_length=10, max_length=5000)


# ─── Response Models ─────────────────────────────────────────

class EvidenceItem(BaseModel):
    title: str
    url: str
    snippet: str
    relevance_score: float = Field(ge=0.0, le=1.0)
    stance: Stance
    source_credibility: SourceCredibility
    source_type: str  # "web", "wikipedia", etc.


class VerifyResponse(BaseModel):
    claim: str
    verdict: Verdict
    confidence_score: float = Field(ge=0.0, le=10.0, description="0=definitely false, 10=definitely true")
    evidence_count: int
    evidence_list: List[EvidenceItem]
    evidence_summary: str
    reasoning: str
    processing_time_seconds: float
    sources_searched: int


class BatchVerifyResponse(BaseModel):
    results: List[VerifyResponse]
    total_claims: int
    processing_time_seconds: float


class SearchResult(BaseModel):
    title: str
    url: str
    snippet: str
    source_type: str


class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]
    total_results: int


class ExtractClaimsResponse(BaseModel):
    original_text: str
    claims: List[str]
    total_claims: int


class HealthResponse(BaseModel):
    status: str
    version: str
    services: dict