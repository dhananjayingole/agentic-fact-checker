"""
Fact Checker Agent - Orchestrates the full verification pipeline.

Pipeline:
  1. Search (DuckDuckGo + Wikipedia)
  2. Evidence analysis (relevance + stance scoring)
  3. Content enrichment (optional page scraping)
  4. LLM judgment (Groq)
  5. Neo4j storage (optional)
"""
import asyncio
import time
from loguru import logger

from src.services.Neo4j_service import Neo4jService
from src.tools.search_tool import SearchTool
from src.tools.evidence_analyzer import EvidenceAnalyzer
from src.services.groq_service import GroqService
from src.models.schemas import (
    VerifyResponse, Verdict, EvidenceItem, Stance, SourceCredibility
)


class FactCheckerAgent:
    """
    Main agent that coordinates search → analysis → LLM judgment.
    Designed to be stateless: instantiate once, call verify() many times.
    """

    def __init__(self):
        self.search_tool = SearchTool()
        self.evidence_analyzer = EvidenceAnalyzer()
        self.groq_service = GroqService()
        self.neo4j_service = Neo4jService()
        logger.info("FactCheckerAgent initialized")

    async def verify(self, claim: str, max_sources: int = 5) -> VerifyResponse:
        """
        Full verification pipeline for a single claim.

        Args:
            claim: The factual claim to verify.
            max_sources: Maximum number of sources to search.

        Returns:
            VerifyResponse with verdict, confidence, evidence.
        """
        start_time = time.time()
        logger.info(f"Verifying claim: {claim[:80]}...")

        # ── Step 1: Search ──────────────────────────────────────
        search_results = await self.search_tool.search_all(claim, max_results=max_sources)
        logger.info(f"Found {len(search_results)} search results")

        if not search_results:
            return self._no_evidence_response(claim, start_time)

        # ── Step 2: Evidence Analysis ──────────────────────────
        evidence_items = self.evidence_analyzer.analyze(search_results, claim)
        logger.info(f"Produced {len(evidence_items)} evidence items")

        # ── Step 3: Content Enrichment (best-effort scraping) ──
        # Only scrape top 3 results to keep latency low
        top_results = search_results[:3]
        enriched_texts = await self.search_tool.get_enriched_content(top_results)

        # Combine enriched texts with snippets from remaining results
        all_evidence_texts = enriched_texts + [
            f"[{r.title}] {r.snippet}"
            for r in search_results[3:]
        ]

        # ── Step 4: LLM Judgment ───────────────────────────────
        llm_result = await self.groq_service.analyze_evidence(claim, all_evidence_texts)

        verdict_str = llm_result.get("verdict", "INCONCLUSIVE")
        try:
            verdict = Verdict(verdict_str)
        except ValueError:
            verdict = Verdict.INCONCLUSIVE

        confidence = float(llm_result.get("confidence_score", 5.0))
        reasoning = llm_result.get("reasoning", "Analysis complete.")
        summary = llm_result.get("summary", "Fact check complete.")

        # ── Step 5: Neo4j Storage (optional) ──────────────────
        asyncio.create_task(
            self._store_async(claim, verdict_str, confidence, evidence_items, summary)
        )

        processing_time = round(time.time() - start_time, 2)
        logger.info(
            f"Verdict: {verdict} | Confidence: {confidence} | "
            f"Time: {processing_time}s"
        )

        return VerifyResponse(
            claim=claim,
            verdict=verdict,
            confidence_score=confidence,
            evidence_count=len(evidence_items),
            evidence_list=evidence_items,
            evidence_summary=summary,
            reasoning=reasoning,
            processing_time_seconds=processing_time,
            sources_searched=len(search_results),
        )

    async def verify_batch(self, claims: list[str], max_sources: int = 3) -> list[VerifyResponse]:
        """Verify multiple claims concurrently (with a semaphore to avoid rate limits)."""
        semaphore = asyncio.Semaphore(3)  # Max 3 concurrent verifications

        async def _guarded_verify(claim):
            async with semaphore:
                try:
                    return await self.verify(claim, max_sources=max_sources)
                except Exception as e:
                    logger.error(f"Batch verify error for '{claim[:50]}': {e}")
                    return self._error_response(claim, str(e))

        tasks = [_guarded_verify(c) for c in claims]
        return await asyncio.gather(*tasks)

    async def _store_async(self, claim, verdict, confidence, evidence_items, summary):
        """Store to Neo4j in background (non-blocking)."""
        try:
            self.neo4j_service.store_verification(
                claim, verdict, confidence, evidence_items, summary
            )
        except Exception as e:
            logger.debug(f"Background Neo4j store failed: {e}")

    def _no_evidence_response(self, claim: str, start_time: float) -> VerifyResponse:
        return VerifyResponse(
            claim=claim,
            verdict=Verdict.UNVERIFIABLE,
            confidence_score=5.0,
            evidence_count=0,
            evidence_list=[],
            evidence_summary="No evidence found for this claim.",
            reasoning="Search returned no results. The claim cannot be verified.",
            processing_time_seconds=round(time.time() - start_time, 2),
            sources_searched=0,
        )

    def _error_response(self, claim: str, error: str) -> VerifyResponse:
        return VerifyResponse(
            claim=claim,
            verdict=Verdict.INCONCLUSIVE,
            confidence_score=5.0,
            evidence_count=0,
            evidence_list=[],
            evidence_summary="An error occurred during verification.",
            reasoning=f"Processing error: {error}",
            processing_time_seconds=0.0,
            sources_searched=0,
        )

    def status(self) -> dict:
        """Return health/status of all components."""
        return {
            "groq_llm": "connected" if self.groq_service.is_available() else "heuristic_mode",
            "neo4j": "connected" if self.neo4j_service.is_available() else "disabled",
            "duckduckgo": "enabled",
            "wikipedia": "enabled",
        }