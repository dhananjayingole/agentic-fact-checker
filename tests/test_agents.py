"""
tests/test_agents.py - Unit tests (no server required).

Run: python -m pytest tests/ -v
"""
import asyncio
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.schemas import SearchResult, Stance, SourceCredibility, Verdict
from src.tools.evidence_analyzer import EvidenceAnalyzer
from src.services.groq_service import GroqService


# ── Evidence Analyzer Tests ───────────────────────────────────

class TestEvidenceAnalyzer:
    def setup_method(self):
        self.analyzer = EvidenceAnalyzer()

    def test_relevance_high_overlap(self):
        result = SearchResult(
            title="Humans use 100% of their brain",
            url="https://www.brainfacts.org/",
            snippet="Neuroscientists confirm humans use virtually all parts of their brain, "
                    "not just 10 percent as the popular myth claims.",
            source_type="web",
        )
        score = self.analyzer._compute_relevance(result, "Humans only use 10% of their brain")
        assert score > 0.3, f"Expected > 0.3, got {score}"

    def test_relevance_low_overlap(self):
        result = SearchResult(
            title="Best pizza recipes",
            url="https://recipes.com/pizza",
            snippet="Make delicious pizza at home with these easy recipes.",
            source_type="web",
        )
        score = self.analyzer._compute_relevance(result, "Humans only use 10% of their brain")
        assert score < 0.2, f"Expected < 0.2, got {score}"

    def test_stance_detection_contradicts(self):
        result = SearchResult(
            title="10% brain myth debunked",
            url="https://snopes.com/brain",
            snippet="This claim is false. Brain imaging shows all parts of the brain are active.",
            source_type="web",
        )
        stance = self.analyzer._detect_stance(result, "Humans only use 10% of their brain")
        assert stance == Stance.CONTRADICTS

    def test_stance_detection_supports(self):
        result = SearchResult(
            title="Water confirmed as H2O",
            url="https://science.org/water",
            snippet="Scientists have confirmed that water is indeed composed of hydrogen and oxygen atoms.",
            source_type="web",
        )
        stance = self.analyzer._detect_stance(result, "Water is made of hydrogen and oxygen")
        assert stance == Stance.SUPPORTS

    def test_credibility_wikipedia(self):
        cred = self.analyzer._classify_credibility("https://en.wikipedia.org/wiki/Brain")
        assert cred == SourceCredibility.HIGH

    def test_credibility_gov(self):
        cred = self.analyzer._classify_credibility("https://www.cdc.gov/health")
        assert cred == SourceCredibility.HIGH

    def test_credibility_unknown(self):
        cred = self.analyzer._classify_credibility("https://somerandomblog.com/post")
        assert cred == SourceCredibility.MEDIUM

    def test_analyze_returns_sorted_by_relevance(self):
        results = [
            SearchResult(title="Unrelated cooking post", url="https://food.com", snippet="pasta recipe", source_type="web"),
            SearchResult(
                title="Brain usage facts",
                url="https://brainfacts.org",
                snippet="Humans use their entire brain, not just 10 percent.",
                source_type="web",
            ),
        ]
        items = self.analyzer.analyze(results, "Humans only use 10% of their brain")
        if len(items) > 1:
            assert items[0].relevance_score >= items[1].relevance_score


# ── Groq Service Tests (heuristic mode, no API key needed) ────

class TestGroqServiceHeuristic:
    def setup_method(self):
        # Force heuristic mode (no real API key)
        self.service = GroqService()
        self.service.client = None  # Ensure heuristic path

    def test_heuristic_no_evidence(self):
        result = self.service._heuristic_verdict("Some claim", [])
        assert result["verdict"] == "INCONCLUSIVE"
        assert result["confidence_score"] == 5.0

    def test_heuristic_negation_evidence(self):
        evidence = ["This claim is false and has been debunked by scientists."]
        result = self.service._heuristic_verdict("Humans only use 10% of their brain", evidence)
        assert result["verdict"] in ("FALSE", "INCONCLUSIVE")

    def test_heuristic_confirmation_evidence(self):
        evidence = ["Scientists have confirmed this is accurate and correct."]
        result = self.service._heuristic_verdict("Water boils at 100 degrees Celsius", evidence)
        assert result["verdict"] in ("TRUE", "INCONCLUSIVE")

    def test_extract_claims_fallback(self):
        text = "The Earth is 4.5 billion years old. Water covers 71% of the surface."
        claims = self.service._simple_sentence_split(text)
        assert isinstance(claims, list)
        assert len(claims) >= 1

    @pytest.mark.asyncio
    async def test_analyze_evidence_async(self):
        result = await self.service.analyze_evidence(
            "Water boils at 100 degrees Celsius at sea level",
            ["Water boiling point is confirmed to be 100 degrees Celsius at standard sea level pressure."],
        )
        assert "verdict" in result
        assert "confidence_score" in result
        assert result["verdict"] in ("TRUE", "FALSE", "INCONCLUSIVE", "UNVERIFIABLE")

    @pytest.mark.asyncio
    async def test_extract_claims_async(self):
        claims = await self.service.extract_claims(
            "The Great Wall of China is 21,196 kilometres long. It was built over many centuries."
        )
        assert isinstance(claims, list)


# ── Schema Validation Tests ───────────────────────────────────

class TestSchemas:
    def test_verify_request_valid(self):
        from src.models.schemas import VerifyRequest
        req = VerifyRequest(claim="The Earth is round", max_sources=5)
        assert req.claim == "The Earth is round"
        assert req.max_sources == 5

    def test_verify_request_strips_whitespace(self):
        from src.models.schemas import VerifyRequest
        req = VerifyRequest(claim="  The Earth is round  ", max_sources=5)
        assert req.claim == "The Earth is round"

    def test_verdict_enum_values(self):
        assert Verdict.TRUE == "TRUE"
        assert Verdict.FALSE == "FALSE"
        assert Verdict.INCONCLUSIVE == "INCONCLUSIVE"
        assert Verdict.UNVERIFIABLE == "UNVERIFIABLE"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])