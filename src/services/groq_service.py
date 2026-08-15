"""
Groq LLM Service - Handles all LLM interactions using Groq's free API.
Model: llama-3.3-70b-versatile
"""
import json
import os
import re
from typing import Optional
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

from src.models.schemas import Verdict


class GroqService:
    """Service for interacting with Groq's LLM API."""

    MODEL = "llama-3.3-70b-versatile"  # Fast, free model
    FALLBACK_MODEL = "llama-3.1-8b-instant"

    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY", "")
        self.client = None
        self._initialize()

    def _initialize(self):
        if not GROQ_AVAILABLE:
            logger.warning("groq package not installed. Using mock responses.")
            return
        if not self.api_key or self.api_key == "your_groq_api_key_here":
            logger.warning("GROQ_API_KEY not set. Using heuristic-only mode.")
            return
        try:
            self.client = Groq(api_key=self.api_key)
            logger.info(f"Groq client initialized with model: {self.MODEL}")
        except Exception as e:
            logger.error(f"Failed to initialize Groq client: {e}")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def analyze_evidence(
        self,
        claim: str,
        evidence_texts: list[str]
    ) -> dict:
        """
        Use LLM to analyze evidence and produce a verdict.
        Returns: {"verdict": "TRUE/FALSE/INCONCLUSIVE", "confidence": 0-10, "reasoning": "..."}
        """
        if not self.client:
            return self._heuristic_verdict(claim, evidence_texts)

        evidence_block = "\n\n".join([
            f"[Source {i+1}]: {text[:800]}"
            for i, text in enumerate(evidence_texts[:5])
        ])

        prompt = f"""You are an expert fact-checker. Analyze the following claim and evidence carefully.

CLAIM: "{claim}"

EVIDENCE:
{evidence_block}

Based ONLY on the evidence provided, determine:
1. Is the claim TRUE, FALSE, or INCONCLUSIVE?
2. Confidence score (0.0 to 10.0): 0=definitely false, 10=definitely true, 5=inconclusive
3. Brief reasoning (2-3 sentences)

Respond with ONLY valid JSON in this exact format:
{{
  "verdict": "TRUE" | "FALSE" | "INCONCLUSIVE",
  "confidence_score": <float 0.0-10.0>,
  "reasoning": "<2-3 sentence explanation>",
  "summary": "<one sentence summary for end users>"
}}"""

        try:
            response = self.client.chat.completions.create(
                model=self.MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=400,
            )
            raw = response.choices[0].message.content.strip()
            # Extract JSON even if there's extra text
            json_match = re.search(r'\{.*\}', raw, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                # Validate verdict
                if result.get("verdict") not in ["TRUE", "FALSE", "INCONCLUSIVE"]:
                    result["verdict"] = "INCONCLUSIVE"
                result["confidence_score"] = float(
                    max(0.0, min(10.0, result.get("confidence_score", 5.0)))
                )
                result["method"] = "llm"
                return result
        except Exception as e:
            logger.error(f"Groq API error: {e}")

        return self._heuristic_verdict(claim, evidence_texts)

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=2, max=8))
    async def extract_claims(self, text: str) -> list[str]:
        """Extract factual claims from a block of text."""
        if not self.client:
            return self._simple_sentence_split(text)

        prompt = f"""Extract all verifiable factual claims from the following text.
Return ONLY a JSON array of strings, each being a single factual claim.
Focus on claims that can be fact-checked (statistics, dates, names, events).
Exclude opinions and predictions.

TEXT: "{text[:2000]}"

Respond with ONLY valid JSON array:
["claim 1", "claim 2", ...]"""

        try:
            response = self.client.chat.completions.create(
                model=self.MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=500,
            )
            raw = response.choices[0].message.content.strip()
            json_match = re.search(r'\[.*\]', raw, re.DOTALL)
            if json_match:
                claims = json.loads(json_match.group())
                return [c for c in claims if isinstance(c, str) and len(c) > 10]
        except Exception as e:
            logger.error(f"Claim extraction error: {e}")

        return self._simple_sentence_split(text)

    def _heuristic_verdict(self, claim: str, evidence_texts: list[str]) -> dict:
        """
        Fallback: keyword-based heuristic when the LLM is unavailable (no API key,
        or a transient Groq outage/rate-limit even after retries).

        IMPORTANT: This is pure string matching with no semantic understanding —
        it can be fooled by a negation word appearing anywhere in unrelated
        context (e.g. "at higher altitudes it is lower" contains no trigger word,
        but a source discussing an unrelated myth being "debunked" could trip a
        false positive on a totally different claim that happens to share nouns).

        Because of that, this fallback NEVER asserts a confident TRUE or FALSE
        verdict. It only ever returns INCONCLUSIVE or UNVERIFIABLE, and always
        flags itself as "method": "heuristic" so callers/UI can surface that this
        verdict was not LLM-verified and should be treated with lower trust.
        """
        result_method = {"method": "heuristic"}

        if not evidence_texts:
            return {
                "verdict": "INCONCLUSIVE",
                "confidence_score": 5.0,
                "reasoning": "No evidence found to verify this claim.",
                "summary": "Could not find sufficient evidence to verify or refute this claim.",
                **result_method,
            }

        claim_lower = claim.lower()
        combined_evidence = " ".join(evidence_texts).lower()

        # Extract key nouns/numbers from claim
        claim_words = set(re.findall(r'\b[a-z]{4,}\b|\b\d+\b', claim_lower))
        evidence_words = set(re.findall(r'\b[a-z]{4,}\b|\b\d+\b', combined_evidence))

        overlap = claim_words & evidence_words
        overlap_ratio = len(overlap) / max(len(claim_words), 1)

        negation_in_evidence = any(
            neg in combined_evidence
            for neg in ["false", "incorrect", "wrong", "myth", "debunked", "not true", "misleading"]
        )
        confirmation_in_evidence = any(
            pos in combined_evidence
            for pos in ["confirmed", "correct", "true", "accurate", "verified", "proven"]
        )

        # NOTE: previously these two branches asserted FALSE / TRUE directly.
        # That is not safe for a keyword-only heuristic (no semantic understanding
        # of negation scope, sarcasm, or unrelated context). Downgraded to
        # INCONCLUSIVE with a lean noted in the reasoning, so the UI/agent layer
        # can decide how to treat it rather than presenting a false-confidence verdict.
        if negation_in_evidence and overlap_ratio > 0.3:
            return {
                "verdict": "INCONCLUSIVE",
                "confidence_score": 4.0,
                "reasoning": (
                    "Heuristic fallback (LLM unavailable): evidence contains language that "
                    "may suggest the claim is incorrect, but this was not confirmed by an LLM "
                    "and should not be treated as a confident verdict."
                ),
                "summary": "Evidence leans against this claim, but verification requires LLM review.",
                **result_method,
            }
        elif confirmation_in_evidence and overlap_ratio > 0.3:
            return {
                "verdict": "INCONCLUSIVE",
                "confidence_score": 6.0,
                "reasoning": (
                    "Heuristic fallback (LLM unavailable): evidence contains language that "
                    "may support the claim, but this was not confirmed by an LLM and should "
                    "not be treated as a confident verdict."
                ),
                "summary": "Evidence leans toward this claim, but verification requires LLM review.",
                **result_method,
            }
        elif overlap_ratio > 0.4:
            return {
                "verdict": "INCONCLUSIVE",
                "confidence_score": 5.0,
                "reasoning": "Evidence found but verdict unclear without LLM analysis.",
                "summary": "Related evidence found but a definitive verdict requires manual review.",
                **result_method,
            }
        else:
            return {
                "verdict": "UNVERIFIABLE",
                "confidence_score": 5.0,
                "reasoning": "Insufficient relevant evidence found to verify or refute this claim.",
                "summary": "Not enough evidence found to make a determination.",
                **result_method,
            }

    def _simple_sentence_split(self, text: str) -> list[str]:
        """Naive sentence splitter as fallback claim extractor."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if len(s.strip()) > 20][:10]

    def is_available(self) -> bool:
        return self.client is not None