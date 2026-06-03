"""
Evidence Analyzer - Scores relevance, detects stance, classifies source credibility.
Pure Python heuristic analysis (no external API needed).
"""
import re
from urllib.parse import urlparse
from loguru import logger

from src.models.schemas import SearchResult, EvidenceItem, Stance, SourceCredibility


# Credibility tiers based on domain reputation
HIGH_CREDIBILITY_DOMAINS = {
    "wikipedia.org", "britannica.com", "reuters.com", "apnews.com",
    "bbc.com", "bbc.co.uk", "nytimes.com", "theguardian.com",
    "washingtonpost.com", "nature.com", "science.org", "pubmed.ncbi.nlm.nih.gov",
    "who.int", "cdc.gov", "nasa.gov", "un.org", "worldbank.org",
    "snopes.com", "factcheck.org", "politifact.com", "fullfact.org",
    "gov.uk", ".gov", "edu", "scholar.google.com",
}

LOW_CREDIBILITY_DOMAINS = {
    "infowars.com", "naturalnews.com", "beforeitsnews.com",
    "yournewswire.com", "worldnewsdailyreport.com",
}

# Stance keywords
SUPPORT_KEYWORDS = [
    "confirmed", "true", "correct", "accurate", "verified", "proven",
    "indeed", "yes", "right", "fact", "evidence shows", "research confirms",
    "studies show", "scientists confirm", "experts agree",
]
CONTRADICT_KEYWORDS = [
    "false", "incorrect", "wrong", "myth", "debunked", "misleading",
    "inaccurate", "untrue", "no evidence", "not true", "disproved",
    "conspiracy", "hoax", "fabricated", "fake", "lacks evidence",
]


class EvidenceAnalyzer:
    """Analyzes search results to produce scored evidence items."""

    def analyze(self, results: list[SearchResult], claim: str) -> list[EvidenceItem]:
        """Analyze all search results and return ranked evidence items."""
        evidence_items = []
        for result in results:
            relevance = self._compute_relevance(result, claim)
            if relevance < 0.05:
                continue  # Skip clearly irrelevant results

            stance = self._detect_stance(result, claim)
            credibility = self._classify_credibility(result.url)

            evidence_items.append(EvidenceItem(
                title=result.title,
                url=result.url,
                snippet=result.snippet[:400],
                relevance_score=round(relevance, 3),
                stance=stance,
                source_credibility=credibility,
                source_type=result.source_type,
            ))

        # Sort by relevance descending
        evidence_items.sort(key=lambda x: x.relevance_score, reverse=True)
        return evidence_items

    def _compute_relevance(self, result: SearchResult, claim: str) -> float:
        """
        Score 0-1 based on keyword overlap between claim and result.
        Uses TF-style weighting (numbers and named entities count more).
        """
        claim_lower = claim.lower()
        result_text = f"{result.title} {result.snippet}".lower()

        # Extract meaningful tokens (skip stopwords)
        stopwords = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "have", "has", "had", "do", "does", "did", "will", "would",
            "could", "should", "may", "might", "shall", "can", "that",
            "this", "these", "those", "it", "its", "in", "on", "at",
            "to", "for", "of", "and", "or", "but", "not",
        }

        def tokenize(text):
            tokens = re.findall(r'\b\w+\b', text.lower())
            return [t for t in tokens if t not in stopwords and len(t) > 2]

        claim_tokens = set(tokenize(claim_lower))
        result_tokens = set(tokenize(result_text))

        if not claim_tokens:
            return 0.0

        # Basic Jaccard overlap
        intersection = claim_tokens & result_tokens
        base_score = len(intersection) / len(claim_tokens)

        # Boost for numbers (dates, statistics)
        claim_numbers = set(re.findall(r'\b\d+\b', claim_lower))
        result_numbers = set(re.findall(r'\b\d+\b', result_text))
        number_overlap = claim_numbers & result_numbers
        number_boost = 0.2 * len(number_overlap) / max(len(claim_numbers), 1) if claim_numbers else 0

        # Boost for Wikipedia (generally reliable)
        wiki_boost = 0.1 if result.source_type == "wikipedia" else 0.0

        return min(1.0, base_score + number_boost + wiki_boost)

    def _detect_stance(self, result: SearchResult, claim: str) -> Stance:
        """Heuristic stance detection based on keyword presence."""
        text = f"{result.title} {result.snippet}".lower()
        claim_lower = claim.lower()

        # Count support vs contradict signals
        support_count = sum(1 for kw in SUPPORT_KEYWORDS if kw in text)
        contradict_count = sum(1 for kw in CONTRADICT_KEYWORDS if kw in text)

        # Check if keywords appear near claim-related text
        claim_words = set(re.findall(r'\b\w{4,}\b', claim_lower))
        context_windows = [text[max(0, text.find(kw) - 50):text.find(kw) + 100]
                          for kw in CONTRADICT_KEYWORDS if kw in text]
        context_claim_match = sum(
            1 for w in context_windows
            for cw in claim_words if cw in w
        )

        if contradict_count > support_count and (contradict_count >= 2 or context_claim_match > 0):
            return Stance.CONTRADICTS
        elif support_count > contradict_count and support_count >= 1:
            return Stance.SUPPORTS
        return Stance.NEUTRAL

    def _classify_credibility(self, url: str) -> SourceCredibility:
        """Classify URL credibility based on known domain tiers."""
        try:
            domain = urlparse(url).netloc.lower()
            # Remove www prefix
            domain = re.sub(r'^www\.', '', domain)

            if any(hd in domain for hd in HIGH_CREDIBILITY_DOMAINS):
                return SourceCredibility.HIGH
            if any(ld in domain for ld in LOW_CREDIBILITY_DOMAINS):
                return SourceCredibility.LOW
            # .gov, .edu, .org get medium-high
            if domain.endswith((".gov", ".edu")):
                return SourceCredibility.HIGH
            if domain.endswith(".org"):
                return SourceCredibility.MEDIUM
        except Exception:
            pass
        return SourceCredibility.MEDIUM