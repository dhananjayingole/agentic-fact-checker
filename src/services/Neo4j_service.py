"""
Neo4j Knowledge Graph Service (Optional)
Stores claim verification history and relationships.
Uses Neo4j AuraDB free tier (50MB, free forever).
"""
import os
import hashlib
from datetime import datetime
from typing import Optional
from loguru import logger

try:
    from neo4j import GraphDatabase
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False


class Neo4jService:
    """Manages knowledge graph storage of claims and evidence."""

    def __init__(self):
        self.driver = None
        self._connect()

    def _connect(self):
        uri = os.getenv("NEO4J_URI", "")
        username = os.getenv("NEO4J_USERNAME", "neo4j")
        password = os.getenv("NEO4J_PASSWORD", "")

        if not NEO4J_AVAILABLE:
            logger.info("neo4j package not installed — knowledge graph disabled")
            return
        if not uri or not password:
            logger.info("Neo4j credentials not set — knowledge graph disabled")
            return
        try:
            self.driver = GraphDatabase.driver(uri, auth=(username, password))
            self.driver.verify_connectivity()
            self._create_constraints()
            logger.info("Neo4j connected successfully")
        except Exception as e:
            logger.warning(f"Neo4j connection failed: {e}. Running without graph storage.")
            self.driver = None

    def _create_constraints(self):
        """Create uniqueness constraints if not present."""
        if not self.driver:
            return
        with self.driver.session() as session:
            try:
                session.run("CREATE CONSTRAINT claim_id IF NOT EXISTS FOR (c:Claim) REQUIRE c.id IS UNIQUE")
                session.run("CREATE CONSTRAINT source_url IF NOT EXISTS FOR (s:Source) REQUIRE s.url IS UNIQUE")
            except Exception as e:
                logger.debug(f"Constraint creation: {e}")

    def store_verification(self, claim: str, verdict: str, confidence: float,
                           evidence_list: list, summary: str) -> Optional[str]:
        """
        Store a verification result in the knowledge graph.
        Returns the claim node ID.
        """
        if not self.driver:
            return None
        claim_id = hashlib.sha256(claim.encode()).hexdigest()[:16]
        try:
            with self.driver.session() as session:
                # Create or update Claim node
                session.run("""
                    MERGE (c:Claim {id: $id})
                    SET c.text = $text,
                        c.verdict = $verdict,
                        c.confidence = $confidence,
                        c.summary = $summary,
                        c.verified_at = $verified_at
                """, {
                    "id": claim_id,
                    "text": claim,
                    "verdict": verdict,
                    "confidence": confidence,
                    "summary": summary,
                    "verified_at": datetime.utcnow().isoformat()
                })

                # Create Source nodes and relationships
                for ev in evidence_list[:5]:
                    session.run("""
                        MERGE (s:Source {url: $url})
                        SET s.title = $title,
                            s.credibility = $credibility,
                            s.source_type = $source_type
                        WITH s
                        MATCH (c:Claim {id: $claim_id})
                        MERGE (c)-[r:HAS_EVIDENCE]->(s)
                        SET r.relevance = $relevance,
                            r.stance = $stance
                    """, {
                        "url": ev.url,
                        "title": ev.title,
                        "credibility": ev.source_credibility,
                        "source_type": ev.source_type,
                        "claim_id": claim_id,
                        "relevance": ev.relevance_score,
                        "stance": ev.stance,
                    })

            logger.info(f"Stored claim {claim_id} in Neo4j")
            return claim_id
        except Exception as e:
            logger.error(f"Neo4j store error: {e}")
            return None

    def find_similar_claims(self, claim: str, limit: int = 3) -> list[dict]:
        """Find previously verified similar claims."""
        if not self.driver:
            return []
        claim_words = set(claim.lower().split())
        try:
            with self.driver.session() as session:
                result = session.run("""
                    MATCH (c:Claim)
                    RETURN c.text AS text, c.verdict AS verdict,
                           c.confidence AS confidence, c.verified_at AS verified_at
                    ORDER BY c.verified_at DESC
                    LIMIT 50
                """)
                records = result.data()

            # Simple word overlap similarity
            similar = []
            for rec in records:
                rec_words = set(rec["text"].lower().split())
                overlap = len(claim_words & rec_words) / max(len(claim_words), 1)
                if overlap > 0.4:
                    similar.append({**rec, "similarity": round(overlap, 2)})

            similar.sort(key=lambda x: x["similarity"], reverse=True)
            return similar[:limit]
        except Exception as e:
            logger.error(f"Neo4j query error: {e}")
            return []

    def is_available(self) -> bool:
        return self.driver is not None

    def close(self):
        if self.driver:
            self.driver.close()