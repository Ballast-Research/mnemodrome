"""
Conflict validation service for bidirectional conflict detection (Feature 025).

Validates claims against stored facts, implements severity scoring algorithm,
and detects different types of conflicts (exact contradiction, partial mismatch, etc.).

Constitutional Principle: Severity scoring must be algorithmic and explainable.
"""

from datetime import datetime, timedelta

from loguru import logger

from src.config import (
    CONFLICT_SEVERITY_MAJOR_MIN,
    CONFLICT_SEVERITY_MINOR_MAX,
)
from src.models.conflict_detection_models import (
    ConflictResult,
    ExtractedClaim,
    StoredFact,
)

# Critical entities for severity boost
CRITICAL_ENTITIES = {
    "Mnemodrome",
    "Neo4j",
    "Qdrant",
    "Redis",
    "server-01",
    "server-02",
    "hypervisor-01",
    "firewall-01",
    "FastMCP",
}


class ConflictValidationService:
    """
    Service for validating claims against stored facts.

    Implements:
    - Conflict detection (exact vs partial mismatch)
    - Severity scoring (0-100)
    - Conflict type classification
    - Recommendation generation

    Attributes:
        severity_minor_max: Threshold for "minor" conflicts (default: 40)
        severity_major_min: Threshold for "major" conflicts (default: 70)
        critical_entities: Set of critical entities for severity boost
    """

    def __init__(
        self,
        severity_minor_max: int = CONFLICT_SEVERITY_MINOR_MAX,
        severity_major_min: int = CONFLICT_SEVERITY_MAJOR_MIN,
        critical_entities: set[str] | None = None,
    ):
        """
        Initialize ConflictValidationService.

        Args:
            severity_minor_max: Threshold for minor conflicts (default: 40)
            severity_major_min: Threshold for major conflicts (default: 70)
            critical_entities: Set of critical entity names (default: CRITICAL_ENTITIES)

        Raises:
            ValueError: If thresholds are invalid
        """
        if not (0 <= severity_minor_max <= 100):
            raise ValueError(
                f"severity_minor_max must be 0-100, got {severity_minor_max}"
            )
        if not (0 <= severity_major_min <= 100):
            raise ValueError(
                f"severity_major_min must be 0-100, got {severity_major_min}"
            )
        if severity_major_min < severity_minor_max:
            raise ValueError(
                f"severity_major_min ({severity_major_min}) must be >= "
                f"severity_minor_max ({severity_minor_max})"
            )

        self.severity_minor_max = severity_minor_max
        self.severity_major_min = severity_major_min
        self.critical_entities = critical_entities or CRITICAL_ENTITIES

        logger.info(
            f"ConflictValidationService initialized: "
            f"minor_max={severity_minor_max}, major_min={severity_major_min}"
        )

    async def validate_claim(
        self,
        claim: ExtractedClaim,
        stored_facts: list[StoredFact],
    ) -> ConflictResult | None:
        """
        Validate a claim against stored facts.

        Returns conflict result if conflict detected, None if no conflict.

        Args:
            claim: ExtractedClaim to validate
            stored_facts: List of StoredFact objects to check against

        Returns:
            ConflictResult if conflict detected, None otherwise

        Raises:
            Exception: If validation fails
        """
        if not stored_facts:
            logger.debug(f"No stored facts to validate against claim: {claim}")
            return None

        # Skip validation for non-assertions
        if claim.claim_type != "assertion":
            logger.debug(
                f"Skipping validation for {claim.claim_type}: "
                f"{claim.entity1} {claim.relation} {claim.entity2}"
            )
            return None

        logger.debug(
            f"Validating claim: {claim.entity1} {claim.relation} {claim.entity2} "
            f"(confidence: {claim.confidence})"
        )

        # Check each stored fact for conflicts
        for fact in stored_facts:
            # Check if this fact relates to the same entities and relationship
            if self._entities_and_relation_match(claim, fact):
                # Check if there's actually a conflict in entity2
                if claim.entity2.lower() != fact.entity2_name.lower():
                    # Conflict detected!
                    severity = self._calculate_severity(claim, fact)
                    conflict_type = self._detect_conflict_type(claim, fact)
                    recommendation = self._generate_recommendation(severity)

                    conflict_result = ConflictResult(
                        conflict_detected=True,
                        conflict_type=conflict_type,
                        severity=severity,
                        recommendation=recommendation,
                        correction_suggestion=self._generate_correction(claim, fact),
                        stored_fact=fact,
                        claim=claim,
                    )

                    logger.info(
                        f"Conflict detected: {claim.entity1} {claim.relation} "
                        f"{claim.entity2} vs {fact.entity2_name} "
                        f"(severity: {severity}, type: {conflict_type})"
                    )

                    return conflict_result

        logger.debug(
            f"No conflict found for claim: {claim.entity1} {claim.relation} {claim.entity2}"
        )
        return None

    def _entities_and_relation_match(
        self, claim: ExtractedClaim, fact: StoredFact
    ) -> bool:
        """
        Check if claim and fact relate to same entities and relationship.

        Handles case-insensitive matching and whitespace normalization.

        Args:
            claim: ExtractedClaim to check
            fact: StoredFact to check against

        Returns:
            True if entities and relation match
        """
        entity1_matches = (
            claim.entity1.lower().strip() == fact.entity1_name.lower().strip()
        )
        relation_matches = claim.relation.upper() == fact.relation.upper()

        return entity1_matches and relation_matches

    def _calculate_severity(self, claim: ExtractedClaim, fact: StoredFact) -> int:
        """
        Calculate conflict severity score (0-100).

        Algorithm:
        - Base contradiction: 30 points
        - Exact entity2 mismatch: +30 points
        - High confidence claim (>0.9): +10 points
        - Recent fact (<7 days): -5 points
        - Temporal conflict indicator: +20 points
        - Critical entity (production): +10 points

        Args:
            claim: ExtractedClaim
            fact: StoredFact for comparison

        Returns:
            Severity score 0-100
        """
        score = 30  # Base contradiction

        # Exact entity2 mismatch
        if claim.entity2.lower() != fact.entity2_name.lower():
            score += 30

        # High confidence claim
        if claim.confidence > 0.9:
            score += 10

        # Recent fact (more trustworthy, less likely to be outdated)
        fact_age = datetime.now() - fact.created_at
        if fact_age < timedelta(days=7):
            score -= 5

        # Check for temporal conflict
        if self._is_temporal_conflict(claim, fact):
            score += 20

        # Critical entity severity boost
        if self._is_critical_entity(claim.entity1) or self._is_critical_entity(
            fact.entity1_name
        ):
            score += 10

        # Clamp to 0-100 range
        return max(0, min(100, score))

    def _detect_conflict_type(self, claim: ExtractedClaim, fact: StoredFact) -> str:
        """
        Detect and classify type of conflict.

        Args:
            claim: ExtractedClaim
            fact: StoredFact

        Returns:
            One of: exact_contradiction, partial_mismatch, temporal_conflict, stale_fact
        """
        # Exact contradiction: completely different entity2
        if claim.entity2.lower() != fact.entity2_name.lower():
            # Check if it's temporal
            if self._is_temporal_conflict(claim, fact):
                return "temporal_conflict"

            # Check if fact is old
            fact_age = datetime.now() - fact.created_at
            if fact_age > timedelta(days=30):
                return "stale_fact"

            # Exact contradiction
            return "exact_contradiction"

        return "partial_mismatch"

    def _is_temporal_conflict(self, claim: ExtractedClaim, fact: StoredFact) -> bool:
        """
        Detect if conflict is temporal (time-based change).

        Temporal conflicts are when:
        - Fact has temporal validity fields
        - Claim relates to timing or change
        - Recent fact but old claim date

        Args:
            claim: ExtractedClaim
            fact: StoredFact

        Returns:
            True if temporal conflict detected
        """
        # Check if fact is temporal
        if not fact.is_temporal:
            return False

        # Check if claim mentions temporal keywords
        temporal_keywords = [
            "will",
            "should",
            "would",
            "planned",
            "migration",
            "moved",
            "moved",
        ]
        claim_text = f"{claim.relation} {claim.entity2}".lower()
        if any(keyword in claim_text for keyword in temporal_keywords):
            return True

        return False

    def _is_critical_entity(self, entity_name: str) -> bool:
        """
        Check if entity is in critical entities set.

        Args:
            entity_name: Entity name to check

        Returns:
            True if entity is critical
        """
        return entity_name in self.critical_entities or any(
            entity_name.lower() == critical.lower()
            for critical in self.critical_entities
        )

    def _generate_recommendation(self, severity: int) -> str:
        """
        Generate recommendation based on severity score.

        Args:
            severity: Severity score (0-100)

        Returns:
            Recommendation action: auto_correct, warn, or block
        """
        if severity < self.severity_minor_max:
            return "skip"
        elif severity < self.severity_major_min:
            return "auto_correct"
        else:
            return "warn"

    def _generate_correction(
        self, claim: ExtractedClaim, fact: StoredFact
    ) -> str | None:
        """
        Generate suggested correction text.

        Replaces claim entity2 with fact entity2.

        Args:
            claim: ExtractedClaim
            fact: StoredFact

        Returns:
            Suggested corrected text or None
        """
        # Simple correction: replace claim entity2 with fact entity2
        original_text = f"{claim.entity1} {claim.relation} {claim.entity2}"
        corrected_text = f"{claim.entity1} {claim.relation} {fact.entity2_name}"

        return corrected_text

    async def validate_claims_batch(
        self,
        claims: list[ExtractedClaim],
        stored_facts: list[StoredFact],
    ) -> list[ConflictResult]:
        """
        Validate multiple claims in batch.

        Args:
            claims: List of ExtractedClaim objects
            stored_facts: List of StoredFact objects to check against

        Returns:
            List of ConflictResult objects for claims with conflicts

        Raises:
            Exception: If validation fails
        """
        conflicts = []

        for claim in claims:
            try:
                conflict = await self.validate_claim(claim, stored_facts)
                if conflict:
                    conflicts.append(conflict)
            except Exception as e:
                logger.error(f"Error validating claim: {claim}, error: {e}")
                continue

        logger.info(
            f"Batch validation complete: {len(claims)} claims checked, "
            f"{len(conflicts)} conflicts found"
        )

        return conflicts
