"""
Claim extraction service for bidirectional conflict detection (Feature 025).

Extracts factual claims from text using LLM, detects claim types (assertion/hypothetical/example),
and provides confidence scoring. Supports entity extraction caching with 300s TTL.

Constitutional Principle: All claims must be classified and confidence-scored before validation.
"""

import json
from typing import TYPE_CHECKING, Optional

from loguru import logger

from src.config import (
    EXTRACTION_MODEL,
)
from src.models.conflict_detection_models import ExtractedClaim

if TYPE_CHECKING:
    from src.services.llm_service import LLMService
    from src.services.redis_service import RedisService


class ClaimExtractionService:
    """
    Service for extracting factual claims from text.

    Implements claim extraction with:
    - LLM-based claim discovery
    - Claim type classification (assertion/hypothetical/example)
    - Confidence scoring (0.0-1.0)
    - Entity extraction caching (300s TTL)

    Attributes:
        llm_service: LLMService for claim extraction
        redis_service: RedisService for entity extraction caching
        cache_ttl_seconds: TTL for cached entity extractions (default: 300)
    """

    CACHE_TTL_SECONDS = 300  # 5 minutes
    CACHE_KEY_PREFIX = "conflict:entity_extraction:"

    def __init__(
        self,
        llm_service: "LLMService",
        redis_service: Optional["RedisService"] = None,
        cache_ttl_seconds: int = CACHE_TTL_SECONDS,
    ):
        """
        Initialize ClaimExtractionService.

        Args:
            llm_service: LLMService instance for claim extraction via LLM
            redis_service: RedisService instance for caching (optional)
            cache_ttl_seconds: TTL for cached extractions (default: 300s)

        Raises:
            ValueError: If llm_service is None
        """
        if llm_service is None:
            raise ValueError("llm_service is required")

        self.llm_service = llm_service
        self.redis_service = redis_service
        self.cache_ttl_seconds = cache_ttl_seconds

        logger.info(
            f"ClaimExtractionService initialized with LLM provider: {llm_service.provider}, "
            f"cache TTL: {cache_ttl_seconds}s"
        )

    async def extract_claims(self, text: str) -> list[ExtractedClaim]:
        """
        Extract factual claims from text using LLM.

        Detects claim type (assertion/hypothetical/example) and confidence scores.

        Args:
            text: Input text to extract claims from

        Returns:
            List of ExtractedClaim objects with type, entities, relations, and confidence

        Raises:
            ValueError: If text is empty
            Exception: If LLM extraction fails
        """
        if not text or not text.strip():
            raise ValueError("Input text cannot be empty")

        logger.debug(f"Extracting claims from text: {text[:100]}...")

        prompt = f"""Extract ALL factual claims from this text.

For each claim:
1. Identify the claim type: "assertion" (definite fact), "hypothetical" (if/suppose/imagine), or "example" (for instance, consider)
2. Find source entity (entity1) and target entity (entity2) with the relationship between them
3. Rate your confidence (0.0-1.0) that this is the actual claim in the text
4. Include the original sentence and surrounding context

Text: {text}

Return a JSON array of claims with this exact structure:
[
  {{
    "claim_type": "assertion|hypothetical|example",
    "entity1": "Source Entity Name",
    "relation": "RELATIONSHIP_TYPE",
    "entity2": "Target Entity Name",
    "confidence": 0.85,
    "sentence": "Original sentence from text",
    "context": "Surrounding context paragraph"
  }},
  ...
]

Rules:
1. Extract ALL meaningful claims (entities and relationships mentioned)
2. claim_type must be one of: assertion, hypothetical, example
3. Preserve exact entity names from the text
4. Use clear, uppercase relationship types (RUNS_ON, USES, IMPLEMENTS, SUPPORTS, etc.)
5. confidence: How certain are you this claim exists in the text? (0.0-1.0)
6. Include original sentence containing the claim
7. Include surrounding context for disambiguation

Return ONLY the JSON array, no additional text."""

        try:
            # Use EXTRACTION_MODEL so claim extraction can be routed to a
            # dedicated fine-tuned model when EXTRACTION_MODEL is set in the
            # environment.  Falls back to llm_service.model (CHAT_MODEL) when
            # EXTRACTION_MODEL is unset — identical to the previous behaviour.
            response = await self.llm_service._call_llm(prompt, model=EXTRACTION_MODEL)

            # Parse JSON response (strip markdown code fences if present)
            cleaned_response = response.strip()
            if cleaned_response.startswith("```json"):
                cleaned_response = cleaned_response[7:]
            if cleaned_response.startswith("```"):
                cleaned_response = cleaned_response[3:]
            if cleaned_response.endswith("```"):
                cleaned_response = cleaned_response[:-3]
            cleaned_response = cleaned_response.strip()

            # ── FAIL LOUD: empty response ──────────────────────────────────
            if not cleaned_response:
                self.llm_service.extraction_degraded_count += 1
                logger.warning(
                    f"LLM returned empty/unparseable output for 'claim_extraction' "
                    f"using model '{EXTRACTION_MODEL}'. This commonly means a "
                    f"REASONING model is emitting hidden thinking tokens and exhausting "
                    f"the token budget. Fix: use a non-reasoning model, ensure "
                    f"LLM_DISABLE_THINKING=true, or raise LLM_MAX_TOKENS. "
                    f"Falling back to empty claims list (DEGRADED extraction quality)."
                )
                return []

            claims_data = json.loads(cleaned_response)

            # Validate and convert to ExtractedClaim objects
            if not isinstance(claims_data, list):
                logger.warning(
                    f"LLM returned non-array response, wrapping: {claims_data}"
                )
                claims_data = [claims_data] if claims_data else []

            extracted_claims = []
            for claim_dict in claims_data:
                try:
                    # Validate required fields
                    if not all(
                        k in claim_dict
                        for k in [
                            "claim_type",
                            "entity1",
                            "relation",
                            "entity2",
                            "confidence",
                            "sentence",
                        ]
                    ):
                        logger.warning(
                            f"Skipping claim with missing fields: {claim_dict}"
                        )
                        continue

                    # Ensure confidence is numeric and within bounds
                    confidence = float(claim_dict.get("confidence", 0.5))
                    if not 0.0 <= confidence <= 1.0:
                        logger.warning(
                            f"Invalid confidence {confidence} for claim, clamping to 0.0-1.0"
                        )
                        confidence = max(0.0, min(1.0, confidence))

                    # Validate claim_type
                    claim_type = claim_dict.get("claim_type", "assertion").lower()
                    if claim_type not in ["assertion", "hypothetical", "example"]:
                        logger.warning(
                            f"Invalid claim_type '{claim_type}', defaulting to 'assertion'"
                        )
                        claim_type = "assertion"

                    claim = ExtractedClaim(
                        claim_type=claim_type,
                        entity1=claim_dict["entity1"],
                        relation=claim_dict["relation"],
                        entity2=claim_dict["entity2"],
                        confidence=confidence,
                        context=claim_dict.get("context", ""),
                        sentence=claim_dict.get("sentence", ""),
                    )

                    extracted_claims.append(claim)

                except Exception as e:
                    logger.warning(f"Failed to parse claim: {claim_dict}, error: {e}")
                    continue

            logger.info(
                f"Extracted {len(extracted_claims)} claims from text "
                f"(assertion: {sum(1 for c in extracted_claims if c.claim_type == 'assertion')}, "
                f"hypothetical: {sum(1 for c in extracted_claims if c.claim_type == 'hypothetical')}, "
                f"example: {sum(1 for c in extracted_claims if c.claim_type == 'example')})"
            )

            return extracted_claims

        except json.JSONDecodeError as e:
            # ── FAIL LOUD: unparseable JSON ────────────────────────────────
            self.llm_service.extraction_degraded_count += 1
            logger.warning(
                f"LLM returned empty/unparseable output for 'claim_extraction' "
                f"using model '{EXTRACTION_MODEL}'. This commonly means a "
                f"REASONING model is emitting hidden thinking tokens and exhausting "
                f"the token budget. Fix: use a non-reasoning model, ensure "
                f"LLM_DISABLE_THINKING=true, or raise LLM_MAX_TOKENS. "
                f"Falling back to empty claims list (DEGRADED extraction quality). "
                f"Parse error: {e}"
            )
            logger.debug(f"Raw response: {response}")
            return []
        except Exception as e:
            logger.error(f"Claim extraction failed: {e}")
            raise

    async def extract_entities_from_claims(
        self, claims: list[ExtractedClaim]
    ) -> dict[str, str]:
        """
        Extract unique entities from a list of claims.

        Supports caching of entity extraction results (300s TTL).

        Args:
            claims: List of ExtractedClaim objects

        Returns:
            Dictionary mapping entity names to types

        Raises:
            Exception: If LLM classification fails
        """
        if not claims:
            return {}

        # Collect unique entity names
        entity_names = set()
        for claim in claims:
            entity_names.add(claim.entity1)
            entity_names.add(claim.entity2)

        logger.debug(f"Extracting types for {len(entity_names)} unique entities")

        # Build cache key (deterministic hash of sorted entity names)
        cache_key = f"{self.CACHE_KEY_PREFIX}{hash(frozenset(entity_names)) % (10**9)}"

        # Try to get from cache
        if self.redis_service and self.redis_service.is_connected:
            try:
                cached = await self.redis_service.get(cache_key)
                if cached:
                    logger.debug(f"Entity extraction cache HIT: {cache_key}")
                    return json.loads(cached)
            except Exception as e:
                logger.debug(f"Cache GET error (graceful degradation): {e}")

        # Build context from all claims for classification
        context = " ".join(claim.sentence for claim in claims)

        # Classify entities using LLM
        entity_types = await self.llm_service.classify_entities_batch(
            [{"name": name, "role": "entity"} for name in entity_names],
            context=context,
        )

        # Build result mapping
        result = {
            name: entity_type
            for name, entity_type in zip(
                sorted(entity_names), entity_types, strict=False
            )
        }

        # Cache result if Redis available
        if self.redis_service and self.redis_service.is_connected:
            try:
                await self.redis_service.set(
                    cache_key,
                    json.dumps(result),
                    ex=self.cache_ttl_seconds,
                )
                logger.debug(f"Entity extraction cached: {cache_key}")
            except Exception as e:
                logger.debug(f"Cache SET error (graceful degradation): {e}")

        logger.debug(f"Extracted entity types: {result}")
        return result

    async def filter_assertions_only(
        self, claims: list[ExtractedClaim]
    ) -> list[ExtractedClaim]:
        """
        Filter out hypothetical and example claims, keeping only assertions.

        Used for validation pipeline - only assertions need conflict checking.

        Args:
            claims: List of ExtractedClaim objects

        Returns:
            List of claims with claim_type == "assertion"
        """
        assertions = [c for c in claims if c.claim_type == "assertion"]
        skipped = len(claims) - len(assertions)

        if skipped > 0:
            logger.debug(
                f"Filtered {skipped} non-assertion claims "
                f"(hypothetical: {sum(1 for c in claims if c.claim_type == 'hypothetical')}, "
                f"example: {sum(1 for c in claims if c.claim_type == 'example')})"
            )

        return assertions
