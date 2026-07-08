"""Unit tests for ClaimExtractionService."""

import json
from unittest.mock import AsyncMock

import pytest

from src.models.conflict_detection_models import ExtractedClaim
from src.services.claim_extraction_service import ClaimExtractionService

# ========================================
# Fixtures
# ========================================


@pytest.fixture
def mock_llm_service():
    """Provide a mock LLMService."""
    service = AsyncMock()
    service.provider = "ollama"
    service._call_llm = AsyncMock()
    service.classify_entities_batch = AsyncMock()
    return service


@pytest.fixture
def mock_redis_service():
    """Provide a mock RedisService with is_connected=True."""
    service = AsyncMock()
    service.is_connected = True
    service.get = AsyncMock(return_value=None)
    service.set = AsyncMock()
    return service


@pytest.fixture
def service(mock_llm_service):
    """Provide ClaimExtractionService with mocked LLM, no Redis."""
    return ClaimExtractionService(llm_service=mock_llm_service)


@pytest.fixture
def service_with_redis(mock_llm_service, mock_redis_service):
    """Provide ClaimExtractionService with mocked LLM and Redis."""
    return ClaimExtractionService(
        llm_service=mock_llm_service,
        redis_service=mock_redis_service,
    )


def _make_claim_json(claims: list[dict]) -> str:
    """Helper to build a valid LLM JSON response."""
    return json.dumps(claims)


def _valid_claim_dict(**overrides) -> dict:
    """Return a valid claim dict with optional overrides."""
    base = {
        "claim_type": "assertion",
        "entity1": "server-01",
        "relation": "RUNS",
        "entity2": "Mnemodrome",
        "confidence": 0.9,
        "sentence": "server-01 runs Mnemodrome.",
        "context": "Server infrastructure context.",
    }
    base.update(overrides)
    return base


# ========================================
# Initialization Tests
# ========================================


class TestClaimExtractionServiceInit:
    """Tests for ClaimExtractionService initialization."""

    def test_init_with_llm_service(self, mock_llm_service):
        svc = ClaimExtractionService(llm_service=mock_llm_service)
        assert svc.llm_service is mock_llm_service
        assert svc.redis_service is None
        assert svc.cache_ttl_seconds == 300

    def test_init_with_redis_service(self, mock_llm_service, mock_redis_service):
        svc = ClaimExtractionService(
            llm_service=mock_llm_service,
            redis_service=mock_redis_service,
            cache_ttl_seconds=600,
        )
        assert svc.redis_service is mock_redis_service
        assert svc.cache_ttl_seconds == 600

    def test_init_raises_without_llm_service(self):
        with pytest.raises(ValueError, match="llm_service is required"):
            ClaimExtractionService(llm_service=None)


# ========================================
# extract_claims Tests
# ========================================


class TestExtractClaims:
    """Tests for the extract_claims method."""

    @pytest.mark.asyncio
    async def test_empty_text_raises(self, service):
        with pytest.raises(ValueError, match="Input text cannot be empty"):
            await service.extract_claims("")

    @pytest.mark.asyncio
    async def test_whitespace_only_raises(self, service):
        with pytest.raises(ValueError, match="Input text cannot be empty"):
            await service.extract_claims("   \n\t  ")

    @pytest.mark.asyncio
    async def test_single_valid_claim(self, service, mock_llm_service):
        claim = _valid_claim_dict()
        mock_llm_service._call_llm.return_value = _make_claim_json([claim])

        result = await service.extract_claims("server-01 runs Mnemodrome.")

        assert len(result) == 1
        assert isinstance(result[0], ExtractedClaim)
        assert result[0].entity1 == "server-01"
        assert result[0].relation == "RUNS"
        assert result[0].entity2 == "Mnemodrome"
        assert result[0].claim_type == "assertion"
        assert result[0].confidence == 0.9

    @pytest.mark.asyncio
    async def test_multiple_claims(self, service, mock_llm_service):
        claims = [
            _valid_claim_dict(entity1="A", entity2="B"),
            _valid_claim_dict(entity1="C", entity2="D", claim_type="hypothetical"),
        ]
        mock_llm_service._call_llm.return_value = _make_claim_json(claims)

        result = await service.extract_claims("A relates to B and C relates to D.")

        assert len(result) == 2
        assert result[0].entity1 == "A"
        assert result[1].claim_type == "hypothetical"

    @pytest.mark.asyncio
    async def test_prompt_contains_input_text(self, service, mock_llm_service):
        mock_llm_service._call_llm.return_value = _make_claim_json([])
        input_text = "Python uses pip for packages."

        await service.extract_claims(input_text)

        call_args = mock_llm_service._call_llm.call_args[0][0]
        assert input_text in call_args
        assert "Extract ALL factual claims" in call_args
        assert "Return ONLY the JSON array" in call_args

    @pytest.mark.asyncio
    async def test_prompt_template_structure(self, service, mock_llm_service):
        """Verify the prompt template includes key structural elements."""
        mock_llm_service._call_llm.return_value = _make_claim_json([])

        await service.extract_claims("Some text.")

        prompt = mock_llm_service._call_llm.call_args[0][0]
        # Verify key prompt elements
        assert "claim_type" in prompt
        assert "assertion|hypothetical|example" in prompt
        assert "entity1" in prompt
        assert "entity2" in prompt
        assert "confidence" in prompt
        assert "sentence" in prompt
        assert "context" in prompt

    # --- JSON parsing edge cases ---

    @pytest.mark.asyncio
    async def test_json_with_markdown_code_fence(self, service, mock_llm_service):
        claim = _valid_claim_dict()
        wrapped = f"```json\n{json.dumps([claim])}\n```"
        mock_llm_service._call_llm.return_value = wrapped

        result = await service.extract_claims("Some text.")

        assert len(result) == 1
        assert result[0].entity1 == "server-01"

    @pytest.mark.asyncio
    async def test_json_with_plain_code_fence(self, service, mock_llm_service):
        claim = _valid_claim_dict()
        wrapped = f"```\n{json.dumps([claim])}\n```"
        mock_llm_service._call_llm.return_value = wrapped

        result = await service.extract_claims("Some text.")

        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_malformed_json_returns_empty(self, service, mock_llm_service):
        mock_llm_service._call_llm.return_value = "this is not json at all"

        result = await service.extract_claims("Some text.")

        assert result == []

    @pytest.mark.asyncio
    async def test_empty_json_array(self, service, mock_llm_service):
        mock_llm_service._call_llm.return_value = "[]"

        result = await service.extract_claims("No claims here.")

        assert result == []

    @pytest.mark.asyncio
    async def test_non_array_response_wrapped(self, service, mock_llm_service):
        """LLM returns a single object instead of array -- should be wrapped."""
        claim = _valid_claim_dict()
        mock_llm_service._call_llm.return_value = json.dumps(claim)

        result = await service.extract_claims("Some text.")

        assert len(result) == 1
        assert result[0].entity1 == "server-01"

    # --- Field validation ---

    @pytest.mark.asyncio
    async def test_missing_required_field_skipped(self, service, mock_llm_service):
        """Claim missing 'entity2' should be skipped."""
        incomplete = {
            "claim_type": "assertion",
            "entity1": "A",
            "relation": "RUNS",
            # missing entity2, confidence, sentence
        }
        mock_llm_service._call_llm.return_value = json.dumps([incomplete])

        result = await service.extract_claims("Some text.")

        assert result == []

    @pytest.mark.asyncio
    async def test_confidence_clamped_above_one(self, service, mock_llm_service):
        claim = _valid_claim_dict(confidence=1.5)
        mock_llm_service._call_llm.return_value = _make_claim_json([claim])

        result = await service.extract_claims("Some text.")

        assert len(result) == 1
        assert result[0].confidence == 1.0

    @pytest.mark.asyncio
    async def test_confidence_clamped_below_zero(self, service, mock_llm_service):
        claim = _valid_claim_dict(confidence=-0.3)
        mock_llm_service._call_llm.return_value = _make_claim_json([claim])

        result = await service.extract_claims("Some text.")

        assert len(result) == 1
        assert result[0].confidence == 0.0

    @pytest.mark.asyncio
    async def test_invalid_claim_type_defaults_to_assertion(
        self, service, mock_llm_service
    ):
        claim = _valid_claim_dict(claim_type="guess")
        mock_llm_service._call_llm.return_value = _make_claim_json([claim])

        result = await service.extract_claims("Some text.")

        assert len(result) == 1
        assert result[0].claim_type == "assertion"

    @pytest.mark.asyncio
    async def test_missing_context_defaults_empty(self, service, mock_llm_service):
        claim = _valid_claim_dict()
        del claim["context"]
        mock_llm_service._call_llm.return_value = _make_claim_json([claim])

        result = await service.extract_claims("Some text.")

        assert len(result) == 1
        assert result[0].context == ""

    @pytest.mark.asyncio
    async def test_mixed_valid_and_invalid_claims(self, service, mock_llm_service):
        """Valid claims kept, invalid ones skipped."""
        claims = [
            _valid_claim_dict(entity1="Good"),
            {"bad": "data"},  # missing required fields
            _valid_claim_dict(entity1="AlsoGood"),
        ]
        mock_llm_service._call_llm.return_value = _make_claim_json(claims)

        result = await service.extract_claims("Some text.")

        assert len(result) == 2
        assert result[0].entity1 == "Good"
        assert result[1].entity1 == "AlsoGood"

    # --- Error handling ---

    @pytest.mark.asyncio
    async def test_llm_exception_propagates(self, service, mock_llm_service):
        mock_llm_service._call_llm.side_effect = ConnectionError("LLM unavailable")

        with pytest.raises(ConnectionError, match="LLM unavailable"):
            await service.extract_claims("Some text.")

    @pytest.mark.asyncio
    async def test_llm_timeout_propagates(self, service, mock_llm_service):
        mock_llm_service._call_llm.side_effect = TimeoutError("Request timed out")

        with pytest.raises(TimeoutError, match="Request timed out"):
            await service.extract_claims("Some text.")


# ========================================
# extract_entities_from_claims Tests
# ========================================


class TestExtractEntitiesFromClaims:
    """Tests for the extract_entities_from_claims method."""

    @pytest.mark.asyncio
    async def test_empty_claims_returns_empty(self, service):
        result = await service.extract_entities_from_claims([])
        assert result == {}

    @pytest.mark.asyncio
    async def test_extracts_unique_entities(self, service, mock_llm_service):
        claims = [
            ExtractedClaim(
                claim_type="assertion",
                entity1="A",
                relation="USES",
                entity2="B",
                confidence=0.9,
                sentence="A uses B.",
            ),
            ExtractedClaim(
                claim_type="assertion",
                entity1="A",
                relation="RUNS",
                entity2="C",
                confidence=0.9,
                sentence="A runs C.",
            ),
        ]
        # 3 unique entities: A, B, C (sorted)
        mock_llm_service.classify_entities_batch.return_value = [
            "Server",
            "Software",
            "Service",
        ]

        result = await service.extract_entities_from_claims(claims)

        assert len(result) == 3
        assert result["A"] == "Server"
        assert result["B"] == "Software"
        assert result["C"] == "Service"

    @pytest.mark.asyncio
    async def test_cache_hit_skips_llm(
        self, service_with_redis, mock_llm_service, mock_redis_service
    ):
        claims = [
            ExtractedClaim(
                claim_type="assertion",
                entity1="X",
                relation="USES",
                entity2="Y",
                confidence=0.9,
                sentence="X uses Y.",
            ),
        ]
        cached_result = json.dumps({"X": "Tool", "Y": "Framework"})
        mock_redis_service.get.return_value = cached_result

        result = await service_with_redis.extract_entities_from_claims(claims)

        assert result == {"X": "Tool", "Y": "Framework"}
        mock_llm_service.classify_entities_batch.assert_not_called()

    @pytest.mark.asyncio
    async def test_cache_miss_calls_llm_and_caches(
        self, service_with_redis, mock_llm_service, mock_redis_service
    ):
        claims = [
            ExtractedClaim(
                claim_type="assertion",
                entity1="P",
                relation="USES",
                entity2="Q",
                confidence=0.9,
                sentence="P uses Q.",
            ),
        ]
        mock_redis_service.get.return_value = None
        mock_llm_service.classify_entities_batch.return_value = ["Person", "Project"]

        result = await service_with_redis.extract_entities_from_claims(claims)

        assert len(result) == 2
        mock_llm_service.classify_entities_batch.assert_called_once()
        mock_redis_service.set.assert_called_once()

    @pytest.mark.asyncio
    async def test_redis_get_error_graceful_degradation(
        self, service_with_redis, mock_llm_service, mock_redis_service
    ):
        """Redis GET failure should not prevent entity extraction."""
        claims = [
            ExtractedClaim(
                claim_type="assertion",
                entity1="M",
                relation="RUNS",
                entity2="N",
                confidence=0.9,
                sentence="M runs N.",
            ),
        ]
        mock_redis_service.get.side_effect = Exception("Redis connection lost")
        mock_llm_service.classify_entities_batch.return_value = ["Server", "Service"]

        result = await service_with_redis.extract_entities_from_claims(claims)

        assert len(result) == 2
        mock_llm_service.classify_entities_batch.assert_called_once()

    @pytest.mark.asyncio
    async def test_redis_set_error_graceful_degradation(
        self, service_with_redis, mock_llm_service, mock_redis_service
    ):
        """Redis SET failure should not prevent returning results."""
        claims = [
            ExtractedClaim(
                claim_type="assertion",
                entity1="R",
                relation="USES",
                entity2="S",
                confidence=0.9,
                sentence="R uses S.",
            ),
        ]
        mock_redis_service.get.return_value = None
        mock_redis_service.set.side_effect = Exception("Redis write error")
        mock_llm_service.classify_entities_batch.return_value = ["Tool", "Library"]

        result = await service_with_redis.extract_entities_from_claims(claims)

        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_no_redis_skips_cache(self, service, mock_llm_service):
        """When redis_service is None, caching is skipped entirely."""
        claims = [
            ExtractedClaim(
                claim_type="assertion",
                entity1="W",
                relation="USES",
                entity2="Z",
                confidence=0.9,
                sentence="W uses Z.",
            ),
        ]
        mock_llm_service.classify_entities_batch.return_value = ["App", "Database"]

        result = await service.extract_entities_from_claims(claims)

        assert len(result) == 2
        mock_llm_service.classify_entities_batch.assert_called_once()


# ========================================
# filter_assertions_only Tests
# ========================================


class TestFilterAssertionsOnly:
    """Tests for the filter_assertions_only method."""

    @pytest.mark.asyncio
    async def test_filters_non_assertions(self, service):
        claims = [
            ExtractedClaim(
                claim_type="assertion",
                entity1="A",
                relation="R",
                entity2="B",
                confidence=0.9,
                sentence="A R B.",
            ),
            ExtractedClaim(
                claim_type="hypothetical",
                entity1="C",
                relation="R",
                entity2="D",
                confidence=0.8,
                sentence="If C R D.",
            ),
            ExtractedClaim(
                claim_type="example",
                entity1="E",
                relation="R",
                entity2="F",
                confidence=0.7,
                sentence="For example E R F.",
            ),
        ]

        result = await service.filter_assertions_only(claims)

        assert len(result) == 1
        assert result[0].entity1 == "A"
        assert result[0].claim_type == "assertion"

    @pytest.mark.asyncio
    async def test_all_assertions_kept(self, service):
        claims = [
            ExtractedClaim(
                claim_type="assertion",
                entity1="A",
                relation="R",
                entity2="B",
                confidence=0.9,
                sentence="A R B.",
            ),
            ExtractedClaim(
                claim_type="assertion",
                entity1="C",
                relation="R",
                entity2="D",
                confidence=0.8,
                sentence="C R D.",
            ),
        ]

        result = await service.filter_assertions_only(claims)

        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_empty_list(self, service):
        result = await service.filter_assertions_only([])
        assert result == []

    @pytest.mark.asyncio
    async def test_no_assertions(self, service):
        claims = [
            ExtractedClaim(
                claim_type="hypothetical",
                entity1="A",
                relation="R",
                entity2="B",
                confidence=0.9,
                sentence="If A R B.",
            ),
            ExtractedClaim(
                claim_type="example",
                entity1="C",
                relation="R",
                entity2="D",
                confidence=0.8,
                sentence="e.g. C R D.",
            ),
        ]

        result = await service.filter_assertions_only(claims)

        assert result == []
