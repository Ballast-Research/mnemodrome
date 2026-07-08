"""Unit tests for AutoCorrectionService."""

from datetime import datetime
from unittest.mock import AsyncMock

import pytest

from src.models.conflict_detection_models import (
    ConflictResult,
    CorrectionResult,
    ExtractedClaim,
    StoredFact,
)
from src.services.auto_correction_service import AutoCorrectionService


@pytest.fixture
def auto_correction_service():
    """Provide AutoCorrectionService with test configuration."""
    return AutoCorrectionService(
        severity_minor_max=40,
        severity_major_min=70,
        confidence_min=0.9,
        event_log_service=None,
    )


@pytest.fixture
def stored_fact():
    """Provide test StoredFact."""
    return StoredFact(
        entity1_uuid="e1-uuid",
        entity1_name="server-01",
        relation="RUNS",
        entity2_uuid="e2-uuid",
        entity2_name="Mnemodrome",
        created_at=datetime.now(),
        fact_uuid="fact-uuid",
        is_temporal=False,
    )


@pytest.fixture
def extracted_claim():
    """Provide test ExtractedClaim."""
    return ExtractedClaim(
        claim_type="assertion",
        entity1="server-01",
        relation="RUNS",
        entity2="ProxyVM",
        confidence=0.95,
        context="",
        sentence="server-01 runs ProxyVM",
    )


class TestAutoCorrectionServiceInit:
    """Test service initialization."""

    def test_init_with_defaults(self):
        """Test initializes with default thresholds."""
        service = AutoCorrectionService()
        assert service.severity_minor_max == 40
        assert service.severity_major_min == 70
        assert service.confidence_min == 0.9

    def test_init_validates_severity_minor_max(self):
        """Test validates severity_minor_max bounds."""
        with pytest.raises(ValueError, match="severity_minor_max must be 0-100"):
            AutoCorrectionService(severity_minor_max=150)

    def test_init_validates_severity_major_min(self):
        """Test validates severity_major_min bounds."""
        with pytest.raises(ValueError, match="severity_major_min must be 0-100"):
            AutoCorrectionService(severity_major_min=-10)

    def test_init_validates_confidence_min(self):
        """Test validates confidence_min bounds."""
        with pytest.raises(ValueError, match="confidence_min must be 0.0-1.0"):
            AutoCorrectionService(confidence_min=1.5)

    def test_init_with_custom_config(self):
        """Test initializes with custom thresholds."""
        service = AutoCorrectionService(
            severity_minor_max=50, severity_major_min=80, confidence_min=0.85
        )
        assert service.severity_minor_max == 50
        assert service.severity_major_min == 80
        assert service.confidence_min == 0.85


class TestCorrectText:
    """Test text correction logic."""

    @pytest.mark.asyncio
    async def test_correct_text_validates_input(self, auto_correction_service):
        """Test validates non-empty text input."""
        with pytest.raises(ValueError, match="original_text cannot be empty"):
            await auto_correction_service.correct_text("", [])

    @pytest.mark.asyncio
    async def test_correct_text_returns_skipped_on_no_conflicts(
        self, auto_correction_service
    ):
        """Test returns 'skipped' action when no conflicts."""
        result = await auto_correction_service.correct_text("Some text", [])

        assert result.action == "skipped"
        assert result.modified_text == "Some text"
        assert result.corrections_applied == []

    @pytest.mark.asyncio
    async def test_correct_text_skips_low_severity_conflicts(
        self, auto_correction_service, stored_fact, extracted_claim
    ):
        """Test skips conflicts below severity_minor_max."""
        conflict = ConflictResult(
            conflict_detected=True,
            conflict_type="exact_contradiction",
            severity=30,  # Below minor_max=40
            recommendation="skip",
            stored_fact=stored_fact,
            claim=extracted_claim,
        )

        result = await auto_correction_service.correct_text(
            "server-01 runs ProxyVM", [conflict]
        )

        # Conflict severity too low, should be skipped
        assert result.action == "skipped"
        assert result.corrections_applied == []

    @pytest.mark.asyncio
    async def test_correct_text_auto_corrects_medium_severity(
        self, auto_correction_service, stored_fact, extracted_claim
    ):
        """Test auto-corrects medium severity conflicts with high confidence."""
        conflict = ConflictResult(
            conflict_detected=True,
            conflict_type="exact_contradiction",
            severity=50,  # Between minor_max and major_min
            recommendation="auto_correct",
            correction_suggestion="server-01 RUNS Mnemodrome",
            stored_fact=stored_fact,
            claim=extracted_claim,
        )

        result = await auto_correction_service.correct_text(
            "server-01 RUNS ProxyVM", [conflict]
        )

        assert result.action == "corrected"
        assert "Mnemodrome" in result.modified_text
        assert len(result.corrections_applied) == 1

    @pytest.mark.asyncio
    async def test_correct_text_warns_on_major_severity(
        self, auto_correction_service, stored_fact, extracted_claim
    ):
        """Test warns instead of correcting major severity conflicts."""
        conflict = ConflictResult(
            conflict_detected=True,
            conflict_type="exact_contradiction",
            severity=80,  # Above major_min=70
            recommendation="warn",
            stored_fact=stored_fact,
            claim=extracted_claim,
        )

        result = await auto_correction_service.correct_text(
            "server-01 RUNS ProxyVM", [conflict]
        )

        assert result.action == "warned"
        assert result.warning_message is not None
        assert "WARNING" in result.warning_message

    @pytest.mark.asyncio
    async def test_correct_text_respects_confidence_threshold(
        self, auto_correction_service, stored_fact
    ):
        """Test skips correction if claim confidence below threshold."""
        low_confidence_claim = ExtractedClaim(
            claim_type="assertion",
            entity1="server-01",
            relation="RUNS",
            entity2="ProxyVM",
            confidence=0.8,  # Below service threshold of 0.9
            context="",
            sentence="server-01 runs ProxyVM",
        )

        conflict = ConflictResult(
            conflict_detected=True,
            conflict_type="exact_contradiction",
            severity=50,
            recommendation="auto_correct",
            correction_suggestion="server-01 RUNS Mnemodrome",
            stored_fact=stored_fact,
            claim=low_confidence_claim,
        )

        result = await auto_correction_service.correct_text(
            "server-01 RUNS ProxyVM", [conflict]
        )

        # Low confidence should skip auto-correction
        assert len(result.corrections_applied) == 0

    @pytest.mark.asyncio
    async def test_correct_text_multiple_conflicts(
        self, auto_correction_service, stored_fact
    ):
        """Test handles multiple conflicts correctly."""
        claim1 = ExtractedClaim(
            claim_type="assertion",
            entity1="server-01",
            relation="RUNS",
            entity2="ProxyVM",
            confidence=0.95,
            context="",
            sentence="server-01 runs ProxyVM",
        )

        claim2 = ExtractedClaim(
            claim_type="assertion",
            entity1="Neo4j",
            relation="STORES",
            entity2="OldData",
            confidence=0.92,
            context="",
            sentence="Neo4j stores OldData",
        )

        fact2 = StoredFact(
            entity1_uuid="neo4j-uuid",
            entity1_name="Neo4j",
            relation="STORES",
            entity2_uuid="data-uuid",
            entity2_name="NewData",
            created_at=datetime.now(),
            fact_uuid="fact2-uuid",
        )

        conflicts = [
            ConflictResult(
                conflict_detected=True,
                conflict_type="exact_contradiction",
                severity=50,
                recommendation="auto_correct",
                correction_suggestion="server-01 RUNS Mnemodrome",
                stored_fact=stored_fact,
                claim=claim1,
            ),
            ConflictResult(
                conflict_detected=True,
                conflict_type="exact_contradiction",
                severity=80,  # Major
                recommendation="warn",
                stored_fact=fact2,
                claim=claim2,
            ),
        ]

        result = await auto_correction_service.correct_text(
            "server-01 RUNS ProxyVM and Neo4j STORES OldData", conflicts
        )

        # Should have 1 correction and major warning
        assert len(result.corrections_applied) == 1
        assert result.warning_message is not None
        assert result.action == "warned"  # Warning takes precedence


class TestCorrectResponseText:
    """Test correction of LLM response text."""

    @pytest.mark.asyncio
    async def test_correct_response_returns_corrected_text(
        self, auto_correction_service, stored_fact, extracted_claim
    ):
        """Test returns corrected text with warning prepended if needed."""
        conflict = ConflictResult(
            conflict_detected=True,
            conflict_type="exact_contradiction",
            severity=80,
            recommendation="warn",
            stored_fact=stored_fact,
            claim=extracted_claim,
        )

        response_text = "According to my understanding, server-01 RUNS ProxyVM"

        result = await auto_correction_service.correct_response_text(
            response_text, [conflict]
        )

        # Should include warning
        assert "WARNING" in result
        assert response_text in result


class TestGenerateAuditSummary:
    """Test audit summary generation."""

    @pytest.mark.asyncio
    async def test_generate_audit_summary_includes_action(
        self, auto_correction_service
    ):
        """Test audit summary includes action."""
        correction_result = CorrectionResult(
            action="corrected",
            modified_text="Corrected text",
            original_text="Original text",
            corrections_applied=[
                {
                    "original": "wrong",
                    "corrected": "right",
                    "reason": "Conflict detected",
                    "confidence": 0.95,
                    "severity": 60,
                }
            ],
            conflicts_detected=1,
        )

        summary = await auto_correction_service.generate_audit_summary(
            correction_result
        )

        assert "Correction Audit Summary" in summary
        assert "Action: corrected" in summary
        assert "Conflicts Detected: 1" in summary

    @pytest.mark.asyncio
    async def test_generate_audit_summary_includes_corrections(
        self, auto_correction_service
    ):
        """Test audit summary includes all corrections applied."""
        correction_result = CorrectionResult(
            action="corrected",
            modified_text="Corrected text",
            original_text="Original text",
            corrections_applied=[
                {
                    "original": "server-01 RUNS ProxyVM",
                    "corrected": "server-01 RUNS Mnemodrome",
                    "reason": "Contradicts stored fact",
                    "confidence": 0.95,
                    "severity": 60,
                }
            ],
            conflicts_detected=1,
        )

        summary = await auto_correction_service.generate_audit_summary(
            correction_result
        )

        assert "server-01 RUNS ProxyVM" in summary
        assert "server-01 RUNS Mnemodrome" in summary
        assert "Contradicts stored fact" in summary

    @pytest.mark.asyncio
    async def test_generate_audit_summary_includes_warnings(
        self, auto_correction_service
    ):
        """Test audit summary includes warning message."""
        correction_result = CorrectionResult(
            action="warned",
            modified_text="Text",
            original_text="Text",
            corrections_applied=[],
            warning_message="WARNING: Conflict detected",
            conflicts_detected=1,
        )

        summary = await auto_correction_service.generate_audit_summary(
            correction_result
        )

        assert "WARNING: Conflict detected" in summary


class TestWithEventLogService:
    """Test integration with EventLogService."""

    @pytest.mark.asyncio
    async def test_logs_auto_correction_events(self):
        """Test logs events when correction applied."""
        event_log_service = AsyncMock()
        service = AutoCorrectionService(event_log_service=event_log_service)

        stored_fact = StoredFact(
            entity1_uuid="e1",
            entity1_name="Entity1",
            relation="RELATES",
            entity2_uuid="e2",
            entity2_name="Entity2",
            created_at=datetime.now(),
            fact_uuid="f-uuid",
        )

        claim = ExtractedClaim(
            claim_type="assertion",
            entity1="Entity1",
            relation="RELATES",
            entity2="Wrong",
            confidence=0.95,
            context="",
            sentence="test",
        )

        conflict = ConflictResult(
            conflict_detected=True,
            conflict_type="exact_contradiction",
            severity=50,
            recommendation="auto_correct",
            correction_suggestion="Entity1 RELATES Entity2",
            stored_fact=stored_fact,
            claim=claim,
        )

        result = await service.correct_text("Entity1 RELATES Wrong", [conflict])

        # Should have called event log
        event_log_service.log_event.assert_called()

    @pytest.mark.asyncio
    async def test_logs_warning_events(self):
        """Test logs events when warning issued."""
        event_log_service = AsyncMock()
        service = AutoCorrectionService(event_log_service=event_log_service)

        stored_fact = StoredFact(
            entity1_uuid="e1",
            entity1_name="Entity1",
            relation="RELATES",
            entity2_uuid="e2",
            entity2_name="Entity2",
            created_at=datetime.now(),
            fact_uuid="f-uuid",
        )

        claim = ExtractedClaim(
            claim_type="assertion",
            entity1="Entity1",
            relation="RELATES",
            entity2="Wrong",
            confidence=0.95,
            context="",
            sentence="test",
        )

        conflict = ConflictResult(
            conflict_detected=True,
            conflict_type="exact_contradiction",
            severity=80,
            recommendation="warn",
            stored_fact=stored_fact,
            claim=claim,
        )

        result = await service.correct_text("Entity1 RELATES Wrong", [conflict])

        # Should have called event log for warning
        event_log_service.log_event.assert_called()
