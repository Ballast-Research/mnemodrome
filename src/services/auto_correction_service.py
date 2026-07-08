"""
Auto-correction service for bidirectional conflict detection (Feature 025).

Applies corrections to text based on conflict severity:
- Minor conflicts (severity <70) → auto-correct with audit logging
- Major conflicts (severity ≥70) → warning banner

Constitutional Principle: Safety through severity-based correction strategy.
"""

from datetime import datetime
from typing import TYPE_CHECKING, Optional

from loguru import logger

from src.config import (
    CONFLICT_AUTO_CORRECT_CONFIDENCE_MIN,
    CONFLICT_SEVERITY_MAJOR_MIN,
    CONFLICT_SEVERITY_MINOR_MAX,
)
from src.models.conflict_detection_models import (
    ConflictResult,
    CorrectionResult,
)

if TYPE_CHECKING:
    from src.services.event_log_service import EventLogService


class AutoCorrectionService:
    """
    Service for applying automatic corrections to text based on conflicts.

    Implements:
    - Severity-based decision (auto-correct vs warn vs block)
    - Confidence-based filtering
    - Audit logging of all corrections
    - Warning banner generation for major conflicts

    Attributes:
        severity_minor_max: Threshold for "minor" conflicts (default: 40)
        severity_major_min: Threshold for "major" conflicts (default: 70)
        confidence_min: Minimum confidence for auto-correction (default: 0.9)
        event_log_service: Optional EventLogService for audit trail
    """

    def __init__(
        self,
        severity_minor_max: int = CONFLICT_SEVERITY_MINOR_MAX,
        severity_major_min: int = CONFLICT_SEVERITY_MAJOR_MIN,
        confidence_min: float = CONFLICT_AUTO_CORRECT_CONFIDENCE_MIN,
        event_log_service: Optional["EventLogService"] = None,
    ):
        """
        Initialize AutoCorrectionService.

        Args:
            severity_minor_max: Threshold for minor conflicts (default: 40)
            severity_major_min: Threshold for major conflicts (default: 70)
            confidence_min: Minimum confidence for auto-correction (default: 0.9)
            event_log_service: Optional EventLogService for audit logging

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
        if not (0.0 <= confidence_min <= 1.0):
            raise ValueError(f"confidence_min must be 0.0-1.0, got {confidence_min}")

        self.severity_minor_max = severity_minor_max
        self.severity_major_min = severity_major_min
        self.confidence_min = confidence_min
        self.event_log_service = event_log_service

        logger.info(
            f"AutoCorrectionService initialized: "
            f"minor_max={severity_minor_max}, major_min={severity_major_min}, "
            f"confidence_min={confidence_min}"
        )

    async def correct_text(
        self,
        original_text: str,
        conflicts: list[ConflictResult],
    ) -> CorrectionResult:
        """
        Apply corrections to text based on conflict severity.

        Strategy:
        - severity < minor_max (40): Skip (no correction needed)
        - minor_max ≤ severity < major_min (40-69): Auto-correct if confidence ≥ 0.9
        - severity ≥ major_min (70): Warn instead of auto-correct

        Args:
            original_text: Original text to potentially correct
            conflicts: List of ConflictResult objects

        Returns:
            CorrectionResult with modified text and audit information

        Raises:
            ValueError: If original_text is empty
            Exception: If correction fails
        """
        if not original_text or not original_text.strip():
            raise ValueError("original_text cannot be empty")

        if not conflicts:
            return CorrectionResult(
                action="skipped",
                modified_text=original_text,
                original_text=original_text,
                corrections_applied=[],
                conflicts_detected=0,
            )

        logger.debug(f"Correcting text with {len(conflicts)} conflicts")

        modified_text = original_text
        corrections_applied = []
        warning_message = None

        # Separate conflicts by severity
        minor_conflicts = [c for c in conflicts if c.severity < self.severity_major_min]
        major_conflicts = [
            c for c in conflicts if c.severity >= self.severity_major_min
        ]

        # Process minor conflicts (auto-correct candidates)
        for conflict in minor_conflicts:
            if conflict.severity < self.severity_minor_max:
                # Severity too low, skip
                logger.debug(
                    f"Skipping correction: severity {conflict.severity} below threshold"
                )
                continue

            # Check if correction suggestion available and confidence high
            if (
                conflict.correction_suggestion
                and conflict.claim
                and conflict.claim.confidence >= self.confidence_min
            ):
                # Apply auto-correction
                logger.info(
                    f"Auto-correcting: {conflict.claim.entity1} {conflict.claim.relation} "
                    f"{conflict.claim.entity2} → {conflict.stored_fact.entity2_name} "
                    f"(severity: {conflict.severity}, confidence: {conflict.claim.confidence})"
                )

                # Find and replace in text
                old_pattern = f"{conflict.claim.entity1} {conflict.claim.relation} {conflict.claim.entity2}"
                new_pattern = conflict.correction_suggestion

                if old_pattern in modified_text:
                    modified_text = modified_text.replace(old_pattern, new_pattern)
                    corrections_applied.append(
                        {
                            "original": old_pattern,
                            "corrected": new_pattern,
                            "reason": (
                                f"Contradicts stored fact: {conflict.stored_fact.entity1_name} "
                                f"{conflict.stored_fact.relation} {conflict.stored_fact.entity2_name}"
                            ),
                            "confidence": float(conflict.claim.confidence),
                            "severity": conflict.severity,
                        }
                    )

                    # Log event if event_log_service available
                    if self.event_log_service:
                        try:
                            await self.event_log_service.log_event(
                                event_type="validation.auto_correction_applied",
                                action="auto_correct_response",
                                context={
                                    "original": old_pattern,
                                    "corrected": new_pattern,
                                    "reason": (
                                        f"Contradicts stored fact: "
                                        f"{conflict.stored_fact.entity1_name} "
                                        f"{conflict.stored_fact.relation} "
                                        f"{conflict.stored_fact.entity2_name}"
                                    ),
                                    "confidence": float(conflict.claim.confidence),
                                    "severity": conflict.severity,
                                },
                                result="success",
                            )
                        except Exception as e:
                            logger.debug(
                                f"Event logging error (graceful degradation): {e}"
                            )

        # Generate warning for major conflicts
        if major_conflicts:
            warning_lines = [
                "WARNING: The following statements may contradict stored knowledge:",
                "",
            ]

            for conflict in major_conflicts:
                warning_lines.append(
                    f"- {conflict.claim.entity1} {conflict.claim.relation} "
                    f"{conflict.claim.entity2} "
                    f"(conflicts with: {conflict.stored_fact.entity1_name} "
                    f"{conflict.stored_fact.relation} {conflict.stored_fact.entity2_name}, "
                    f"severity: {conflict.severity}/100)"
                )

            warning_message = "\n".join(warning_lines)
            logger.warning(
                f"Generated warning for {len(major_conflicts)} major conflicts"
            )

            # Log event if event_log_service available
            if self.event_log_service:
                try:
                    await self.event_log_service.log_event(
                        event_type="validation.conflict_warning",
                        action="warn_major_conflict",
                        context={
                            "conflicts_count": len(major_conflicts),
                            "avg_severity": sum(c.severity for c in major_conflicts)
                            // len(major_conflicts),
                            "conflicts": [
                                {
                                    "entity1": c.claim.entity1 if c.claim else None,
                                    "relation": c.claim.relation if c.claim else None,
                                    "entity2": c.claim.entity2 if c.claim else None,
                                    "severity": c.severity,
                                }
                                for c in major_conflicts
                            ],
                        },
                        result="success",
                    )
                except Exception as e:
                    logger.debug(f"Event logging error (graceful degradation): {e}")

        # Determine action taken
        if corrections_applied and not major_conflicts:
            action = "corrected"
        elif major_conflicts and not corrections_applied:
            action = "warned"
        elif corrections_applied and major_conflicts:
            action = "warned"  # Warn takes precedence when both present
        else:
            action = "skipped"

        result = CorrectionResult(
            action=action,
            modified_text=modified_text,
            original_text=original_text,
            corrections_applied=corrections_applied,
            warning_message=warning_message,
            conflicts_detected=len(conflicts),
        )

        logger.info(
            f"Text correction complete: action={action}, "
            f"corrections={len(corrections_applied)}, "
            f"warnings={len(major_conflicts)}"
        )

        return result

    async def correct_response_text(
        self,
        response_text: str,
        conflicts: list[ConflictResult],
    ) -> str:
        """
        Apply corrections to LLM response text.

        Returns corrected text with warning banner prepended for major conflicts.

        Args:
            response_text: LLM response text to correct
            conflicts: List of ConflictResult objects from validation

        Returns:
            Corrected/warned response text

        Raises:
            ValueError: If response_text is empty
        """
        correction_result = await self.correct_text(response_text, conflicts)

        # Prepend warning if present
        if correction_result.warning_message:
            return f"{correction_result.warning_message}\n\n{correction_result.modified_text}"

        return correction_result.modified_text

    async def generate_audit_summary(
        self,
        correction_result: CorrectionResult,
    ) -> str:
        """
        Generate human-readable audit summary of corrections applied.

        Args:
            correction_result: CorrectionResult from correct_text()

        Returns:
            Formatted audit summary

        Raises:
            Exception: If summary generation fails
        """
        summary_lines = [
            "Correction Audit Summary",
            "========================",
            f"Action: {correction_result.action}",
            f"Timestamp: {datetime.now().isoformat()}",
            f"Conflicts Detected: {correction_result.conflicts_detected}",
            f"Corrections Applied: {len(correction_result.corrections_applied)}",
            "",
        ]

        if correction_result.corrections_applied:
            summary_lines.append("Corrections:")
            for i, correction in enumerate(correction_result.corrections_applied, 1):
                summary_lines.append(f"  {i}. {correction['original']}")
                summary_lines.append(f"     → {correction['corrected']}")
                summary_lines.append(f"     Reason: {correction['reason']}")
                summary_lines.append(f"     Confidence: {correction['confidence']:.2f}")
                summary_lines.append("")

        if correction_result.warning_message:
            summary_lines.append("Warnings:")
            summary_lines.append(correction_result.warning_message)

        return "\n".join(summary_lines)
