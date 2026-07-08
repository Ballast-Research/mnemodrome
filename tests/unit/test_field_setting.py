"""
Unit tests for temporal field setting decision tree (Feature 006).

Tests the set_temporal_fields function that determines which fields to set
based on invalidation_type parameter.

Test Coverage:
- Natural invalidation (sets invalid_at only)
- System invalidation (sets expired_at only)
- Both invalidation (sets both fields)
- Default parameter behavior (should default to "both")
- Invalid invalidation type (should raise ValueError)
- Timestamp generation (ISO 8601 format with timezone)
- Return value structure (dict with correct field names)
"""

from datetime import UTC, datetime

import pytest


def set_temporal_fields(
    relationship_uuid: str, invalidation_type: str = "both", reason: str = ""
) -> dict[str, datetime]:
    """
    Set temporal validity fields based on invalidation type.

    Args:
        relationship_uuid: UUID of the relationship being invalidated
        invalidation_type: Type of invalidation ("natural" | "system" | "both")
        reason: Human-readable reason for invalidation

    Returns:
        Dict with fields that were set and their datetime values

    Raises:
        ValueError: If invalidation_type is not one of the valid options
    """
    now = datetime.now(UTC)
    fields_set = {}

    if invalidation_type == "natural":
        # Fact ended naturally (real-world change)
        fields_set["invalid_at"] = now

    elif invalidation_type == "system":
        # System invalidated fact (error correction)
        fields_set["expired_at"] = now

    elif invalidation_type == "both":
        # Both fields set (default behavior)
        fields_set["invalid_at"] = now
        fields_set["expired_at"] = now

    else:
        raise ValueError(f"Invalid invalidation_type: {invalidation_type}")

    return fields_set


class TestSetTemporalFields:
    """Unit tests for set_temporal_fields decision tree."""

    def test_natural_invalidation_sets_only_invalid_at(self):
        """
        Test that "natural" invalidation type sets only invalid_at field.

        Scenario: Real-world fact ended naturally (e.g., project completed)
        Expected: Only invalid_at is set, expired_at is NOT set
        """
        result = set_temporal_fields(
            relationship_uuid="test-uuid-123",
            invalidation_type="natural",
            reason="Project completed",
        )

        # Assert only invalid_at is present
        assert "invalid_at" in result
        assert "expired_at" not in result

        # Assert invalid_at is a datetime with timezone
        assert isinstance(result["invalid_at"], datetime)
        assert result["invalid_at"].tzinfo is not None

        # Assert timestamp is recent (within last second)
        now = datetime.now(UTC)
        time_diff = (now - result["invalid_at"]).total_seconds()
        assert time_diff < 1.0, "Timestamp should be very recent"

    def test_system_invalidation_sets_only_expired_at(self):
        """
        Test that "system" invalidation type sets only expired_at field.

        Scenario: System corrected an error (e.g., duplicate fact detected)
        Expected: Only expired_at is set, invalid_at is NOT set
        """
        result = set_temporal_fields(
            relationship_uuid="test-uuid-456",
            invalidation_type="system",
            reason="Duplicate fact detected",
        )

        # Assert only expired_at is present
        assert "expired_at" in result
        assert "invalid_at" not in result

        # Assert expired_at is a datetime with timezone
        assert isinstance(result["expired_at"], datetime)
        assert result["expired_at"].tzinfo is not None

        # Assert timestamp is recent (within last second)
        now = datetime.now(UTC)
        time_diff = (now - result["expired_at"]).total_seconds()
        assert time_diff < 1.0, "Timestamp should be very recent"

    def test_both_invalidation_sets_both_fields(self):
        """
        Test that "both" invalidation type sets both invalid_at and expired_at.

        Scenario: Strongest invalidation - fact ended AND system invalidated
        Expected: Both fields are set with same timestamp
        """
        result = set_temporal_fields(
            relationship_uuid="test-uuid-789",
            invalidation_type="both",
            reason="Infrastructure migrated",
        )

        # Assert both fields are present
        assert "invalid_at" in result
        assert "expired_at" in result

        # Assert both are datetime objects with timezone
        assert isinstance(result["invalid_at"], datetime)
        assert isinstance(result["expired_at"], datetime)
        assert result["invalid_at"].tzinfo is not None
        assert result["expired_at"].tzinfo is not None

        # Assert timestamps are identical (set at same time)
        time_diff = abs((result["invalid_at"] - result["expired_at"]).total_seconds())
        assert time_diff < 0.001, "Both timestamps should be set simultaneously"

    def test_default_parameter_is_both(self):
        """
        Test that omitting invalidation_type parameter defaults to "both".

        Scenario: User doesn't specify invalidation_type
        Expected: Defaults to "both" (strongest invalidation)
        """
        result = set_temporal_fields(
            relationship_uuid="test-uuid-default", reason="Default invalidation"
        )

        # Assert both fields are present (default behavior)
        assert "invalid_at" in result
        assert "expired_at" in result

        # Assert both are datetime objects
        assert isinstance(result["invalid_at"], datetime)
        assert isinstance(result["expired_at"], datetime)

    def test_invalid_invalidation_type_raises_value_error(self):
        """
        Test that invalid invalidation_type raises ValueError.

        Scenario: User provides invalid invalidation type
        Expected: ValueError with descriptive message
        """
        invalid_types = [
            "invalid",
            "unknown",
            "NATURAL",  # Case sensitive
            "SYSTEM",  # Case sensitive
            "BOTH",  # Case sensitive
            "natural_system",
            "",
            "null",
            "none",
        ]

        for invalid_type in invalid_types:
            with pytest.raises(ValueError) as exc_info:
                set_temporal_fields(
                    relationship_uuid="test-uuid-error",
                    invalidation_type=invalid_type,
                    reason="Test invalid type",
                )

            # Assert error message includes the invalid value
            assert invalid_type in str(exc_info.value)
            assert "Invalid invalidation_type" in str(exc_info.value)

    def test_timestamp_iso8601_format_with_timezone(self):
        """
        Test that timestamps are in ISO 8601 format with UTC timezone.

        Scenario: Verify timestamp format matches Neo4j datetime requirements
        Expected: ISO 8601 string with timezone (e.g., 2025-10-11T12:34:56.789+00:00)
        """
        result = set_temporal_fields(
            relationship_uuid="test-uuid-format",
            invalidation_type="both",
            reason="Format test",
        )

        # Test invalid_at format
        invalid_at = result["invalid_at"]
        iso_string = invalid_at.isoformat()

        # Assert ISO 8601 format components
        assert "T" in iso_string, "Should have T separator"
        assert "+" in iso_string or "Z" in iso_string, "Should have timezone indicator"

        # Assert UTC timezone
        assert invalid_at.tzinfo == UTC

        # Assert can be serialized to string and parsed back
        parsed = datetime.fromisoformat(iso_string)
        assert parsed.tzinfo is not None, "Parsed datetime should retain timezone"

    def test_fields_set_return_value_structure(self):
        """
        Test that return value is a dict with correct field names.

        Scenario: Verify return value structure for all invalidation types
        Expected: Dict with snake_case field names matching Neo4j property names
        """
        # Test natural
        natural_result = set_temporal_fields(
            relationship_uuid="test-1", invalidation_type="natural", reason="Test"
        )
        assert isinstance(natural_result, dict)
        assert set(natural_result.keys()) == {"invalid_at"}

        # Test system
        system_result = set_temporal_fields(
            relationship_uuid="test-2", invalidation_type="system", reason="Test"
        )
        assert isinstance(system_result, dict)
        assert set(system_result.keys()) == {"expired_at"}

        # Test both
        both_result = set_temporal_fields(
            relationship_uuid="test-3", invalidation_type="both", reason="Test"
        )
        assert isinstance(both_result, dict)
        assert set(both_result.keys()) == {"invalid_at", "expired_at"}

    def test_all_three_decision_tree_paths(self):
        """
        Comprehensive test verifying all three paths through decision tree.

        Scenario: Execute all three branches in single test
        Expected: Each path produces correct output
        """
        # Path 1: Natural invalidation
        natural = set_temporal_fields(
            relationship_uuid="path-1",
            invalidation_type="natural",
            reason="Path 1 test",
        )
        assert "invalid_at" in natural and "expired_at" not in natural

        # Path 2: System invalidation
        system = set_temporal_fields(
            relationship_uuid="path-2", invalidation_type="system", reason="Path 2 test"
        )
        assert "expired_at" in system and "invalid_at" not in system

        # Path 3: Both invalidation
        both = set_temporal_fields(
            relationship_uuid="path-3", invalidation_type="both", reason="Path 3 test"
        )
        assert "invalid_at" in both and "expired_at" in both

        # Verify all timestamps are distinct (different calls = different times)
        # Note: They may be identical if executed too quickly, so we just verify they exist
        all_timestamps = [
            natural["invalid_at"],
            system["expired_at"],
            both["invalid_at"],
            both["expired_at"],
        ]
        for ts in all_timestamps:
            assert isinstance(ts, datetime)
            assert ts.tzinfo == UTC

    def test_relationship_uuid_parameter_not_used_in_logic(self):
        """
        Test that relationship_uuid parameter doesn't affect field setting logic.

        Scenario: Different UUIDs should produce same field setting behavior
        Expected: UUID is for reference only, doesn't change decision tree
        """
        uuid1 = "00000000-0000-0000-0000-000000000001"
        uuid2 = "99999999-9999-9999-9999-999999999999"

        result1 = set_temporal_fields(uuid1, "natural", "Test")
        result2 = set_temporal_fields(uuid2, "natural", "Test")

        # Assert same fields set (timestamps will differ)
        assert set(result1.keys()) == set(result2.keys())
        assert "invalid_at" in result1
        assert "invalid_at" in result2

    def test_reason_parameter_not_used_in_logic(self):
        """
        Test that reason parameter doesn't affect field setting logic.

        Scenario: Different reasons should produce same field setting behavior
        Expected: Reason is for logging only, doesn't change decision tree
        """
        reason1 = "Short reason"
        reason2 = "Very long reason with lots of detail about why this fact is being invalidated"

        result1 = set_temporal_fields("uuid", "system", reason1)
        result2 = set_temporal_fields("uuid", "system", reason2)

        # Assert same fields set
        assert set(result1.keys()) == set(result2.keys())
        assert "expired_at" in result1
        assert "expired_at" in result2


class TestTemporalFieldSemantics:
    """Tests verifying correct understanding of field semantics."""

    def test_invalid_at_semantic_natural_end(self):
        """
        Test that invalid_at represents when fact stopped being true in real world.

        Semantic: Natural end of fact's validity (e.g., project completed, person left company)
        """
        # This is a documentation test - verifies correct field usage
        result = set_temporal_fields(
            relationship_uuid="semantic-test-1",
            invalidation_type="natural",
            reason="Project completed (natural end)",
        )

        # invalid_at should be set for natural ends
        assert "invalid_at" in result
        # expired_at should NOT be set (this was natural, not system correction)
        assert "expired_at" not in result

    def test_expired_at_semantic_system_correction(self):
        """
        Test that expired_at represents when system invalidated fact (admin/error).

        Semantic: System-initiated invalidation (e.g., error correction, duplicate removal)
        """
        # This is a documentation test - verifies correct field usage
        result = set_temporal_fields(
            relationship_uuid="semantic-test-2",
            invalidation_type="system",
            reason="Duplicate fact detected (system correction)",
        )

        # expired_at should be set for system corrections
        assert "expired_at" in result
        # invalid_at should NOT be set (fact might still be true, but system removed it)
        assert "invalid_at" not in result

    def test_both_semantic_strongest_invalidation(self):
        """
        Test that "both" represents strongest invalidation (default).

        Semantic: Both natural end AND system invalidation (complete invalidation)
        """
        # This is a documentation test - verifies correct field usage
        result = set_temporal_fields(
            relationship_uuid="semantic-test-3",
            invalidation_type="both",
            reason="Infrastructure migrated (both natural change and system update)",
        )

        # Both fields should be set for strongest invalidation
        assert "invalid_at" in result
        assert "expired_at" in result


# Edge case tests
class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_string_reason(self):
        """Test that empty reason string doesn't cause issues."""
        result = set_temporal_fields(
            relationship_uuid="edge-1", invalidation_type="natural", reason=""
        )
        assert "invalid_at" in result

    def test_very_long_uuid(self):
        """Test with unusually long UUID string."""
        long_uuid = "a" * 1000
        result = set_temporal_fields(
            relationship_uuid=long_uuid, invalidation_type="system", reason="Test"
        )
        assert "expired_at" in result

    def test_special_characters_in_reason(self):
        """Test that special characters in reason don't break logic."""
        special_reason = (
            "Test with 'quotes', \"double quotes\", and \n newlines \t tabs"
        )
        result = set_temporal_fields(
            relationship_uuid="edge-2", invalidation_type="both", reason=special_reason
        )
        assert "invalid_at" in result
        assert "expired_at" in result
