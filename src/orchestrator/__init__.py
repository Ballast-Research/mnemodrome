"""
Orchestration Layer Module

Phase 3 MVP: Multi-Agent Validation Pipeline

Exports the agent coordinator and supporting classes for use throughout the system.
"""

from src.orchestrator.agent_coordinator import (
    AgentCoordinator,
    AggregatedResult,
    ContextPackage,
    EntityConsistencyValidator,
    JobStatus,
    OrchestrationJob,
    PatternComplianceValidator,
    TestCoverageValidator,
    ValidationResult,
    ValidationRole,
    ValidationSeverity,
    ValidationViolation,
    validate_code,
)

__all__ = [
    "AgentCoordinator",
    "OrchestrationJob",
    "ValidationResult",
    "AggregatedResult",
    "ValidationViolation",
    "ContextPackage",
    "ValidationRole",
    "JobStatus",
    "ValidationSeverity",
    "PatternComplianceValidator",
    "TestCoverageValidator",
    "EntityConsistencyValidator",
    "validate_code",
]
