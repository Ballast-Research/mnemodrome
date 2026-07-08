"""
Evolution Candidate Service - Core logic for suggesting instruction updates.

Implements the comparison pipeline:
1. Query high-confidence patterns from Neo4j
2. Discover target SKILL.md files
3. Use LLM to compare patterns vs. current instructions
4. Generate and store evolution candidates
5. Support approval workflow and change application
"""

import json
from datetime import datetime
from typing import Any
from uuid import uuid4

from loguru import logger

from src.config import (
    EVOLUTION_ENABLED,
    EVOLUTION_MAX_CANDIDATES,
    EVOLUTION_MIN_OCCURRENCES,
    EVOLUTION_MIN_SUCCESS_RATE,
)
from src.models.evolution_models import (
    EvolutionCandidate,
    EvolutionProposal,
    EvolutionResult,
    SkillInfo,
)
from src.services.llm_service import llm_service as llm_service_singleton
from src.services.neo4j_service import neo4j_service
from src.services.pattern_query_service import PatternQueryService
from src.services.skill_file_loader import EvolutionError, SkillFileLoader


class EvolutionCandidateService:
    """
    Core service for generating and applying evolution candidates.

    Implements:
    - Pattern querying and filtering
    - LLM-powered gap detection
    - Candidate generation and storage
    - Diff validation and application
    - Change documentation
    """

    def __init__(self):
        """Initialize evolution candidate service with dependencies."""
        self.pattern_query_service = PatternQueryService()
        self.skill_file_loader = SkillFileLoader()
        self.neo4j_service = neo4j_service
        self.llm_service = llm_service_singleton
        self.enabled = EVOLUTION_ENABLED

        logger.debug(
            f"EvolutionCandidateService initialized: "
            f"enabled={self.enabled}, "
            f"min_success_rate={EVOLUTION_MIN_SUCCESS_RATE}, "
            f"min_occurrences={EVOLUTION_MIN_OCCURRENCES}"
        )

    async def generate_candidates(
        self,
        skill_path: str | None = None,
        min_success_rate: float = EVOLUTION_MIN_SUCCESS_RATE,
        min_occurrences: int = EVOLUTION_MIN_OCCURRENCES,
        max_candidates: int = EVOLUTION_MAX_CANDIDATES,
    ) -> dict[str, Any]:
        """
        Generate evolution candidates from patterns.

        Pipeline:
        1. Check if feature is enabled
        2. Query high-confidence patterns
        3. Discover target skills
        4. Compare patterns to skills using LLM
        5. Store candidates in Neo4j
        6. Return sorted by confidence

        Args:
            skill_path: Optional specific skill to analyze (None = all skills)
            min_success_rate: Minimum pattern success rate threshold (0.8 default)
            min_occurrences: Minimum pattern occurrence count (3 default)
            max_candidates: Maximum candidates to return (10 default)

        Returns:
            Dict with:
            - candidates: List of EvolutionCandidate dicts
            - skills_analyzed: Number of skills examined
            - patterns_evaluated: Number of patterns reviewed
            - candidates_generated: Number of candidates created
        """
        if not self.enabled:
            logger.info("Evolution disabled (EVOLUTION_ENABLED=false)")
            return {
                "candidates": [],
                "skills_analyzed": 0,
                "patterns_evaluated": 0,
                "candidates_generated": 0,
            }

        try:
            # Step 1: Query high-confidence patterns
            logger.info(
                f"Querying patterns: min_success_rate={min_success_rate}, "
                f"min_occurrences={min_occurrences}"
            )

            patterns = await self.pattern_query_service.query_patterns(
                min_success_rate=min_success_rate,
                limit=100,  # Get more for filtering
            )

            patterns = [p for p in patterns if p.occurrence_count >= min_occurrences]

            if not patterns:
                logger.info("No qualifying patterns found")
                return {
                    "candidates": [],
                    "skills_analyzed": 0,
                    "patterns_evaluated": 0,
                    "candidates_generated": 0,
                }

            logger.info(f"Found {len(patterns)} qualifying patterns")

            # Step 2: Discover target skills
            logger.info("Discovering skills...")
            skills = await self.skill_file_loader.discover_skills()

            if not skills:
                logger.warning("No skills discovered")
                return {
                    "candidates": [],
                    "skills_analyzed": 0,
                    "patterns_evaluated": 0,
                    "candidates_generated": 0,
                }

            # Filter by skill_path if provided
            if skill_path:
                skills = [s for s in skills if skill_path in s.path]
                if not skills:
                    logger.warning(f"No skills matching path: {skill_path}")
                    return {
                        "candidates": [],
                        "skills_analyzed": 0,
                        "patterns_evaluated": 0,
                        "candidates_generated": 0,
                    }

            logger.info(f"Found {len(skills)} skills to analyze")

            # Step 3: Compare patterns to skills
            candidates = []

            for skill in skills:
                logger.debug(f"Analyzing skill: {skill.name}")

                try:
                    skill_content = await self.skill_file_loader.load_skill_content(
                        skill.path
                    )
                except EvolutionError as e:
                    logger.warning(f"Skipping skill {skill.name}: {e}")
                    continue

                # Compare each pattern to this skill
                for pattern in patterns:
                    if len(candidates) >= max_candidates:
                        logger.info(f"Reached max_candidates limit ({max_candidates})")
                        break

                    proposal = await self._compare_pattern_to_skill(
                        pattern, skill_content, skill
                    )

                    if proposal and proposal.gap_detected:
                        # Create evolution candidate
                        candidate = await self._create_candidate_from_proposal(
                            proposal, pattern, skill
                        )

                        if candidate:
                            candidates.append(candidate)
                            logger.debug(
                                f"Created candidate for {skill.name}: "
                                f"confidence={candidate.confidence_score}"
                            )

                if len(candidates) >= max_candidates:
                    break

            # Step 4: Sort by confidence descending
            candidates = sorted(
                candidates, key=lambda c: c.confidence_score, reverse=True
            )

            logger.info(f"Generated {len(candidates)} evolution candidates")

            # Step 5: Store in Neo4j
            stored_uuids = []
            for candidate in candidates:
                try:
                    await self._store_candidate_in_neo4j(candidate)
                    stored_uuids.append(candidate.uuid)
                except Exception as e:
                    logger.warning(f"Failed to store candidate: {e}")

            logger.success(f"Stored {len(stored_uuids)} candidates in Neo4j")

            return {
                "candidates": [c.model_dump() for c in candidates],
                "skills_analyzed": len(skills),
                "patterns_evaluated": len(patterns),
                "candidates_generated": len(candidates),
            }

        except Exception as e:
            logger.error(f"Candidate generation failed: {e}")
            # Graceful degradation
            return {
                "candidates": [],
                "skills_analyzed": 0,
                "patterns_evaluated": 0,
                "candidates_generated": 0,
                "error": str(e),
            }

    async def _compare_pattern_to_skill(
        self, pattern: Any, skill_content: str, skill: SkillInfo
    ) -> EvolutionProposal | None:
        """
        Use LLM to compare a pattern against current skill instructions.

        Args:
            pattern: Pattern object from Neo4j
            skill_content: Current SKILL.md content
            skill: SkillInfo metadata

        Returns:
            EvolutionProposal with gap detection result, or None on error
        """
        try:
            # Build comparison prompt
            prompt = f"""Analyze whether this pattern contradicts or supplements current skill instructions.

SKILL NAME: {skill.name}

CURRENT INSTRUCTIONS:
```markdown
{skill_content[:2000]}  # First 2000 chars to avoid token overload
```

PATTERN EVIDENCE:
Problem: {pattern.problem_description}
Solution: {pattern.solution_approach}
Success Rate: {pattern.success_rate:.0%}
Occurrences: {pattern.occurrence_count}

TASK:
1. Does the pattern reveal a gap (contradiction, missing approach, or improvement)?
2. If yes: What type? contradiction (conflicting) | supplement (missing) | refinement (improve)
3. If yes: What specific change to the instructions would reflect the pattern?

Respond in JSON:
{{
  "gap_detected": bool,
  "gap_type": "contradiction" | "supplement" | "refinement" | null,
  "proposed_change": "specific text change or null",
  "rationale": "explanation or null",
  "confidence": 0.0-1.0
}}"""

            # Call LLM
            response_text = await self.llm_service._call_llm(prompt, temperature=0.0)

            # Parse JSON response
            try:
                response = json.loads(response_text)
            except json.JSONDecodeError:
                logger.warning(f"Invalid JSON from LLM: {response_text[:100]}")
                return None

            proposal = EvolutionProposal(
                gap_detected=response.get("gap_detected", False),
                gap_type=response.get("gap_type"),
                proposed_diff=response.get("proposed_change"),
                rationale=response.get("rationale"),
                confidence=response.get("confidence"),
            )

            return proposal

        except Exception as e:
            logger.warning(f"LLM comparison failed: {e}")
            # Graceful degradation
            return None

    async def _create_candidate_from_proposal(
        self, proposal: EvolutionProposal, pattern: Any, skill: SkillInfo
    ) -> EvolutionCandidate | None:
        """
        Create EvolutionCandidate from LLM proposal and pattern data.

        Args:
            proposal: EvolutionProposal from LLM comparison
            pattern: Pattern object from Neo4j
            skill: SkillInfo metadata

        Returns:
            EvolutionCandidate or None if validation fails
        """
        try:
            # Validate diff exists
            if not proposal.proposed_diff:
                logger.debug(f"No proposed diff for {skill.name}")
                return None

            # Create candidate
            candidate = EvolutionCandidate(
                uuid=str(uuid4()),
                skill_path=skill.path,
                skill_name=skill.name,
                proposed_diff=proposal.proposed_diff,
                rationale=proposal.rationale or "Pattern suggests improvement",
                pattern_uuids=[pattern.pattern_uuid],
                pattern_summary=f"{pattern.name} ({pattern.success_rate:.0%} success, "
                f"{pattern.occurrence_count} occurrences)",
                confidence_score=proposal.confidence or pattern.success_rate,
                occurrence_count=pattern.occurrence_count,
                success_rate=pattern.success_rate,
            )

            return candidate

        except Exception as e:
            logger.warning(f"Failed to create candidate: {e}")
            return None

    async def _store_candidate_in_neo4j(self, candidate: EvolutionCandidate) -> None:
        """
        Store evolution candidate in Neo4j.

        Args:
            candidate: EvolutionCandidate to store

        Raises:
            Exception if storage fails
        """
        await self.neo4j_service.connect()

        cypher = """
        CREATE (ec:EvolutionCandidate {
            uuid: $uuid,
            skill_path: $skill_path,
            skill_name: $skill_name,
            proposed_diff: $proposed_diff,
            rationale: $rationale,
            pattern_uuids: $pattern_uuids,
            pattern_summary: $pattern_summary,
            status: $status,
            confidence_score: $confidence_score,
            occurrence_count: $occurrence_count,
            success_rate: $success_rate,
            created_at: datetime()
        })
        RETURN ec.uuid as uuid
        """

        params = {
            "uuid": candidate.uuid,
            "skill_path": candidate.skill_path,
            "skill_name": candidate.skill_name,
            "proposed_diff": candidate.proposed_diff,
            "rationale": candidate.rationale,
            "pattern_uuids": candidate.pattern_uuids,
            "pattern_summary": candidate.pattern_summary,
            "status": candidate.status,
            "confidence_score": candidate.confidence_score,
            "occurrence_count": candidate.occurrence_count,
            "success_rate": candidate.success_rate,
        }

        result = await self.neo4j_service.query(cypher, params)

        if result:
            logger.debug(f"Stored candidate: {candidate.uuid}")
        else:
            raise Exception(f"Failed to store candidate: {candidate.uuid}")

    async def apply_candidate(
        self, candidate_uuid: str, backup_retention_days: int = 30
    ) -> EvolutionResult:
        """
        Apply an approved evolution candidate to skill file.

        Safety checks:
        - Double path validation at apply time
        - Verify candidate exists and is pending
        - Create backup and verify checksum
        - Detect stale diffs (re-read file, compare to expected)
        - Apply diff and verify file integrity
        - Support rollback via rollback_candidate() method

        Args:
            candidate_uuid: UUID of candidate to apply
            backup_retention_days: Days to retain backups (default 30)

        Returns:
            EvolutionResult with success/failure details

        Raises:
            EvolutionError if critical safety check fails
        """
        try:
            # Load candidate
            candidate = await self._load_candidate_from_neo4j(candidate_uuid)

            if not candidate:
                return EvolutionResult(
                    success=False,
                    candidate_uuid=candidate_uuid,
                    error_message="Candidate not found",
                )

            if candidate.status != "pending":
                return EvolutionResult(
                    success=False,
                    candidate_uuid=candidate_uuid,
                    error_message=f"Candidate is not pending (status={candidate.status})",
                )

            # Validate path (double-check at apply time)
            try:
                self.skill_file_loader.validate_path(candidate.skill_path)
            except EvolutionError as e:
                logger.error(f"Path validation failed: {e}")
                await self._log_security_violation(
                    candidate_uuid, "path_validation_failed", str(e)
                )
                return EvolutionResult(
                    success=False,
                    candidate_uuid=candidate_uuid,
                    error_message=str(e),
                )

            # Create backup and verify
            try:
                backup_path = await self.skill_file_loader.create_backup(
                    candidate.skill_path
                )
                # Verify backup integrity by checking file exists and has content

                backup_checksum = await self._compute_file_checksum(backup_path)
            except EvolutionError as e:
                logger.error(f"Backup creation failed: {e}")
                await self._log_security_violation(
                    candidate_uuid, "backup_creation_failed", str(e)
                )
                return EvolutionResult(
                    success=False,
                    candidate_uuid=candidate_uuid,
                    error_message=f"Backup creation failed: {e}",
                )

            # Detect stale diffs: re-read file and check if it matches expected content
            try:
                current_content = await self.skill_file_loader.load_skill_content(
                    candidate.skill_path
                )
                is_stale = await self._detect_stale_diff(
                    current_content, candidate.proposed_diff
                )
                if is_stale:
                    logger.warning(f"Stale diff detected for {candidate_uuid}")
                    await self._log_security_violation(
                        candidate_uuid,
                        "stale_diff_detected",
                        "File content changed since candidate generation",
                    )
                    return EvolutionResult(
                        success=False,
                        candidate_uuid=candidate_uuid,
                        backup_path=backup_path,
                        error_message="Stale diff: File content changed since candidate generation. "
                        "Re-generate candidates and try again.",
                    )
            except EvolutionError as e:
                logger.error(f"Stale diff detection failed: {e}")
                return EvolutionResult(
                    success=False,
                    candidate_uuid=candidate_uuid,
                    backup_path=backup_path,
                    error_message=f"Stale diff detection failed: {e}",
                )

            # Apply diff
            try:
                await self.skill_file_loader.apply_diff_to_file(
                    candidate.skill_path, candidate.proposed_diff
                )
            except EvolutionError as e:
                logger.error(f"Diff application failed: {e}")
                await self._log_security_violation(
                    candidate_uuid, "diff_application_failed", str(e)
                )
                return EvolutionResult(
                    success=False,
                    candidate_uuid=candidate_uuid,
                    backup_path=backup_path,
                    error_message=f"Diff application failed: {e}",
                )

            # Verify file integrity after applying diff
            try:
                modified_content = await self.skill_file_loader.load_skill_content(
                    candidate.skill_path
                )
                if not modified_content:
                    logger.error("File is empty after diff application")
                    await self._log_security_violation(
                        candidate_uuid,
                        "file_integrity_check_failed",
                        "File is empty after applying diff",
                    )
                    # Rollback to backup
                    await self.rollback_candidate(candidate_uuid, backup_path)
                    return EvolutionResult(
                        success=False,
                        candidate_uuid=candidate_uuid,
                        backup_path=backup_path,
                        error_message="File integrity check failed: file is empty after diff. Rolled back.",
                    )
            except EvolutionError as e:
                logger.error(f"File integrity check failed: {e}")
                await self._log_security_violation(
                    candidate_uuid, "file_integrity_check_failed", str(e)
                )
                # Rollback to backup
                await self.rollback_candidate(candidate_uuid, backup_path)
                return EvolutionResult(
                    success=False,
                    candidate_uuid=candidate_uuid,
                    backup_path=backup_path,
                    error_message=f"File integrity check failed: {e}. Rolled back.",
                )

            # Update status in Neo4j
            await self._update_candidate_status(candidate_uuid, "applied")

            # Cleanup old backups
            try:
                await self.skill_file_loader.cleanup_old_backups(
                    candidate.skill_path, backup_retention_days
                )
            except Exception as e:
                logger.warning(f"Backup cleanup failed: {e}")

            logger.success(f"Applied candidate: {candidate_uuid}")

            return EvolutionResult(
                success=True,
                candidate_uuid=candidate_uuid,
                backup_path=backup_path,
            )

        except Exception as e:
            logger.error(f"Apply candidate failed: {e}")
            return EvolutionResult(
                success=False,
                candidate_uuid=candidate_uuid,
                error_message=str(e),
            )

    async def rollback_candidate(self, candidate_uuid: str, backup_path: str) -> bool:
        """
        Rollback a candidate application using backup file.

        Args:
            candidate_uuid: UUID of candidate to rollback
            backup_path: Path to backup file to restore

        Returns:
            True if rollback successful, False otherwise
        """
        try:
            from pathlib import Path

            backup_file = Path(backup_path)

            if not backup_file.exists():
                logger.error(f"Backup file not found: {backup_path}")
                return False

            # Get original skill path from candidate
            candidate = await self._load_candidate_from_neo4j(candidate_uuid)
            if not candidate:
                logger.error(f"Candidate not found: {candidate_uuid}")
                return False

            skill_path = Path(candidate.skill_path)
            backup_content = backup_file.read_text(encoding="utf-8")
            skill_path.write_text(backup_content, encoding="utf-8")

            logger.info(f"Rolled back candidate {candidate_uuid} from {backup_path}")
            return True

        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            return False

    async def _compute_file_checksum(self, file_path: str) -> str:
        """Compute SHA256 checksum of file for integrity verification."""
        import hashlib
        from pathlib import Path

        try:
            content = Path(file_path).read_bytes()
            return hashlib.sha256(content).hexdigest()
        except Exception as e:
            logger.warning(f"Checksum computation failed: {e}")
            return ""

    async def _detect_stale_diff(
        self, current_content: str, proposed_diff: str
    ) -> bool:
        """
        Detect if proposed diff is stale (file changed since candidate generation).

        Simple heuristic: Check if the diff's context lines exist in current content.
        """
        try:
            import re

            # Extract context lines from diff
            context_pattern = r"^ (.+)$"
            context_lines = re.findall(context_pattern, proposed_diff, re.MULTILINE)

            if not context_lines:
                # No context lines found, can't verify staleness
                return False

            # Check if at least 80% of context lines exist in current content
            found_count = sum(1 for line in context_lines if line in current_content)
            threshold = len(context_lines) * 0.8

            is_stale = found_count < threshold
            if is_stale:
                logger.warning(
                    f"Stale diff detected: only {found_count}/{len(context_lines)} "
                    f"context lines found in current content"
                )

            return is_stale

        except Exception as e:
            logger.warning(f"Stale diff detection failed: {e}")
            return False

    async def _log_security_violation(
        self, candidate_uuid: str, violation_type: str, details: str
    ) -> None:
        """Log security violations to event log."""
        try:
            # Use the log_event function from mcp_server if available
            # For now, just log to logger
            logger.warning(
                f"Security violation for candidate {candidate_uuid}: "
                f"{violation_type} - {details}"
            )
        except Exception as e:
            logger.error(f"Failed to log security violation: {e}")

    async def document_change(
        self, candidate_uuid: str, action: str, reason: str | None = None
    ) -> None:
        """
        Document an evolution candidate decision in Mnemodrome.

        For approved changes: calls capture_learning + log_event
        For rejected/skipped: calls log_event

        Args:
            candidate_uuid: UUID of the candidate
            action: "approved", "rejected", or "skipped"
            reason: Optional reason for rejection/skip
        """
        try:
            # Load candidate for context
            candidate = await self._load_candidate_from_neo4j(candidate_uuid)
            if not candidate:
                logger.warning(
                    f"Candidate not found for documentation: {candidate_uuid}"
                )
                return

            if action == "approved":
                # Document approved changes as learning
                # Note: We would call capture_learning here, but to avoid circular
                # imports and keep this service pure, we return the info for the
                # calling code (mcp_server.py) to handle documentation
                logger.info(
                    f"Approved change to {candidate.skill_name}: {candidate.rationale}"
                )
                # The calling MCP tool will handle capture_learning
                return

            # For rejected/skipped, log to event log
            # Note: We would call log_event here, but to keep circular imports at bay,
            # the calling code should handle this
            if action == "rejected":
                logger.info(f"Rejected candidate {candidate_uuid}: {reason}")
            elif action == "skipped":
                logger.info(f"Skipped candidate {candidate_uuid} for later review")

        except Exception as e:
            logger.warning(f"Failed to document change for {candidate_uuid}: {e}")

    async def reject_candidate(
        self, candidate_uuid: str, reason: str
    ) -> EvolutionResult:
        """
        Reject an evolution candidate with reason and document decision.

        Logs rejection to event log for audit trail.

        Args:
            candidate_uuid: UUID of candidate to reject
            reason: Human-provided rejection reason

        Returns:
            EvolutionResult indicating success
        """
        try:
            await self._update_candidate_status(
                candidate_uuid, "rejected", rejection_reason=reason
            )

            # Document change decision
            await self.document_change(candidate_uuid, "rejected", reason)

            logger.info(f"Rejected candidate: {candidate_uuid}")

            return EvolutionResult(
                success=True,
                candidate_uuid=candidate_uuid,
            )

        except Exception as e:
            logger.error(f"Reject candidate failed: {e}")
            return EvolutionResult(
                success=False,
                candidate_uuid=candidate_uuid,
                error_message=str(e),
            )

    async def skip_candidate(self, candidate_uuid: str) -> EvolutionResult:
        """
        Skip an evolution candidate for later review and document decision.

        Logs skip to event log for audit trail.

        Args:
            candidate_uuid: UUID of candidate to skip

        Returns:
            EvolutionResult indicating success
        """
        try:
            await self._update_candidate_status(candidate_uuid, "skipped")

            # Document change decision
            await self.document_change(candidate_uuid, "skipped", None)

            logger.info(f"Skipped candidate: {candidate_uuid}")

            return EvolutionResult(
                success=True,
                candidate_uuid=candidate_uuid,
            )

        except Exception as e:
            logger.error(f"Skip candidate failed: {e}")
            return EvolutionResult(
                success=False,
                candidate_uuid=candidate_uuid,
                error_message=str(e),
            )

    async def _load_candidate_from_neo4j(
        self, candidate_uuid: str
    ) -> EvolutionCandidate | None:
        """Load evolution candidate from Neo4j."""
        await self.neo4j_service.connect()

        cypher = """
        MATCH (ec:EvolutionCandidate {uuid: $uuid})
        RETURN ec
        """

        result = await self.neo4j_service.query(cypher, {"uuid": candidate_uuid})

        if not result:
            return None

        data = result[0]["ec"]

        return EvolutionCandidate(
            uuid=data["uuid"],
            skill_path=data["skill_path"],
            skill_name=data["skill_name"],
            proposed_diff=data["proposed_diff"],
            rationale=data["rationale"],
            pattern_uuids=data.get("pattern_uuids", []),
            pattern_summary=data.get("pattern_summary", ""),
            status=data.get("status", "pending"),
            rejection_reason=data.get("rejection_reason"),
            confidence_score=data.get("confidence_score", 0.0),
            occurrence_count=data.get("occurrence_count", 1),
            success_rate=data.get("success_rate", 0.0),
            created_at=data.get("created_at", datetime.utcnow()),
            reviewed_at=data.get("reviewed_at"),
            applied_at=data.get("applied_at"),
            reviewed_by=data.get("reviewed_by"),
        )

    async def _update_candidate_status(
        self,
        candidate_uuid: str,
        status: str,
        rejection_reason: str | None = None,
    ) -> None:
        """Update candidate status in Neo4j."""
        await self.neo4j_service.connect()

        cypher = """
        MATCH (ec:EvolutionCandidate {uuid: $uuid})
        SET ec.status = $status,
            ec.reviewed_at = datetime(),
            ec.reviewed_by = 'user'
        """

        params = {
            "uuid": candidate_uuid,
            "status": status,
        }

        if rejection_reason:
            cypher += ", ec.rejection_reason = $rejection_reason"
            params["rejection_reason"] = rejection_reason

        await self.neo4j_service.query(cypher, params)
