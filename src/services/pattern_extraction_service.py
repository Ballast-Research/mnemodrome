"""
Pattern Extraction Service for Feature 026.

Extracts recurring patterns from session summaries using LLM analysis.
Handles pattern structuring, confidence scoring, and validation.
"""

import json

from loguru import logger

from src.config import PATTERN_RECOGNITION_ENABLED
from src.models.pattern_models import ExtractedPattern
from src.services.llm_service import LLMService


class PatternExtractionService:
    """
    Extract patterns from session summaries using LLM analysis.

    Converts unstructured session text into structured patterns with
    confidence scores and outcome classification.
    """

    def __init__(self):
        """Initialize pattern extraction service with LLM dependency."""
        self.llm_service = LLMService()
        self.enabled = PATTERN_RECOGNITION_ENABLED

    async def extract_patterns_from_sessions(
        self, session_summaries: list[dict]
    ) -> list[ExtractedPattern]:
        """
        Extract patterns from multiple session summaries.

        Args:
            session_summaries: List of session dicts with 'uuid', 'conversation_summary',
                             and 'key_insights' fields

        Returns:
            List of ExtractedPattern objects with confidence scores

        Raises:
            ValueError: If session_summaries is empty or invalid
        """
        if not self.enabled:
            logger.debug("Pattern recognition disabled, returning empty list")
            return []

        if not session_summaries:
            raise ValueError("session_summaries cannot be empty")

        all_patterns = []

        for session in session_summaries:
            try:
                patterns = await self._extract_patterns_from_session(session)
                all_patterns.extend(patterns)
                logger.debug(
                    f"Extracted {len(patterns)} patterns from session {session.get('uuid', 'unknown')}"
                )
            except Exception as e:
                logger.error(
                    f"Error extracting patterns from session {session.get('uuid', 'unknown')}: {e}"
                )
                continue

        logger.info(
            f"Total patterns extracted: {len(all_patterns)} from {len(session_summaries)} sessions"
        )
        return all_patterns

    async def _extract_patterns_from_session(
        self, session: dict
    ) -> list[ExtractedPattern]:
        """
        Extract patterns from a single session.

        Args:
            session: Dict with 'uuid', 'conversation_summary', 'key_insights'

        Returns:
            List of ExtractedPattern objects
        """
        session_uuid = session.get("uuid")
        summary = session.get("conversation_summary", "")
        insights = session.get("key_insights", [])

        if not summary:
            logger.warning(f"Session {session_uuid} has empty conversation_summary")
            return []

        # Combine summary and insights into single text for LLM
        combined_text = (
            f"{summary}\n\nKey Insights:\n"
            + "\n".join([f"- {insight}" for insight in insights])
            if insights
            else summary
        )

        # Limit text size for API
        combined_text = combined_text[:2000]

        # Build extraction prompt
        prompt = self._build_extraction_prompt(combined_text)

        try:
            # Call LLM for pattern extraction
            response = await self.llm_service._call_llm(prompt)
            patterns = self._parse_extraction_response(response, session_uuid)
            return patterns
        except Exception as e:
            logger.error(f"LLM call failed for session {session_uuid}: {e}")
            return []

    def _build_extraction_prompt(self, session_text: str) -> str:
        """
        Build the extraction prompt for LLM.

        Args:
            session_text: Combined session summary and insights

        Returns:
            Formatted prompt for pattern extraction
        """
        return f"""Analyze the following session summary and extract recurring patterns.

SESSION SUMMARY:
{session_text}

Task: Extract all distinct patterns/approaches mentioned in this session. For each pattern:

1. Identify the problem type (choose one: infrastructure, code, architecture, deployment, data, integration, testing, other)
2. Describe what problem was being solved
3. Describe what approach/solution was attempted
4. Classify the outcome (SUCCESS, FAILED, PARTIAL, INCONCLUSIVE)
5. Assign a confidence score (0.0-1.0) for how clearly this pattern is described
6. Extract any metrics (latency improvement, error rate change, etc.) if mentioned

Return JSON array with this structure:
[
  {{
    "pattern_type": "infrastructure|code|architecture|deployment|data|integration|testing|other",
    "problem_description": "clear description of the problem",
    "solution_approach": "description of the solution attempted",
    "outcome": "SUCCESS|FAILED|PARTIAL|INCONCLUSIVE",
    "confidence": 0.85,
    "metrics": {{"metric_name": value}} or null
  }}
]

Important:
- Extract multiple patterns if the session describes multiple distinct problems/solutions
- Only include patterns with clear problem statements and solution attempts
- Confidence should reflect clarity of the problem/solution description
- Be conservative with confidence - penalize ambiguity
- Return empty array [] if no clear patterns found"""

    def _parse_extraction_response(
        self, response: str, session_uuid: str
    ) -> list[ExtractedPattern]:
        """
        Parse JSON response from LLM into ExtractedPattern objects.

        Args:
            response: LLM response text
            session_uuid: UUID of source session

        Returns:
            List of validated ExtractedPattern objects
        """
        try:
            # Extract JSON from response
            # LLM might return markdown code blocks
            json_str = response
            if "```json" in json_str:
                json_str = json_str.split("```json")[1].split("```")[0]
            elif "```" in json_str:
                json_str = json_str.split("```")[1].split("```")[0]

            patterns_data = json.loads(json_str)
            if not isinstance(patterns_data, list):
                logger.warning(f"Expected JSON array, got {type(patterns_data)}")
                return []

            patterns = []
            for data in patterns_data:
                try:
                    # Add session UUID to pattern data
                    data["session_uuid"] = session_uuid

                    # Validate and create model
                    pattern = ExtractedPattern(**data)

                    # Filter low-confidence patterns (below 0.5)
                    if pattern.confidence >= 0.5:
                        patterns.append(pattern)
                    else:
                        logger.debug(
                            f"Skipping low-confidence pattern (confidence={pattern.confidence})"
                        )

                except ValueError as e:
                    logger.warning(f"Invalid pattern data: {e}")
                    continue

            return patterns

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            logger.debug(f"Response was: {response[:200]}")
            return []
        except Exception as e:
            logger.error(f"Error parsing extraction response: {e}")
            return []

    def calculate_confidence_score(
        self,
        base_confidence: float,
        clarity_score: float = 1.0,
        specificity_boost: float = 1.0,
        recency_weight: float = 1.0,
    ) -> float:
        """
        Calculate refined confidence score using multiple factors.

        Args:
            base_confidence: Extraction LLM confidence (0.0-1.0)
            clarity_score: How clear is the description (0.5-1.0)
            specificity_boost: Boost for specific metrics/details (1.0-1.2)
            recency_weight: Weight for recent patterns (0.8-1.0)

        Returns:
            Final confidence score (0.0-1.0)
        """
        score = base_confidence * clarity_score * specificity_boost * recency_weight
        return min(1.0, max(0.0, score))
