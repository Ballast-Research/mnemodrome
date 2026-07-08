"""
LLM Service - Unified interface for entity extraction and classification (T053)

Supports multiple LLM providers configurable via .env:
- Anthropic (Claude)
- OpenAI (GPT-4)
- Ollama (local models)
- openai-compatible (OpenRouter, xAI/Grok, Groq, Together, DeepSeek, Mistral, vLLM, SGLang, LM Studio, …)

Environment Configuration:
  LLM_PROVIDER=anthropic|openai|ollama|openai-compatible
  CHAT_MODEL=claude-3-5-sonnet-20241022|gpt-4|llama3.1|mistral-7b-instruct
  LLM_API_KEY=<api_key>      # Not required for Ollama; required for all others
  LLM_BASE_URL=<url>         # Required for ollama and openai-compatible (e.g. http://localhost:11434)
  OLLAMA_BASE_URL=<url>      # Back-compat alias for LLM_BASE_URL (existing installs)
  LLM_TEMPERATURE=0.0
  LLM_MAX_TOKENS=1000

Optional per-role model overrides (both default to CHAT_MODEL when unset):
  EXTRACTION_MODEL=<model>      # Model used for entity/claim extraction
  CLASSIFICATION_MODEL=<model>  # Model used for entity classification
"""

import asyncio
import functools
import json
import re
import time

from loguru import logger

from src.config import (
    CHAT_MODEL,
    CLASSIFICATION_MODEL,
    EXTRACTION_MODEL,
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_DISABLE_THINKING,
    LLM_MAX_RETRIES,
    LLM_MAX_TOKENS,
    LLM_PROVIDER,
    LLM_TEMPERATURE,
    LLM_TIMEOUT,
    OLLAMA_FALLBACK_URL,
)

# gpu_cache_manager disabled — job manager handles GPU dispatch and KEEP_ALIVE=5s handles eviction


class LLMService:
    """Unified LLM interface for entity extraction and classification."""

    def __init__(self):
        self.provider = LLM_PROVIDER
        self.model = CHAT_MODEL
        # Per-role model overrides; both equal self.model when EXTRACTION_MODEL /
        # CLASSIFICATION_MODEL are not set in the environment (back-compat default).
        self.extraction_model = EXTRACTION_MODEL
        self.classification_model = CLASSIFICATION_MODEL
        self.api_key = LLM_API_KEY
        self.base_url = LLM_BASE_URL
        self.fallback_url = OLLAMA_FALLBACK_URL
        self.temperature = LLM_TEMPERATURE
        self.max_tokens = LLM_MAX_TOKENS
        self.timeout = LLM_TIMEOUT
        self.max_retries = LLM_MAX_RETRIES
        self.disable_thinking = LLM_DISABLE_THINKING

        # Lazy initialization - client created on first use
        self.client = None
        self._fallback_client = None
        self._using_fallback = False
        self._initialized = False
        self._initialization_error = None

        # Observability: counts how many times each LLM stage fell back to a
        # degraded result because the model returned empty or unparseable
        # output.  Non-zero means real data was lost silently.  Extraction and
        # classification are counted separately — a single-label fine-tuned
        # classifier that cannot emit a batch JSON array (and correctly falls
        # back to sequential classification) must not read as an extraction
        # failure.  Both are exposed via health_check() and the /health endpoint.
        self.extraction_degraded_count = 0
        self.classification_degraded_count = 0

    def _initialize_client(self):
        """
        Initialize LLM provider client based on configuration.

        Lazy initialization - only called on first LLM use.
        Stores initialization error to avoid repeated attempts.
        """
        if self._initialized:
            return

        try:
            if self.provider == "anthropic":
                try:
                    from anthropic import Anthropic

                    if not self.api_key:
                        raise ValueError("LLM_API_KEY required for Anthropic provider")
                    self.client = Anthropic(api_key=self.api_key)
                    logger.info(f"LLM service initialized: Anthropic ({self.model})")
                except ImportError:
                    self._initialization_error = (
                        "anthropic package not installed. Run: pip install anthropic"
                    )
                    raise ImportError(self._initialization_error)

            elif self.provider == "openai":
                try:
                    from openai import OpenAI

                    if not self.api_key:
                        raise ValueError("LLM_API_KEY required for OpenAI provider")
                    self.client = OpenAI(api_key=self.api_key)
                    logger.info(f"LLM service initialized: OpenAI ({self.model})")
                except ImportError:
                    self._initialization_error = (
                        "openai package not installed. Run: pip install openai"
                    )
                    raise ImportError(self._initialization_error)

            elif self.provider == "ollama":
                try:
                    from openai import OpenAI

                    # Ensure base_url ends with /v1 for OpenAI SDK compatibility
                    base_url = self.base_url or "http://localhost:11434"
                    if not base_url.endswith("/v1"):
                        base_url = base_url.rstrip("/") + "/v1"
                    self.client = OpenAI(
                        base_url=base_url,
                        api_key="ollama",  # Ollama doesn't require real API key
                        timeout=self.timeout,  # Issue #28: Add timeout protection
                        max_retries=self.max_retries,  # Issue #28: Retry transient failures
                        default_headers={
                            "X-Dispatch-Strategy": "burst",
                            "X-Client-ID": "mnemodrome-llm",
                        },
                    )
                    logger.info(
                        f"LLM service initialized: Ollama ({self.model} at {base_url}, "
                        f"timeout={self.timeout}s, max_retries={self.max_retries})"
                    )
                    # Initialize fallback client if configured
                    if self.fallback_url:
                        fallback_base = self.fallback_url
                        if not fallback_base.endswith("/v1"):
                            fallback_base = fallback_base.rstrip("/") + "/v1"
                        self._fallback_client = OpenAI(
                            base_url=fallback_base,
                            api_key="ollama",
                            timeout=self.timeout,
                            max_retries=self.max_retries,
                            default_headers={
                                "X-Dispatch-Strategy": "burst",
                                "X-Client-ID": "mnemodrome-llm-fallback",
                            },
                        )
                        logger.info(f"Ollama fallback client ready: {fallback_base}")
                except ImportError:
                    self._initialization_error = (
                        "openai package required for Ollama. Run: pip install openai"
                    )
                    raise ImportError(self._initialization_error)

            elif self.provider == "openai-compatible":
                try:
                    from openai import OpenAI

                    if not self.api_key:
                        raise ValueError(
                            "LLM_API_KEY required for openai-compatible provider"
                        )
                    if not self.base_url:
                        raise ValueError(
                            "LLM_BASE_URL required for openai-compatible provider"
                        )
                    self.client = OpenAI(
                        base_url=self.base_url,
                        api_key=self.api_key,
                        timeout=self.timeout,
                        max_retries=self.max_retries,
                    )
                    logger.info(
                        f"LLM service initialized: openai-compatible ({self.model} at {self.base_url})"
                    )
                except ImportError:
                    self._initialization_error = "openai package required for openai-compatible. Run: pip install openai"
                    raise ImportError(self._initialization_error)

            else:
                self._initialization_error = (
                    f"Unsupported LLM provider: {self.provider}"
                )
                raise ValueError(self._initialization_error)

            self._initialized = True

            # Log the resolved model trio so operators always know which model
            # handles each role.  A single compact line is emitted when all
            # three roles map to the same model (the common single-model case).
            if (
                self.extraction_model == self.model
                and self.classification_model == self.model
            ):
                logger.info(
                    f"LLM models — chat/extraction/classification: {self.model}"
                )
            else:
                logger.info(
                    f"LLM models — chat: {self.model}, "
                    f"extraction: {self.extraction_model}, "
                    f"classification: {self.classification_model}"
                )

        except Exception as e:
            self._initialization_error = str(e)
            raise

    async def extract_entities_and_relationships(self, context: str) -> dict:
        """
        Extract entities and relationships from context using LLM.

        Replaces heuristic extraction in graph_factory._extract_with_llm()

        Args:
            context: Natural language text

        Returns:
            {
                "entities": [{"name": str, "role": str}, ...],
                "relationships": [{"source": str, "relation": str, "target": str}, ...]
            }
        """
        prompt = f"""Extract ALL entities and relationships from this context.

Context: {context}

Return a JSON object with this exact structure:
{{
  "entities": [
    {{"name": "Entity Name", "role": "subject|object|entity"}},
    ...
  ],
  "relationships": [
    {{"source": "Entity1", "relation": "RELATION_TYPE", "target": "Entity2"}},
    ...
  ]
}}

Rules:
1. Extract ALL meaningful entities (people, projects, technologies, concepts, etc.)
2. Preserve multi-word entity names exactly (e.g., "Gang of Four", not "Gang")
3. Use clear, uppercase relationship types (e.g., "USES", "IMPLEMENTS", "COLLABORATED_WITH")
4. Only include relationships explicitly mentioned or strongly implied
5. Use "role": "subject" for actors, "object" for acted-upon, "entity" for neutral

Return ONLY the JSON, no additional text."""

        # Lazy initialize on first use
        if not self._initialized:
            self._initialize_client()

        try:
            response = await self._call_llm(
                prompt, model=self.extraction_model, json_mode=True
            )

            # Parse JSON response
            try:
                # Strip markdown code fences if present (some models like Gemma wrap JSON)
                cleaned_response = response.strip()
                if cleaned_response.startswith("```json"):
                    cleaned_response = cleaned_response[7:]  # Remove ```json
                if cleaned_response.startswith("```"):
                    cleaned_response = cleaned_response[3:]  # Remove ```
                if cleaned_response.endswith("```"):
                    cleaned_response = cleaned_response[:-3]  # Remove trailing ```
                cleaned_response = cleaned_response.strip()

                # Defence: if there is still leading non-JSON text (e.g. a
                # model that ignores the no-think hint and emits prose before
                # the object), skip to the first '{' or '['.
                json_start = min(
                    (
                        cleaned_response.find(c)
                        for c in ("{", "[")
                        if cleaned_response.find(c) != -1
                    ),
                    default=-1,
                )
                if json_start > 0:
                    cleaned_response = cleaned_response[json_start:]

                # ── FAIL LOUD: empty response ──────────────────────────────
                if not cleaned_response:
                    self.extraction_degraded_count += 1
                    logger.warning(
                        f"LLM returned empty/unparseable output for 'entity_extraction' "
                        f"using model '{self.extraction_model}'. This commonly means a REASONING model "
                        f"is emitting hidden thinking tokens and exhausting the token budget. "
                        f"Fix: use a non-reasoning model, ensure LLM_DISABLE_THINKING=true, "
                        f"or raise LLM_MAX_TOKENS. Falling back to empty result "
                        f"(DEGRADED extraction quality)."
                    )
                    return {"entities": [], "relationships": []}

                result = json.loads(cleaned_response)

                # Validate structure
                if "entities" not in result or "relationships" not in result:
                    raise ValueError("Missing required fields in LLM response")

                logger.debug(
                    f"LLM extracted {len(result['entities'])} entities, "
                    f"{len(result['relationships'])} relationships"
                )

                return result

            except json.JSONDecodeError as e:
                # ── FAIL LOUD: unparseable JSON ────────────────────────────
                self.extraction_degraded_count += 1
                logger.warning(
                    f"LLM returned empty/unparseable output for 'entity_extraction' "
                    f"using model '{self.extraction_model}'. This commonly means a REASONING model "
                    f"is emitting hidden thinking tokens and exhausting the token budget. "
                    f"Fix: use a non-reasoning model, ensure LLM_DISABLE_THINKING=true, "
                    f"or raise LLM_MAX_TOKENS. Falling back to empty result "
                    f"(DEGRADED extraction quality). Parse error: {e}"
                )
                logger.debug(f"Raw response: {response}")
                # Fallback to empty extraction
                return {"entities": [], "relationships": []}

        except Exception as e:
            logger.error(f"LLM extraction failed: {e}")
            raise

    # Known-good probe for the startup canary. A non-reasoning instruct model
    # returns at least one entity from this sentence; a reasoning model that
    # emits hidden thinking instead of JSON returns none.
    CANARY_TEXT = (
        "Alice Johnson, a senior engineer at Acme Corporation in Paris, works "
        "with her colleague Bob Chen to build a recommendation system using "
        "PostgreSQL and Redis."
    )
    CANARY_MIN_ENTITIES = 2
    # Headroom a reasoning model needs to emit JSON *after* its thinking, used
    # as the last self-heal step before giving up.
    CANARY_HEALED_MAX_TOKENS = 4096

    async def extraction_canary(self) -> dict:
        """
        Prove the configured extraction model actually returns entities before
        the server starts serving — the safeguard against the silent
        reasoning-model trap, where every capture stores nothing.

        Self-healing escalation (so a healthy model never fails the install):
          1. Run extraction with the current settings.
          2. If empty and thinking is not yet suppressed (ollama /
             openai-compatible only): enable no-think, retry.
          3. If still empty: raise the token budget — reasoning models exhaust
             it on hidden thinking before emitting JSON — and retry.
          4. If still empty: report failure; the caller decides hard-fail vs
             warn.

        Any setting that fixes it is persisted on this singleton AND surfaced in
        the return value so the operator can make it permanent in .env.

        A model that cannot be REACHED is reported as an error (connectivity),
        which is distinct from a model that responds but yields no entities (the
        trap). Probe attempts do not pollute extraction_degraded_count.

        Returns:
            {
              "ok": bool,            # entities came back
              "healed": bool,        # a setting had to be auto-adjusted
              "entity_count": int,
              "adjustments": [str],  # human-readable settings changed
              "model": str,
              "error": str | None,   # set only when the model was unreachable
            }
        """
        adjustments: list[str] = []
        degraded_before = self.extraction_degraded_count

        def _result(ok: bool, count: int, error: str | None = None) -> dict:
            # Probe attempts must not inflate the real "data was lost" metric.
            self.extraction_degraded_count = degraded_before
            return {
                "ok": ok,
                "healed": bool(adjustments) and ok,
                "entity_count": count,
                "adjustments": adjustments,
                "model": self.extraction_model,
                "error": error,
            }

        async def _probe() -> int:
            result = await self.extract_entities_and_relationships(self.CANARY_TEXT)
            return len(result.get("entities", []))

        # Attempt 1: as configured. A raise here means the model is unreachable
        # (connectivity), not the trap.
        try:
            count = await _probe()
        except Exception as e:
            return _result(False, 0, error=f"extraction model unreachable: {e}")
        if count >= self.CANARY_MIN_ENTITIES:
            return _result(True, count)

        # Attempt 2: force no-think (only meaningful for local/compatible providers).
        if not self.disable_thinking and self.provider in (
            "ollama",
            "openai-compatible",
        ):
            self.disable_thinking = True
            adjustments.append("LLM_DISABLE_THINKING=true")
            logger.warning(
                "Extraction canary returned 0 entities — auto-enabling "
                "LLM_DISABLE_THINKING and retrying."
            )
            try:
                count = await _probe()
            except Exception:
                count = 0
            if count >= self.CANARY_MIN_ENTITIES:
                return _result(True, count)

        # Attempt 3: raise the token budget so a reasoning model still emits JSON
        # after its hidden thinking.
        if self.max_tokens < self.CANARY_HEALED_MAX_TOKENS:
            original_max_tokens = self.max_tokens
            self.max_tokens = self.CANARY_HEALED_MAX_TOKENS
            adjustments.append(
                f"LLM_MAX_TOKENS {original_max_tokens}->{self.max_tokens}"
            )
            logger.warning(
                f"Extraction canary still empty — raising LLM_MAX_TOKENS to "
                f"{self.max_tokens} and retrying."
            )
            try:
                count = await _probe()
            except Exception:
                count = 0
            if count >= self.CANARY_MIN_ENTITIES:
                return _result(True, count)

        # Reachable, but yields nothing usable even after self-heal — the trap.
        return _result(False, 0)

    async def classify_entity(
        self, name: str, context: str = "", role: str = "entity"
    ) -> str:
        """
        Classify entity type using LLM.

        Replaces heuristic classification in classification_service._heuristic_classify()

        Args:
            name: Entity name
            context: Context where entity appears
            role: Role hint (subject, object, entity)

        Returns:
            Entity type: Person, Project, Organization, Concept, Technology, Event, Location, Document, Entity
        """
        valid_types = [
            "Person",
            "Project",
            "Organization",
            "Concept",
            "Technology",
            "Event",
            "Location",
            "Document",
            "Entity",
        ]

        prompt = f"""Classify this entity into ONE of these types:
{", ".join(valid_types)}

Entity: {name}
Context: {context}
Role: {role}

Return ONLY the type name, no explanation.

Examples:
- "Alice" → Person
- "Neo4j" → Technology
- "Phase 4.5" → Project
- "Factory Pattern" → Concept
- "Gang of Four" → Organization (or Document if referring to the book)
- "entity duplication problem" → Event

Classification:"""

        # Lazy initialize on first use
        if not self._initialized:
            self._initialize_client()

        try:
            response = await self._call_llm(prompt, model=self.classification_model)

            # Clean response (remove quotes, whitespace)
            entity_type = response.strip().strip('"').strip("'")

            # ── FAIL LOUD: empty response ──────────────────────────────────
            if not entity_type:
                self.classification_degraded_count += 1
                logger.warning(
                    f"LLM returned empty/unparseable output for 'entity_classification' "
                    f"using model '{self.classification_model}'. This commonly means a REASONING model "
                    f"is emitting hidden thinking tokens and exhausting the token budget. "
                    f"Fix: use a non-reasoning model, ensure LLM_DISABLE_THINKING=true, "
                    f"or raise LLM_MAX_TOKENS. Falling back to 'Entity' "
                    f"(DEGRADED classification quality)."
                )
                return "Entity"

            # Validate response
            if entity_type not in valid_types:
                logger.warning(
                    f"LLM returned invalid type '{entity_type}' for '{name}', "
                    f"defaulting to 'Entity'"
                )
                return "Entity"

            logger.debug(f"LLM classified: {name} → {entity_type}")
            return entity_type

        except Exception as e:
            logger.error(f"LLM classification failed for '{name}': {e}")
            return "Entity"  # Fallback

    async def classify_entities_batch(
        self, entities: list[dict], context: str
    ) -> list[str]:
        """
        Classify multiple entities in a single LLM call (T057).

        Performance: O(1) vs O(n) for sequential classification.

        Sends all entities in a single LLM call for O(1) classification.

        Args:
            entities: List of {"name": str, "role": str} dicts
            context: Full context string

        Returns:
            List of entity type strings in same order as input

        Example:
            entities = [
                {"name": "Alice", "role": "subject"},
                {"name": "Neo4j", "role": "object"}
            ]
            types = await llm.classify_entities_batch(entities, context)
            # Returns: ["Person", "Technology"]
        """
        if not entities:
            return []

        # Lazy initialize on first use
        if not self._initialized:
            self._initialize_client()

        valid_types = [
            "Person",
            "Project",
            "Organization",
            "Concept",
            "Technology",
            "Event",
            "Location",
            "Document",
            "Entity",
        ]

        # Build batch classification prompt
        entities_list = "\n".join(
            f"{i + 1}. {e['name']} (role: {e.get('role', 'entity')})"
            for i, e in enumerate(entities)
        )

        prompt = f"""Classify each entity into ONE of these types:
{", ".join(valid_types)}

Context: {context}

Entities to classify:
{entities_list}

Return ONLY a JSON array of types in the same order:
["Type1", "Type2", ...]

Examples:
- "Alice" → Person
- "Neo4j" → Technology
- "Phase 4.5" → Project
- "Factory Pattern" → Concept

Classification:"""

        try:
            response = await self._call_llm(prompt, model=self.classification_model)

            # Parse JSON response (strip markdown fences)
            cleaned_response = response.strip()
            if cleaned_response.startswith("```json"):
                cleaned_response = cleaned_response[7:]
            if cleaned_response.startswith("```"):
                cleaned_response = cleaned_response[3:]
            if cleaned_response.endswith("```"):
                cleaned_response = cleaned_response[:-3]
            cleaned_response = cleaned_response.strip()

            # Defence: skip any leading non-JSON prose before the '['.
            json_start = cleaned_response.find("[")
            if json_start > 0:
                cleaned_response = cleaned_response[json_start:]

            # ── FAIL LOUD: empty response ──────────────────────────────────
            if not cleaned_response:
                # NOT a degradation: many classifiers — notably single-label
                # fine-tunes like entity-classifier-v5 — cannot emit a JSON
                # array. The sequential fallback below classifies each entity
                # correctly, so this is an expected optimization miss, not lost
                # data. Real classification degradation is counted in
                # classify_entity() when an individual classification fails.
                logger.debug(
                    f"Batch classification returned no JSON array from "
                    f"'{self.classification_model}'; using sequential per-entity "
                    f"classification instead."
                )
                return await self._sequential_classify_fallback(entities, context)

            # ── FAIL LOUD: unparseable JSON ────────────────────────────────
            try:
                types = json.loads(cleaned_response)
            except json.JSONDecodeError as parse_err:
                # NOT a degradation — see note above; the sequential fallback
                # classifies each entity correctly. Real degradation is counted
                # per-entity in classify_entity().
                logger.debug(
                    f"Batch classification from '{self.classification_model}' was not "
                    f"valid JSON ({parse_err}); using sequential per-entity "
                    f"classification instead."
                )
                return await self._sequential_classify_fallback(entities, context)

            # Validate response length
            if len(types) != len(entities):
                logger.warning(
                    f"Batch classification returned {len(types)} types for {len(entities)} entities, "
                    f"falling back to sequential"
                )
                return await self._sequential_classify_fallback(entities, context)

            # Validate all types are valid
            validated_types = []
            for t in types:
                if t in valid_types:
                    validated_types.append(t)
                else:
                    logger.warning(
                        f"Invalid type '{t}' in batch response, using 'Entity'"
                    )
                    validated_types.append("Entity")

            logger.debug(
                f"Batch classified {len(entities)} entities: {validated_types}"
            )
            return validated_types

        except Exception as e:
            logger.error(
                f"Batch classification failed: {e}, falling back to sequential"
            )
            return await self._sequential_classify_fallback(entities, context)

    async def health_check(self) -> dict:
        """
        Comprehensive health check including LLM connectivity and GPU cache status.

        Feature 030: Includes GPU cache management statistics.
        Issue #29: Validates LLM connectivity with short timeout to prevent hung operations.

        Use before calling capture_learning, capture_context, or other expensive MCP tools.

        Returns:
            Dict with health status, LLM status, and GPU cache stats

        Example:
            health = await llm_service.health_check()
            if not health.get("llm_healthy"):
                logger.error(f"LLM service unavailable: {health}")
                return {"status": "error", "message": "LLM service unavailable"}
        """
        health_status = {
            "llm_healthy": False,
            "provider": self.provider,
            "model": self.model,
            "gpu_cache_stats": {},  # gpu_cache_manager disabled — job manager handles GPU management
            "extraction_degraded_count": self.extraction_degraded_count,
            "classification_degraded_count": self.classification_degraded_count,
        }

        if not self._initialized:
            try:
                self._initialize_client()
            except Exception as e:
                logger.error(f"LLM health check failed during initialization: {e}")
                health_status["error"] = str(e)
                return health_status

        try:
            # Test with minimal prompt and 5s timeout
            test_prompt = "test"
            response = await asyncio.wait_for(self._call_llm(test_prompt), timeout=5.0)

            if response:
                logger.debug("LLM health check passed")
                health_status["llm_healthy"] = True
                return health_status
            else:
                logger.warning("LLM health check returned empty response")
                health_status["error"] = "Empty response"
                return health_status

        except TimeoutError:
            logger.error("LLM health check timed out after 5s")
            health_status["error"] = "Health check timeout"
            return health_status
        except Exception as e:
            logger.error(f"LLM health check failed: {e}")
            health_status["error"] = str(e)
            return health_status

    async def _sequential_classify_fallback(
        self, entities: list[dict], context: str
    ) -> list[str]:
        """
        Parallel classification fallback when batch fails.

        Sends all entity classification requests concurrently via asyncio.gather,
        leveraging burst dispatch to spread across multiple GPUs.

        Args:
            entities: List of {"name": str, "role": str} dicts
            context: Full context string

        Returns:
            List of entity type strings
        """
        logger.info(
            f"Using parallel classification fallback for {len(entities)} entities"
        )
        tasks = [
            self.classify_entity(
                name=entity["name"], context=context, role=entity.get("role", "entity")
            )
            for entity in entities
        ]
        return list(await asyncio.gather(*tasks))

    @staticmethod
    def _strip_thinking(text: str) -> str:
        """
        Strip hidden reasoning tokens from model output.

        Removes <think>...</think> blocks emitted by reasoning models
        (Qwen3, DeepSeek-R1, etc.) so that downstream JSON parsers and
        plain-text parsers never see raw reasoning tokens.  Applied to
        every LLM response before it is returned to callers.

        Args:
            text: Raw LLM response string

        Returns:
            Response with thinking blocks removed and leading/trailing
            whitespace stripped.
        """
        stripped = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        return stripped.strip()

    async def _call_llm(
        self, prompt: str, model: str | None = None, json_mode: bool = False
    ) -> str:
        """
        Call LLM provider with unified interface and performance monitoring.

        GPU management delegated to job manager (burst dispatch + KEEP_ALIVE=5s).
        Fallback client used if primary times out.

        BUG-FIX (event-loop blocking): All synchronous SDK calls are now
        wrapped in asyncio.to_thread so the event loop is never blocked
        during the full model round-trip.  This also makes asyncio.wait_for
        timeouts functional again.

        BUG-FIX (reasoning-model empty response): For ollama and
        openai-compatible providers, extra_body={"think": False} is passed
        when LLM_DISABLE_THINKING=true (default) to prevent hidden <think>
        tokens from exhausting the token budget.  A secondary defence
        (_strip_thinking) removes any <think> blocks that still leak through.

        Args:
            prompt: System/user prompt
            model: Optional model name override.  When provided, this model is
                   used instead of self.model for this single call.  Callers
                   use this to route extraction vs classification to different
                   fine-tuned models (EXTRACTION_MODEL / CLASSIFICATION_MODEL).
                   Defaults to None, which falls back to self.model (CHAT_MODEL)
                   — identical to the previous single-model behaviour.

        Returns:
            LLM response text (thinking blocks stripped)
        """
        # Resolve the model to use for this call.  When no override is passed
        # (the common case when only CHAT_MODEL is set) this is simply self.model,
        # so behaviour is completely unchanged for single-model deployments.
        effective_model = model or self.model

        start_time = time.time()

        while True:
            try:
                if self.provider == "anthropic":
                    # BUG-FIX: wrap synchronous Anthropic SDK call so the
                    # event loop is not blocked during the model round-trip.
                    response = await asyncio.to_thread(
                        functools.partial(
                            self.client.messages.create,
                            model=effective_model,
                            max_tokens=self.max_tokens,
                            temperature=self.temperature,
                            messages=[{"role": "user", "content": prompt}],
                        )
                    )
                    result = response.content[0].text

                elif self.provider in ("openai", "ollama", "openai-compatible"):
                    # Build kwargs for the chat completion call.
                    call_kwargs: dict = {
                        "model": effective_model,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": self.temperature,
                        "max_tokens": self.max_tokens,
                    }
                    # BUG-FIX (reasoning models): suppress <think> tokens on
                    # ollama / openai-compatible only.  NEVER sent to the real
                    # openai or anthropic providers (would cause an API error).
                    if self.disable_thinking and self.provider in (
                        "ollama",
                        "openai-compatible",
                    ):
                        call_kwargs["extra_body"] = {"think": False}

                    # Force syntactically valid JSON for structured calls (entity
                    # extraction). Small local models reliably emit entities but
                    # frequently produce malformed JSON (missing delimiters) on
                    # complex inputs; constrained decoding fixes this. Supported by
                    # OpenAI and by Ollama's OpenAI-compatible endpoint.
                    if json_mode:
                        call_kwargs["response_format"] = {"type": "json_object"}

                    # BUG-FIX: wrap synchronous OpenAI SDK call so the event
                    # loop is not blocked during the model round-trip.
                    response = await asyncio.to_thread(
                        functools.partial(
                            self.client.chat.completions.create,
                            **call_kwargs,
                        )
                    )
                    result = response.choices[0].message.content

                else:
                    raise ValueError(f"Unsupported provider: {self.provider}")

                # BUG-FIX (reasoning-model empty response, defence layer):
                # Strip any <think>...</think> blocks that still leaked
                # through (e.g. when disable_thinking=False, or on a
                # provider that ignores the hint).
                result = self._strip_thinking(result)

                # Performance monitoring (Issue #30)
                duration = time.time() - start_time
                if duration > 10.0:
                    logger.warning(
                        f"Slow LLM inference detected: {duration:.2f}s for model {effective_model} "
                        f"(provider: {self.provider})"
                    )
                else:
                    logger.debug(f"LLM inference completed in {duration:.2f}s")

                return result

            except TimeoutError as e:
                duration = time.time() - start_time

                # Try fallback Ollama before giving up
                if (
                    self.provider == "ollama"
                    and self._fallback_client
                    and not self._using_fallback
                ):
                    logger.warning(
                        f"Primary Ollama timeout after {duration:.2f}s, "
                        f"switching to fallback at {self.fallback_url}"
                    )
                    self._using_fallback = True
                    self.client = self._fallback_client
                    start_time = time.time()  # Reset timer for fallback
                    continue

                logger.error(
                    f"LLM inference timeout after {duration:.2f}s for model {effective_model}: {e}"
                )
                raise

            except Exception as e:
                duration = time.time() - start_time

                # Try fallback Ollama before giving up
                if (
                    self.provider == "ollama"
                    and self._fallback_client
                    and not self._using_fallback
                ):
                    logger.warning(
                        f"Primary Ollama failed after {duration:.2f}s ({e}), "
                        f"switching to fallback at {self.fallback_url}"
                    )
                    self._using_fallback = True
                    self.client = self._fallback_client
                    start_time = time.time()  # Reset timer for fallback
                    continue

                # Log and raise the error
                logger.error(
                    f"LLM inference failed after {duration:.2f}s for model {effective_model}: {e}"
                )
                raise


# Global service instance
llm_service = LLMService()
