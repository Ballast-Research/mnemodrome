"""
Unit tests for the startup extraction canary (LLMService.extraction_canary).

The canary is the safeguard against the silent reasoning-model trap: a model
that emits hidden thinking instead of JSON returns zero entities, and every
capture stores nothing. The canary proves extraction works before the server
serves, self-heals when it can (auto no-think, raised token budget), and reports
a hard failure otherwise.

These tests drive the canary entirely through a mocked _call_llm so no live LLM
is required.
"""

from unittest.mock import AsyncMock

import pytest

from src.services.llm_service import LLMService

VALID_JSON = (
    '{"entities": ['
    '{"name": "Alice Johnson", "role": "subject"}, '
    '{"name": "Acme Corporation", "role": "object"}], '
    '"relationships": ['
    '{"source": "Alice Johnson", "relation": "WORKS_AT", "target": "Acme Corporation"}]}'
)


def _make_service(provider="ollama", disable_thinking=True, max_tokens=1000):
    svc = LLMService()
    svc.provider = provider
    svc.disable_thinking = disable_thinking
    svc.max_tokens = max_tokens
    svc._initialized = True  # skip real client construction
    return svc


@pytest.mark.asyncio
async def test_canary_passes_for_healthy_model():
    svc = _make_service()
    svc._call_llm = AsyncMock(return_value=VALID_JSON)

    result = await svc.extraction_canary()

    assert result["ok"] is True
    assert result["healed"] is False
    assert result["entity_count"] >= 1
    assert result["adjustments"] == []
    assert result["error"] is None


@pytest.mark.asyncio
async def test_canary_self_heals_by_enabling_no_think():
    # Thinking starts ON; model returns nothing until no-think is enabled.
    svc = _make_service(disable_thinking=False)

    def fake_call(prompt, model=None, **kwargs):
        return VALID_JSON if svc.disable_thinking else ""

    svc._call_llm = AsyncMock(side_effect=fake_call)

    result = await svc.extraction_canary()

    assert result["ok"] is True
    assert result["healed"] is True
    assert svc.disable_thinking is True  # persisted on the singleton
    assert "LLM_DISABLE_THINKING=true" in result["adjustments"]


@pytest.mark.asyncio
async def test_canary_self_heals_by_raising_token_budget():
    # No-think already on; model returns nothing until the token budget is raised.
    svc = _make_service(disable_thinking=True, max_tokens=1000)

    def fake_call(prompt, model=None, **kwargs):
        return VALID_JSON if svc.max_tokens >= svc.CANARY_HEALED_MAX_TOKENS else ""

    svc._call_llm = AsyncMock(side_effect=fake_call)

    result = await svc.extraction_canary()

    assert result["ok"] is True
    assert result["healed"] is True
    assert svc.max_tokens == svc.CANARY_HEALED_MAX_TOKENS
    assert any("LLM_MAX_TOKENS" in a for a in result["adjustments"])


@pytest.mark.asyncio
async def test_canary_hard_fails_on_unrecoverable_reasoning_model():
    # Reachable but always empty, even after every self-heal step = the trap.
    svc = _make_service(disable_thinking=False, max_tokens=1000)
    svc._call_llm = AsyncMock(return_value="")

    result = await svc.extraction_canary()

    assert result["ok"] is False
    assert result["error"] is None  # reachable, just empty
    assert result["entity_count"] == 0


@pytest.mark.asyncio
async def test_canary_reports_unreachable_model_as_error_not_trap():
    svc = _make_service()
    svc._call_llm = AsyncMock(side_effect=ConnectionError("connection refused"))

    result = await svc.extraction_canary()

    assert result["ok"] is False
    assert result["error"] is not None
    assert "unreachable" in result["error"]


@pytest.mark.asyncio
async def test_canary_probes_do_not_pollute_degraded_count():
    svc = _make_service(disable_thinking=True, max_tokens=1000)
    svc.extraction_degraded_count = 7
    svc._call_llm = AsyncMock(return_value="")  # empty probes would otherwise increment

    await svc.extraction_canary()

    assert svc.extraction_degraded_count == 7  # snapshot restored
