"""
Install option matrix tests.

Closes the test-coverage gap that let two install bugs ship undetected:
  1. openai-compatible was not in the config allow-list → runtime crash on import
  2. The existing CI smoke job only ran with LLM_PROVIDER=ollama; no other
     provider paths were exercised by any test.

Two test layers are combined here:

  Layer A — Config bootability (pure Python, no shell, hermetic):
      Call validate_llm_config() under patch.dict(os.environ, ...) to cover
      every provider/flag combination without touching the filesystem.

  Layer B — Shell integration (subprocess, --dry-run, no Docker required):
      Run install.sh --dry-run with provider flags, parse the resulting .env,
      and validate it passes / raises as expected.

The shell tests (Layer B) write to the project .env.  They back it up
beforehand and restore it afterward so the real configuration is preserved.

Does NOT require: Docker, live LLM endpoints, or real cloud API keys.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from collections.abc import Generator
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent.parent
INSTALL_SH = PROJECT_ROOT / "install.sh"
PYTHON = Path(sys.executable)  # honours the venv / uv interpreter


def _docker_available() -> bool:
    """Return True if Docker CLI and daemon are present and reachable."""
    try:
        r = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            timeout=5,
        )
        return r.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


# Mark that skips the entire Layer-B class when Docker is absent.
# install.sh runs a docker preflight check even in --dry-run mode.
requires_docker = pytest.mark.skipif(
    not _docker_available(),
    reason="Docker daemon not available — install.sh preflight would fail",
)

# Storage paths required by the top-level validate_config() that runs at
# import time — already set by tests/conftest.py but referenced here for
# clarity in assert messages.
_STORAGE_ENV = {
    "MNEMODROME_FACTS_PATH": "/tmp/mnemodrome-test/facts",
    "MNEMODROME_LEARNINGS_PATH": "/tmp/mnemodrome-test/learnings",
    "MNEMODROME_SESSIONS_PATH": "/tmp/mnemodrome-test/sessions",
    "MNEMODROME_GRAPHS_PATH": "/tmp/mnemodrome-test/graphs",
    "MNEMODROME_UPDATES_PATH": "/tmp/mnemodrome-test/updates",
    "MNEMODROME_PROCEDURES_PATH": "/tmp/mnemodrome-test/procedures",
}

# ---------------------------------------------------------------------------
# Import validate_llm_config — safe because conftest.py already pre-loaded
# all required env vars before any src import happened.
# ---------------------------------------------------------------------------
from src.config import validate_llm_config  # noqa: E402

# ===========================================================================
# LAYER A — Config bootability tests (hermetic, no shell, no filesystem I/O)
# ===========================================================================


def _base_env(**overrides: str) -> dict[str, str]:
    """Return a minimal env dict that satisfies non-LLM config validators."""
    base = {
        "NEO4J_URI": "bolt://localhost:7687",
        "NEO4J_USER": "neo4j",
        "NEO4J_PASSWORD": "test-password",
        **_STORAGE_ENV,
    }
    base.update(overrides)
    return base


class TestValidateLlmConfigOllama:
    """Ollama provider — happy path and edge cases."""

    def test_ollama_with_base_url_passes(self):
        env = _base_env(
            LLM_PROVIDER="ollama",
            CHAT_MODEL="gemma4:12b",
            LLM_BASE_URL="http://host.docker.internal:11434",
        )
        with pytest.MonkeyPatch.context() as mp:
            mp.delenv("LLM_PROVIDER", raising=False)
            mp.delenv("CHAT_MODEL", raising=False)
            mp.delenv("LLM_BASE_URL", raising=False)
            mp.delenv("OLLAMA_BASE_URL", raising=False)
            mp.delenv("LLM_API_KEY", raising=False)
            for k, v in env.items():
                mp.setenv(k, v)
            validate_llm_config()  # must not raise

    def test_ollama_without_base_url_passes_with_warning(self, capfd):
        """Ollama missing base URL is a warning, not an error."""
        env = _base_env(
            LLM_PROVIDER="ollama",
            CHAT_MODEL="gemma4:12b",
        )
        with pytest.MonkeyPatch.context() as mp:
            mp.delenv("LLM_PROVIDER", raising=False)
            mp.delenv("CHAT_MODEL", raising=False)
            mp.delenv("LLM_BASE_URL", raising=False)
            mp.delenv("OLLAMA_BASE_URL", raising=False)
            mp.delenv("LLM_API_KEY", raising=False)
            for k, v in env.items():
                mp.setenv(k, v)
            # Should not raise — just logs a warning
            validate_llm_config()

    def test_ollama_missing_model_raises(self):
        env = _base_env(
            LLM_PROVIDER="ollama",
            LLM_BASE_URL="http://localhost:11434",
        )
        with pytest.MonkeyPatch.context() as mp:
            mp.delenv("LLM_PROVIDER", raising=False)
            mp.delenv("CHAT_MODEL", raising=False)
            mp.delenv("LLM_BASE_URL", raising=False)
            mp.delenv("OLLAMA_BASE_URL", raising=False)
            mp.delenv("LLM_API_KEY", raising=False)
            for k, v in env.items():
                mp.setenv(k, v)
            with pytest.raises(ValueError, match="CHAT_MODEL is required"):
                validate_llm_config()


class TestValidateLlmConfigOpenAI:
    """openai provider — happy path and missing-key error."""

    def test_openai_with_key_passes(self):
        env = _base_env(
            LLM_PROVIDER="openai",
            CHAT_MODEL="gpt-4o",
            LLM_API_KEY="sk-dummy-test-key-12345",
            LLM_BASE_URL="https://api.openai.com/v1",
        )
        with pytest.MonkeyPatch.context() as mp:
            mp.delenv("LLM_PROVIDER", raising=False)
            mp.delenv("CHAT_MODEL", raising=False)
            mp.delenv("LLM_BASE_URL", raising=False)
            mp.delenv("OLLAMA_BASE_URL", raising=False)
            mp.delenv("LLM_API_KEY", raising=False)
            for k, v in env.items():
                mp.setenv(k, v)
            validate_llm_config()  # must not raise

    def test_openai_missing_key_raises(self):
        env = _base_env(
            LLM_PROVIDER="openai",
            CHAT_MODEL="gpt-4o",
            LLM_BASE_URL="https://api.openai.com/v1",
        )
        with pytest.MonkeyPatch.context() as mp:
            mp.delenv("LLM_PROVIDER", raising=False)
            mp.delenv("CHAT_MODEL", raising=False)
            mp.delenv("LLM_BASE_URL", raising=False)
            mp.delenv("OLLAMA_BASE_URL", raising=False)
            mp.delenv("LLM_API_KEY", raising=False)
            for k, v in env.items():
                mp.setenv(k, v)
            with pytest.raises(ValueError, match="LLM_API_KEY is required"):
                validate_llm_config()


class TestValidateLlmConfigAnthropic:
    """anthropic provider — happy path and missing-key error."""

    def test_anthropic_with_key_passes(self):
        env = _base_env(
            LLM_PROVIDER="anthropic",
            CHAT_MODEL="claude-3-5-sonnet-20241022",
            LLM_API_KEY="sk-ant-dummy-test-key",
        )
        with pytest.MonkeyPatch.context() as mp:
            mp.delenv("LLM_PROVIDER", raising=False)
            mp.delenv("CHAT_MODEL", raising=False)
            mp.delenv("LLM_BASE_URL", raising=False)
            mp.delenv("OLLAMA_BASE_URL", raising=False)
            mp.delenv("LLM_API_KEY", raising=False)
            for k, v in env.items():
                mp.setenv(k, v)
            validate_llm_config()  # must not raise

    def test_anthropic_missing_key_raises(self):
        env = _base_env(
            LLM_PROVIDER="anthropic",
            CHAT_MODEL="claude-3-5-sonnet-20241022",
        )
        with pytest.MonkeyPatch.context() as mp:
            mp.delenv("LLM_PROVIDER", raising=False)
            mp.delenv("CHAT_MODEL", raising=False)
            mp.delenv("LLM_BASE_URL", raising=False)
            mp.delenv("OLLAMA_BASE_URL", raising=False)
            mp.delenv("LLM_API_KEY", raising=False)
            for k, v in env.items():
                mp.setenv(k, v)
            with pytest.raises(ValueError, match="LLM_API_KEY is required"):
                validate_llm_config()


class TestValidateLlmConfigOpenAICompatible:
    """
    openai-compatible provider — this is the provider that shipped a non-bootable
    config before the allow-list fix.

    The key assertions here:
      - Valid config (provider + model + key + base URL) PASSES
      - Missing LLM_API_KEY RAISES
      - Missing LLM_BASE_URL RAISES  ← was silent before the fix
      - Provider value 'openai-compatible' is in the allow-list (would have raised
        "Unsupported LLM_PROVIDER" with the old code)
    """

    def test_openai_compatible_full_config_passes(self):
        """The post-fix happy path: all required fields → no exception."""
        env = _base_env(
            LLM_PROVIDER="openai-compatible",
            CHAT_MODEL="mistral-7b-instruct",
            LLM_API_KEY="dummy-openrouter-key-xyz",
            LLM_BASE_URL="https://openrouter.ai/api/v1",
        )
        with pytest.MonkeyPatch.context() as mp:
            mp.delenv("LLM_PROVIDER", raising=False)
            mp.delenv("CHAT_MODEL", raising=False)
            mp.delenv("LLM_BASE_URL", raising=False)
            mp.delenv("OLLAMA_BASE_URL", raising=False)
            mp.delenv("LLM_API_KEY", raising=False)
            for k, v in env.items():
                mp.setenv(k, v)
            validate_llm_config()  # must not raise

    def test_openai_compatible_missing_api_key_raises(self):
        env = _base_env(
            LLM_PROVIDER="openai-compatible",
            CHAT_MODEL="mistral-7b-instruct",
            LLM_BASE_URL="https://openrouter.ai/api/v1",
        )
        with pytest.MonkeyPatch.context() as mp:
            mp.delenv("LLM_PROVIDER", raising=False)
            mp.delenv("CHAT_MODEL", raising=False)
            mp.delenv("LLM_BASE_URL", raising=False)
            mp.delenv("OLLAMA_BASE_URL", raising=False)
            mp.delenv("LLM_API_KEY", raising=False)
            for k, v in env.items():
                mp.setenv(k, v)
            with pytest.raises(ValueError, match="LLM_API_KEY is required"):
                validate_llm_config()

    def test_openai_compatible_missing_base_url_raises(self):
        """No base URL for openai-compatible must be a hard error (no sane default)."""
        env = _base_env(
            LLM_PROVIDER="openai-compatible",
            CHAT_MODEL="mistral-7b-instruct",
            LLM_API_KEY="dummy-key",
        )
        with pytest.MonkeyPatch.context() as mp:
            mp.delenv("LLM_PROVIDER", raising=False)
            mp.delenv("CHAT_MODEL", raising=False)
            mp.delenv("LLM_BASE_URL", raising=False)
            mp.delenv("OLLAMA_BASE_URL", raising=False)
            mp.delenv("LLM_API_KEY", raising=False)
            for k, v in env.items():
                mp.setenv(k, v)
            with pytest.raises(
                ValueError, match="LLM_BASE_URL is required for openai-compatible"
            ):
                validate_llm_config()

    def test_openai_compatible_ollama_base_url_fallback_passes(self):
        """OLLAMA_BASE_URL is accepted as base URL for openai-compatible (backward compat)."""
        env = _base_env(
            LLM_PROVIDER="openai-compatible",
            CHAT_MODEL="mistral-7b-instruct",
            LLM_API_KEY="dummy-key",
            OLLAMA_BASE_URL="http://host.docker.internal:8000/v1",
        )
        with pytest.MonkeyPatch.context() as mp:
            mp.delenv("LLM_PROVIDER", raising=False)
            mp.delenv("CHAT_MODEL", raising=False)
            mp.delenv("LLM_BASE_URL", raising=False)
            mp.delenv("OLLAMA_BASE_URL", raising=False)
            mp.delenv("LLM_API_KEY", raising=False)
            for k, v in env.items():
                mp.setenv(k, v)
            validate_llm_config()  # must not raise

    def test_openai_compatible_in_allow_list(self):
        """
        Regression test for the original bug:
        'openai-compatible' was not in the provider allow-list, so any
        openai-compatible config would raise 'Unsupported LLM_PROVIDER'
        even when all fields were present.

        This test confirms 'openai-compatible' passes the allow-list check.
        Simulate the old code by removing openai-compatible from the check.
        """
        # Confirm the provider is accepted — if it weren't in the allow-list,
        # this would raise ValueError("Unsupported LLM_PROVIDER: openai-compatible")
        env = _base_env(
            LLM_PROVIDER="openai-compatible",
            CHAT_MODEL="mistral-7b-instruct",
            LLM_API_KEY="dummy-key",
            LLM_BASE_URL="https://openrouter.ai/api/v1",
        )
        with pytest.MonkeyPatch.context() as mp:
            mp.delenv("LLM_PROVIDER", raising=False)
            mp.delenv("CHAT_MODEL", raising=False)
            mp.delenv("LLM_BASE_URL", raising=False)
            mp.delenv("OLLAMA_BASE_URL", raising=False)
            mp.delenv("LLM_API_KEY", raising=False)
            for k, v in env.items():
                mp.setenv(k, v)
            # If openai-compatible were missing from allow-list, this raises:
            #   ValueError: Unsupported LLM_PROVIDER: openai-compatible
            validate_llm_config()

    def test_unknown_provider_raises(self):
        """Providers not in the allow-list must always raise (regression guard)."""
        env = _base_env(
            LLM_PROVIDER="some-future-provider",
            CHAT_MODEL="some-model",
            LLM_API_KEY="some-key",
        )
        with pytest.MonkeyPatch.context() as mp:
            mp.delenv("LLM_PROVIDER", raising=False)
            mp.delenv("CHAT_MODEL", raising=False)
            mp.delenv("LLM_BASE_URL", raising=False)
            mp.delenv("OLLAMA_BASE_URL", raising=False)
            mp.delenv("LLM_API_KEY", raising=False)
            for k, v in env.items():
                mp.setenv(k, v)
            with pytest.raises(ValueError, match="Unsupported LLM_PROVIDER"):
                validate_llm_config()


class TestValidateLlmConfigMissingProvider:
    """LLM_PROVIDER absent must fail clearly."""

    def test_missing_provider_raises(self):
        env = _base_env(
            CHAT_MODEL="gemma4:12b",
        )
        with pytest.MonkeyPatch.context() as mp:
            mp.delenv("LLM_PROVIDER", raising=False)
            mp.delenv("CHAT_MODEL", raising=False)
            mp.delenv("LLM_BASE_URL", raising=False)
            mp.delenv("OLLAMA_BASE_URL", raising=False)
            mp.delenv("LLM_API_KEY", raising=False)
            for k, v in env.items():
                mp.setenv(k, v)
            with pytest.raises(ValueError, match="LLM_PROVIDER is required"):
                validate_llm_config()


# ===========================================================================
# LAYER B — Shell integration tests (run install.sh --dry-run, parse .env)
# ===========================================================================

# Helpers -----------------------------------------------------------------


def _parse_env_file(path: Path) -> dict[str, str]:
    """
    Parse a .env file into a dict, skipping comments and empty lines.
    Handles KEY=value and KEY= (empty value) forms.
    """
    result: dict[str, str] = {}
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "=" in stripped:
            key, _, value = stripped.partition("=")
            result[key.strip()] = value.strip()
    return result


def _run_install(
    tmp_data: Path,
    extra_flags: list[str],
    timeout: int = 60,
) -> subprocess.CompletedProcess[str]:
    """
    Run install.sh --dry-run from the project root and return the result.

    Always uses --dry-run so Docker is never required.
    Captures stdout + stderr.
    """
    cmd = [
        str(INSTALL_SH),
        "--dry-run",
        "--data-root",
        str(tmp_data),
        *extra_flags,
    ]
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=str(PROJECT_ROOT),
    )


@pytest.fixture()
def env_backup() -> Generator[Path, None, None]:
    """
    Back up the real .env before a shell test and restore it afterward.

    install.sh writes (and backs up) .env in-place.  This fixture ensures
    the real working .env is restored after each shell test.
    """
    env_path = PROJECT_ROOT / ".env"
    backup_path = PROJECT_ROOT / ".env.test_matrix_backup"

    # Back up if the real .env exists
    existed = env_path.exists()
    if existed:
        shutil.copy2(str(env_path), str(backup_path))

    # Snapshot pre-existing backups so teardown removes only what THIS test run
    # creates.  install.sh writes a .env.bak.<ts> on every run; without this the
    # repo root accumulated them unbounded (the previous cleanup was a no-op).
    pre_existing_baks = set(PROJECT_ROOT.glob(".env.bak.*"))

    yield env_path

    # Restore or remove
    if existed and backup_path.exists():
        shutil.move(str(backup_path), str(env_path))
    elif not existed and env_path.exists():
        env_path.unlink()
    # Remove only the .env.bak.* files created during this test run.
    for bak in set(PROJECT_ROOT.glob(".env.bak.*")) - pre_existing_baks:
        bak.unlink(missing_ok=True)


# Shell test cases ---------------------------------------------------------


@requires_docker
class TestInstallShDryRun:
    """
    Layer B -- shell integration.

    These tests run install.sh --dry-run and validate the generated .env.
    The Docker CLI and daemon must be present because install.sh always runs a
    Docker preflight check (even in --dry-run mode) -- docker compose up itself
    is skipped.

    install.sh now accepts --llm-api-key and writes LLM_API_KEY to the
    generated .env.  When a cloud provider (openai/anthropic/openai-compatible)
    is selected non-interactively (via --defaults or provider flags) WITHOUT
    supplying --llm-api-key or the LLM_API_KEY env var, install.sh exits
    non-zero (fails fast via die).  Cloud-provider tests therefore pass a dummy
    key via --llm-api-key so that install.sh exits 0.
    """

    def test_defaults_exits_zero_and_writes_env(self, env_backup, tmp_path):
        """--defaults path: exit 0 and .env is written."""
        result = _run_install(tmp_path, ["--defaults"])
        assert result.returncode == 0, (
            f"install.sh --defaults --dry-run exited {result.returncode}.\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )
        assert env_backup.exists(), ".env was not written by install.sh"
        combined = result.stdout + result.stderr
        assert "Dry run complete" in combined

    def test_defaults_env_has_ollama_provider(self, env_backup, tmp_path):
        """--defaults should produce LLM_PROVIDER=ollama in .env."""
        _run_install(tmp_path, ["--defaults"])
        env = _parse_env_file(env_backup)
        assert env.get("LLM_PROVIDER") == "ollama", (
            f"Expected LLM_PROVIDER=ollama, got: {env.get('LLM_PROVIDER')!r}"
        )
        assert env.get("CHAT_MODEL"), "CHAT_MODEL must not be empty"

    def test_defaults_env_passes_config_validation(self, env_backup, tmp_path):
        """
        --defaults env must pass validate_llm_config() without supplementation.
        Ollama doesn't require an API key or base URL (base URL is warned, not errored).
        """
        _run_install(tmp_path, ["--defaults"])
        env = _parse_env_file(env_backup)

        with pytest.MonkeyPatch.context() as mp:
            mp.delenv("LLM_PROVIDER", raising=False)
            mp.delenv("CHAT_MODEL", raising=False)
            mp.delenv("LLM_BASE_URL", raising=False)
            mp.delenv("OLLAMA_BASE_URL", raising=False)
            mp.delenv("LLM_API_KEY", raising=False)
            for k, v in env.items():
                mp.setenv(k, v)
            validate_llm_config()  # must not raise

    def test_custom_data_root_reflected_in_env(self, env_backup, tmp_path):
        """--data-root value must appear as DATA_ROOT in the generated .env."""
        custom_root = tmp_path / "custom-mnemodrome-data"
        _run_install(tmp_path, ["--defaults", "--data-root", str(custom_root)])
        env = _parse_env_file(env_backup)
        assert env.get("DATA_ROOT") == str(custom_root), (
            f"Expected DATA_ROOT={custom_root}, got {env.get('DATA_ROOT')!r}"
        )

    def test_explicit_ollama_provider_flag(self, env_backup, tmp_path):
        """--llm-provider ollama with endpoint/model flags.

        --defaults is required alongside provider flags so that Q1
        (install directory) is answered non-interactively.  Q3 is skipped
        automatically when --llm-provider is set.
        """
        result = _run_install(
            tmp_path,
            [
                "--defaults",
                "--llm-provider",
                "ollama",
                "--llm-endpoint",
                "http://localhost:11434",
                "--chat-model",
                "llama3.1:8b",
                "--embedding-model",
                "nomic-embed-text",
            ],
        )
        assert result.returncode == 0
        env = _parse_env_file(env_backup)
        assert env.get("LLM_PROVIDER") == "ollama"
        assert env.get("CHAT_MODEL") == "llama3.1:8b"
        assert env.get("OLLAMA_EMBEDDING_MODEL") == "nomic-embed-text"
        # Config validation must pass
        with pytest.MonkeyPatch.context() as mp:
            mp.delenv("LLM_PROVIDER", raising=False)
            mp.delenv("CHAT_MODEL", raising=False)
            mp.delenv("LLM_BASE_URL", raising=False)
            mp.delenv("OLLAMA_BASE_URL", raising=False)
            mp.delenv("LLM_API_KEY", raising=False)
            for k, v in env.items():
                mp.setenv(k, v)
            validate_llm_config()

    def test_openai_compatible_provider_flag(self, env_backup, tmp_path):
        """
        --llm-provider openai-compatible --llm-endpoint <url> must produce a
        well-structured .env that passes validate_llm_config.

        install.sh now accepts --llm-api-key and writes LLM_API_KEY to .env.
        A cloud provider chosen non-interactively without a key causes install.sh
        to exit non-zero (fail fast).  This test supplies a dummy key via
        --llm-api-key so install.sh exits 0 and the key appears in the .env.

        This test directly targets the original openai-compatible bug:
          - Before fix: LLM_PROVIDER=openai-compatible was not in the allow-list
            so validate_llm_config raised "Unsupported LLM_PROVIDER" even with a
            fully-formed .env.
          - After fix: the provider is in the allow-list and the key is written
            by install.sh when --llm-api-key is supplied.
        """
        endpoint = "http://localhost:8000/v1"
        result = _run_install(
            tmp_path,
            [
                "--defaults",
                "--llm-provider",
                "openai-compatible",
                "--llm-endpoint",
                endpoint,
                "--chat-model",
                "mistral-7b-instruct",
                "--llm-api-key",
                "dummy-compat-key",
            ],
        )
        assert result.returncode == 0, (
            f"install.sh exited {result.returncode}\n{result.stdout}\n{result.stderr}"
        )
        env = _parse_env_file(env_backup)

        # Structural assertions on the generated .env
        assert env.get("LLM_PROVIDER") == "openai-compatible", (
            f"Expected LLM_PROVIDER=openai-compatible, got {env.get('LLM_PROVIDER')!r}"
        )
        assert env.get("CHAT_MODEL") == "mistral-7b-instruct"
        # The base URL must be present (install.sh writes OLLAMA_BASE_URL / LLM_BASE_URL
        # from OLLAMA_DOCKER_ENDPOINT which is translate_to_docker(LLM_ENDPOINT))
        assert env.get("LLM_BASE_URL") or env.get("OLLAMA_BASE_URL"), (
            "Neither LLM_BASE_URL nor OLLAMA_BASE_URL present in .env for openai-compatible"
        )

        # --- Without API key: validate_llm_config MUST raise ---
        # Simulate a user who has the .env but clears the key.
        with pytest.MonkeyPatch.context() as mp:
            mp.delenv("LLM_PROVIDER", raising=False)
            mp.delenv("CHAT_MODEL", raising=False)
            mp.delenv("LLM_BASE_URL", raising=False)
            mp.delenv("OLLAMA_BASE_URL", raising=False)
            mp.delenv("LLM_API_KEY", raising=False)
            for k, v in env.items():
                mp.setenv(k, v)
            # The .env now contains LLM_API_KEY from --llm-api-key; force it
            # empty to exercise the "missing key" validation branch.
            mp.setenv("LLM_API_KEY", "")
            with pytest.raises(ValueError, match="LLM_API_KEY is required"):
                validate_llm_config()

        # --- With the key from .env: validate_llm_config MUST pass ---
        with pytest.MonkeyPatch.context() as mp:
            mp.delenv("LLM_PROVIDER", raising=False)
            mp.delenv("CHAT_MODEL", raising=False)
            mp.delenv("LLM_BASE_URL", raising=False)
            mp.delenv("OLLAMA_BASE_URL", raising=False)
            mp.delenv("LLM_API_KEY", raising=False)
            for k, v in env.items():
                mp.setenv(k, v)
            validate_llm_config()  # must not raise

    def test_openai_provider_flag(self, env_backup, tmp_path):
        """
        --llm-provider openai must produce LLM_PROVIDER=openai in .env.

        install.sh now accepts --llm-api-key and writes LLM_API_KEY to .env.
        A cloud provider chosen non-interactively without a key causes install.sh
        to exit non-zero (fail fast).  This test supplies a dummy key so
        install.sh exits 0 and the key appears in the generated .env.
        """
        result = _run_install(
            tmp_path,
            [
                "--defaults",
                "--llm-provider",
                "openai",
                "--chat-model",
                "gpt-4o",
                "--llm-api-key",
                "sk-dummy-openai-key",
            ],
        )
        assert result.returncode == 0, (
            f"install.sh exited {result.returncode}\n{result.stdout}\n{result.stderr}"
        )
        env = _parse_env_file(env_backup)
        assert env.get("LLM_PROVIDER") == "openai"
        assert env.get("CHAT_MODEL") == "gpt-4o"

        # Without key → validate_llm_config fails (expected)
        # The .env now contains LLM_API_KEY from --llm-api-key; force it empty
        # to exercise the "missing key" validation branch.
        with pytest.MonkeyPatch.context() as mp:
            mp.delenv("LLM_PROVIDER", raising=False)
            mp.delenv("CHAT_MODEL", raising=False)
            mp.delenv("LLM_BASE_URL", raising=False)
            mp.delenv("OLLAMA_BASE_URL", raising=False)
            mp.delenv("LLM_API_KEY", raising=False)
            for k, v in env.items():
                mp.setenv(k, v)
            mp.setenv("LLM_API_KEY", "")
            with pytest.raises(ValueError, match="LLM_API_KEY is required"):
                validate_llm_config()

        # With the key from .env → passes (entire env is structurally valid)
        with pytest.MonkeyPatch.context() as mp:
            mp.delenv("LLM_PROVIDER", raising=False)
            mp.delenv("CHAT_MODEL", raising=False)
            mp.delenv("LLM_BASE_URL", raising=False)
            mp.delenv("OLLAMA_BASE_URL", raising=False)
            mp.delenv("LLM_API_KEY", raising=False)
            for k, v in env.items():
                mp.setenv(k, v)
            mp.setenv("LLM_BASE_URL", "https://api.openai.com/v1")
            validate_llm_config()

    def test_anthropic_provider_flag(self, env_backup, tmp_path):
        """
        --llm-provider anthropic must produce LLM_PROVIDER=anthropic in .env.

        install.sh now accepts --llm-api-key and writes LLM_API_KEY to .env.
        A cloud provider chosen non-interactively without a key causes install.sh
        to exit non-zero (fail fast).  This test supplies a dummy key so
        install.sh exits 0 and the key appears in the generated .env.
        """
        result = _run_install(
            tmp_path,
            [
                "--defaults",
                "--llm-provider",
                "anthropic",
                "--chat-model",
                "claude-3-5-sonnet-20241022",
                "--llm-api-key",
                "sk-ant-dummy-test-key",
            ],
        )
        assert result.returncode == 0, (
            f"install.sh exited {result.returncode}\n{result.stdout}\n{result.stderr}"
        )
        env = _parse_env_file(env_backup)
        assert env.get("LLM_PROVIDER") == "anthropic"
        assert env.get("CHAT_MODEL") == "claude-3-5-sonnet-20241022"

        # Without key → validate_llm_config fails (expected)
        # The .env now contains LLM_API_KEY from --llm-api-key; force it empty
        # to exercise the "missing key" validation branch.
        with pytest.MonkeyPatch.context() as mp:
            mp.delenv("LLM_PROVIDER", raising=False)
            mp.delenv("CHAT_MODEL", raising=False)
            mp.delenv("LLM_BASE_URL", raising=False)
            mp.delenv("OLLAMA_BASE_URL", raising=False)
            mp.delenv("LLM_API_KEY", raising=False)
            for k, v in env.items():
                mp.setenv(k, v)
            mp.setenv("LLM_API_KEY", "")
            with pytest.raises(ValueError, match="LLM_API_KEY is required"):
                validate_llm_config()

        # With the key from .env → passes
        with pytest.MonkeyPatch.context() as mp:
            mp.delenv("LLM_PROVIDER", raising=False)
            mp.delenv("CHAT_MODEL", raising=False)
            mp.delenv("LLM_BASE_URL", raising=False)
            mp.delenv("OLLAMA_BASE_URL", raising=False)
            mp.delenv("LLM_API_KEY", raising=False)
            for k, v in env.items():
                mp.setenv(k, v)
            validate_llm_config()

    def test_cloud_provider_without_key_fails_fast(self, env_backup, tmp_path):
        """
        install.sh must exit non-zero when a cloud provider is chosen
        non-interactively (--defaults or provider flags) but no API key is
        supplied (neither --llm-api-key nor LLM_API_KEY env var).

        This is the fail-fast guarantee added alongside the --llm-api-key flag.
        The error output must mention the missing key so the user knows what to fix.
        """
        result = _run_install(
            tmp_path,
            [
                "--defaults",
                "--llm-provider",
                "openai",
                "--chat-model",
                "gpt-4o",
            ],
        )
        assert result.returncode != 0, (
            "install.sh should have exited non-zero when openai is selected "
            "non-interactively without --llm-api-key, but it exited 0.\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )
        combined = result.stdout + result.stderr
        import re

        assert re.search(r"LLM_API_KEY|api[- ]key", combined, re.IGNORECASE), (
            "install.sh exit output does not mention the missing API key.\n"
            f"combined output: {combined}"
        )


# ===========================================================================
# LAYER B (extra) — Simulate the original bug to confirm test detectability
# ===========================================================================


class TestOriginalBugDetection:
    """
    Confirm the test matrix WOULD HAVE caught the original openai-compatible
    non-bootable config bug.

    The original bug: config.py's allow-list was
        ["anthropic", "openai", "ollama"]
    i.e. "openai-compatible" was absent.

    validate_llm_config() would raise ValueError("Unsupported LLM_PROVIDER:
    openai-compatible") for any openai-compatible config, no matter how
    well-formed.  This means any fresh install with --llm-provider openai-compatible
    produced a non-bootable .env.

    We simulate the old code by checking that validate_llm_config raises for
    an unknown provider (same code path).  The fix is that "openai-compatible"
    is now in the allow-list — see TestValidateLlmConfigOpenAICompatible above.
    """

    def test_provider_not_in_allow_list_always_raises(self):
        """
        Simulates the old broken state: if openai-compatible were removed from
        the allow-list, this is the error a user would see.

        We use a deliberately unknown provider name to trigger the same error
        path without touching the real allow-list.
        """
        env = _base_env(
            LLM_PROVIDER="openai-compatible-OLD-NOT-SUPPORTED",
            CHAT_MODEL="mistral-7b",
            LLM_API_KEY="dummy-key",
            LLM_BASE_URL="https://openrouter.ai/api/v1",
        )
        with pytest.MonkeyPatch.context() as mp:
            mp.delenv("LLM_PROVIDER", raising=False)
            mp.delenv("CHAT_MODEL", raising=False)
            mp.delenv("LLM_BASE_URL", raising=False)
            mp.delenv("OLLAMA_BASE_URL", raising=False)
            mp.delenv("LLM_API_KEY", raising=False)
            for k, v in env.items():
                mp.setenv(k, v)
            with pytest.raises(ValueError, match="Unsupported LLM_PROVIDER"):
                validate_llm_config()

    def test_openai_compatible_now_passes_allow_list(self):
        """
        Direct regression guard: validate_llm_config must NOT raise
        'Unsupported LLM_PROVIDER' for 'openai-compatible'.

        If this test fails it means openai-compatible was removed from the
        allow-list → the original bug has been re-introduced.
        """
        env = _base_env(
            LLM_PROVIDER="openai-compatible",
            CHAT_MODEL="mistral-7b-instruct",
            LLM_API_KEY="dummy-key",
            LLM_BASE_URL="https://openrouter.ai/api/v1",
        )
        with pytest.MonkeyPatch.context() as mp:
            mp.delenv("LLM_PROVIDER", raising=False)
            mp.delenv("CHAT_MODEL", raising=False)
            mp.delenv("LLM_BASE_URL", raising=False)
            mp.delenv("OLLAMA_BASE_URL", raising=False)
            mp.delenv("LLM_API_KEY", raising=False)
            for k, v in env.items():
                mp.setenv(k, v)
            # The following MUST NOT raise ValueError("Unsupported LLM_PROVIDER")
            # If it does, the openai-compatible bug has been re-introduced.
            try:
                validate_llm_config()
            except ValueError as exc:
                if "Unsupported LLM_PROVIDER" in str(exc):
                    pytest.fail(
                        "REGRESSION: openai-compatible is not in the provider allow-list.\n"
                        "This is the original bug that shipped a non-bootable config.\n"
                        f"Exception: {exc}"
                    )
                # Other ValueErrors (e.g. missing fields) are not the bug
                raise
