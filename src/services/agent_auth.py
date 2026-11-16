"""Per-agent API keys for the MCP HTTP server (Feature 040).

One key per agent. The keys file maps an agent name to the SHA-256 digest of
its key; plaintext keys are shown once by the CLI and never stored. Every
request to the MCP HTTP app (``/mcp``, SSE, and the hook routes) passes
through :class:`AgentKeyAuthMiddleware`, which resolves the presented key to
an agent name and, depending on the mode, serves, warns, or rejects.

The authenticated agent name is the provenance for captures: read it with
:func:`current_agent` wherever a capture is written.

This module deliberately does not import ``src.config`` so the key
management CLI can run without a full server configuration.

Keys file format (JSON, version 1)::

    {
      "version": 1,
      "agents": {
        "claude-code": {
          "key_sha256": "<64 hex chars>",
          "created_at": "2026-10-10T12:00:00+00:00",
          "rotated_at": null,
          "scopes": ["read", "write"]
        }
      }
    }

``scopes`` is reserved for a later per-key permission check (for example a
read-only agent). It is stored and listed, but not enforced yet: every valid
key has full access.
"""

from __future__ import annotations

import collections
import contextlib
import contextvars
import fcntl
import hashlib
import hmac
import json
import os
import re
import secrets
import tempfile
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from loguru import logger

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

AUTH_MODES = ("off", "warn", "enforce")
KEY_PREFIX = "mnd_"
KEY_HEADER = "x-mnemodrome-key"
LEGACY_AGENT_NAME = "legacy-hook-token"
STATE_KEY = "mnemodrome_agent"
DEFAULT_SCOPES = ["read", "write"]
FILE_VERSION = 1

_AGENT_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")

# Agent name for the current request. Set by the middleware for HTTP routes;
# MCP tool calls run in the session task, so current_agent() also looks at the
# MCP request context (which carries the Starlette request).
_agent_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "mnemodrome_agent", default=None
)


# ---------------------------------------------------------------------------
# Key primitives
# ---------------------------------------------------------------------------


def generate_key() -> str:
    """Return a new high-entropy key (256 bits of randomness)."""
    return KEY_PREFIX + secrets.token_urlsafe(32)


def hash_key(key: str) -> str:
    """Return the hex SHA-256 digest of a key.

    Keys are random 256-bit tokens, so a fast unsalted digest is sufficient:
    there is nothing to brute-force from a leaked digest.
    """
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def keys_match(presented: str, stored_digest: str) -> bool:
    """Constant-time comparison of a presented key against a stored digest."""
    return hmac.compare_digest(hash_key(presented), stored_digest)


def validate_agent_name(name: str) -> str:
    """Return ``name`` if valid, else raise ``ValueError``."""
    if not isinstance(name, str) or not _AGENT_NAME_RE.match(name):
        raise ValueError(
            "Agent name must be 1-64 characters: letters, digits, '.', '_' or '-', "
            "starting with a letter or digit."
        )
    if name == LEGACY_AGENT_NAME:
        raise ValueError(f"'{LEGACY_AGENT_NAME}' is reserved.")
    return name


def normalize_mode(value: str | None) -> str:
    """Normalize an auth mode string; raise ``ValueError`` if unknown."""
    mode = (value or "off").strip().lower()
    if mode not in AUTH_MODES:
        raise ValueError(
            f"MNEMODROME_AUTH_MODE must be one of {', '.join(AUTH_MODES)}; "
            f"got '{value}'."
        )
    return mode


def extract_presented_key(headers: dict[str, str]) -> str | None:
    """Return the key presented in request headers, if any.

    ``Authorization: Bearer <key>`` takes precedence; ``X-Mnemodrome-Key``
    is accepted for clients that cannot set Authorization. ``headers`` keys
    must be lower-case.
    """
    auth = headers.get("authorization", "")
    if auth:
        scheme, _, value = auth.partition(" ")
        if scheme.lower() == "bearer" and value.strip():
            return value.strip()
    alt = headers.get(KEY_HEADER, "").strip()
    return alt or None


# ---------------------------------------------------------------------------
# Keys file
# ---------------------------------------------------------------------------


def empty_document() -> dict[str, Any]:
    return {"version": FILE_VERSION, "agents": {}}


def load_document(path: Path) -> dict[str, Any]:
    """Load and validate a keys file. A missing file is an empty document."""
    if not path.exists():
        return empty_document()
    with path.open("r", encoding="utf-8") as fh:
        doc = json.load(fh)
    if not isinstance(doc, dict) or not isinstance(doc.get("agents"), dict):
        raise ValueError("keys file must be a JSON object with an 'agents' object")
    if doc.get("version", FILE_VERSION) != FILE_VERSION:
        raise ValueError(f"unsupported keys file version: {doc.get('version')}")
    for name, entry in doc["agents"].items():
        validate_agent_name(name)
        if not isinstance(entry, dict) or not _HEX64_RE.match(
            str(entry.get("key_sha256", ""))
        ):
            raise ValueError(f"agent '{name}' has no valid key_sha256")
    return doc


def save_document(path: Path, doc: dict[str, Any]) -> None:
    """Write a keys file atomically with owner-only permissions."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".agent-keys-", dir=str(path.parent))
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(doc, fh, indent=2, sort_keys=True)
            fh.write("\n")
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


@contextlib.contextmanager
def locked_document(path: Path) -> Iterator[None]:
    """Exclusive inter-process lock around a read-modify-write of the keys file.

    Uses ``flock`` on a sibling ``<name>.lock`` file (mode 0600) so concurrent
    ``add`` / ``rotate`` / ``revoke`` runs cannot lose each other's updates.
    The data file itself is replaced atomically by :func:`save_document`.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(path.name + ".lock")
    fd = os.open(str(lock_path), os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def _now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def add_agent(path: Path, name: str, scopes: list[str] | None = None) -> str:
    """Add an agent and return its new plaintext key (shown once)."""
    validate_agent_name(name)
    with locked_document(path):
        doc = load_document(path)
        if name in doc["agents"]:
            raise ValueError(f"agent '{name}' already exists; use rotate")
        key = generate_key()
        doc["agents"][name] = {
            "key_sha256": hash_key(key),
            "created_at": _now_iso(),
            "rotated_at": None,
            "scopes": list(scopes or DEFAULT_SCOPES),
        }
        save_document(path, doc)
    return key


def rotate_agent(path: Path, name: str) -> str:
    """Replace an agent's key and return the new plaintext key (shown once)."""
    with locked_document(path):
        doc = load_document(path)
        if name not in doc["agents"]:
            raise KeyError(f"agent '{name}' not found")
        key = generate_key()
        doc["agents"][name]["key_sha256"] = hash_key(key)
        doc["agents"][name]["rotated_at"] = _now_iso()
        save_document(path, doc)
    return key


def revoke_agent(path: Path, name: str) -> None:
    """Remove an agent's key. Other agents are untouched."""
    with locked_document(path):
        doc = load_document(path)
        if name not in doc["agents"]:
            raise KeyError(f"agent '{name}' not found")
        del doc["agents"][name]
        save_document(path, doc)


def list_agents(path: Path) -> list[dict[str, Any]]:
    """Return agent metadata (never digests), sorted by name."""
    doc = load_document(path)
    return [
        {
            "name": name,
            "created_at": entry.get("created_at"),
            "rotated_at": entry.get("rotated_at"),
            "scopes": entry.get("scopes", DEFAULT_SCOPES),
        }
        for name, entry in sorted(doc["agents"].items())
    ]


# ---------------------------------------------------------------------------
# Runtime key store (hot reload on file change)
# ---------------------------------------------------------------------------


class AgentKeyStore:
    """Resolves presented keys to agent names; reloads the file when it changes."""

    def __init__(
        self,
        path: str | Path | None,
        legacy_token: str | None = None,
        check_interval_s: float = 1.0,
        fail_closed: bool = False,
    ) -> None:
        # fail_closed: when the keys file becomes unreadable after a good
        # load, drop every file-based key instead of keeping the last good
        # set (which could keep a just-revoked key valid). Used in enforce.
        self.fail_closed = fail_closed
        self._load_failed = False
        self.path = Path(path) if path else None
        self._legacy_digest = hash_key(legacy_token) if legacy_token else None
        self._check_interval_s = check_interval_s
        self._digests: dict[str, str] = {}
        self._signature: tuple[int, int] | None = None
        self._last_check = 0.0
        self._lock = threading.Lock()
        self.reload(force=True)

    @property
    def agent_count(self) -> int:
        return len(self._digests)

    @property
    def has_legacy_token(self) -> bool:
        return self._legacy_digest is not None

    def reload(self, force: bool = False) -> None:
        """Reload the keys file if its mtime/size changed (or ``force``)."""
        if self.path is None:
            return
        with self._lock:
            now = time.monotonic()
            if not force and now - self._last_check < self._check_interval_s:
                return
            self._last_check = now
            try:
                st = self.path.stat()
                signature = (st.st_mtime_ns, st.st_size)
            except FileNotFoundError:
                signature = None
            if not force and signature is not None and signature == self._signature:
                return
            if signature is None:
                if self._digests:
                    logger.warning("Agent keys file removed; no agent keys loaded")
                self._digests = {}
                self._signature = None
                self._load_failed = False
                return
            try:
                doc = load_document(self.path)
            except Exception as exc:
                if self.fail_closed:
                    if not self._load_failed:  # log once per failure episode
                        logger.critical(
                            f"Agent keys file could not be loaded "
                            f"({type(exc).__name__}); FAILING CLOSED: dropping "
                            f"{len(self._digests)} loaded key(s). All file-based "
                            "agent keys are rejected until the file is fixed."
                        )
                    self._load_failed = True
                    self._digests = {}
                    # Forget the signature: a transient error (EACCES, EMFILE,
                    # mid-edit) is retried on the next poll, and restoring a
                    # backup with identical mtime/size (cp -p) is not mistaken
                    # for "unchanged since the last good load".
                    self._signature = None
                    return
                else:
                    logger.error(
                        f"Agent keys file could not be loaded ({type(exc).__name__}); "
                        f"keeping {len(self._digests)} previously loaded key(s)"
                    )
                self._signature = signature
                return
            self._load_failed = False
            self._digests = {
                name: entry["key_sha256"] for name, entry in doc["agents"].items()
            }
            self._signature = signature
            logger.info(f"Loaded {len(self._digests)} agent key(s)")

    def authenticate(self, presented: str | None) -> str | None:
        """Return the agent name for a presented key, or ``None``.

        Compares against every stored digest without early exit so timing
        does not reveal which (or whether any) entry matched.
        """
        if not presented:
            return None
        self.reload()
        digest = hash_key(presented)
        match: str | None = None
        for name, stored in self._digests.items():
            if hmac.compare_digest(digest, stored) and match is None:
                match = name
        if self._legacy_digest is not None:
            if hmac.compare_digest(digest, self._legacy_digest) and match is None:
                match = LEGACY_AGENT_NAME
        return match


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


def current_agent() -> str | None:
    """Return the authenticated agent name for the current request, if any."""
    # Per-request state first: on a streamable-http session the session task
    # (and its contextvars) is created by the first request, so the contextvar
    # alone could attribute later callers to the first one.
    try:
        from mcp.server.lowlevel.server import request_ctx  # noqa: PLC0415

        ctx = request_ctx.get()
    except (LookupError, ImportError):
        ctx = None
    if ctx is not None:
        request = getattr(ctx, "request", None)
        scope = getattr(request, "scope", None)
        if isinstance(scope, dict):
            # A request context exists: its own state is authoritative. An
            # anonymous caller (served in warn mode) must not inherit the
            # identity a session's contextvar got from an earlier caller.
            state = scope.get("state")
            value = state.get(STATE_KEY) if isinstance(state, dict) else None
            return value if isinstance(value, str) else None
    return _agent_var.get()


def set_current_agent(name: str | None) -> contextvars.Token:
    """Set the agent for the current context (used by tests and stdio)."""
    return _agent_var.set(name)


def reset_current_agent(token: contextvars.Token) -> None:
    _agent_var.reset(token)


# ---------------------------------------------------------------------------
# ASGI middleware
# ---------------------------------------------------------------------------


@dataclass
class AuthStats:
    authenticated: int = 0
    missing: int = 0
    invalid: int = 0
    rejected: int = 0
    _last_warn: collections.OrderedDict[tuple[str, str], float] = field(
        default_factory=collections.OrderedDict
    )

    def as_dict(self) -> dict[str, int]:
        return {
            "authenticated": self.authenticated,
            "unauthenticated_missing": self.missing,
            "unauthenticated_invalid": self.invalid,
            "rejected": self.rejected,
        }


class AgentKeyAuthMiddleware:
    """Pure ASGI middleware (keeps streaming responses intact)."""

    def __init__(
        self,
        app: Any,
        store: AgentKeyStore,
        mode: str = "off",
        exempt_paths: tuple[str, ...] = ("/health",),
        warn_interval_s: float = 60.0,
        legacy_paths: tuple[str, ...] | None = None,
        max_warn_keys: int = 1024,
    ) -> None:
        # legacy_paths: when set, the legacy hook token authenticates only
        # these paths (the hook routes); elsewhere it counts as an invalid key.
        self.legacy_paths = tuple(legacy_paths) if legacy_paths is not None else None
        self.max_warn_keys = max_warn_keys
        self.app = app
        self.store = store
        self.mode = normalize_mode(mode)
        self.exempt_paths = tuple(exempt_paths)
        self.warn_interval_s = warn_interval_s
        self.stats = AuthStats()

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        scope_type = scope.get("type")
        if scope_type not in ("http", "websocket") or self.mode == "off":
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")
        headers = {
            k.decode("latin-1").lower(): v.decode("latin-1")
            for k, v in scope.get("headers", [])
        }
        presented = extract_presented_key(headers)
        agent = self.store.authenticate(presented)
        if (
            agent == LEGACY_AGENT_NAME
            and self.legacy_paths is not None
            and path not in self.legacy_paths
        ):
            agent = None  # legacy token is valid on the hook routes only

        if agent is None and path in self.exempt_paths:
            await self.app(scope, receive, send)
            return

        if agent is None:
            reason = "missing" if presented is None else "invalid"
            if reason == "missing":
                self.stats.missing += 1
            else:
                self.stats.invalid += 1
            client = scope.get("client") or ("unknown", 0)
            client_host = str(client[0])
            if self.mode == "enforce":
                self.stats.rejected += 1
                self._warn(client_host, path, reason, rejected=True)
                if scope_type == "websocket":
                    # Close before accept; code 1008 = policy violation.
                    await send({"type": "websocket.close", "code": 1008})
                else:
                    await _send_401(send)
                return
            self._warn(client_host, path, reason, rejected=False)
            await self.app(scope, receive, send)
            return

        self.stats.authenticated += 1
        state = scope.setdefault("state", {})
        if isinstance(state, dict):
            state[STATE_KEY] = agent
        token = _agent_var.set(agent)
        try:
            await self.app(scope, receive, send)
        finally:
            _agent_var.reset(token)

    def _warn(self, client: str, path: str, reason: str, rejected: bool) -> None:
        now = time.monotonic()
        key = (client[:64], reason)
        last = self.stats._last_warn.get(key)
        if last is not None and now - last < self.warn_interval_s:
            return
        self.stats._last_warn[key] = now
        self.stats._last_warn.move_to_end(key)
        while len(self.stats._last_warn) > self.max_warn_keys:
            self.stats._last_warn.popitem(last=False)
        # The path is attacker-controlled: log it escaped (repr) and capped.
        path = _safe_log_value(path)
        client = _safe_log_value(client)
        action = "rejected (401)" if rejected else "served"
        counted = self.stats.missing if reason == "missing" else self.stats.invalid
        logger.warning(
            f"AUTH {self.mode}: unauthenticated request {action} "
            f"(reason={reason}_key, path={path}, client={client}, "
            f"{reason}_total={counted}). Give this client an agent key: "
            f"python -m src.cli.agent_keys add <name>"
        )


def _safe_log_value(value: str, limit: int = 200) -> str:
    """Escape (repr) and length-cap an untrusted value for log lines."""
    text = repr(str(value))
    if len(text) > limit:
        text = text[: limit - 3] + "..."
    return text


async def _send_401(send: Any) -> None:
    body = b'{"error": "Unauthorized"}'
    await send(
        {
            "type": "http.response.start",
            "status": 401,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
                (b"www-authenticate", b'Bearer realm="mnemodrome"'),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})
