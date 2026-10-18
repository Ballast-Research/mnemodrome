"""
Admin UI authentication (Feature 039).

Two ways in, one identity (ADMIN_USERNAME / ADMIN_PASSWORD):

- **Form login + session cookie** for people in a browser. A normal HTML form
  with ``autocomplete="username"`` / ``"current-password"`` fields, so password
  managers can save and fill it. The cookie is HMAC-signed (stdlib only),
  HttpOnly, SameSite=Strict, and Secure when the request arrived over HTTPS
  (directly or via a reverse proxy that sets X-Forwarded-Proto).
- **HTTP Basic auth** for scripts and API clients (``curl -u``). Unchanged.

Unauthenticated browser page loads are redirected to /admin/login. API calls
made by the Admin UI's own JavaScript (``X-Requested-With: mnemodrome-admin``)
get a plain 401 so the browser never shows its Basic-auth pop-up; any other
client still gets the standard 401 + ``WWW-Authenticate: Basic`` challenge.

Changing ADMIN_PASSWORD invalidates every existing session.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import os
import secrets
import time
from urllib.parse import quote

from fastapi import HTTPException, Request, status
from fastapi.responses import JSONResponse, RedirectResponse, Response

from src import config

SESSION_COOKIE = "mnemodrome_admin_session"
UI_HEADER = "x-requested-with"
UI_HEADER_VALUE = "mnemodrome-admin"
SESSION_TTL_SECONDS = int(float(os.getenv("ADMIN_SESSION_TTL_HOURS", "12")) * 3600)

# Signing secret: set ADMIN_SESSION_SECRET to keep sessions across restarts.
# Without it a random per-process secret is used (everyone logs in again after
# a restart, which is acceptable for a single-operator admin page).
_PROCESS_SECRET = secrets.token_bytes(32)

# Login throttling: per client address, in memory.
LOGIN_MAX_FAILURES = 5
LOGIN_WINDOW_SECONDS = 300
_failures: dict[str, list[float]] = {}


class LoginRequiredError(Exception):
    """Raised by the auth dependency; turned into a redirect or a 401."""

    def __init__(self, browser_page: bool, ui_call: bool):
        self.browser_page = browser_page
        self.ui_call = ui_call


# ---------------------------------------------------------------------------
# Credentials and tokens
# ---------------------------------------------------------------------------


def _secret() -> bytes:
    configured = os.getenv("ADMIN_SESSION_SECRET", "")
    base = configured.encode() if configured else _PROCESS_SECRET
    # Bind sessions to the current credentials: a password change logs
    # everyone out.
    fingerprint = hashlib.sha256(
        f"{config.ADMIN_USERNAME}\0{config.ADMIN_PASSWORD}".encode()
    ).digest()
    return hashlib.sha256(base + fingerprint).digest()


def credentials_valid(username: str, password: str) -> bool:
    """Constant-time check against the configured admin credentials."""
    if not config.ADMIN_PASSWORD:
        return False
    user_ok = secrets.compare_digest(
        username.encode("utf8"), config.ADMIN_USERNAME.encode("utf8")
    )
    pass_ok = secrets.compare_digest(
        password.encode("utf8"), config.ADMIN_PASSWORD.encode("utf8")
    )
    return user_ok and pass_ok


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _unb64(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def make_session_token(username: str, now: float | None = None) -> str:
    expires = int((now or time.time()) + SESSION_TTL_SECONDS)
    payload = f"{username}|{expires}".encode()
    sig = hmac.new(_secret(), payload, hashlib.sha256).digest()
    return f"{_b64(payload)}.{_b64(sig)}"


def read_session_token(token: str | None, now: float | None = None) -> str | None:
    """Return the username for a valid, unexpired token, else None."""
    if not token or "." not in token:
        return None
    try:
        payload_b64, sig_b64 = token.split(".", 1)
        payload = _unb64(payload_b64)
        sig = _unb64(sig_b64)
    except Exception:
        return None
    expected = hmac.new(_secret(), payload, hashlib.sha256).digest()
    if not hmac.compare_digest(sig, expected):
        return None
    try:
        username, expires = payload.decode().rsplit("|", 1)
        if int(expires) < (now or time.time()):
            return None
    except ValueError:
        return None
    if not secrets.compare_digest(username.encode(), config.ADMIN_USERNAME.encode()):
        return None
    return username


# ---------------------------------------------------------------------------
# Request helpers
# ---------------------------------------------------------------------------


def is_https(request: Request) -> bool:
    proto = request.headers.get("x-forwarded-proto", "").split(",")[0].strip()
    return request.url.scheme == "https" or proto.lower() == "https"


def client_key(request: Request) -> str:
    forwarded = request.headers.get("x-real-ip") or ""
    host = request.client.host if request.client else "unknown"
    return forwarded.strip() or host


def wants_html_page(request: Request) -> bool:
    return request.method == "GET" and "text/html" in request.headers.get("accept", "")


def is_ui_call(request: Request) -> bool:
    return request.headers.get(UI_HEADER, "").lower() == UI_HEADER_VALUE


def safe_next(target: str | None) -> str:
    """Only allow redirects back into the Admin UI."""
    if (
        target
        and target.startswith("/admin")
        and not target.startswith("//")
        and "\\" not in target
        and not target.startswith("/admin/login")
    ):
        return target
    return "/admin/memory"


def origin_ok(request: Request) -> bool:
    """For cookie-authenticated writes: Origin, when sent, must be this host."""
    origin = request.headers.get("origin")
    if not origin:
        return True
    host = request.headers.get("x-forwarded-host") or request.headers.get("host", "")
    return origin.split("://", 1)[-1].rstrip("/") == host.split(",")[0].strip()


# ---------------------------------------------------------------------------
# Throttling
# ---------------------------------------------------------------------------


def throttled(key: str, now: float | None = None) -> bool:
    now = now or time.time()
    recent = [t for t in _failures.get(key, []) if now - t < LOGIN_WINDOW_SECONDS]
    _failures[key] = recent
    return len(recent) >= LOGIN_MAX_FAILURES


def record_failure(key: str, now: float | None = None) -> None:
    _failures.setdefault(key, []).append(now or time.time())


def clear_failures(key: str) -> None:
    _failures.pop(key, None)


# ---------------------------------------------------------------------------
# Responses
# ---------------------------------------------------------------------------


def set_session_cookie(response: Response, request: Request, username: str) -> None:
    response.set_cookie(
        SESSION_COOKIE,
        make_session_token(username),
        max_age=SESSION_TTL_SECONDS,
        httponly=True,
        secure=is_https(request),
        samesite="strict",
        path="/",
    )


def clear_session_cookie(response: Response) -> None:
    response.delete_cookie(SESSION_COOKIE, path="/")


def login_required_response(request: Request, exc: LoginRequiredError) -> Response:
    if exc.browser_page:
        target = request.url.path
        if request.url.query:
            target += "?" + request.url.query
        return RedirectResponse(
            url=f"/admin/login?next={quote(safe_next(target), safe='')}",
            status_code=303,
        )
    headers = {} if exc.ui_call else {"WWW-Authenticate": "Basic"}
    return JSONResponse(
        status_code=status.HTTP_401_UNAUTHORIZED,
        content={"detail": "Not authenticated"},
        headers=headers,
    )


def forbidden(detail: str) -> HTTPException:
    return HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=detail)
