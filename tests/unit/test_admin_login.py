"""Tests for the Admin UI form login + session cookie (Feature 039)."""

import base64
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from src import admin_auth
from src.admin_ui import app

BASIC = {"Authorization": "Basic " + base64.b64encode(b"admin:test-password").decode()}
HTML = {"Accept": "text/html,application/xhtml+xml"}
UI = {"X-Requested-With": "mnemodrome-admin"}


@pytest.fixture(autouse=True)
def _reset_throttle():
    admin_auth._failures.clear()
    yield
    admin_auth._failures.clear()


@pytest.fixture
def client():
    with patch("src.admin_ui.initialize_config_manager"):
        with TestClient(app, base_url="http://admin.test") as c:
            yield c


def _login(client, password="test-password", next_="/admin/memory/facts"):  # noqa: S107 (test value)
    return client.post(
        "/admin/login",
        data={"username": "admin", "password": password, "next": next_},
        follow_redirects=False,
    )


class TestTokens:
    def test_round_trip(self):
        token = admin_auth.make_session_token("admin")
        assert admin_auth.read_session_token(token) == "admin"

    def test_expired(self):
        token = admin_auth.make_session_token("admin", now=1000.0)
        assert admin_auth.read_session_token(token) is None

    def test_tampered(self):
        token = admin_auth.make_session_token("admin")
        payload, sig = token.split(".")
        forged = admin_auth._b64(b"admin|99999999999") + "." + sig
        assert admin_auth.read_session_token(forged) is None
        assert admin_auth.read_session_token("garbage") is None
        assert admin_auth.read_session_token(None) is None

    def test_password_change_invalidates(self):
        token = admin_auth.make_session_token("admin")
        with patch.object(admin_auth.config, "ADMIN_PASSWORD", "a-new-password-123"):
            assert admin_auth.read_session_token(token) is None

    def test_safe_next(self):
        assert (
            admin_auth.safe_next("/admin/memory/facts?x=1") == "/admin/memory/facts?x=1"
        )
        for bad in [
            None,
            "",
            "https://evil.example",
            "//evil.example",
            "/other",
            "/admin/login",
            "/admin\\..\\x",
        ]:
            assert admin_auth.safe_next(bad) == "/admin/memory"


class TestLoginFlow:
    def test_login_page_has_password_manager_fields(self, client):
        r = client.get("/admin/login")
        assert r.status_code == 200
        assert 'autocomplete="username"' in r.text
        assert 'autocomplete="current-password"' in r.text
        assert 'method="post"' in r.text
        assert r.headers["cache-control"] == "no-store"

    def test_browser_page_redirects_to_login(self, client):
        r = client.get(
            "/admin/memory/facts?entity=x", headers=HTML, follow_redirects=False
        )
        assert r.status_code == 303
        assert r.headers["location"].startswith("/admin/login?next=")
        assert "%2Fadmin%2Fmemory%2Ffacts" in r.headers["location"]

    def test_ui_api_call_gets_401_without_basic_challenge(self, client):
        r = client.get("/admin/api/memory/overview", headers=UI)
        assert r.status_code == 401
        assert "www-authenticate" not in {k.lower() for k in r.headers}

    def test_script_api_call_keeps_basic_challenge(self, client):
        r = client.get("/admin/api/memory/overview")
        assert r.status_code == 401
        assert r.headers["WWW-Authenticate"] == "Basic"
        assert r.json()["detail"] == "Not authenticated"

    def test_basic_auth_still_works(self, client):
        assert client.get("/admin/memory", headers=BASIC).status_code == 200

    def test_successful_login_sets_cookie_and_redirects(self, client):
        r = _login(client)
        assert r.status_code == 303
        assert r.headers["location"] == "/admin/memory/facts"
        cookie = r.headers["set-cookie"].lower()
        assert admin_auth.SESSION_COOKIE in cookie
        assert "httponly" in cookie
        assert "samesite=strict" in cookie
        assert "secure" not in cookie  # plain http in this test
        # cookie now authenticates pages and API calls
        assert client.get("/admin/memory", headers=HTML).status_code == 200
        assert client.get("/admin/health-check").status_code == 200

    def test_cookie_secure_behind_tls_proxy(self, client):
        r = client.post(
            "/admin/login",
            data={"username": "admin", "password": "test-password"},
            headers={"X-Forwarded-Proto": "https"},
            follow_redirects=False,
        )
        assert "secure" in r.headers["set-cookie"].lower()

    def test_wrong_password(self, client):
        r = _login(client, password="nope")  # noqa: S106 (wrong password on purpose)
        assert r.status_code == 303
        assert "error=invalid" in r.headers["location"]
        assert "set-cookie" not in r.headers

    def test_open_redirect_blocked(self, client):
        r = _login(client, next_="https://evil.example/")
        assert r.headers["location"] == "/admin/memory"

    def test_throttle_after_repeated_failures(self, client):
        for _ in range(admin_auth.LOGIN_MAX_FAILURES):
            _login(client, password="nope")  # noqa: S106 (wrong password on purpose)
        r = _login(client)  # right password, still throttled
        assert "error=throttled" in r.headers["location"]
        assert "set-cookie" not in r.headers

    def test_logout_clears_cookie(self, client):
        _login(client)
        r = client.get("/admin/logout", follow_redirects=False)
        assert r.status_code == 303
        assert "logged_out=1" in r.headers["location"]
        assert admin_auth.SESSION_COOKIE in r.headers["set-cookie"]
        r = client.get("/admin/memory", headers=HTML, follow_redirects=False)
        assert r.status_code == 303

    def test_logged_in_user_skips_login_form(self, client):
        _login(client)
        r = client.get("/admin/login?next=/admin/memory/search", follow_redirects=False)
        assert r.status_code == 303
        assert r.headers["location"] == "/admin/memory/search"

    def test_cookie_write_from_other_origin_refused(self, client):
        _login(client)
        r = client.post(
            "/admin/api/memory/files/refresh",
            headers={**UI, "Origin": "https://evil.example"},
        )
        assert r.status_code == 403
        r = client.post(
            "/admin/api/memory/files/refresh",
            headers={**UI, "Origin": "http://admin.test"},
        )
        assert r.status_code != 403
