#!/usr/bin/env python3
"""
Mnemodrome Deployment Verification Script

Verifies that the containerized Mnemodrome MCP server is fully operational by
performing a real MCP protocol handshake, listing tools, and executing the
echo tool end-to-end, plus optional Admin UI and Docker health checks.

Transport-aware: defaults to the current Streamable HTTP transport (served at
/mcp). Pass --transport sse to verify a server still on the deprecated SSE
transport (served at /sse).

Usage:
    uv run python scripts/verify_deployment.py
    uv run python scripts/verify_deployment.py --host your-server-ip --port 8765
    uv run python scripts/verify_deployment.py --transport sse
"""

import argparse
import asyncio
import contextlib
import subprocess
import sys
from datetime import datetime


@contextlib.asynccontextmanager
async def _mcp_streams(base_url: str, transport: str):
    """Yield (read, write) MCP streams for the chosen transport."""
    if transport == "sse":
        from mcp.client.sse import sse_client

        async with sse_client(f"{base_url}/sse") as (read, write):
            yield read, write
    else:
        from mcp.client.streamable_http import streamablehttp_client

        # streamablehttp_client yields a 3rd element (session-id getter) we don't need.
        async with streamablehttp_client(f"{base_url}/mcp") as (read, write, _):
            yield read, write


async def verify_deployment(
    host: str, port: int, transport: str, admin_port: int | None = None
) -> bool:
    """Run full deployment verification battery against the chosen transport."""
    from mcp import ClientSession

    base_url = f"http://{host}:{port}"
    endpoint = "/sse" if transport == "sse" else "/mcp"
    results: list[tuple[str, bool, str]] = []
    passed = 0
    failed = 0

    print(f"\n{'=' * 60}")
    print("  Mnemodrome Deployment Verification")
    print(f"  Target:    {base_url}{endpoint}")
    print(f"  Transport: {transport}")
    print(f"  Time:      {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 60}\n")

    def record(name: str, ok: bool, detail: str):
        nonlocal passed, failed
        results.append((name, ok, detail))
        if ok:
            passed += 1
        else:
            failed += 1
        status = "\033[32mPASS\033[0m" if ok else "\033[31mFAIL\033[0m"
        print(f"   [{status}] {detail}")

    # Tests 1-3: full MCP protocol over the chosen transport
    print(f"1-3. MCP handshake / tools / echo over {transport} ({endpoint})")
    tool_count = 0
    try:
        async with _mcp_streams(base_url, transport) as (read, write):
            async with ClientSession(read, write) as session:
                init = await session.initialize()
                si = init.serverInfo
                record("MCP initialize", True, f"Server: {si.name} v{si.version}")

                tools = await session.list_tools()
                tool_count = len(tools.tools)
                names = sorted(t.name for t in tools.tools)
                preview = ", ".join(names[:5])
                if tool_count > 5:
                    preview += f" ... (+{tool_count - 5} more)"
                record(
                    "MCP tools/list", tool_count > 0, f"{tool_count} tools: {preview}"
                )

                if "echo" in names:
                    test_msg = "Mnemodrome deployment verified!"
                    res = await session.call_tool("echo", {"message": test_msg})
                    text = " ".join(getattr(c, "text", "") for c in (res.content or []))
                    # MCP tool failures come back as a normal result with
                    # isError=True — guard against a false pass.
                    ok = not getattr(res, "isError", False) and test_msg in text
                    record("Echo tool", ok, f"Response: {text[:80]}")
                else:
                    record("Echo tool", False, "echo tool not registered")
    except Exception as e:
        record("MCP handshake", False, f"{type(e).__name__}: {e}")
        for name in ("MCP tools/list", "Echo tool"):
            if not any(r[0] == name for r in results):
                record(name, False, "Skipped — handshake failed")

    # Test 4: Admin UI (optional)
    test_num = 4
    if admin_port:
        print(f"{test_num}. Admin Web UI (port {admin_port})")
        try:
            import httpx

            async with httpx.AsyncClient() as client:
                resp = await client.get(f"http://{host}:{admin_port}/", timeout=5.0)
                ok = resp.status_code in (200, 307, 401)
                record("Admin Web UI", ok, f"HTTP {resp.status_code}")
        except Exception as e:
            record("Admin Web UI", False, str(e))
        test_num += 1

    # Test 5: Docker health
    print(f"{test_num}. Docker Health Status")
    try:
        result = subprocess.run(
            [
                "docker",
                "inspect",
                "--format",
                "{{.State.Health.Status}}",
                "mnemodrome-mcp-server",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        health = result.stdout.strip()
        record(
            "Docker healthcheck", health == "healthy", f"Status: {health or 'unknown'}"
        )
    except Exception as e:
        record("Docker healthcheck", False, str(e))

    # Summary
    total = passed + failed
    print(f"\n{'=' * 60}")
    if failed == 0:
        print(f"  \033[32mALL {total} TESTS PASSED\033[0m")
    else:
        print(f"  \033[31m{failed}/{total} FAILED\033[0m  |  {passed}/{total} passed")
    print(f"{'=' * 60}\n")

    return failed == 0


def main():
    parser = argparse.ArgumentParser(description="Mnemodrome Deployment Verification")
    parser.add_argument(
        "--host", default="localhost", help="MCP server host (default: localhost)"
    )
    parser.add_argument(
        "--port", type=int, default=8765, help="MCP server port (default: 8765)"
    )
    parser.add_argument(
        "--admin-port", type=int, default=None, help="Admin UI port (optional)"
    )
    parser.add_argument(
        "--transport",
        choices=("streamable-http", "sse"),
        default="streamable-http",
        help="MCP transport to verify (default: streamable-http; use 'sse' for legacy servers)",
    )
    args = parser.parse_args()

    try:
        import mcp  # noqa: F401
    except ImportError:
        print(
            "ERROR: the 'mcp' package is required. Run via: uv run python scripts/verify_deployment.py"
        )
        sys.exit(1)

    success = asyncio.run(
        verify_deployment(args.host, args.port, args.transport, args.admin_port)
    )
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
