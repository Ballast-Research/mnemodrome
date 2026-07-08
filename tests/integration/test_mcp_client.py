#!/usr/bin/env python3
"""
Test MCP client to simulate what Claude Code does
"""

import asyncio
import json
import sys
from pathlib import Path


async def test_mcp_connection():
    """Simulate Claude Code connecting to MCP server"""

    # Start the server as subprocess
    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "src.mcp_server",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=str(Path(__file__).parent.parent),
    )

    print("Server process started", file=sys.stderr)

    # Send initialize request
    init_request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "test-client", "version": "1.0"},
        },
    }

    request_json = json.dumps(init_request) + "\n"
    print(f"Sending: {request_json.strip()}", file=sys.stderr)

    proc.stdin.write(request_json.encode())
    await proc.stdin.drain()

    # Read response
    response_line = await asyncio.wait_for(proc.stdout.readline(), timeout=5.0)
    print(f"Received: {response_line.decode().strip()}", file=sys.stderr)

    response = json.loads(response_line)
    print("\n✅ Server responded successfully!", file=sys.stderr)
    print(f"Server info: {response['result']['serverInfo']}", file=sys.stderr)

    # Send tools/list request
    tools_request = {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}

    request_json = json.dumps(tools_request) + "\n"
    print(f"\nSending: {request_json.strip()}", file=sys.stderr)

    proc.stdin.write(request_json.encode())
    await proc.stdin.drain()

    response_line = await asyncio.wait_for(proc.stdout.readline(), timeout=5.0)
    print(f"Received: {response_line.decode().strip()}", file=sys.stderr)

    tools_response = json.loads(response_line)
    print("\n✅ Tools list:", file=sys.stderr)
    for tool in tools_response["result"]["tools"]:
        print(f"  - {tool['name']}: {tool['description']}", file=sys.stderr)

    # Cleanup
    proc.stdin.close()
    await proc.wait()

    print("\n✅ Connection test successful!", file=sys.stderr)


if __name__ == "__main__":
    try:
        asyncio.run(test_mcp_connection())
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        sys.exit(1)
