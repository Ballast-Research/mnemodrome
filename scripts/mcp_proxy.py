"""
MCP Proxy for Claude Desktop
Bridges Claude Desktop (STDIO) to Mnemodrome's Streamable HTTP server.
"""

import os

from fastmcp import Client, FastMCP

# Create client pointing to your running Streamable HTTP server
# (fastmcp auto-detects the transport from the URL). Set MNEMODROME_MCP_URL to override.
MNEMODROME_URL = os.environ.get("MNEMODROME_MCP_URL", "http://localhost:8765/mcp")
client = Client(MNEMODROME_URL, name="Mnemodrome")

# Create proxy that runs via STDIO for Claude Desktop
proxy = FastMCP.as_proxy(client, name="Mnemodrome Proxy")

if __name__ == "__main__":
    proxy.run()  # STDIO transport for Claude Desktop
