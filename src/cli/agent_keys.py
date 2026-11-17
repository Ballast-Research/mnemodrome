#!/usr/bin/env python3
"""
Agent key management - Feature 040 (per-agent MCP keys)

Manages the keys file named by MNEMODROME_AGENT_KEYS_FILE (or --file).
The file stores only SHA-256 digests. A new key is printed ONCE, on stdout,
by `add` and `rotate`; store it in your secret manager right away.

Usage:
    python -m src.cli.agent_keys add claude-code      # new agent, prints its key once
    python -m src.cli.agent_keys list                 # names and dates, never digests
    python -m src.cli.agent_keys rotate claude-code   # new key, old one stops working
    python -m src.cli.agent_keys revoke claude-code   # remove one agent's key
    python -m src.cli.agent_keys --file ./agent-keys.json list

The running server notices file changes within about a second; no restart.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from src.services import agent_auth

ENV_VAR = "MNEMODROME_AGENT_KEYS_FILE"


def _resolve_path(arg: str | None) -> Path:
    if arg:
        return Path(arg).expanduser()
    value = os.getenv(ENV_VAR, "")
    if not value:
        try:  # honour a local .env like the server does
            from dotenv import load_dotenv  # noqa: PLC0415

            load_dotenv()
            value = os.getenv(ENV_VAR, "")
        except ImportError:
            pass
    if not value:
        raise SystemExit(f"error: set {ENV_VAR} or pass --file PATH")
    return Path(value).expanduser()


def _print_key(name: str, key: str, verb: str) -> None:
    print(key)
    print(
        f"{verb} key for agent '{name}'. This is the only time it is shown; "
        "store it now. Clients send it as 'Authorization: Bearer <key>' "
        "(or 'X-Mnemodrome-Key: <key>').",
        file=sys.stderr,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.cli.agent_keys",
        description="Manage Mnemodrome per-agent API keys.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Usage:", 1)[1] if __doc__ else None,
    )
    parser.add_argument("--file", help=f"keys file (default: ${ENV_VAR})")
    sub = parser.add_subparsers(dest="command", required=True)

    add = sub.add_parser("add", help="add an agent and print its new key once")
    add.add_argument("name")
    sub.add_parser("list", help="list agents (never shows keys or digests)")
    rotate = sub.add_parser("rotate", help="replace an agent's key")
    rotate.add_argument("name")
    revoke = sub.add_parser("revoke", help="remove an agent's key")
    revoke.add_argument("name")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    path = _resolve_path(args.file)

    try:
        if args.command == "add":
            key = agent_auth.add_agent(path, args.name)
            _print_key(args.name, key, "Created")
        elif args.command == "rotate":
            key = agent_auth.rotate_agent(path, args.name)
            _print_key(args.name, key, "Rotated")
        elif args.command == "revoke":
            agent_auth.revoke_agent(path, args.name)
            print(f"Revoked key for agent '{args.name}'.")
        elif args.command == "list":
            agents = agent_auth.list_agents(path)
            if not agents:
                print("No agent keys.")
            else:
                print(f"{'NAME':<32} {'CREATED':<26} {'ROTATED':<26} SCOPES")
                for a in agents:
                    print(
                        f"{a['name']:<32} {a['created_at'] or '-':<26} "
                        f"{a['rotated_at'] or '-':<26} {','.join(a['scopes'])}"
                    )
    except (ValueError, KeyError) as exc:
        msg = exc.args[0] if exc.args else str(exc)
        print(f"error: {msg}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"error: {exc.strerror or exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
