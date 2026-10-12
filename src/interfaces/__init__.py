"""
Mnemodrome Interface Adapters

This package contains alternative interface adapters for Mnemodrome's core services:
- code_first: Direct Python API for Claude Code (token-efficient, code-first pattern)
- pipeline: Open Web UI Pipeline integration (future)

All interfaces share the same core service layer for consistency.
"""

from src.interfaces.code_first import MnemodromeAPI

__all__ = ["MnemodromeAPI"]
