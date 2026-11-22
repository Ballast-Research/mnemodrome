# Contributing to Mnemodrome

Thank you for your interest in contributing to Mnemodrome! This document provides guidelines for contributing to the project.

## Code of Conduct

Be respectful, professional, and constructive in all interactions.

## How to Contribute

### Reporting Bugs

1. Check [existing issues](https://github.com/Ballast-Research/mnemodrome/issues) first
2. Create a new issue with:
   - Clear description of the bug
   - Steps to reproduce
   - Expected vs actual behavior
   - Environment details (OS, Python version, LLM provider)
   - Relevant logs or error messages

### Suggesting Features

1. Check [existing feature requests](https://github.com/Ballast-Research/mnemodrome/issues?q=is%3Aissue+label%3Aenhancement)
2. Create a new issue with:
   - Clear use case description
   - Expected behavior
   - Why this would benefit users
   - Implementation suggestions (optional)

### Pull Requests

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Follow the design principles**: Read [`docs/product/PHILOSOPHY.md`](docs/product/PHILOSOPHY.md)
4. **Write tests**: All new features require tests
5. **Update documentation**: Keep README, INSTALLATION.md, and inline docs current
6. **Commit with good messages**: Follow conventional commits format
7. **Submit PR**: Include description of changes and link to related issues

## Development Setup

### Prerequisites

- Python 3.11+
- Neo4j 5.0+
- Redis 7.0+ (optional but recommended)
- Ollama (for embeddings)

### Local Development

```bash
# Clone your fork
git clone https://github.com/Ballast-Research/mnemodrome.git
cd mnemodrome

# Install dependencies (requires uv: https://docs.astral.sh/uv/)
uv sync

# Configure environment
cp .env.example .env
# Edit .env with your settings

# Run tests
uv run pytest tests/unit/

# Run linting
uv run ruff check .
```

## Project Structure

```
mnemodrome/
├── src/
│   ├── config.py              # Configuration management
│   ├── mcp_server.py          # MCP server entry point
│   ├── models/                # Pydantic models
│   └── services/              # Business logic services
├── tests/
│   ├── unit/                  # Unit tests
│   ├── integration/           # Integration tests
│   └── contract/              # Contract tests
├── scripts/                   # Utility scripts
└── docs/assets/               # Demo media
```

## Coding Standards

### Python Style

- Follow PEP 8
- Use type hints (Python 3.11+ syntax)
- Maximum line length: 100 characters
- Use descriptive variable names
- Document all public functions/classes

### Example

```python
def capture_learning(
    title: str, content: str, category: str = "general", auto_deduplicate: bool = True
) -> str:
    """
    Capture structured learning in dual memory layers.

    Args:
        title: Brief, descriptive learning title
        content: Full learning content (markdown supported)
        category: Organization category
        auto_deduplicate: Enable pre-capture validation

    Returns:
        Success message with file path and Neo4j UUID

    Raises:
        ValueError: If title or content is empty
    """
    # Implementation
```

### Testing

- Write unit tests for all new functions
- Write integration tests for MCP tools
- Use contract tests for critical interfaces
- Aim for >80% coverage
- Test with all supported LLM providers (Anthropic, OpenAI, Ollama)

### Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat(capture): add deduplication to capture_learning
fix(vector): resolve Qdrant connection timeout
docs(readme): update installation instructions
test(conflict): add contract tests for bidirectional detection
refactor(services): extract entity factory pattern
```

## Feature Development Process

Feature specs are maintained by the core team outside this repo. Contributors
open an issue; no spec files are needed in a PR. See
[ADR-003](docs/decisions/ADR-003-specs-maintained-privately.md).

1. Open an issue describing the use case (see "Suggesting Features").
2. Wait for a maintainer to agree on the approach before writing code.
3. Implement with tests, update user documentation, and add a `CHANGELOG.md`
   entry under `[Unreleased]`.
4. State in the PR whether it changes the public API surface
   ([versioning policy](docs/release/versioning-policy.md)).

## Documentation

### Code Documentation

- Docstrings for all public APIs
- Inline comments for complex logic
- Type hints for all function signatures

### User Documentation

Update relevant files:
- `README.md` - For new features
- `INSTALLATION.md` - For setup changes
- Tool docstrings in `src/mcp_server.py` - For new or changed MCP tools (tools are self-describing via the MCP protocol)

## Testing Strategy

### Unit Tests

```bash
# Run all unit tests
uv run pytest tests/unit/

# Run specific test file
uv run pytest tests/unit/test_vector_service.py

# Run with coverage
uv run pytest --cov=src --cov-report=html
```

### Integration Tests

```bash
# Requires running Neo4j, Redis, Qdrant
uv run pytest tests/integration/

# Test specific MCP tool
uv run pytest tests/integration/test_capture_learning.py
```

### Contract Tests

```bash
# Test API contracts
uv run pytest tests/contract/
```

## LLM Provider Testing

Test with all supported providers:

```bash
# Test with Anthropic (Claude)
LLM_PROVIDER=anthropic uv run pytest

# Test with OpenAI
LLM_PROVIDER=openai uv run pytest

# Test with Ollama (local)
LLM_PROVIDER=ollama uv run pytest
```

## Release Process

Releases are cut by maintainers. Contributors never change the version: it lives
only in `pyproject.toml` and changes only in the release commit. Add your entry
under `[Unreleased]` in `CHANGELOG.md` instead.

- Policy: [docs/release/versioning-policy.md](docs/release/versioning-policy.md)
- Procedure: [docs/release/cut-a-release.md](docs/release/cut-a-release.md)
- Overview: [docs/release/README.md](docs/release/README.md)

## Need Help?

- **Documentation**: Start with [README.md](README.md) and [INSTALLATION.md](INSTALLATION.md)
- **GitHub Issues**: [Report bugs or request features](https://github.com/Ballast-Research/mnemodrome/issues)
- **Discussions**: [Ask questions](https://github.com/Ballast-Research/mnemodrome/discussions)

## License

By contributing, you agree that your contributions will be licensed under the Apache License 2.0.
