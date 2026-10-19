# Mnemodrome Interface Adapters

Alternative interfaces for Mnemodrome's dual-memory system, each optimized for specific use cases while sharing the same core service layer.

## Available Interfaces

### 1. Code-First API (`code_first.py`)

**Purpose**: Token-efficient Python interface for Claude Code sessions

**Token Efficiency**: Achieves **98.7% reduction** vs MCP protocol (150k → 2k tokens)

**Best For**:
- Claude Code interactive sessions
- Python applications with direct Mnemodrome access
- Token-constrained environments
- Fast iteration and development

### 2. MCP Server (Baseline)

**Purpose**: Model Context Protocol standard interface

**Best For**:
- MCP-compatible AI tools
- Cross-tool compatibility
- Standard protocol adherence

### 3. Pipeline Interface (Future)

**Purpose**: Open Web UI Pipeline integration

**Best For**:
- Conversational memory in Open Web UI
- Inlet/outlet processing
- Chat session persistence

---

## Code-First API

### Quick Start

```python
from src.interfaces.code_first import MnemodromeAPI

# Initialize API
api = MnemodromeAPI()
await api.initialize()

# Load recent context
context = await api.load_context("Ollama infrastructure", days=7)
print(context)

# Capture a fact
result = await api.capture_fact(
    "Ollama", "RUNS_ON", "server-01", context="User-level service for S-drive access"
)

# Cleanup
await api.close()
```

### Async Context Manager (Recommended)

```python
from src.interfaces.code_first import MnemodromeAPI

async with MnemodromeAPI() as api:
    # Load context
    context = await api.load_context("current topic")

    # Capture facts as you learn
    await api.capture_fact("Entity1", "RELATES_TO", "Entity2")

    # Capture complex scenarios
    await api.capture_context(
        "Alice and Bob collaborated with Charlie to implement Phase 4a "
        "using FastMCP and Neo4j"
    )

    # Auto-cleanup on exit
```

---

## API Reference

### Initialization & Lifecycle

#### `__init__()`
Create MnemodromeAPI instance (services not initialized yet).

#### `initialize(redis_host=None, redis_port=None, redis_db=None, redis_max_connections=10)`
Initialize Neo4j (required) and Redis (optional with graceful degradation).

**Parameters**:
- `redis_host`: Redis host (default from config)
- `redis_port`: Redis port (default from config)
- `redis_db`: Redis database number (default from config)
- `redis_max_connections`: Connection pool size

**Raises**: `RuntimeError` if Neo4j connection fails

#### `close()`
Close all service connections. Safe to call multiple times (idempotent).

---

### Memory Operations

#### `load_context(prompt="", days=7, limit=10) -> str`
Load recent context using temporal-first strategy with entity extraction.

**Parameters**:
- `prompt`: Optional prompt for entity/keyword filtering
- `days`: Time window for temporal context (default: 7)
- `limit`: Maximum relationships to return (default: 10)

**Returns**: Formatted markdown with recent context

**Example**:
```python
# Load context about specific topic
context = await api.load_context("Ollama infrastructure", days=7)

# Load all recent activity
recent = await api.load_context(days=3, limit=20)
```

---

#### `capture_fact(entity1, relation, entity2, context="", entity1_type=None, entity2_type=None) -> dict`
Capture factual relationship between two entities.

**Parameters**:
- `entity1`: Source entity name
- `relation`: Relationship type
- `entity2`: Target entity name
- `context`: Optional context/evidence
- `entity1_type`: Optional entity type override
- `entity2_type`: Optional entity type override

**Returns**: Dict with capture results
```python
{
    "status": "complete" | "partial",
    "entity1_uuid": str,
    "entity2_uuid": str,
    "relationship": dict,
    "markdown_path": str | None,
    "duration_ms": float,
}
```

**Example**:
```python
# Auto-classify entity types
result = await api.capture_fact(
    "Ollama", "RUNS_ON", "server-01", context="User-level service"
)

# Override entity types
result = await api.capture_fact(
    "Alice", "DEVELOPS", "Mnemodrome", entity1_type="Person", entity2_type="Project"
)
```

---

#### `capture_context(context) -> dict`
Extract and capture multi-entity graph from rich natural language.

**Parameters**:
- `context`: Natural language description with multiple entities/relationships

**Returns**: Dict with extraction results
```python
{
    "entities_created": int,
    "relationships_created": int,
    "entity_uuids": dict[str, str],
    "extraction_time_ms": float,
}
```

**Example**:
```python
result = await api.capture_context(
    "Alice and Bob collaborated with Charlie to implement Phase 4a "
    "using FastMCP and Neo4j, which resolved the entity duplication bug"
)
print(f"Created {result['entities_created']} entities")
```

---

#### `capture_learning(title, content, category="general") -> dict`
Capture structured learning in dual-memory layers (Neo4j + markdown).

**Parameters**:
- `title`: Learning title (brief, descriptive)
- `content`: Full learning content (markdown supported)
- `category`: Category (e.g., insight, solution, pattern, breakthrough)

**Returns**: Dict with capture results
```python
{"uuid": str, "file_path": str, "duration_ms": float}
```

**Example**:
```python
result = await api.capture_learning(
    title="Factory Pattern Eliminates Entity Duplication",
    content="Using Gang of Four Factory Pattern with MERGE ensures...",
    category="breakthrough",
)
```

---

#### `capture_session(conversation_summary, key_insights=None) -> dict`
Capture end-of-session knowledge with graph integration.

**Parameters**:
- `conversation_summary`: High-level summary of session
- `key_insights`: Optional list of key takeaways

**Returns**: Dict with capture results

**Example**:
```python
result = await api.capture_session(
    conversation_summary="Implemented Phase 4.5 factory pattern",
    key_insights=[
        "Gang of Four pattern eliminated duplicates",
        "UUID-based resolution ensures single canonical node",
    ],
)
```

---

### Temporal Validity Operations

#### `invalidate_fact(entity1, relation, entity2, reason, invalidation_type="both") -> dict`
Mark fact as no longer valid (non-destructive, preserves history).

**Parameters**:
- `entity1`: Source entity name
- `relation`: Relationship type
- `entity2`: Target entity name
- `reason`: Human-readable reason for invalidation
- `invalidation_type`: "natural", "system", or "both"

**Returns**: Dict with invalidation status

**Example**:
```python
result = await api.invalidate_fact(
    "Mnemodrome",
    "RUNS_ON",
    "server-02",
    reason="Migrated to server-01",
    invalidation_type="natural",
)
```

---

#### `update_fact(entity1, relation, entity2_old, entity2_new, context="", conflict_resolution="prompt") -> dict`
Update fact by atomically superseding it (invalidate old + create new).

**Parameters**:
- `entity1`: Source entity name
- `relation`: Relationship type
- `entity2_old`: Old target entity (being replaced)
- `entity2_new`: New target entity (replacement)
- `context`: Optional context for new fact
- `conflict_resolution`: "prompt", "auto", or "error"

**Returns**: Dict with update status

**Example**:
```python
result = await api.update_fact(
    "Mnemodrome",
    "RUNS_ON",
    entity2_old="server-02",
    entity2_new="server-01",
    context="Infrastructure migration",
)
```

---

### Utility Methods

#### `health() -> dict`
Health check for monitoring and orchestration.

**Returns**: Service connectivity status
```python
{
    "status": "healthy" | "degraded",
    "neo4j": "connected" | "disconnected",
    "redis": "connected" | "disconnected",
    "timestamp": float,
}
```

#### `discover_schema() -> dict`
Discover actual Neo4j schema (source of truth).

**Returns**: Sample entities and relationships from production graph

---

## MCP vs Code-First Comparison

| Aspect | MCP Server | Code-First API |
|--------|------------|----------------|
| **Token Usage** | ~150,000 tokens | ~2,000 tokens |
| **Reduction** | Baseline | **98.7%** |
| **Initialization** | Protocol overhead | Direct import |
| **Method Calls** | JSON-RPC | Native Python `await` |
| **Type Safety** | Runtime validation | Python type hints |
| **IDE Support** | Limited | Full autocomplete |
| **Documentation** | External | Inline docstrings |
| **Best For** | MCP tools | Claude Code, Python apps |

---

## Performance Characteristics

### Token Efficiency

**MCP Protocol** (baseline):
```
- Tool definitions: ~150,000 tokens
- Per-session overhead: Constant
- Load time: All tools upfront
```

**Code-First API**:
```
- Import statement: ~200 tokens
- Initialization: ~500 tokens
- Method calls: ~100-300 tokens each
- Total typical session: ~2,000 tokens
```

### Latency

- **Initialization**: <500ms (Neo4j + Redis)
- **Context loading**: 50-200ms (depends on query complexity)
- **Fact capture**: 100-300ms (includes classification + markdown)
- **Graph extraction**: 500-1500ms (LLM-based entity extraction)

### Memory Usage

- Shared services with MCP server (no duplication)
- Connection pooling (Neo4j: default, Redis: 10 max)
- Graceful degradation when Redis unavailable

---

## Troubleshooting

### Initialization Errors

**Problem**: `RuntimeError: Neo4j connection failed`

**Solution**:
```python
# Check Neo4j is running
await neo4j_service.ping()

# Verify configuration
from src.config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

print(f"Connecting to: {NEO4J_URI}")
```

---

### Redis Unavailable

**Behavior**: API continues with graceful degradation (no caching)

**Check Status**:
```python
status = await api.health()
print(f"Redis: {status['redis']}")  # "connected" or "disconnected"
```

---

### Method Calls Before Initialization

**Problem**: `RuntimeError: MnemodromeAPI not initialized`

**Solution**: Always call `initialize()` before using methods
```python
api = MnemodromeAPI()
await api.initialize()  # ← Required!
context = await api.load_context()
```

Or use async context manager:
```python
async with MnemodromeAPI() as api:  # Auto-initializes
    context = await api.load_context()
```

---

### Type Hints Not Working

**Problem**: IDE not showing autocomplete

**Solution**: Ensure type hints are visible
```python
from src.interfaces.code_first import MnemodromeAPI
from typing import Any

api: MnemodromeAPI = MnemodromeAPI()
result: dict[str, Any] = await api.capture_fact(...)
```

---

## Architecture Notes

### Thin Wrapper Pattern

Code-first interface is a **thin wrapper** over Mnemodrome's service layer:

```
MnemodromeAPI (interface)
    ↓
Service Layer (business logic)
    ↓
Neo4j + Redis (storage)
```

**Benefits**:
- No business logic duplication
- Consistent behavior across interfaces
- Easy maintenance and updates
- Shared connection pooling

### Service Delegation

All methods delegate to existing services:
- `graph_factory` - Entity extraction and graph building
- `classification_service` - Entity type classification
- `entity_factory` - Entity deduplication and resolution
- `relationship_factory` - Temporal relationship creation
- `file_service` - Markdown file writing
- `neo4j_service` - Graph queries and writes
- `redis_service` - Caching (optional)

---

## Examples

### Complete Session Example

```python
from src.interfaces.code_first import MnemodromeAPI


async def analyze_and_capture():
    async with MnemodromeAPI() as api:
        # 1. Load context from previous work
        context = await api.load_context("Ollama deployment", days=7)
        print(f"Found context:\n{context}\n")

        # 2. Capture new facts as we learn
        await api.capture_fact(
            "Ollama",
            "RUNS_AS",
            "user-level service",
            context="Needs S-drive access for models",
        )

        # 3. Capture complex scenario
        await api.capture_context(
            "User-level service runs on gpu-server with CUDA GPUs 0,1,2 "
            "and reads models from /data/ollama-models"
        )

        # 4. Capture breakthrough learning
        await api.capture_learning(
            title="User-Level Service for Storage Access",
            content="""
            # Why User-Level Service?

            System-level services run as 'ollama' user without access to
            mounted storage. User-level services inherit user permissions,
            enabling S-drive access for model storage.

            ## Implementation
            - Service: `~/.config/systemd/user/ollama.service`
            - Models: `/data/ollama-models`
            - GPUs: CUDA_VISIBLE_DEVICES=0,1,2
            """,
            category="breakthrough",
        )

        # 5. End-of-session synthesis
        await api.capture_session(
            conversation_summary="Debugged Ollama service, discovered user-level requirement",
            key_insights=[
                "User-level services needed for mounted storage access",
                "System services lack permissions for S-drive",
                "Configuration in ~/.config/systemd/user/",
            ],
        )


# Run it
import asyncio

asyncio.run(analyze_and_capture())
```

---

## Contributing

When adding new methods to the code-first interface:

1. **Delegate to services** - Don't duplicate business logic
2. **Return structured dicts** - Not strings (unlike MCP)
3. **Add comprehensive docstrings** - Include examples
4. **Add type hints** - Use `dict[str, Any]` for returns
5. **Check initialization** - Call `self._check_initialized()`
6. **Add tests** - Mock services, test behavior
7. **Update this README** - Document new methods

---

## License

Part of Mnemodrome dual-memory system.
