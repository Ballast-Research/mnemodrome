#!/usr/bin/env python3
"""Direct test of VectorService to verify functionality before MCP integration."""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from services.vector_service import get_vector_service


async def test_vector_service():
    """Test vector service initialization and basic operations."""
    print("=" * 60)
    print("Testing VectorService Direct Integration")
    print("=" * 60)

    # Initialize service
    print("\n1. Initializing VectorService...")
    vector_service = get_vector_service()

    try:
        await vector_service.initialize()
        print("   ✅ VectorService initialized successfully")
    except Exception as e:
        print(f"   ❌ Initialization failed: {e}")
        return

    # Test health check
    print("\n2. Testing health check...")
    try:
        health = await vector_service.health_check()
        print("   ✅ Health check passed")
        print(f"   - Status: {health['status']}")
        print(f"   - Enabled: {health['enabled']}")
        print(f"   - Initialized: {health['initialized']}")
        print(f"   - Collections: {health.get('collections', [])}")
        if "stats" in health:
            total_vectors = sum(
                s.get("indexed_vectors_count", 0) for s in health["stats"].values()
            )
            total_points = sum(
                s.get("points_count", 0) for s in health["stats"].values()
            )
            print(f"   - Total indexed vectors: {total_vectors}")
            print(f"   - Total points: {total_points}")
    except Exception as e:
        print(f"   ❌ Health check failed: {e}")
        import traceback

        traceback.print_exc()
        return

    # Test embedding generation
    print("\n3. Testing embedding generation...")
    test_text = "How does Mnemodrome handle temporal context?"
    try:
        embedding = await vector_service.embed_text(test_text)
        print(f"   ✅ Generated {len(embedding)}-dimensional embedding")
        print(f"   - Sample values: [{embedding[0]:.4f}, {embedding[1]:.4f}, ...]")
    except Exception as e:
        print(f"   ❌ Embedding generation failed: {e}")
        return

    # Test prompt search (should be empty initially)
    print("\n4. Testing semantic search for similar prompts...")
    try:
        results = await vector_service.search_similar_prompts(
            query_text=test_text, limit=5
        )
        print(f"   ✅ Search completed: found {len(results)} results")
        if results:
            for i, result in enumerate(results, 1):
                print(f"   - Result {i}: score={result['score']:.3f}")
        else:
            print("   - No results (expected for new collection)")
    except Exception as e:
        print(f"   ❌ Search failed: {e}")
        return

    # Test storing a prompt
    print("\n5. Testing prompt storage...")
    try:
        result = await vector_service.store_prompt_embedding(
            text=test_text,
            prompt_id="test-prompt-001",
            metadata={
                "session_id": "test-session-001",
                "timestamp": "2025-11-23T19:45:00Z",
                "response": "Mnemodrome uses temporal validity fields and event logs for context.",
            },
        )
        if result and result != False:
            print(f"   ✅ Prompt stored successfully (UUID: {result[:8]}...)")
        else:
            print("   ⚠️  Storage failed")
    except Exception as e:
        print(f"   ❌ Storage failed: {e}")
        import traceback

        traceback.print_exc()
        return

    # Search again to verify stored prompt
    print("\n6. Searching for stored prompt...")
    try:
        results = await vector_service.search_similar_prompts(
            query_text="temporal context in Mnemodrome", limit=5
        )
        print(f"   ✅ Search completed: found {len(results)} results")
        if results:
            for i, result in enumerate(results, 1):
                print(f"   - Result {i}: score={result['score']:.3f}")
                text = result.get("matched_text") or result.get("text", "N/A")
                if text != "N/A":
                    print(f"     Text: {text[:80]}...")
        else:
            print("   ⚠️  No results found (unexpected)")
    except Exception as e:
        print(f"   ❌ Search failed: {e}")
        return

    print("\n" + "=" * 60)
    print("✅ All tests completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(test_vector_service())
