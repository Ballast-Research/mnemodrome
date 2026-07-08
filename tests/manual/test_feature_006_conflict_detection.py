"""
Manual Test Script for Feature 006: Conflict Detection in capture_fact

Run this script with the MCP server running to test conflict detection functionality.

Prerequisites:
- Mnemodrome MCP server running
- Neo4j accessible
- Python environment with httpx installed

Usage:
    .venv/bin/python tests/manual/test_feature_006_conflict_detection.py
"""

import asyncio
import json
from datetime import datetime

import httpx


class MnemodromeTester:
    """Test client for Mnemodrome MCP server."""

    def __init__(self, base_url="http://localhost:8765"):
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=30.0)

    async def call_tool(self, tool_name: str, **kwargs):
        """Call an MCP tool via HTTP."""
        response = await self.client.post(
            f"{self.base_url}/tools/{tool_name}", json=kwargs
        )
        response.raise_for_status()
        return response.json()

    async def capture_fact(
        self, entity1, relation, entity2, context="", auto_resolve_conflicts=False
    ):
        """Wrapper for capture_fact tool."""
        return await self.call_tool(
            "capture_fact",
            entity1=entity1,
            relation=relation,
            entity2=entity2,
            context=context,
            auto_resolve_conflicts=auto_resolve_conflicts,
        )

    async def cleanup_test_data(self):
        """Clean up test entities from Neo4j."""
        # This would require a cleanup tool or direct Neo4j access
        print("⚠️  Manual cleanup required: Delete TestApp* entities from Neo4j")


async def test_conflict_detection():
    """Test conflict detection without auto-resolution."""
    print("\n" + "=" * 80)
    print("TEST 1: Conflict Detection (auto_resolve=False)")
    print("=" * 80)

    tester = MnemodromeTester()
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    test_entity = f"TestApp-{timestamp}"

    try:
        # Step 1: Create initial fact
        print(f"\n1️⃣  Creating initial fact: {test_entity} RUNS_ON server-alpha")
        result1 = await tester.capture_fact(
            entity1=test_entity,
            relation="RUNS_ON",
            entity2="server-alpha",
            context="Initial deployment for testing",
            auto_resolve_conflicts=False,
        )
        print("✅ Initial fact created")
        print(f"   Response type: {type(result1)}")

        # Step 2: Try to create conflicting fact (should detect conflict)
        print(f"\n2️⃣  Attempting conflicting fact: {test_entity} RUNS_ON server-beta")
        result2 = await tester.capture_fact(
            entity1=test_entity,
            relation="RUNS_ON",
            entity2="server-beta",
            context="Conflicting deployment",
            auto_resolve_conflicts=False,  # Explicit
        )

        # Verify conflict detection
        if isinstance(result2, dict) and result2.get("status") == "conflict_detected":
            print("✅ PASS: Conflict detected correctly")
            print(f"   Complexity score: {result2['conflict']['complexity_score']}")
            print(f"   Auto-resolvable: {result2['conflict']['auto_resolvable']}")
            print(f"   Existing facts: {len(result2['conflict']['existing_facts'])}")
            return True
        else:
            print("❌ FAIL: Expected conflict_detected, got:")
            print(json.dumps(result2, indent=2))
            return False

    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False
    finally:
        await tester.client.aclose()


async def test_auto_resolution():
    """Test automatic conflict resolution."""
    print("\n" + "=" * 80)
    print("TEST 2: Auto-Resolution (auto_resolve=True)")
    print("=" * 80)

    tester = MnemodromeTester()
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    test_entity = f"TestApp-AutoResolve-{timestamp}"

    try:
        # Step 1: Create initial fact
        print(f"\n1️⃣  Creating initial fact: {test_entity} RUNS_ON server-alpha")
        result1 = await tester.capture_fact(
            entity1=test_entity,
            relation="RUNS_ON",
            entity2="server-alpha",
            context="Initial deployment",
            auto_resolve_conflicts=False,
        )
        print("✅ Initial fact created")

        # Step 2: Create conflicting fact with auto-resolution
        print("\n2️⃣  Creating conflicting fact WITH auto-resolution")
        result2 = await tester.capture_fact(
            entity1=test_entity,
            relation="RUNS_ON",
            entity2="server-beta",
            context="Migration with auto-resolve",
            auto_resolve_conflicts=True,  # Enable auto-resolution
        )

        # Verify auto-resolution
        if isinstance(result2, dict) and result2.get("conflict_auto_resolved") is True:
            print("✅ PASS: Conflict auto-resolved successfully")
            print(f"   Invalidated facts: {len(result2['invalidated_facts'])}")
            print(f"   New fact UUID: {result2['new_fact_uuid'][:8]}...")
            print(f"   Complexity score: {result2['complexity_score']}")
            return True
        else:
            print("❌ FAIL: Expected auto-resolution, got:")
            print(json.dumps(result2, indent=2))
            return False

    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False
    finally:
        await tester.client.aclose()


async def test_no_conflict():
    """Test normal operation without conflicts."""
    print("\n" + "=" * 80)
    print("TEST 3: Normal Operation (No Conflicts)")
    print("=" * 80)

    tester = MnemodromeTester()
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    test_entity = f"TestApp-NoConflict-{timestamp}"

    try:
        print(f"\n1️⃣  Creating fact: {test_entity} RUNS_ON server-gamma")
        result = await tester.capture_fact(
            entity1=test_entity,
            relation="RUNS_ON",
            entity2="server-gamma",
            context="No conflict expected",
            auto_resolve_conflicts=False,
        )

        # Verify normal string response
        if isinstance(result, str) and "✅ Fact captured successfully" in result:
            print("✅ PASS: Normal operation without conflicts")
            return True
        else:
            print("❌ FAIL: Expected string response, got:")
            print(f"   Type: {type(result)}")
            print(f"   Content: {result}")
            return False

    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False
    finally:
        await tester.client.aclose()


async def main():
    """Run all tests."""
    print("\n" + "🧪 " * 40)
    print("Feature 006: Relationship Conflict Detection Tests")
    print("🧪 " * 40)

    results = []

    # Test 1: Conflict detection
    results.append(("Conflict Detection", await test_conflict_detection()))

    # Test 2: Auto-resolution
    results.append(("Auto-Resolution", await test_auto_resolution()))

    # Test 3: No conflict
    results.append(("Normal Operation", await test_no_conflict()))

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")

    print(f"\n📊 Results: {passed}/{total} tests passed ({passed / total * 100:.1f}%)")

    if passed == total:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print("\n⚠️  Some tests failed. Review output above.")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
