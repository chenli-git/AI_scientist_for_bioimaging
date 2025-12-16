"""
Test usage tracking functionality
"""

import sys
sys.path.insert(0, '/Users/chenli/Documents/pythonprojects/AI_scientist_for_bioimaging')

import aibioagent as aba

# Test 1: Reset and verify
print("=" * 70)
print("Test 1: Reset usage stats")
print("=" * 70)
aba.reset_usage_stats()
stats = aba.get_usage_stats()
print(f"After reset - Total calls: {stats['total_calls']}")
print(f"After reset - Total cost: ${stats['total_cost_usd']:.4f}")
assert stats['total_calls'] == 0, "Reset should set calls to 0"
assert stats['total_cost_usd'] == 0.0, "Reset should set cost to 0"
print("✅ Reset test passed\n")

# Test 2: Manual tracking
print("=" * 70)
print("Test 2: Manual token tracking")
print("=" * 70)
from core.usage_tracker import get_tracker
tracker = get_tracker()

# Simulate an LLM call
tracker.track_llm_call(
    model="gpt-4o-mini",
    input_tokens=100,
    output_tokens=50
)

stats = aba.get_usage_stats()
print(f"After 1 LLM call:")
print(f"  Total calls: {stats['total_calls']}")
print(f"  Total tokens: {stats['total_tokens']}")
print(f"  LLM calls: {stats['llm_calls']}")
print(f"  Total cost: ${stats['total_cost_usd']:.6f}")

assert stats['total_calls'] == 1, "Should have 1 call"
assert stats['total_tokens'] == 150, "Should have 150 tokens"
assert stats['llm_calls'] == 1, "Should have 1 LLM call"
assert stats['total_cost_usd'] > 0, "Cost should be > 0"
print("✅ Tracking test passed\n")

# Test 3: Multiple calls
print("=" * 70)
print("Test 3: Multiple API calls")
print("=" * 70)
tracker.track_llm_call("gpt-4o-mini", 200, 100)
tracker.track_embedding_call("text-embedding-3-small", 500)
tracker.track_vision_call("gpt-4o", 300, 150)

stats = aba.get_usage_stats()
print(f"After 4 total calls:")
print(f"  Total calls: {stats['total_calls']}")
print(f"  LLM calls: {stats['llm_calls']}")
print(f"  Embedding calls: {stats['embedding_calls']}")
print(f"  Vision calls: {stats['vision_calls']}")
print(f"  Total tokens: {stats['total_tokens']:,}")
print(f"  Total cost: ${stats['total_cost_usd']:.6f}")

assert stats['total_calls'] == 4, "Should have 4 calls"
# Vision calls are also counted as LLM calls (they use the same underlying API)
assert stats['llm_calls'] == 3, "Should have 3 LLM calls (2 regular + 1 vision)"
assert stats['embedding_calls'] == 1, "Should have 1 embedding call"
assert stats['vision_calls'] == 1, "Should have 1 vision call"
print("✅ Multiple calls test passed\n")

# Test 4: Print summary
print("=" * 70)
print("Test 4: Print summary")
print("=" * 70)
aba.get_usage_stats(print_summary=True)
print("✅ Summary test passed\n")

# Test 5: Save to file
print("=" * 70)
print("Test 5: Save to file")
print("=" * 70)
import tempfile
import os
import json

temp_file = tempfile.mktemp(suffix=".json")
aba.get_usage_stats(save_to_file=temp_file)

# Verify file was created and contains valid JSON
assert os.path.exists(temp_file), "File should be created"
with open(temp_file, 'r') as f:
    saved_data = json.load(f)
    assert saved_data['total_calls'] == 4, "Saved data should match"
    print(f"Saved to: {temp_file}")
    print(f"Saved data contains {len(saved_data)} keys")

os.remove(temp_file)
print("✅ Save to file test passed\n")

# Test 6: Model breakdown
print("=" * 70)
print("Test 6: Model breakdown")
print("=" * 70)
stats = aba.get_usage_stats()
print(f"Models used: {list(stats['by_model'].keys())}")
for model, model_stats in stats['by_model'].items():
    print(f"  {model}: {model_stats['calls']} calls, ${model_stats['cost_usd']:.6f}")
print("✅ Model breakdown test passed\n")

print("\n" + "=" * 70)
print("🎉 ALL TESTS PASSED!")
print("=" * 70)
print("\nUsage tracking is working correctly!")
print("Users can now:")
print("  1. Track token usage with get_usage_stats()")
print("  2. View costs with print_summary=True")
print("  3. Save logs with save_to_file='path.json'")
print("  4. Reset tracking with reset_usage_stats()")
