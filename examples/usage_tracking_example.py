"""
Example: Using Usage Tracking & Cost Management

This example shows how to track token usage and costs
when using the aibioagent package.
"""

import aibioagent as aba

# Step 1: Set up API key (replace with your key)
# aba.set_api_key("sk-your-openai-key-here")

print("=" * 70)
print("Example: Token Usage & Cost Tracking")
print("=" * 70)

# Step 2: Reset usage stats to start fresh
print("\n1️⃣ Resetting usage statistics...")
aba.reset_usage_stats()

# Step 3: Do some work (this would normally make real API calls)
print("\n2️⃣ Simulating API usage...")
print("   (In real usage, you would call aba.ask(), aba.add_papers(), etc.)")

# Simulate usage for demonstration
from core.usage_tracker import get_tracker
tracker = get_tracker()

# Simulate a few queries
tracker.track_llm_call("gpt-4o-mini", input_tokens=250, output_tokens=150)
tracker.track_llm_call("gpt-4o-mini", input_tokens=300, output_tokens=200)
tracker.track_embedding_call("text-embedding-3-small", tokens=1200)
tracker.track_vision_call("gpt-4o", input_tokens=400, output_tokens=180)

print("   ✓ Simulated 4 API calls")

# Step 4: Check usage statistics
print("\n3️⃣ Checking usage statistics...")
stats = aba.get_usage_stats()

print(f"\n📊 Quick Stats:")
print(f"   Total API calls:  {stats['total_calls']}")
print(f"   Total tokens:     {stats['total_tokens']:,}")
print(f"   Estimated cost:   ${stats['total_cost_usd']:.4f}")

# Step 5: Print detailed summary
print("\n4️⃣ Detailed Summary:")
aba.get_usage_stats(print_summary=True)

# Step 6: Save to file for record keeping
print("5️⃣ Saving usage log to file...")
aba.get_usage_stats(save_to_file="usage_log.json")

# Step 7: Model breakdown
print("\n6️⃣ Cost breakdown by model:")
for model, model_stats in stats['by_model'].items():
    print(f"   {model}:")
    print(f"      Calls:  {model_stats['calls']}")
    print(f"      Tokens: {model_stats['total_tokens']:,}")
    print(f"      Cost:   ${model_stats['cost_usd']:.4f}")

print("\n" + "=" * 70)
print("💡 Usage Tips:")
print("=" * 70)
print("""
1. Reset before specific operations:
   aba.reset_usage_stats()
   # ... do work ...
   aba.get_usage_stats(print_summary=True)

2. Monitor costs in real-time:
   stats = aba.get_usage_stats()
   if stats['total_cost_usd'] > 1.0:
       print("Warning: Cost exceeded $1")

3. Keep logs for analysis:
   import datetime
   timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
   aba.get_usage_stats(save_to_file=f"logs/usage_{timestamp}.json")

4. Compare model costs:
   # Try different models and compare costs
   aba.set_llm_model("gpt-4o-mini")  # Cheaper
   aba.set_llm_model("gpt-4o")       # Better quality, higher cost
""")

print("\n✅ Example complete!")
