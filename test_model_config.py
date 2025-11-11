"""
Quick test script for new model configuration functions.
"""

# Test that the functions are defined and have proper signatures
import inspect

print("Checking new model configuration functions...\n")

# Temporarily bypass dotenv requirement for testing
import sys
sys.modules['dotenv'] = type(sys)('dotenv')
sys.modules['dotenv'].load_dotenv = lambda: None

# Mock settings to avoid import errors
import config
config.settings = type(sys)('settings')
config.settings.OPENAI_API_KEY = "test-key"
config.settings.LLM_MODEL = "gpt-5-nano"
config.settings.VISION_LLM_MODEL = "gpt-4o-mini"
config.settings.EMBED_MODEL = "text-embedding-3-small"
config.settings.CHROMA_DIR = "data/chroma_db"

# Import functions
from aibioagent import (
    set_llm_model,
    set_vision_model, 
    set_embed_model,
    get_models,
    info
)

print("✅ All new functions imported successfully!\n")

# Check function signatures
functions = {
    "set_llm_model": set_llm_model,
    "set_vision_model": set_vision_model,
    "set_embed_model": set_embed_model,
    "get_models": get_models,
    "info": info
}

print("Function Signatures:")
print("=" * 60)
for name, func in functions.items():
    sig = inspect.signature(func)
    print(f"{name}{sig}")
    doc = func.__doc__
    if doc:
        first_line = doc.strip().split('\n')[0]
        print(f"  → {first_line}")
    print()

print("=" * 60)
print("\n✅ All syntax checks passed!")
print("\nNew API functions:")
print("  • aba.set_llm_model(name)      - Set text generation model")
print("  • aba.set_vision_model(name)   - Set vision analysis model")
print("  • aba.set_embed_model(name)    - Set embedding model")
print("  • aba.get_models()             - Get current models")
print("  • aba.info()                   - Now shows embedding model!")
