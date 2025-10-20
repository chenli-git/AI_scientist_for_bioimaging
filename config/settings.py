import os
from dotenv import load_dotenv

# Load .env file into environment variables
load_dotenv()

# Access the API key and store constants
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LLM_MODEL = "gpt-5-nano"        # or "gpt-4o"
EMBED_MODEL = "text-embedding-3-small"    # vector embedding
CHROMA_DIR = "data/chroma_db"

if not OPENAI_API_KEY:
    raise ValueError("Missing OPENAI_API_KEY in .env")
