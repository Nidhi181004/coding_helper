import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")

INDEX_DIR = Path(os.getenv("INDEX_DIR", "data/index")).resolve()
INCLUDE_EXT = [e.strip() for e in os.getenv("INCLUDE_EXT", ".py,.js,.ts,.jsx,.tsx,.java,.md").split(",")]

INDEX_DIR.mkdir(parents=True, exist_ok=True)
