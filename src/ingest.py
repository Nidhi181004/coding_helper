from pathlib import Path
from typing import Iterable
from .utils import docs_from_dir
from .vectorstore import build_faiss_from_docs, save_faiss

def ingest_directory(dir_path: str, include_ext: Iterable[str]) -> int:
    p = Path(dir_path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"Directory not found: {p}")
    docs = docs_from_dir(p, include_ext)
    if not docs:
        return 0
    vs = build_faiss_from_docs(docs)
    save_faiss(vs)
    return len(docs)
