from pathlib import Path
from typing import List
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from .config import EMBED_MODEL_NAME, INDEX_DIR

def _embeddings():
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)

def build_faiss_from_docs(docs: List[Document]) -> FAISS:
    return FAISS.from_documents(docs, _embeddings())

def save_faiss(vs: FAISS, path: Path = INDEX_DIR) -> None:
    vs.save_local(str(path))

def load_faiss(path: Path = INDEX_DIR) -> FAISS:
    return FAISS.load_local(str(path), _embeddings(), allow_dangerous_deserialization=True)

