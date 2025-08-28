from typing import List
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

def retrieve_context(vs: FAISS, question: str, k: int = 4) -> List[Document]:
    return vs.as_retriever(search_kwargs={"k": k}).get_relevant_documents(question)

def join_context(docs: List[Document]) -> str:
    return "\n\n".join([f"--- {d.metadata.get('source','')} ---\n{d.page_content}" for d in docs])
