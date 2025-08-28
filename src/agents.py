from langchain_ollama import ChatOllama
from langchain_community.vectorstores import FAISS
from .prompts import EXPLAIN_TEMPLATE, BUGS_TEMPLATE, REFACTOR_TEMPLATE
from .rag import retrieve_context, join_context
from .config import OLLAMA_MODEL

class CodingAgents:
    def __init__(self, vs: FAISS):
        self.vs = vs
        self.llm = ChatOllama(model=OLLAMA_MODEL, temperature=0.2)

    def _run(self, template, question: str, k: int = 4):
        docs = retrieve_context(self.vs, question, k=k)
        context = join_context(docs) if docs else "No context found."
        prompt = template.format_messages(question=question, context=context)
        resp = self.llm.invoke(prompt)
        return resp.content

    def explain(self, question: str): return self._run(EXPLAIN_TEMPLATE, question)
    def bugs(self, question: str): return self._run(BUGS_TEMPLATE, question)
    def refactor(self, question: str): return self._run(REFACTOR_TEMPLATE, question)

