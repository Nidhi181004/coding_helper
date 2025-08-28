import typer
from rich import print as rprint
from rich.markdown import Markdown
from rich.panel import Panel

from .config import INCLUDE_EXT, INDEX_DIR
from .vectorstore import load_faiss
from .ingest import ingest_directory
from .agents import CodingAgents
from .rag import retrieve_context, join_context

app = typer.Typer(help="Autonomous Coding Helper (Offline: Ollama + HuggingFace)")

@app.command()
def ingest(path: str):
    total = ingest_directory(path, INCLUDE_EXT)
    rprint(Panel.fit(f"[bold green]Ingested {total} code chunks[/bold green]\nIndex saved at {INDEX_DIR}"))

def _agents():
    vs = load_faiss(INDEX_DIR)
    return CodingAgents(vs)

@app.command()
def explain(question: str):
    ans = _agents().explain(question)
    rprint(Markdown(f"### 🧠 Explanation\n\n{ans}"))

@app.command()
def bugs(question: str):
    ans = _agents().bugs(question)
    rprint(Markdown(f"### 🐞 Issues\n\n{ans}"))

@app.command()
def refactor(question: str):
    ans = _agents().refactor(question)
    rprint(Markdown(f"### 🛠️ Refactor\n\n{ans}"))

@app.command()
def search(term: str):
    docs = retrieve_context(load_faiss(INDEX_DIR), term, k=5)
    ctx = join_context(docs)
    rprint(Markdown(f"### 🔎 Context\n\n```\n{ctx}\n```"))

if __name__ == "__main__":
    app()
