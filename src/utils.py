from pathlib import Path
from typing import Iterable, List, Tuple
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

DEFAULT_SEPARATORS = ["\nclass ", "\ndef ", "\nasync def ", "\nfunction ", "\nexport function ", "\n/* ", "\n# ", "\n// ", "\n"]

def read_files(root: Path, include_ext: Iterable[str]) -> List[Tuple[Path, str]]:
    results = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix in include_ext:
            try:
                text = p.read_text(encoding="utf-8", errors="ignore")
                results.append((p, text))
            except:
                pass
    return results

def split_code_text(text: str, chunk_size: int = 1200, chunk_overlap: int = 150):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap,
        separators=DEFAULT_SEPARATORS, length_function=len, is_separator_regex=False,
    )
    return splitter.split_text(text)

def docs_from_dir(dir_path: Path, include_ext: Iterable[str]):
    docs = []
    files = read_files(dir_path, include_ext)
    for path, text in files:
        for chunk in split_code_text(text):
            docs.append(Document(page_content=chunk, metadata={"source": str(path)}))
    return docs
