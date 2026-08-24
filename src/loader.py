import os
import glob
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import DATA_DIR, CHUNK_SIZE, CHUNK_OVERLAP


def load_all_docs(folder=DATA_DIR):
    folder = str(folder)
    docs = []

    pdf_paths = sorted(glob.glob(os.path.join(folder, "*.pdf")))
    txt_paths = sorted(glob.glob(os.path.join(folder, "*.txt")))

    for path in pdf_paths:
        try:
            loaded = PyPDFLoader(path).load()
            fname = os.path.basename(path)
            date_hint = fname.replace("_", " ").replace(".pdf", "")
            for d in loaded:
                d.metadata["source_file"] = fname
                d.page_content = f"[Document: {date_hint}]\n{d.page_content}"
            docs.extend(loaded)
        except Exception as e:
            print(f"[skip] {os.path.basename(path)} -> {e}")

    for path in txt_paths:
        try:
            loaded = TextLoader(path, encoding="utf-8").load()
            for d in loaded:
                d.metadata["source_file"] = os.path.basename(path)
            docs.extend(loaded)
        except Exception as e:
            print(f"[skip] {os.path.basename(path)} -> {e}")

    print(f"files found  -> pdf: {len(pdf_paths)} | txt: {len(txt_paths)}")
    print(f"pages loaded -> {len(docs)}")
    return docs


def split_docs(docs, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    chunks = [c for c in chunks if len(c.page_content.strip()) > 50]
    print(f"chunks created -> {len(chunks)}")
    return chunks

    