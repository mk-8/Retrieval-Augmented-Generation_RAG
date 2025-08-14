import argparse
import os
import shutil
from typing import Iterator
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document as LCDocument
from docling.document_converter import DocumentConverter
from langchain.vectorstores.chroma import Chroma
from get_embedding_function import get_embedding_function
from langchain_text_splitters import RecursiveCharacterTextSplitter

CHROMA_PATH = "./chroma"
DATA_PATH = "./Data/dps-report-2.pdf"

class DoclingPDFLoader(BaseLoader):
    def __init__(self, file_path: str | list[str]) -> None:
        self._file_paths = file_path if isinstance(file_path, list) else [file_path]
        self._converter = DocumentConverter()

    def lazy_load(self) -> Iterator[LCDocument]:
        for source in self._file_paths:
            dl_doc = self._converter.convert(source).document
            text = dl_doc.export_to_markdown()
            yield LCDocument(page_content=text)

    def load(self):
        return list(self.lazy_load())

def split_documents(documents: list[LCDocument]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)

def calculate_chunk_ids(chunks):
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id
        chunk.metadata["id"] = chunk_id

    return chunks

def add_to_chroma(chunks: list[LCDocument]):
    db = Chroma(
        persist_directory=CHROMA_PATH, 
        embedding_function=get_embedding_function()
    )

    chunks_with_ids = calculate_chunk_ids(chunks)
    existing_items = db.get(include=[])
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    new_chunks = [chunk for chunk in chunks_with_ids 
                 if chunk.metadata["id"] not in existing_ids]

    if new_chunks:
        print(f"👉 Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
        db.persist()
    else:
        print("No new documents to add")

def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    parser.add_argument("--file_path", type=str, default=None, help="Path to single PDF file")
    args = parser.parse_args()

    if args.reset:
        print("✨ Clearing Database")
        clear_database()

    # Use either single file or directory
    if args.file_path:
        loader = DoclingPDFLoader(file_path=args.file_path)
    else:
        loader = DoclingPDFLoader(file_path=[os.path.join(DATA_PATH, f) 
                                           for f in os.listdir(DATA_PATH) 
                                           if f.endswith('.pdf')])

    documents = loader.load()
    chunks = split_documents(documents)
    add_to_chroma(chunks)

if __name__ == "__main__":
    main()