# build_vectordb.py (Corrected)

import json
import shutil
from pathlib import Path
import argparse
from typing import List, Dict
import os
import sqlite3
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm import tqdm

try:
    from rag_system.config import settings
except (ImportError, ModuleNotFoundError):
    print("无法从rag_system.config导入设置，将使用文件内的默认路径。")


    class SettingsFallback:
        SQLITE_DB_PATH = "../data/database/literature_materials.db"


    settings = SettingsFallback()

EMBEDDING_MODEL_NAME = "BAAI/bge-large-zh-v1.5"
EMBEDDING_DEVICE = "mps"
CHUNK_SIZE = 750
CHUNK_OVERLAP = 75


def get_authoritative_titles_from_sqlite(db_path: Path) -> Dict[str, str]:
    if not db_path.exists():
        print(f"❌ 错误: SQLite数据库在路径 '{db_path}' 未找到。无法获取权威标题。")
        return {}
    print(f"--- 正在从SQLite数据库 '{db_path}' 获取权威标题... ---")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT doi, title FROM Papers")
    results = cursor.fetchall()
    conn.close()
    title_map = {row[0]: row[1] for row in results}
    print(f"✅ 成功从SQLite加载了 {len(title_map)} 个权威标题。")
    return title_map


def load_and_prepare_documents(json_path: Path, title_map: Dict[str, str]) -> List[Document]:
    print(f"--- 正在从 {json_path} 加载论文全文内容... ---")
    if not json_path.exists():
        print(f"Error: Source JSON file not found at {json_path}")
        return []
    with open(json_path, 'r', encoding='utf-8') as f:
        papers_data = json.load(f)
    documents = []
    for paper in tqdm(papers_data, desc="Preparing documents with authoritative titles"):
        page_content = paper.get("llm_ready_fulltext_cleaned", "")
        if not page_content:
            continue
        doi = paper.get("doi", "N/A")
        authoritative_title = title_map.get(doi, paper.get("retrieved_title", "N/A"))
        metadata = {
            "doi": doi,
            "title": authoritative_title,
            "year": paper.get("retrieved_year", "N/A"),
            "journal": paper.get("retrieved_journal", "N/A"),
            "authors": ", ".join(paper.get("retrieved_authors", [])),
            "keywords": ", ".join(paper.get("extracted_keywords", [])),
            "filename": paper.get("filename", "N/A"),
            "local_path": paper.get("local_path", "N/A"),
        }
        doc = Document(page_content=page_content, metadata=metadata)
        documents.append(doc)
    print(f"✅ 成功加载并准备了 {len(documents)} 个文档。")
    return documents


def chunk_documents(documents: List[Document]) -> List[Document]:
    print("--- Chunking documents ---")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True
    )
    chunked_documents = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunked_documents)} chunks.")
    return chunked_documents


def get_embedding_function() -> HuggingFaceEmbeddings:
    print(f"--- Initializing embedding model: {EMBEDDING_MODEL_NAME} ---")
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    model_kwargs = {"device": EMBEDDING_DEVICE}
    encode_kwargs = {"normalize_embeddings": True}
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
    )
    print(f"Embedding model loaded successfully on device: '{EMBEDDING_DEVICE}'")
    return embeddings


def run_build_pipeline(source_path: Path, db_path: Path):
    authoritative_title_map = get_authoritative_titles_from_sqlite(Path(settings.SQLITE_DB_PATH))
    if not authoritative_title_map:
        print("无法继续，因为未能从SQLite获取权威标题。")
        return

    # ================== [ 关 键 修 复 ] ==================
    # The call now correctly passes the authoritative_title_map
    documents = load_and_prepare_documents(source_path, authoritative_title_map)
    # =====================================================

    if not documents:
        print("No documents to process. Exiting.")
        return

    chunked_docs = chunk_documents(documents)
    embedding_function = get_embedding_function()

    print(f"--- Building vector database at {db_path} ---")
    if db_path.exists():
        print(f"Found existing database. Removing to create a new one...")
        shutil.rmtree(db_path)
    db = Chroma.from_documents(
        documents=chunked_docs, embedding=embedding_function, persist_directory=str(db_path)
    )

    print("\n🎉 Vector database build complete!")
    print(f"   Database stored at: {db_path}")
    print(f"   Total vectors in DB: {db._collection.count()}")


def main():
    IDE_RUN = True
    IDE_DEFAULT_SOURCE_JSON_PATH = "../data/processed_text/processed_papers.json"
    IDE_DEFAULT_VECTOR_DB_PATH = "../data/vector_db/chroma_db"

    if IDE_RUN:
        print("Running in IDE mode with hardcoded paths.")
        source_path_str = IDE_DEFAULT_SOURCE_JSON_PATH
        db_path_str = IDE_DEFAULT_VECTOR_DB_PATH
    else:
        # (Command-line argument parsing remains unchanged)
        # For simplicity, I'm omitting the argparse code here, but it stays the same.
        parser = argparse.ArgumentParser(description="Build a Chroma vector database from processed JSON data.")
        parser.add_argument("--source", required=True, help="Path to the source processed JSON file.")
        parser.add_argument("--output", required=True, help="Path to the output folder for the Chroma database.")
        args = parser.parse_args()
        source_path_str = args.source
        db_path_str = args.output

    source_path = Path(source_path_str)
    db_path = Path(db_path_str)
    run_build_pipeline(source_path, db_path)


if __name__ == "__main__":
    main()