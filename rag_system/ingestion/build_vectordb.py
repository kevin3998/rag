import os
import shutil
import argparse
from pathlib import Path
from typing import List

# 1. 导入 tqdm 库
from tqdm import tqdm

# 从我们统一的配置文件中导入所有设置
from rag_system.config import settings

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter


# --- CORE FUNCTIONS (函数内部将使用导入的 settings) ---

def load_and_prepare_documents(json_path: Path) -> List[Document]:
    """
    Loads paper data from the JSON file and prepares them as LangChain Document objects.
    """
    print(f"--- Loading data from {json_path} ---")

    if not json_path.exists():
        print(f"Error: Source JSON file not found at {json_path}")
        return []

    with open(json_path, 'r', encoding='utf-8') as f:
        papers_data = json.load(f)

    documents = []
    # 2. 使用 tqdm 包裹循环，添加进度条
    # desc 参数为进度条添加了描述性标题
    for paper in tqdm(papers_data, desc="[Step 1/3] Preparing documents"):
        page_content = paper.get("llm_ready_fulltext_cleaned", "")
        if not page_content:
            continue

        metadata = {
            "doi": paper.get("doi", "N/A"),
            "title": paper.get("retrieved_title", "N/A"),
            "year": paper.get("retrieved_year", "N/A"),
            "journal": paper.get("retrieved_journal", "N/A"),
            "authors": ", ".join(paper.get("retrieved_authors", [])),
            "keywords": ", ".join(paper.get("extracted_keywords", [])),
            "filename": paper.get("filename", "N/A"),
            "local_path": paper.get("local_path", "N/A"),
        }

        doc = Document(page_content=page_content, metadata=metadata)
        documents.append(doc)

    print(f"Successfully loaded and prepared {len(documents)} documents.")
    return documents


def chunk_documents(documents: List[Document]) -> List[Document]:
    """Splits the loaded documents into smaller chunks."""
    print("--- [Step 2/3] Chunking documents ---")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        add_start_index=True
    )
    chunked_documents = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunked_documents)} chunks.")
    return chunked_documents


def get_embedding_function() -> HuggingFaceEmbeddings:
    """Initializes and returns the embedding model function."""
    print(f"--- Initializing embedding model: {settings.EMBEDDING_MODEL_NAME} ---")
    print("(第一次运行时会自动下载模型，请耐心等待，后续会直接从缓存加载)")
    model_kwargs = {"device": settings.EMBEDDING_DEVICE}
    encode_kwargs = {"normalize_embeddings": True}

    embeddings = HuggingFaceEmbeddings(
        model_name=settings.EMBEDDING_MODEL_NAME,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    print(f"Embedding model loaded successfully on device: '{settings.EMBEDDING_DEVICE}'")
    return embeddings


def run_build_pipeline(source_path: Path, db_path: Path):
    """
    Orchestrates the vector DB creation process with progress feedback.
    """
    documents = load_and_prepare_documents(source_path)
    if not documents:
        print("No documents to process. Exiting.")
        return

    chunked_docs = chunk_documents(documents)
    embedding_function = get_embedding_function()

    print(f"--- [Step 3/3] Building vector database at {db_path} ---")
    if db_path.exists():
        print(f"Found existing database. Removing to create a new one...")
        shutil.rmtree(db_path)

    # 3. 分批处理以显示嵌入进度
    # Chroma.from_documents 是一个单独的操作，无法直接显示进度。
    # 我们通过分批添加文档的方式，来手动创建一个可以被tqdm包裹的循环。

    # 定义批处理大小
    batch_size = 128

    # 创建文档批次
    batches = [chunked_docs[i:i + batch_size] for i in range(0, len(chunked_docs), batch_size)]

    # 初始化一个空的Chroma数据库
    db = Chroma(
        persist_directory=str(db_path),
        embedding_function=embedding_function
    )

    # 遍历批次并使用tqdm显示进度
    for batch in tqdm(batches, desc="Embedding and ingesting chunks"):
        db.add_documents(documents=batch)

    # 确保所有数据都已持久化到磁盘
    db.persist()

    print("\n🎉 Vector database build complete!")
    print(f"   Database stored at: {db_path}")
    print(f"   Total vectors in DB: {db._collection.count()}")


# --- MAIN EXECUTION LOGIC (保持不变) ---
def main():
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    IDE_RUN = True
    if IDE_RUN:
        print("Running in IDE mode, using paths from settings.py.")
        source_path = settings.SOURCE_DATA_PATH
        db_path = settings.VECTOR_DB_PATH
    else:
        print("Running with command-line arguments.")
        parser = argparse.ArgumentParser(description="Build a Chroma vector database from processed JSON data.")
        parser.add_argument("--source", default=str(settings.SOURCE_DATA_PATH),
                            help="Path to the source processed JSON file.")
        parser.add_argument("--output", default=str(settings.VECTOR_DB_PATH),
                            help="Path to the output folder for the Chroma database.")
        args = parser.parse_args()
        source_path = Path(args.source)
        db_path = Path(args.output)
    run_build_pipeline(source_path, db_path)


if __name__ == "__main__":
    main()