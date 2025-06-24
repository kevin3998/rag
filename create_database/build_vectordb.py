import json
import shutil
from pathlib import Path
import argparse
from typing import List
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# The new, recommended way to import HuggingFace embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm import tqdm
import os
# --- MODEL AND CHUNKING CONFIGURATION ---
# Use the CORRECT, full model identifier from Hugging Face
EMBEDDING_MODEL_NAME = "BAAI/bge-large-zh-v1.5"
# For your Mac: "mps". For Nvidia GPU: "cuda". For CPU-only: "cpu".
EMBEDDING_DEVICE = "mps"

# Text chunking settings
CHUNK_SIZE = 750
CHUNK_OVERLAP = 75


# --- CORE FUNCTIONS ---

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
    for paper in tqdm(papers_data, desc="Preparing documents"):
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
    """Initializes and returns the embedding model function using the new package."""
    print(f"--- Initializing embedding model: {EMBEDDING_MODEL_NAME} ---")
    # Use the new HuggingFaceEmbeddings class from langchain-huggingface
    model_kwargs = {"device": EMBEDDING_DEVICE}
    encode_kwargs = {"normalize_embeddings": True}

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    print(f"Embedding model loaded successfully on device: '{EMBEDDING_DEVICE}'")
    return embeddings


def run_build_pipeline(source_path: Path, db_path: Path):
    """
    The main function to orchestrate the vector DB creation process using provided paths.
    """
    # Step 1: Load and prepare documents from the JSON file
    documents = load_and_prepare_documents(source_path)
    if not documents:
        print("No documents to process. Exiting.")
        return

    # Step 2: Split the documents into manageable chunks
    chunked_docs = chunk_documents(documents)

    # Step 3: Initialize the embedding model
    embedding_function = get_embedding_function()

    # Step 4: Build and persist the vector database
    print(f"--- Building vector database at {db_path} ---")

    if db_path.exists():
        print(f"Found existing database. Removing to create a new one...")
        shutil.rmtree(db_path)

    db = Chroma.from_documents(
        documents=chunked_docs,
        embedding=embedding_function,
        persist_directory=str(db_path)
    )

    print("\n🎉 Vector database build complete!")
    print(f"   Database stored at: {db_path}")
    print(f"   Total vectors in DB: {db._collection.count()}")


# --- MAIN EXECUTION LOGIC ---
def main():
    # --- Configuration for direct IDE run ---
    IDE_RUN = True

    # --- Default values for IDE execution (Modify these as needed) ---
    IDE_DEFAULT_SOURCE_JSON_PATH = "../data/processed_text/processed_papers_test.json"
    IDE_DEFAULT_VECTOR_DB_PATH = "../data/vector_db_test/chroma_db_test.db"

    if IDE_RUN:
        print("Running in IDE mode with hardcoded paths.")
        source_path_str = IDE_DEFAULT_SOURCE_JSON_PATH
        db_path_str = IDE_DEFAULT_VECTOR_DB_PATH
    else:
        print("Running with command-line arguments.")
        parser = argparse.ArgumentParser(description="Build a Chroma vector database from processed JSON data.")
        parser.add_argument("--source", required=True, help="Path to the source processed JSON file.")
        parser.add_argument("--output", required=True, help="Path to the output folder for the Chroma database.")
        args = parser.parse_args()
        source_path_str = args.source
        db_path_str = args.output

    # --- Run the pipeline ---
    source_path = Path(source_path_str)
    db_path = Path(db_path_str)

    run_build_pipeline(source_path, db_path)


if __name__ == "__main__":
    main()