import os
import shutil
import argparse
from pathlib import Path
from typing import List

from tqdm import tqdm

# 从统一的配置文件中导入所有设置
from rag_system.config import settings
from rag_system.ingestion.build_vectordb import (
    load_and_prepare_documents,  # 我们复用之前的函数
    chunk_documents,
    get_embedding_function
)

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document


def update_database():
    """
    对向量数据库执行增量更新。
    只添加源文件中存在但数据库中不存在的新文档。
    """
    # 设置镜像，以备不时之需
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

    source_path = settings.SOURCE_DATA_PATH
    db_path = settings.VECTOR_DB_PATH

    # --- 1. 加载所有源文档的“预期”状态 ---
    print(f"--- [Step 1/3] 正在从 {source_path} 加载源文档信息... ---")
    if not source_path.exists():
        print(f"错误：源数据文件未找到: {source_path}")
        return

    # 我们这里不直接创建Document对象，而是先加载原始JSON数据
    with open(source_path, 'r', encoding='utf-8') as f:
        all_source_papers = json.load(f)
    print(f"在源文件中找到 {len(all_source_papers)} 篇论文。")

    # --- 2. 连接到现有数据库并获取所有已存在的文档ID ---
    print(f"--- [Step 2/3] 正在连接到数据库 {db_path} 并检查现有文档... ---")
    embedding_function = get_embedding_function()  # 需要嵌入函数来连接

    # 连接到现有数据库，如果不存在会自动创建
    db = Chroma(
        persist_directory=str(db_path),
        embedding_function=embedding_function
    )

    # 获取数据库中所有条目的元数据，我们只关心 local_path
    # .get()方法在数据量大时可能较慢，但对于几千篇论文是可行的
    existing_items = db.get(include=["metadatas"])
    existing_paths = {meta['local_path'] for meta in existing_items['metadatas'] if 'local_path' in meta}
    print(f"数据库中已存在 {len(existing_paths)} 篇唯一论文。")

    # --- 3. 筛选、处理并添加新文档 ---
    print("--- [Step 3/3] 正在筛选并添加新文档... ---")
    new_papers_to_add = []
    for paper in all_source_papers:
        # 使用 local_path 作为唯一标识符
        if paper.get('local_path') not in existing_paths:
            new_papers_to_add.append(paper)

    if not new_papers_to_add:
        print("数据库已是最新，无需添加新文档。")
        return

    print(f"发现 {len(new_papers_to_add)} 篇新论文，准备进行处理...")

    # 将筛选出的新论文数据转换为LangChain Document对象
    new_documents = []
    for paper in tqdm(new_papers_to_add, desc="准备新文档"):
        page_content = paper.get("llm_ready_fulltext_cleaned", "")
        if not page_content: continue
        metadata = {k: v for k, v in paper.items() if k != "llm_ready_fulltext_cleaned"}
        new_documents.append(Document(page_content=page_content, metadata=metadata))

    # 对新文档进行切分
    chunked_new_docs = chunk_documents(new_documents)

    # 将切分好的新文档块添加到数据库
    if chunked_new_docs:
        print(f"正在将 {len(chunked_new_docs)} 个新文本块添加到数据库...")
        # 定义批处理大小以显示进度
        batch_size = 128
        batches = [chunked_new_docs[i:i + batch_size] for i in range(0, len(chunked_new_docs), batch_size)]

        for batch in tqdm(batches, desc="嵌入并存储新文本块"):
            db.add_documents(documents=batch)

    # 确保数据持久化
    db.persist()
    print(f"\n✅ 数据库更新完成！成功添加了 {len(new_papers_to_add)} 篇新论文。")
    print(f"数据库当前总条目数: {db._collection.count()}")


if __name__ == "__main__":
    import json

    update_database()