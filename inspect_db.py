import os
import chromadb
import pandas as pd
from pprint import pprint

# --- 配置 ---
# 确保这个路径指向您之前创建的数据库文件夹
DB_PATH = "data/vector_db/chroma_db"


# --- 主逻辑 ---
def inspect_chroma_db(db_path: str):
    """
    连接到持久化的ChromaDB并查看其内容。
    """
    if not os.path.exists(db_path):
        print(f"错误：数据库路径不存在: {db_path}")
        return

    print(f"正在连接到数据库: {db_path}\n")
    # 1. 创建一个持久化的客户端
    client = chromadb.PersistentClient(path=db_path)

    # 2. 获取数据库中的集合（Collection）
    # LangChain默认创建的集合名称通常是 "langchain"
    # 我们可以先列出所有集合来确认
    collections = client.list_collections()
    if not collections:
        print("错误：数据库中没有找到任何集合。")
        return

    collection_name = collections[0].name  # 获取第一个集合的名称
    collection = client.get_collection(name=collection_name)

    print(f"成功连接到集合: '{collection_name}'")
    print(f"集合中的条目总数: {collection.count()}\n")

    # 3. 从集合中取出数据
    # 我们只取出前5条作为示例，并包含它们的元数据和文档内容
    # include 参数非常有用，可以指定你想要看的内容
    results = collection.get(
        limit=5,
        include=["metadatas", "documents"]
    )

    print("--- 数据库前5条内容示例 ---")

    # 4. 格式化并打印结果
    ids = results['ids']
    documents = results['documents']
    metadatas = results['metadatas']

    for i in range(len(ids)):
        print(f"--- Entry {i + 1} ---")
        print(f"ID: {ids[i]}")

        print("\nMETADATA:")
        pprint(metadatas[i])  # 使用pprint美化打印字典

        print("\nDOCUMENT (文本块内容):")
        # 只显示前400个字符以保持简洁
        print(documents[i][:400] + "..." if len(documents[i]) > 400 else documents[i])
        print("-" * 20 + "\n")

    # 5. (推荐) 使用Pandas更清晰地展示
    print("\n--- 使用Pandas DataFrame更清晰地展示 ---")
    # 将结果转换为一个更易读的表格
    df = pd.DataFrame({
        'id': results['ids'],
        'document': [doc[:150] + '...' for doc in results['documents']],  # 截断长文本
        'metadata': results['metadatas']
    })

    # 设置Pandas显示选项以完整显示内容
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)

    print(df)


if __name__ == "__main__":
    inspect_chroma_db(DB_PATH)