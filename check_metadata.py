# check_metadata.py
# 这是一个诊断脚本，用于检查并打印出您向量数据库中存储的确切元数据。

import sys
import os
import pprint

# --- 路径设置 ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from langchain_community.vectorstores import Chroma
from rag_system.config import settings
from rag_system.ingestion.embedding import get_embedding_function


def check_vector_db_metadata():
    """连接到ChromaDB并打印出所有文档的元数据。"""
    print("--- 正在连接到向量数据库... ---")
    if not os.path.exists(settings.VECTOR_DB_PATH):
        print(f"❌ 错误: 在路径 '{settings.VECTOR_DB_PATH}' 下找不到向量数据库。")
        print("   请先运行 'create_database/build_vectordb.py' 来构建数据库。")
        return

    try:
        embedding_function = get_embedding_function()
        vector_db = Chroma(
            persist_directory=str(settings.VECTOR_DB_PATH),
            embedding_function=embedding_function
        )

        print("✅ 连接成功！正在获取所有元数据...")

        # .get()方法可以获取数据库中的条目，这里我们获取所有内容
        # include=["metadatas"] 表示我们只关心元数据部分
        all_metadata = vector_db.get(include=["metadatas"])

        # 从返回结果中提取元数据列表
        metadata_list = all_metadata.get('metadatas', [])

        if not metadata_list:
            print("❌ 数据库中没有找到任何元数据。")
            return

        # 我们只关心标题，所以提取所有不重复的标题
        unique_titles = set()
        for meta in metadata_list:
            if 'title' in meta:
                unique_titles.add(meta['title'])

        print("\n" + "=" * 80)
        print("🔍 以下是您知识库中存储的所有唯一论文标题（已去除重复项）:")
        print("=" * 80)

        # 使用pprint美观地打印出所有标题
        pprint.pprint(sorted(list(unique_titles)))

        print("\n" + "=" * 80)
        print("下一步操作建议:")
        print("1. 从上面的列表中，找到您想在测试中使用的那两个标题。")
        print("2. **直接、完整地复制**这两个标题字符串。")
        print("3. 将它们粘贴到 `test_tool.py` 文件中，替换掉旧的 `titles` 列表。")
        print("4. 这样就能确保100%匹配，解决检索失败的问题。")
        print("=" * 80)

    except Exception as e:
        print(f"❌ 在检查元数据时发生错误: {e}")


if __name__ == "__main__":
    check_vector_db_metadata()