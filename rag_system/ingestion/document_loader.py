import json
from pathlib import Path
from typing import List
from langchain_core.documents import Document


def load_json_documents(file_path: Path) -> List[Document]:
    """
    加载并解析JSON文件，将其转换为LangChain的Document对象列表。

    Args:
        file_path (Path): a `processed_papers.json` file.

    Returns:
        List[Document]: A list of Document objects.
    """
    documents = []
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for item in data:
        # 确保JSON文件中的每个对象都有 'page_content' 和 'metadata' 键
        page_content = item.get("page_content", "")
        metadata = item.get("metadata", {})

        if not page_content:
            continue  # Skip entries without content

        doc = Document(page_content=page_content, metadata=metadata)
        documents.append(doc)

    print(f"成功加载 {len(documents)} 个文档来源自 {file_path}")
    return documents


if __name__ == '__main__':
    # 用于直接测试该模块
    from rag_system.config import settings

    if not settings.SOURCE_DATA_PATH.exists():
        print(f"错误: 源数据文件未找到，请检查路径: {settings.SOURCE_DATA_PATH}")
        # 创建一个假的json文件用于测试
        sample_data = [
            {"page_content": "这是第一篇关于高分子膜材料的介绍性文献。",
             "metadata": {"source": "paper_1.pdf", "page": 1}},
            {"page_content": "第二篇文献详细讨论了该材料在水处理中的应用。",
             "metadata": {"source": "paper_2.pdf", "page": 5}}
        ]
        settings.SOURCE_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(settings.SOURCE_DATA_PATH, 'w', encoding='utf-8') as f:
            json.dump(sample_data, f, ensure_ascii=False, indent=2)
        print(f"已创建示例文件: {settings.SOURCE_DATA_PATH}")

    loaded_docs = load_json_documents(settings.SOURCE_DATA_PATH)
    print("\n第一个文档内容:")
    print(loaded_docs[0].page_content)
    print("\n第一个文档元数据:")
    print(loaded_docs[0].metadata)