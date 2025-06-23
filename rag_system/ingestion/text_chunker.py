from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from rag_system.config import settings


def chunk_documents(documents: List[Document]) -> List[Document]:
    """
    将文档列表切分成更小的文本块。

    Args:
        documents (List[Document]): A list of documents to be chunked.

    Returns:
        List[Document]: A list of smaller document chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        add_start_index=True  # 在元数据中添加块的起始位置，便于追溯
    )

    chunked_documents = text_splitter.split_documents(documents)
    print(f"成功将 {len(documents)} 个文档切分为 {len(chunked_documents)} 个文本块。")
    return chunked_documents


if __name__ == '__main__':
    # 用于直接测试该模块
    from rag_system.ingestion.document_loader import load_json_documents

    # 确保测试文件存在
    if not settings.SOURCE_DATA_PATH.exists():
        sample_data = [{
                           "page_content": "这是一段非常非常长的测试文本，旨在模拟一篇完整的科学论文。它需要足够长，以便RecursiveCharacterTextSplitter能够根据设定的chunk_size和chunk_overlap将其有效地分割成多个块。我们将不断重复这句话来增加长度。" * 20,
                           "metadata": {"source": "test_paper.pdf"}}]
        settings.SOURCE_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(settings.SOURCE_DATA_PATH, 'w', encoding='utf-8') as f:
            json.dump(sample_data, f)

    docs = load_json_documents(settings.SOURCE_DATA_PATH)
    chunks = chunk_documents(docs)
    print("\n第一个文本块示例:")
    print(chunks[0])
    print(f"\n元数据: {chunks[0].metadata}")