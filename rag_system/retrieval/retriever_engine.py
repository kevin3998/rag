from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.vectorstores import VectorStoreRetriever

from rag_system.config import settings


class RetrieverEngine:
    """
    负责从持久化的向量数据库中检索相关文档。
    """

    def __init__(self):
        """
        初始化检索器，加载向量数据库和嵌入模型。
        """
        if not settings.VECTOR_DB_PATH.exists():
            raise FileNotFoundError(
                f"向量数据库未找到，请先运行 build_vectordb.py。路径: {settings.VECTOR_DB_PATH}"
            )

        # 1. 加载嵌入函数 (必须与构建时完全一致)
        embedding_function = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL_NAME,
            model_kwargs={'device': settings.EMBEDDING_DEVICE},
            encode_kwargs={'normalize_embeddings': True}
        )

        # 2. 加载持久化的向量数据库
        self.vector_store = Chroma(
            persist_directory=str(settings.VECTOR_DB_PATH),
            embedding_function=embedding_function
        )
        print("RetrieverEngine: 向量数据库加载成功。")

    def as_retriever(self) -> VectorStoreRetriever:
        """
        将向量数据库转换为一个LangChain的Retriever对象。

        将使用在 settings.py 中定义的 K 值。
        """
        return self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": settings.RETRIEVER_K}
        )


if __name__ == '__main__':
    # 用于直接测试该模块
    try:
        print("测试 RetrieverEngine...")
        engine = RetrieverEngine()
        retriever = engine.as_retriever()

        test_query = "graphene oxide membrane"
        results = retriever.invoke(test_query)

        print(f"\n针对查询 '{test_query}' 的前 {len(results)} 条检索结果:")
        for i, doc in enumerate(results):
            print(f"\n--- 结果 {i + 1} ---")
            print(f"来源: {doc.metadata.get('title', 'N/A')}")
            print(f"内容片段: {doc.page_content[:250]}...")

    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"发生未知错误: {e}")