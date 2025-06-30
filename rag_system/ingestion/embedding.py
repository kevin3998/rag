from langchain_huggingface import HuggingFaceEmbeddings
from rag_system.config import settings  # 注意，这里要用相对路径导入settings


def get_embedding_function() -> HuggingFaceEmbeddings:
    """
    初始化并返回用于文本向量化的HuggingFace嵌入模型函数。
    这是一个核心的、可被多处复用的组件。
    """
    print(
        f"--- Initializing embedding model: {settings.EMBEDDING_MODEL_NAME} on device: {settings.EMBEDDING_DEVICE} ---")

    model_kwargs = {"device": settings.EMBEDDING_DEVICE}
    encode_kwargs = {"normalize_embeddings": True}  # 归一化对于相似度计算很重要

    embeddings = HuggingFaceEmbeddings(
        model_name=settings.EMBEDDING_MODEL_NAME,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    print("✅ Embedding model loaded successfully.")
    return embeddings