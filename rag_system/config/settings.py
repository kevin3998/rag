import os
from pathlib import Path
from dotenv import load_dotenv

# --- 基础设置：加载环境变量 ---
# 这段代码会自动找到项目根目录下的 .env 文件并加载它
# 这使得API密钥等敏感信息可以与代码本身分离
env_path = Path(__file__).parent.parent.parent / '.env'
load_dotenv(dotenv_path=env_path)


# --- 路径设置 (Paths) ---
# 使用 pathlib 来构建与操作系统无关的、健壮的路径
# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.parent

# 数据输入和输出路径
# SOURCE_DATA_PATH 指向由 2.data_processing.py 生成的JSON文件
SOURCE_DATA_PATH = PROJECT_ROOT / "data" / "processed_text" / "processed_papers.json"
# VECTOR_DB_PATH 指向持久化向量数据库的存储位置
VECTOR_DB_PATH = PROJECT_ROOT / "data" / "vector_db" / "chroma_db"
SQLITE_DB_PATH = PROJECT_ROOT / "data" / "database" / "literature_materials.db"


# --- 模型设置 (Models) ---
# 用于问答生成的大语言模型 (LLM for Generation)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
LOCAL_LLM_MODEL_NAME = "qwen3-tuned:latest"

# 用于将文本转换为向量的嵌入模型 (Embedding Model)
# 注意：这里是包含了组织名称的、正确的Hugging Face模型ID
EMBEDDING_MODEL_NAME = "BAAI/bge-large-zh-v1.5"
# 针对您的macOS系统，使用 "mps" 进行硬件加速。如果是Nvidia显卡用 "cuda"，纯CPU用 "cpu"
EMBEDDING_DEVICE = "mps"


# --- RAG系统参数 (RAG Parameters) ---
# 1. 数据注入/切分 (Ingestion / Chunking)
CHUNK_SIZE = 750  # 每个文本块的目标大小（字符数）
CHUNK_OVERLAP = 75 # 相邻文本块之间的重叠大小（字符数）

# 2. 检索 (Retrieval)
# 在从数据库中检索时，返回最相似的 top_k 个文本块
RETRIEVER_K = 10

# 3. 生成 (Generation)
# 这是提供给LLM的、包含上下文和问题的提示词模板
PROMPT_TEMPLATE = """
请严格根据以下提供的上下文信息来回答问题。
你的回答应该清晰、简洁，并且完全基于所提供的上下文。
如果你在上下文中找不到问题的答案，请明确说明“根据提供的资料，我无法回答这个问题”，不要试图编造任何答案。

---
上下文信息:
{context}
---

问题:
{question}
"""


# Agent循环相关设置
MAX_ITERATIONS = 5  # Agent最大循环次数，防止无限循环
MAX_RETRIES = 2     # 单个工具的最大重试次数

# 【新增】反思与决策相关设置
REFLECTION_CONFIDENCE_THRESHOLD = 0.6
# ================== [ 关 键 新 增 ] ==================
# 添加您的SQLite数据库文件的路径。
# 请根据您的项目结构修改这个路径。
# 假设您的数据库文件在 data/ 目录下。
