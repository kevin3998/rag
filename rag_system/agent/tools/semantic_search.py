from typing import Optional, List
from langchain_core.tools import tool
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama

from rag_system.config import settings
from rag_system.ingestion.embedding import get_embedding_function

# --- 全局组件初始化 ---
try:
    print("--- Initializing components for semantic_search_tool ---")
    embedding_function = get_embedding_function()
    # 【核心修正】确保将Path对象转换为字符串，再传递给Chroma
    vector_db = Chroma(
        persist_directory=str(settings.VECTOR_DB_PATH),
        embedding_function=embedding_function
    )
    llm = ChatOllama(model=settings.LOCAL_LLM_MODEL_NAME, temperature=0.1)

    SUMMARIZE_PROMPT = PromptTemplate.from_template(
        """# 任务
        你是一个专业的科研论文分析师。请根据下面提供的【上下文信息】，清晰、准确、有条理地回答用户的【总结要求】。
        你的回答必须严格基于【上下文信息】，不得引入外部知识。

        ---
        【上下文信息】:
        {context}
        ---
        【总结要求】:
        {question}
        ---
        分析总结:
        """
    )
    summarization_chain = SUMMARIZE_PROMPT | llm
    print("✅ semantic_search_tool components initialized successfully.")

except Exception as e:
    print(f"❌ Error initializing components for semantic_search_tool: {e}")
    vector_db = None
    summarization_chain = None


@tool
def semantic_search_tool(
        query: str,
        context: Optional[str] = None,
        paper_titles: Optional[List[str]] = None
) -> str:
    """
    一个多功能语义分析工具，具备三种工作模式：
    1. 【开放式搜索模式】: 如果只提供`query`，它会在整个向量数据库中进行语义搜索，回答概念性问题。
    2. 【按需总结模式】: 如果提供了`context`（通常是上一步的直接文本输出），它会根据`query`的要求总结这段上下文。
    3. 【按图索骥模式 (数据库互通)】: 如果提供了`paper_titles`列表，它会忽略`context`，转而从向量数据库中精确查找这些论文的全文内容，然后根据`query`进行深度总结。
    """
    if not vector_db or not summarization_chain:
        return "错误：semantic_search_tool 的核心组件未能成功初始化，无法执行任务。"

    # --- 模式3：按图索骥 (数据库互通的核心) ---
    if paper_titles and isinstance(paper_titles, list) and len(paper_titles) > 0:
        print(f"--- [Tool Log] semantic_search_tool: Activating 'Title Fetch' mode for {len(paper_titles)} titles.")
        try:
            documents = vector_db.get(
                where={"title": {"$in": paper_titles}},
                include=["documents"]
            )['documents']

            if not documents:
                return f"根据您提供的 {len(paper_titles)} 个论文标题，未能在向量数据库中找到任何对应的详细内容。"

            rich_context = "\n\n---\n\n".join(documents)
            print(f"--- [Tool Log] Successfully fetched content for {len(documents)} chunks from vector DB.")

            response = summarization_chain.invoke({"context": rich_context, "question": query})
            return response.content

        except Exception as e:
            return f"在根据论文标题从向量数据库检索内容时发生错误: {e}"

    # --- 模式2：按需总结 ---
    elif context:
        print("--- [Tool Log] semantic_search_tool: Activating 'Context Summarize' mode.")
        response = summarization_chain.invoke({"context": context, "question": query})
        return response.content

    # --- 模式1：开放式搜索 ---
    else:
        print("--- [Tool Log] semantic_search_tool: Activating 'Open Search' mode.")
        try:
            results = vector_db.similarity_search(query, k=5)
            if not results:
                return "在向量数据库中未能找到与您的问题相关的任何信息。"

            search_context = "\n\n---\n\n".join([doc.page_content for doc in results])
            response = summarization_chain.invoke({"context": search_context, "question": query})
            return response.content
        except Exception as e:
            return f"在执行开放式语义搜索时发生错误: {e}"
