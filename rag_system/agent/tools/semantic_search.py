# rag_system/agent/tools/semantic_search.py (已修复)

from typing import Optional, List, Any
from langchain_core.tools import tool
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama
from pydantic import BaseModel, Field

from rag_system.config import settings
from rag_system.ingestion.embedding import get_embedding_function


# --- 全局组件初始化 (保持不变) ---
def get_tool_components():
    """Initializes and returns the core components for the tool."""
    try:
        print("--- Initializing components for semantic_search_tool ---")
        embedding_function = get_embedding_function()
        vector_db = Chroma(
            persist_directory=str(settings.VECTOR_DB_PATH),
            embedding_function=embedding_function
        )
        llm = ChatOllama(model=settings.LOCAL_LLM_MODEL_NAME, temperature=0.1)

        reasoning_prompt = PromptTemplate.from_template(
            """# 角色
            你是一位顶尖的材料科学家，你的任务是基于下面提供的【相关文献摘要】，对用户的【核心问题】进行一次深入的、有逻辑的分析和推理。

            # 任务
            1.  **综合信息**: 仔细阅读所有的【相关文献摘要】。
            2.  **逻辑推理**: 根据这些摘要中的线索，推导出能够回答用户【核心问题】的机理或结论。
            3.  **结构化输出**: 你的回答应该条理清晰、逻辑严谨，并明确指出你的结论是基于提供的文献信息得出的。如果信息不足以得出结论，请明确指出。

            ---
            【相关文献摘要】:
            {context}
            ---
            【核心问题】:
            {question}
            ---
            你的分析与推理:
            """
        )
        reasoning_chain = reasoning_prompt | llm

        print("✅ semantic_search_tool components initialized successfully.")
        return vector_db, reasoning_chain
    except Exception as e:
        print(f"❌ Error initializing components for semantic_search_tool: {e}")
        return None, None


vector_db, reasoning_chain = get_tool_components()


class SemanticSearchInput(BaseModel):
    query: str = Field(description="一个需要进行深度分析和总结的核心问题。")
    context: Optional[Any] = Field(None,
                                   description="可选的上下文，通常是一个包含论文标题的列表 (List[str])，由前一个工具提供。")


@tool(args_schema=SemanticSearchInput)
def semantic_search_tool(query: str, context: Optional[Any] = None) -> str:
    """
    一个强大的分析与推理工具。它首先根据上下文（如论文标题列表）从知识库中检索详细信息，
    然后基于这些信息对用户的核心问题进行深入的分析和总结。
    如果未提供上下文，它会先进行开放式搜索，然后进行分析。
    """
    if not vector_db or not reasoning_chain:
        return "出现错误: semantic_search_tool 的核心组件未能成功初始化，无法执行任务。"

    retrieved_context = ""

    # ================== [ 关 键 修 复 ] ==================
    # 模式1: 基于上下文的精确检索 (最优先)
    if isinstance(context, list) and context:
        print(f"--- [Tool Log] semantic_search_tool: Activating 'Title Fetch' mode for {len(context)} titles.")
        all_fetched_docs = []
        try:
            paper_titles = [str(item) for item in context if isinstance(item, str)]
            if paper_titles:
                # 遍历标题列表，为每个标题单独进行检索
                for title in paper_titles:
                    print(f"    - Retrieving content for title: '{title[:50]}...'")
                    # 使用 'where' 过滤器精确匹配单个标题
                    docs_for_title = vector_db.get(
                        where={"title": title},
                        include=["documents"]
                    ).get('documents', [])

                    if docs_for_title:
                        # 将这篇论文的所有文档块作为一个整体，用换行符连接起来
                        paper_full_text = "\n".join(docs_for_title)
                        # 在文档开头明确标注来源标题，帮助LLM区分
                        all_fetched_docs.append(f"--- 内容来源: {title} ---\n{paper_full_text}")
                        print(f"    - Found {len(docs_for_title)} chunks for this title.")
                    else:
                        print(f"    - No content found for title: '{title[:50]}...'")

                if all_fetched_docs:
                    # 使用特定的分隔符将不同论文的内容分开
                    retrieved_context = "\n\n<<< --- NEW PAPER --- >>>\n\n".join(all_fetched_docs)
                    print(f"--- [Tool Log] Successfully fetched content for {len(all_fetched_docs)} unique papers.")
                else:
                    print(f"--- [Tool Log] 根据标题列表未找到任何详细内容，将转为开放式搜索。")
            else:
                print(f"--- [Tool Log] 上下文列表为空，将转为开放式搜索。")
        except Exception as e:
            print(f"--- [Tool Warning] 根据标题检索时发生错误: {e}。将转为开放式搜索。")
    # =====================================================

    # 模式2: 开放式搜索 (当没有有效上下文时)
    if not retrieved_context:
        print("--- [Tool Log] semantic_search_tool: Activating 'Open Search' mode.")
        try:
            results = vector_db.similarity_search(query, k=settings.RETRIEVER_K)
            if results:
                retrieved_context = "\n\n---\n\n".join([doc.page_content for doc in results])
        except Exception as e:
            return f"在执行开放式语义搜索时发生错误: {e}"

    # --- 最终的分析与推理步骤 ---
    if not retrieved_context:
        return "在整个知识库中未能找到与您问题相关的任何信息，无法进行分析。"

    print("--- [Tool Log] Invoking reasoning chain with retrieved context...")
    response = reasoning_chain.invoke({"context": retrieved_context, "question": query})
    return response.content