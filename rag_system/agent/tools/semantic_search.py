# rag_system/agent/tools/semantic_search.py (最终修复版)

from typing import Optional, List, Any
from langchain_core.tools import tool
# ... 其他导入保持不变 ...
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama
from pydantic import BaseModel, Field
from rag_system.config import settings
from rag_system.ingestion.embedding import get_embedding_function


# --- 全局组件初始化 (保持不变) ---
def get_tool_components():
    # ... 此函数内容完全不变 ...
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
            你是一位顶尖的材料科学家，你的任务是基于下面提供的单篇【相关文献摘要】，对用户的【核心问题】进行一次深入的、有逻辑的分析和推理。

            # 任务
            1.  **专注单篇**: 仔细阅读下面提供的【相关文献摘要】，它只包含一篇论文的内容。
            2.  **逻辑推理**: 根据这篇摘要中的线索，推导出能够回答用户【核心问题】的机理或结论。
            3.  **结构化输出**: 你的回答应该条理清晰、逻辑严谨，并明确指出你的结论是基于提供的文献信息得出的。

            ---
            【相关文献摘要】:
            {context}
            ---
            【核心问题】:
            {question}
            ---
            你对这篇论文的分析与推理:
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

    if isinstance(context, list) and context:
        print(
            f"--- [Tool Log] semantic_search_tool: Activating '逐篇分析 (Per-Title Analysis)' mode for {len(context)} titles.")
        individual_summaries = []
        try:
            paper_titles = [str(item) for item in context if isinstance(item, str)]
            if not paper_titles:
                print(f"--- [Tool Log] 上下文列表为空，将转为开放式搜索。")
            else:
                for i, title in enumerate(paper_titles, 1):
                    print(f"\n--- 正在分析第 {i}/{len(paper_titles)} 篇论文: '{title[:50]}...' ---")
                    docs_for_title = vector_db.get(
                        where={"title": title}, include=["documents"]
                    ).get('documents', [])

                    if not docs_for_title:
                        summary = f"### 关于《{title}》的总结:\n未能从知识库中找到该论文的详细内容。"
                        individual_summaries.append(summary)
                        continue

                    single_paper_context = "\n".join(docs_for_title)
                    print(f"    - 已检索到 {len(docs_for_title)} 个内容块，正在提交给LLM进行分析...")

                    # ================== [ 最终修复 ] ==================
                    # 为子任务创建一个新的、更具体的指令，而不是使用原始的总指令。
                    sub_task_query = f"请总结这篇标题为《{title}》的论文的核心内容、方法和结论。"
                    # =====================================================

                    response = reasoning_chain.invoke({
                        "context": single_paper_context,
                        "question": sub_task_query  # <-- 使用新的、具体的子任务指令
                    })

                    summary = f"### 关于《{title}》的总结:\n{response.content}"
                    individual_summaries.append(summary)
                    print(f"    - 分析完成。")

                if not individual_summaries:
                    print(f"--- [Tool Log] 未能根据任何标题找到内容，将转为开放式搜索。")
                else:
                    final_report = "\n\n---\n\n".join(individual_summaries)
                    return f"已完成对 {len(individual_summaries)} 篇论文的逐一分析报告：\n\n{final_report}"
        except Exception as e:
            print(f"--- [Tool Warning] 在逐篇分析时发生错误: {e}。将转为开放式搜索。")

    # ... 开放式搜索部分的代码保持不变 ...
    print("--- [Tool Log] semantic_search_tool: Activating 'Open Search' mode.")
    try:
        results = vector_db.similarity_search(query, k=settings.RETRIEVER_K)
        if not results:
            return "在整个知识库中未能找到与您问题相关的任何信息，无法进行分析。"
        open_search_context = "\n\n---\n\n".join([doc.page_content for doc in results])
        print("--- [Tool Log] Invoking reasoning chain with retrieved context...")
        response = reasoning_chain.invoke({"context": open_search_context, "question": query})
        return response.content
    except Exception as e:
        return f"在执行开放式语义搜索时发生错误: {e}"