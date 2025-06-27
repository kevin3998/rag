# rag_system/agent/tools/semantic_search.py

from langchain.tools import tool
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from typing import Optional
import re

from rag_system.generation.qa_chain import AdvancedQAChain
from rag_system.config import settings

llm = ChatOllama(model=settings.LOCAL_LLM_MODEL_NAME, temperature=0.1)

CONTEXTUAL_SUMMARY_PROMPT = PromptTemplate.from_template(
"""你是一位专业的科研论文分析师。你的任务是根据下面提供的【上下文信息】，清晰、准确地回答用户的【具体问题】。
【最高指令】: 你的回答必须完全基于下面提供的【上下文信息】，严禁使用任何外部知识。

### 上下文信息:
{context}

### 具体问题:
{question}

### 你的分析和总结:
"""
)
contextual_summary_chain = CONTEXTUAL_SUMMARY_PROMPT | llm | StrOutputParser()

_qa_chain_instance = AdvancedQAChain()
rag_chain = _qa_chain_instance.rag_chain

def _clean_final_answer(text: str) -> str:
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

@tool
def semantic_search_tool(query: str, context: Optional[str] = None) -> str:
    """
    一个多功能语义分析工具，它有两种工作模式：
    1.  **开放式搜索模式 (默认)**: 当只提供`query`参数时，它会在整个知识库中进行语义搜索。
    2.  **指定上下文精读模式**: 当同时提供`context`和`query`参数时，它会只针对`context`中提供的具体文本内容，来回答`query`提出的问题。
    """
    if context:
        print(f"--- [Tool: semantic_search_tool | Mode: Contextual Summary] ---")
        try:
            response = contextual_summary_chain.invoke({
                "context": context,
                "question": query
            })
            return _clean_final_answer(response)
        except Exception as e:
            return f"在对指定上下文进行摘要时出现错误: {e}"
    else:
        print(f"--- [Tool: semantic_search_tool | Mode: RAG Search] ---")
        try:
            response = rag_chain.invoke(query)
            return _clean_final_answer(str(response)) if response else "未能从文献库中找到相关信息。"
        except Exception as e:
            return f"执行开放式语义检索时出现错误: {e}"