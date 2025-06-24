from langchain.tools import tool
from rag_system.generation.qa_chain import AdvancedQAChain

_qa_chain_instance = AdvancedQAChain()
# 直接使用qa_chain里经过我们反复调试的、能正常工作的rag_chain组件
pure_rag_chain = _qa_chain_instance.rag_chain

@tool
def semantic_search_tool(query: str) -> str:
    """
    当需要根据概念、主题或模糊描述来查找相关的详细文献段落时，使用此工具。
    适用于回答开放式、需要深入解释的问题。
    """
    print(f"--- [Tool: semantic_search_tool] 正在用问题 '{query[:30]}...' 调用RAG链 ---")
    try:
        response = pure_rag_chain.invoke(query)
        return response if response else "未能从文献库中找到相关信息。"
    except Exception as e:
        return f"执行语义检索时出现错误: {e}"