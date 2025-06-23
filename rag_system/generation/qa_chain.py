from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.documents import Document
from langchain_ollama import ChatOllama
from typing import List

from rag_system.config import settings
from rag_system.retrieval.retriever_engine import RetrieverEngine


class AdvancedQAChain:
    """
    高级问答链的最终稳定版本。
    修复了RAG链中的数据流问题，并恢复了完整的三路路由功能。
    """

    def __init__(self):
        self.llm = ChatOllama(model="qwen3-14b-f16:latest")  # 请确保这里的名字和您ollama list中的一致
        print(f"AdvancedQAChain: 本地LLM模型 '{self.llm.model}' 初始化成功。")

        self.retriever = RetrieverEngine().as_retriever()
        self._setup_components()
        print("AdvancedQAChain: 智能问答组件构建完成。")

    def _setup_components(self):
        # --- 【关键修复】在这里定义一个辅助函数，用于格式化文档 ---
        def format_docs(docs: List[Document]) -> str:
            """将检索到的文档列表格式化为单一的字符串。"""
            return "\n\n".join(
                f"--- 文档来源: {doc.metadata.get('title', 'N/A')} ---\n{doc.page_content}" for doc in docs)

        # --- 组件1：RAG链 (已修复) ---
        rag_prompt = ChatPromptTemplate.from_template(settings.PROMPT_TEMPLATE)
        self.rag_chain = (
            # 这里的 RunnablePassthrough() 会接收原始的query字符串
            # 然后并行地：
            # 1. 将query传递给retriever，然后用format_docs格式化结果，赋值给context
            # 2. 将query本身赋值给question
                {"context": self.retriever | RunnableLambda(format_docs), "question": RunnablePassthrough()}
                | rag_prompt
                | self.llm
                | StrOutputParser()
        )

        # --- 组件2：通用对话链 (经测试是健康的) ---
        chat_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are Qwen, a helpful and conversational AI assistant developed by Alibaba Cloud."),
            ("human", "{question}")
        ])
        self.chat_chain = (
                {"question": RunnablePassthrough()}
                | chat_prompt
                | self.llm
                | StrOutputParser()
        )

        # --- 组件3：礼貌拒绝链 (经测试是健康的) ---
        self.rejection_chain = RunnableLambda(
            lambda x: "很抱歉，作为一个专攻膜材料领域的AI助手，我无法回答关于其他专业领域的问题。")

        # --- 组件4：路由器 (经测试是健康的) ---
        router_prompt_text = """
        根据用户的问题，将其严格分类为以下三种类型之一：
        1. "membrane_science_rag": 如果问题明确关于科学、技术，特别是膜材料、石墨烯、纳米过滤、渗透、分离技术等相关专业领域。
        2. "other_domain_rejection": 如果问题是关于其他专业领域，例如金融、法律、医学、天文学、历史等，这些不是你的专业范围。
        3. "general_chat": 如果是问候（你好）、感谢、关于你自身（你是谁）、或者其他所有不属于以上两类的日常闲聊。

        只输出 "membrane_science_rag", "other_domain_rejection", "general_chat" 这三个标签中的一个。

        用户问题: {question}
        分类:
        """
        router_prompt = ChatPromptTemplate.from_template(router_prompt_text)
        self.router_chain = (
                {"question": RunnablePassthrough()}
                | router_prompt
                | self.llm
                | StrOutputParser()
        )

    def stream_answer(self, query: str):
        """
        以流式方式获取答案，恢复使用清晰的 if/else 路由逻辑。
        """
        topic = self.router_chain.invoke(query)
        print(f"路由决策: 问题 '{query[:30]}...' 被分类为 -> {topic}")

        if "membrane_science_rag" in topic.lower():
            yield {"type": "status", "message": "正在从数据库检索相关信息..."}
            for chunk in self.rag_chain.stream(query):
                yield chunk

        elif "other_domain_rejection" in topic.lower():
            yield {"type": "status", "message": "正在判断问题领域..."}
            # stream() 方法可以处理RunnableLambda
            for chunk in self.rejection_chain.stream(query):
                yield chunk

        else:  # general_chat
            yield {"type": "status", "message": "正在生成通用回答..."}
            for chunk in self.chat_chain.stream(query):
                yield chunk