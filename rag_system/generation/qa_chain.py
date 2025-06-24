from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.documents import Document
from langchain_ollama import ChatOllama
from typing import List

from rag_system.config import settings
from rag_system.retrieval.retriever_engine import RetrieverEngine

class AdvancedQAChain:
    def __init__(self):
        self.llm = ChatOllama(model=settings.LOCAL_LLM_MODEL_NAME)
        self.retriever = RetrieverEngine().as_retriever()
        self._setup_components()

    def _setup_components(self):
        def format_docs(docs: List[Document]) -> str:
            return "\n\n".join(f"--- 文档来源: {doc.metadata.get('title', 'N/A')} ---\n{doc.page_content}" for doc in docs)

        rag_prompt = ChatPromptTemplate.from_template(settings.PROMPT_TEMPLATE)
        self.rag_chain = (
            {"context": self.retriever | RunnableLambda(format_docs), "question": RunnablePassthrough()}
            | rag_prompt
            | self.llm
            | StrOutputParser()
        )