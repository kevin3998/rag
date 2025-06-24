from langchain.agents import AgentExecutor, create_react_agent
from langchain_ollama import ChatOllama

# 导入我们最终版本的Prompt和工具
from rag_system.agent.prompt import react_prompt
from rag_system.agent.tools.semantic_search import semantic_search_tool
from rag_system.agent.tools.structured_query import structured_data_query_tool
from rag_system.config import settings


class MaterialScienceAgent:
    """
    一个稳定、可靠、具备多工具使用能力的ReAct智能体。
    """

    def __init__(self):
        self.llm = ChatOllama(model=settings.LOCAL_LLM_MODEL_NAME)
        self.tools = [semantic_search_tool, structured_data_query_tool]

        # 将LLM、工具、和Prompt绑定在一起，创建Agent的核心逻辑
        agent = create_react_agent(self.llm, self.tools, react_prompt)

        # 创建智能体执行器，它负责运行ReAct的思考-行动循环
        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,  # 在终端打印详细思考过程，便于观察
            handle_parsing_errors=True  # 增强对LLM输出格式错误的容忍度
        )
        print("MaterialScienceAgent (ReAct): 智能体执行器构建完成。")

    def run(self, user_input: str):
        """
        运行智能体并流式返回其输出。
        """
        # Agent Executor的流式输出包含了每一步的思考、行动和最终结果
        return self.agent_executor.stream({"input": user_input})