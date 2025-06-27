# rag_system/main_controller.py

import re
from typing import Generator, Optional, Dict, Any

# 导入必要的模块
from rag_system.config import settings
from rag_system.planner.planner import Planner
from rag_system.executor.executor import Executor
from rag_system.reflector.reflector import Reflector
from rag_system.state import AgentState

# 导入意图分类器所需的组件 (假设这部分代码仍然存在且有效)
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --- 意图分类器的提示词 (保持不变) ---
ROUTER_PROMPT_TEMPLATE = """
你是一个AI助手的意图分类路由。根据用户输入的问题，将其分类为以下三种类型之一：
1.  `domain_specific_question`: 关于科学、技术，特别是材料科学、化学、物理等领域的专业问题。
2.  `daily_conversation`: 日常问候、闲聊、或者询问AI自身身份的对话。
3.  `out_of_domain_question`: 不属于上述两类的其他领域问题，例如询问历史、金融、娱乐等。

你的回答只能是 `domain_specific_question`、`daily_conversation`、`out_of_domain_question` 这三个词中的一个，不能包含任何其他文字。

【任务】
用户: "{query}"
回答:
"""


class MainController:
    """
    最终版的、具备动态信息流处理能力的、简化的主控制器。
    它负责编排Planner, Executor, 和 Reflector的整个工作流程。
    """

    def __init__(self, max_loops: int = 3):  # 复杂的计划通常不需要太多循环
        """
        初始化主控制器。
        """
        self.planner = Planner()
        self.executor = Executor()
        self.reflector = Reflector()
        self.max_loops = max_loops

        # 初始化用于意图分类和闲聊的组件
        self.llm = ChatOllama(model=settings.LOCAL_LLM_MODEL_NAME, temperature=0.1)
        router_prompt = PromptTemplate.from_template(ROUTER_PROMPT_TEMPLATE)
        self.router_chain = router_prompt | self.llm | StrOutputParser()
        chat_prompt = PromptTemplate.from_template(
            "你是一个名为“膜科学智能研究助手”的AI，请友好地回答用户的问题。\n用户: {query}\nAI:")
        self.chat_chain = chat_prompt | self.llm | StrOutputParser()

        print("✅ MainController initialized with a simplified and robust execution loop.")

    def _prepare_next_input(self, tool_input: Dict[str, Any], previous_step_result: str) -> Dict[str, Any]:
        """
        【核心】动态地将上一步的结果注入到下一步的输入中。
        它会查找特殊的占位符并进行替换。
        """
        prepared_input = {}
        # 将上一步的结果序列化为字符串，以便注入
        context_str = str(previous_step_result)

        for key, value in tool_input.items():
            if isinstance(value, str) and "{previous_step_result}" in value:
                # 如果输入字符串中包含占位符，则用上一步结果替换它
                # 这允许更灵活的提示，例如 "基于以下信息：{previous_step_result}，请..."
                prepared_input[key] = value.replace("{previous_step_result}", context_str)
            else:
                prepared_input[key] = value
        return prepared_input

    def _clean_final_answer(self, text: str) -> str:
        """移除最终答案中可能存在的<think>标签，给用户一个干净的结果。"""
        return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

    def run(self, query: str) -> Generator[str, None, None]:
        """
        执行完整的 Plan -> Act -> Reflect 循环来处理用户请求。
        """
        # --- 阶段 0: 意图分类 ---
        yield "--- [阶段 0: 意图分类] ---"
        yield f"收到用户目标: {query}"
        try:
            raw_intent_output = self.router_chain.invoke({"query": query})
            intent = raw_intent_output.strip().split()[-1].lower()
            yield f"意图分类结果: {intent}"
        except Exception as e:
            yield f"❌ 意图分类失败: {e}"
            return

        # 根据意图进行分流
        if intent == "daily_conversation":
            yield "--- [处理: 日常对话] ---"
            response_generator = self.chat_chain.stream({"query": query})
            yield from response_generator
            return
        elif intent == "out_of_domain_question":
            yield "--- [处理: 领域外问题] ---"
            yield "非常抱歉，我是一个专注于膜材料科学领域的研究助手，无法为您提供其他领域的专业信息。"
            return
        elif intent == "domain_specific_question":
            yield "--- [处理: 专业领域问题，启动规划流程] ---"
        else:
            yield f"--- [处理: 未知意图] ---"
            yield f"抱歉，我暂时无法理解您的问题意图（分类结果：{intent}）。"
            return

        # --- 阶段 1: 规划 ---
        agent_state = AgentState(initial_query=query)
        yield "--- [阶段 1: 规划] ---"
        agent_state = self.planner.generate_plan(agent_state)
        if not agent_state.plan or not agent_state.plan.steps:
            yield "❌ 规划失败：未能生成有效的计划。"
            return
        plan_summary = "\n".join([f"  - 步骤 {s.step_id}: {s.reasoning}" for s in agent_state.plan.steps])
        yield f"生成计划如下:\n{plan_summary}"

        # --- 阶段 2: 简化的线性执行循环 ---
        yield "--- [阶段 2: 执行与反思循环] ---"
        last_step_result = ""
        for step in agent_state.plan.steps:
            yield f"\n--- [执行步骤 {step.step_id}/{len(agent_state.plan.steps)}] ---"

            # 如果不是第一步，就准备输入（注入上一步结果）
            if step.step_id > 1:
                yield "🔧 正在准备下一步输入（注入上一步结果）..."
                step.tool_input = self._prepare_next_input(step.tool_input, last_step_result)

            # 更新当前步骤指针并执行
            agent_state.current_step_id = step.step_id
            yield f"▶️ 执行工具: {step.tool_name}"
            agent_state = self.executor.execute_step(agent_state)

            # 获取更新后的步骤状态
            executed_step = agent_state.get_step_by_id(step.step_id)
            yield f"  - 结果: {executed_step.result if executed_step.is_success else executed_step.error_message}"
            last_step_result = executed_step.result if executed_step.is_success else executed_step.error_message

            # 【失败熔断机制】如果任何一步失败，则立即终止任务
            if not executed_step.is_success:
                yield "--- [阶段 3: 终止] ---"
                yield f"🛑 关键步骤 {executed_step.step_id} 执行失败，任务中止。"
                final_answer = f"任务在步骤 {executed_step.step_id} 失败。\n\n**错误详情:**\n{last_step_result}"
                yield f"\n\n---\n**最终答案:**\n\n{final_answer}"
                return

        # --- 阶段 3: 所有步骤成功完成 ---
        yield "\n--- [阶段 3: 完成] ---"
        yield "✅ 计划的所有步骤已成功执行。"
        # 进行最后一次反思，对整个成功的工作流进行总结
        yield "🤔 进行最后总结性反思..."
        agent_state = self.reflector.reflect(agent_state)
        latest_reflection = agent_state.history[-1]
        yield f"  - 最终评审: {latest_reflection.critique}"

        # 将最后一步的、干净的结果作为最终答案
        final_answer = self._clean_final_answer(last_step_result)
        agent_state.final_answer = final_answer
        yield f"\n\n---\n**最终答案:**\n\n{final_answer}"