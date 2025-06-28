# rag_system/main_controller.py

import re
from typing import Generator, Dict, Any

# 导入所有必要的模块
from rag_system.config import settings
from rag_system.state import AgentState, Reflection # 确保从state导入Reflection
from rag_system.planner.planner import Planner
from rag_system.executor.executor import Executor
from rag_system.reflector.reflector import Reflector
from rag_system.decider.decider import Decider

from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

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
    集成了 Decider 的高级主控制器。
    它负责编排 Planner -> Executor -> Reflector -> Decider 的完整工作流程。
    """

    def __init__(self, max_loops: int = 5):  # 增加循环次数以允许重规划
        """
        初始化主控制器，现在包含 Decider。
        """
        self.planner = Planner()
        self.executor = Executor()
        self.reflector = Reflector()
        self.decider = Decider()  # <-- 实例化我们基于规则的Decider
        self.max_loops = max_loops

        self.llm = ChatOllama(model=settings.LOCAL_LLM_MODEL_NAME, temperature=0.1)
        router_prompt = PromptTemplate.from_template(ROUTER_PROMPT_TEMPLATE)
        self.router_chain = router_prompt | self.llm | StrOutputParser()
        chat_prompt = PromptTemplate.from_template(
            "你是一个名为“膜科学智能研究助手”的AI，我是基于你微调的Qwen3模型，请友好地回答用户的问题。\n用户: {query}\nAI:")
        self.chat_chain = chat_prompt | self.llm | StrOutputParser()
        print("✅ MainController initialized with advanced execution loop (Plan-Execute-Reflect-Decide).")

    def _prepare_next_input(self, tool_input: Dict[str, Any], previous_step_result: str) -> Dict[str, Any]:
        """动态地将上一步的结果注入到下一步的输入中。"""
        prepared_input = {}
        # 将上一步的结果序列化为字符串，以便注入
        context_str = str(previous_step_result)

        for key, value in tool_input.items():
            # 您原有的占位符是 __PREVIOUS_STEP_RESULT__，这里我们保持兼容
            if isinstance(value, str) and "__PREVIOUS_STEP_RESULT__" in value:
                prepared_input[key] = value.replace("__PREVIOUS_STEP_RESULT__", context_str)
            else:
                prepared_input[key] = value
        return prepared_input

    def _clean_final_answer(self, text: str) -> str:
        """移除最终答案中可能存在的<think>标签，给用户一个干净的结果。"""
        return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

    def run(self, query: str) -> Generator[str, None, None]:
        """
        执行完整的 Plan -> Execute -> Reflect -> Decide 循环。
        """
        # --- 阶段 0: 意图分类 (保持不变) ---
        yield "--- [阶段 0: 意图分类] ---"
        # ... (这部分代码与之前完全相同) ...
        yield f"收到用户目标: {query}"
        try:
            raw_intent_output = self.router_chain.invoke({"query": query})
            intent = raw_intent_output.strip().split()[-1].lower()
            yield f"意图分类结果: {intent}"
        except Exception as e:
            yield f"❌ 意图分类失败: {e}"
            return

        if intent == "daily_conversation":
            yield "--- [处理: 日常对话] ---"
            response_generator = self.chat_chain.stream({"query": query})
            yield from response_generator
            return
        elif intent == "out_of_domain_question":
            yield "--- [处理: 领域外问题] ---"
            yield "非常抱歉，我是一个专注于膜材料科学领域的研究助手，无法为您提供其他领域的专业信息。"
            return
        elif intent != "domain_specific_question":
            yield f"--- [处理: 未知意图] ---"
            yield f"抱歉，我暂时无法理解您的问题意图（分类结果：{intent}）。"
            return

        # --- 阶段 1: 初始规划 ---
        yield "--- [处理: 专业领域问题，启动规划流程] ---"
        agent_state = AgentState(initial_query=query)
        yield "--- [阶段 1: 规划] ---"
        agent_state = self.planner.generate_plan(agent_state)
        if not agent_state.plan or not agent_state.plan.steps:
            yield "❌ 初始规划失败：未能生成有效计划。"
            return
        plan_summary = "\n".join([f"  - 步骤 {s.step_id}: {s.reasoning}" for s in agent_state.plan.steps])
        yield f"生成初始计划:\n{plan_summary}"

        # --- 阶段 2: 动态执行、反思与决策循环 ---
        yield "\n--- [阶段 2: 动态执行、反思与决策循环] ---"
        loop_count = 0
        last_step_result = ""

        while loop_count < self.max_loops:
            loop_count += 1
            yield f"\n--- [循环 {loop_count}/{self.max_loops}] ---"

            step_to_execute = agent_state.get_next_step()
            if not step_to_execute:
                yield "✅ 所有计划步骤已成功执行。"
                break

            yield f"--- [执行步骤 {step_to_execute.step_id}/{len(agent_state.plan.steps)}] ---"

            if step_to_execute.step_id > 1:
                step_to_execute.tool_input = self._prepare_next_input(step_to_execute.tool_input, last_step_result)

            yield f"▶️ 执行工具: {step_to_execute.tool_name}"
            agent_state = self.executor.execute_step(agent_state)
            executed_step = agent_state.get_step_by_id(step_to_execute.step_id)

            result_str = executed_step.result if executed_step.is_success else executed_step.error_message
            yield f"  - 结果: {result_str}"
            last_step_result = result_str

            yield "🤔 进行反思..."
            agent_state = self.reflector.reflect(agent_state)
            latest_reflection = agent_state.history[-1]
            if not isinstance(latest_reflection, Reflection):
                yield "❌ 错误：历史记录的最新条目不是一个有效的Reflection对象。"
                break
            yield f"  - 评审: {latest_reflection.critique}"

            yield "▶️ 进行决策..."
            decision = self.decider.decide(latest_reflection)
            yield f"  - 决策结果: {decision}"

            # ---整个逻辑块现在只依赖 `decision` 字符串 ---
            if decision == "FINISH":
                yield "✅ 决策为FINISH，任务完成。"
                break

            elif decision == "PROCEED":
                if agent_state.is_plan_completed():
                    yield "✅ 已是最后一步且决策为PROCEED，任务完成。"
                    break
                else:
                    yield "✅ 决策为PROCEED，准备执行下一步。"
                    agent_state.current_step_id += 1
                    continue

            elif decision == "REPLAN":
                yield "🔄 决策为REPLAN，启动重新规划..."
                agent_state.plan = None
                agent_state = self.planner.generate_plan(agent_state)
                if not agent_state.plan or not agent_state.plan.steps:
                    yield "❌ 重新规划失败：未能生成有效计划。"
                    break
                plan_summary = "\n".join([f"  - 步骤 {s.step_id}: {s.reasoning}" for s in agent_state.plan.steps])
                yield f"生成新计划:\n{plan_summary}"
                agent_state.current_step_id = 1
                last_step_result = ""
                continue

            elif decision == "RETRY":
                # 由于基于规则的Decider不提供修正后的输入，我们只记录重试的决策
                yield "⚠️ 决策为RETRY。正在重试当前步骤..."
                continue

        # --- 阶段 3: 生成最终答案 ---
        yield "\n--- [阶段 3: 生成最终答案] ---"
        if last_step_result:
             final_answer = self._clean_final_answer(last_step_result)
             agent_state.final_answer = final_answer
             yield f"\n\n---\n**最终答案:**\n\n{final_answer}"
        else:
             yield "未能得出最终答案。可能是由于规划失败或在最大循环次数内未完成任务。"