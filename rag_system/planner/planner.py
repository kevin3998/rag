# rag_system/planner/planner.py

import json
import re
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from rag_system.config import settings
from rag_system.state import AgentState, Plan, Step, Reflection
from rag_system.planner.prompt import PROMPT_TEMPLATE  # 导入我们新的、包含示例的Prompt


class Planner:
    """
    规划器 (Planner)
    负责将用户的复杂目标分解为一系列可执行的步骤（Plan）。
    这个版本通过few-shot示例，被教会了如何从历史失败中学习并调整计划。
    """

    def __init__(self):
        self.llm = ChatOllama(model=settings.LOCAL_LLM_MODEL_NAME, temperature=0.0)
        # 解析器现在需要解析出Plan对象
        self.output_parser = PydanticOutputParser(pydantic_object=Plan)
        # 注意：这里的模板输入变量已更新，以匹配新的PROMPT_TEMPLATE
        self.prompt_template = PromptTemplate(
            template=PROMPT_TEMPLATE,
            input_variables=["user_goal", "history_str"],
            # 我们不再需要 partial_variables，因为格式指令已硬编码在Prompt的示例中
        )
        print("✅ Planner initialized with few-shot history-aware prompt.")

    def _extract_json_from_response(self, text: str) -> str:
        """
        从LLM的原始输出中稳健地提取JSON字符串。
        """
        match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            return match.group(1)

        # 作为备用方案，查找第一个 '{' 和最后一个 '}'
        start_index = text.find('{')
        end_index = text.rfind('}')
        if start_index != -1 and end_index != -1 and end_index > start_index:
            return text[start_index: end_index + 1]

        raise ValueError("Response does not contain a valid JSON object.")

    def _format_history_for_prompt(self, agent_state: AgentState) -> str:
        """
        将AgentState中的历史记录格式化为简洁的字符串，以便注入到Prompt中。
        这可以防止Prompt因过长的工具输出而变得过于臃肿。
        """
        if not agent_state.history:
            return "[]"  # 如果没有历史，返回一个空列表字符串

        simplified_history = []
        for item in agent_state.history:
            if isinstance(item, Step):
                simplified_history.append({
                    "step_id": item.step_id,
                    "tool_name": item.tool_name,
                    # 截断过长的结果，只保留核心信息
                    "result": (item.result[:200] + '...') if item.result and len(item.result) > 200 else item.result,
                    "is_success": item.is_success,
                    "error_message": item.error_message
                })
            elif isinstance(item, Reflection):
                # 只包含对决策最重要的字段
                simplified_history.append({
                    "critique": item.critique,
                    "suggestion": item.suggestion
                })

        return json.dumps(simplified_history, indent=2, ensure_ascii=False)

    def generate_plan(self, agent_state: AgentState) -> AgentState:
        """
        根据用户目标和执行历史，生成一个新的任务计划。
        """
        print("🤔 Planner starting to generate a plan...")

        try:
            # 1. 格式化历史记录以注入Prompt
            history_str = self._format_history_for_prompt(agent_state)

            # 2. 调用Prompt模板
            prompt_value = self.prompt_template.invoke({
                "user_goal": agent_state.initial_query,
                "history_str": history_str
            })

            # 3. 调用LLM
            raw_output = self.llm.invoke(prompt_value).content

            # 4. 提取并解析JSON
            json_string = self._extract_json_from_response(raw_output)
            plan = self.output_parser.parse(json_string)

            # 5. 更新AgentState
            agent_state.plan = plan
            print(f"✅ Planner generated a new plan with {len(plan.steps)} steps.")

        except Exception as e:
            # 如果任何步骤失败，打印错误但保持agent_state.plan为None
            # 主控制器将捕获到plan为None并报告规划失败
            print(f"❌ Planner failed to generate a valid plan: {e}")
            agent_state.plan = None

        return agent_state