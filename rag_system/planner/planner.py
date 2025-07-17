# rag_system/planner/planner.py (结构化修复版)

import json
import re
from typing import List
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.messages import BaseMessage
from langchain_core.tools import BaseTool

from rag_system.graph_state import GraphState, Plan, Step, Reflection
from rag_system.config import settings
from rag_system.planner.prompt import PROMPT_TEMPLATE


def _format_tools_description(tools: List[BaseTool]) -> str:
    """格式化工具列表，便于Planner识别。"""
    descriptions = []
    for tool in tools:
        schema = tool.args_schema.schema()
        props = schema.get('properties', {})
        required_params = schema.get('required', [])

        desc = (
            f"工具名称: `{tool.name}`\n"
            f"  - 描述: {tool.description}\n"
            f"  - 参数:\n"
        )
        for param_name, param_info in props.items():
            is_required = " (必需)" if param_name in required_params else " (可选)"
            param_type = param_info.get('type', 'N/A')
            param_desc = param_info.get('description', '')
            desc += f"    - `{param_name}`{is_required}: {param_desc} (类型: {param_type})\n"
        descriptions.append(desc)
    return "\n".join(descriptions)


class Planner:
    def __init__(self, tools: List[BaseTool]):
        self.llm = ChatOllama(model=settings.LOCAL_LLM_MODEL_NAME, temperature=0.0)
        self.output_parser = PydanticOutputParser(pydantic_object=Plan)

        # 🚀 [关键修复] 增加 query 和 context 两个输入变量
        self.prompt_template = PromptTemplate(
            template=PROMPT_TEMPLATE,
            input_variables=[
                "user_goal", "tools_description",
                "history_str", "chat_history_str",
                "query", "context"  # 新增
            ],
        )
        self.tools_description = _format_tools_description(tools)
        print("✅ Planner initialized with tool-aware prompt.")

    def _extract_json_from_response(self, text: str) -> str:
        match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            return match.group(1)
        start_index = text.find('{')
        end_index = text.rfind('}')
        if start_index != -1 and end_index != -1 and end_index > start_index:
            return text[start_index:end_index + 1]
        raise ValueError("Response does not contain a valid JSON object.")

    def _format_agent_history(self, history: list) -> str:
        if not history:
            return "[]"
        simplified_history = []
        for item in history:
            if isinstance(item, Step):
                simplified_history.append({
                    "step_id": item.step_id,
                    "tool_name": item.tool_name,
                    "result": str(item.result)[:200] + '...',
                    "is_success": item.is_success
                })
            elif isinstance(item, Reflection):
                simplified_history.append({
                    "critique": item.critique,
                    "suggestion": item.suggestion
                })
        return json.dumps(simplified_history, indent=2, ensure_ascii=False)

    def _format_chat_history(self, chat_history: List[BaseMessage]) -> str:
        if not chat_history:
            return "[]"
        return json.dumps([
            {"role": msg.type, "content": msg.content}
            for msg in chat_history
        ], indent=2, ensure_ascii=False)

    def generate_plan(self, user_query: str, history: list, chat_history: List[BaseMessage]) -> Plan:
        print("🤔 Planner starting to generate a plan (with tool descriptions)...")

        agent_history_str = self._format_agent_history(history)
        chat_history_str = self._format_chat_history(chat_history)

        # 🚀 [关键修复] 这里传入 query 和 context（先简化 context）
        prompt_value = self.prompt_template.invoke({
            "user_goal": user_query,
            "tools_description": self.tools_description,
            "history_str": agent_history_str,
            "chat_history_str": chat_history_str,
            "query": user_query,
            "context": agent_history_str  # 可扩展为其他上下文
        })

        raw_output = self.llm.invoke(prompt_value).content
        json_string = self._extract_json_from_response(raw_output)
        plan = self.output_parser.parse(json_string)

        print(f"✅ Planner generated a new plan with {len(plan.steps)} steps.")
        return plan


def plan_node(state: GraphState, planner_instance: Planner) -> dict:
    print("--- [节点: Planner] ---")
    try:
        plan_result = planner_instance.generate_plan(
            user_query=state['initial_query'],
            history=state['history'],
            chat_history=state['chat_history']
        )
        return {
            "plan": plan_result,
            "history": state['history'] + [f"Log: Successfully generated a plan for '{plan_result.goal}'."]
        }
    except Exception as e:
        print(f"❌ Planner failed to generate a valid plan: {e}")
        error_count = state.get('error_count', 0) + 1
        return {
            "plan": None,
            "history": state['history'] + [f"Error: Planner failed. Reason: {e}"],
            "error_count": error_count
        }
