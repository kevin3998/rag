# rag_system/reflector/reflector.py

import json
import re
from pydantic import BaseModel, Field

from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from rag_system.graph_state import GraphState, Step, Reflection, Plan
from rag_system.config import settings
from rag_system.reflector.prompt import PROMPT_TEMPLATE


class ReflectionOutput(BaseModel):
    critique: str = Field(description="对上一步执行结果的批判性评估。")
    is_success: bool = Field(description="判断上一步是否成功达到了其预期目标。")
    confidence: float = Field(description="对成功判断的置信度，范围0.0到1.0。")
    suggestion: str = Field(description="基于评估，为下一步提出的具体建议。")
    is_finished: bool = Field(description="判断整个任务是否已经完成。")


class Reflector:
    def __init__(self):
        self.llm = ChatOllama(model=settings.LOCAL_LLM_MODEL_NAME, temperature=0.1)
        self.output_parser = PydanticOutputParser(pydantic_object=ReflectionOutput)

        # ================== [ 关 键 修 复 ] ==================
        # 这里的变量名必须与invoke时使用的 "context_str" 保持一致
        self.prompt_template = PromptTemplate(
            template=PROMPT_TEMPLATE,
            input_variables=["context_str"],  # <--- 确保这里是 "context_str"
            partial_variables={"format_instructions": self.output_parser.get_format_instructions()}
        )
        # =====================================================
        print("✅ Reflector initialized successfully.")

    def _extract_json_from_response(self, text: str) -> str:
        match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match: return match.group(1)
        start_index = text.find('{')
        end_index = text.rfind('}')
        if start_index != -1 and end_index != -1 and end_index > start_index:
            return text[start_index: end_index + 1]
        raise ValueError("Response does not contain a valid JSON object.")

    def _format_context_for_prompt(self, initial_query: str, plan: Plan, step_to_reflect: Step) -> str:
        context = {
            "initial_query": initial_query,
            "plan_goal": plan.goal,
            "full_plan_steps": [s.model_dump(include={'step_id', 'tool_name', 'tool_input', 'reasoning'}) for s in
                                plan.steps],
            "step_to_reflect": step_to_reflect.model_dump()
        }
        return json.dumps(context, indent=2, ensure_ascii=False)

    def generate_reflection(self, initial_query: str, plan: Plan, step_to_reflect: Step) -> Reflection:
        print("🤔 Reflector starting to generate a reflection...")
        context_str = self._format_context_for_prompt(initial_query, plan, step_to_reflect)
        try:
            prompt_value = self.prompt_template.invoke({"context_str": context_str})
            raw_output = self.llm.invoke(prompt_value).content
            json_string = self._extract_json_from_response(raw_output)
            parsed_output: ReflectionOutput = self.output_parser.parse(json_string)
            reflection = Reflection(critique=parsed_output.critique, is_success=parsed_output.is_success,
                                    confidence=parsed_output.confidence, suggestion=parsed_output.suggestion)
            reflection.is_finished = parsed_output.is_finished
        except Exception as e:
            print(f"❌ Reflector failed to get a valid reflection from LLM: {e}")
            reflection = Reflection(
                critique="反思模块在尝试解析LLM输出时遇到内部错误。这通常是由于上一步的工具执行失败导致的。",
                is_success=False,
                confidence=1.0,
                suggestion="由于内部错误，建议立即重新规划以尝试不同的方法。",
            )
            reflection.is_finished = False
        return reflection


def reflect_node(state: GraphState, reflector_instance: Reflector) -> dict:
    print("--- [节点: Reflector] ---")
    last_step = next((item for item in reversed(state['history']) if isinstance(item, Step)), None)
    if not last_step:
        print("❌ Reflector Error: No step found in history to reflect upon.")
        return {}
    reflection = reflector_instance.generate_reflection(initial_query=state['initial_query'], plan=state['plan'],
                                                        step_to_reflect=last_step)
    history = state['history'] + [reflection]
    decision = "FINISH" if reflection.is_finished else "PROCEED"
    return {"history": history, "decision": decision}
