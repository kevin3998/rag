# rag_system/reflector/reflector.py

import json
import re
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from rag_system.config import settings
from rag_system.state import AgentState, Reflection  # 从state导入Reflection
from rag_system.reflector.prompt import PROMPT_TEMPLATE
from rag_system.state import ReflectionOutput  # 假设您有一个schema文件


class Reflector:
    def __init__(self):
        self.llm = ChatOllama(model=settings.LOCAL_LLM_MODEL_NAME, temperature=0.1)
        # 注意：这里的pydantic_object应该是LLM直接输出的结构，而不是最终的Reflection对象
        self.output_parser = PydanticOutputParser(pydantic_object=ReflectionOutput)
        self.prompt_template = PromptTemplate(
            template=PROMPT_TEMPLATE,
            input_variables=["agent_state_str"],
            partial_variables={"format_instructions": self.output_parser.get_format_instructions()}
        )
        print("✅ Reflector initialized successfully.")

    def _extract_json_from_response(self, text: str) -> str:
        # (这个函数与我们在其他模块中使用的完全相同)
        match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match: return match.group(1)
        start_index = text.find('{')
        end_index = text.rfind('}')
        if start_index != -1 and end_index != -1 and end_index > start_index:
            return text[start_index: end_index + 1]
        raise ValueError("Response does not contain a valid JSON object.")

    def _format_agent_state_for_prompt(self, agent_state: AgentState) -> str:
        # 只包含与反思相关的信息，减少token消耗
        step_to_reflect = agent_state.get_step_by_id(agent_state.current_step_id)
        state_for_prompt = {
            "initial_query": agent_state.initial_query,
            "plan": agent_state.plan.model_dump(include={'goal', 'steps'}),
            "current_step_id": agent_state.current_step_id,
            "step_to_reflect": step_to_reflect.model_dump() if step_to_reflect else None
        }
        return json.dumps(state_for_prompt, indent=2, ensure_ascii=False)

    def reflect(self, agent_state: AgentState) -> AgentState:
        """
        对当前步骤的执行结果进行反思，并将结果添加到AgentState的历史记录中。
        """
        print("🤔 Reflector starting to reflect...")

        # 获取当前需要反思的步骤
        step_to_reflect = agent_state.get_step_by_id(agent_state.current_step_id)
        if not step_to_reflect:
            print("⚠️ Reflector: No step to reflect upon. Skipping.")
            return agent_state

        agent_state_str = self._format_agent_state_for_prompt(agent_state)

        try:
            # --- 主流程 ---
            prompt_value = self.prompt_template.invoke({"agent_state_str": agent_state_str})
            raw_output = self.llm.invoke(prompt_value).content
            json_string = self._extract_json_from_response(raw_output)
            parsed_output: ReflectionOutput = self.output_parser.parse(json_string)

            # 【核心修正】将LLM的输出与步骤的元数据合并，创建完整的Reflection对象
            reflection = Reflection(
                step_id=step_to_reflect.step_id,
                critique=parsed_output.critique,
                is_success=parsed_output.is_success,
                confidence=parsed_output.confidence,
                suggestion=parsed_output.suggestion,
                is_finished=parsed_output.is_finished  # 从LLM的输出中获取
            )
            print(f"✅ Reflection successful: {reflection.critique}")

        except Exception as e:
            # --- 备用方案 ---
            print(f"❌ Reflector failed to get a valid reflection from LLM: {e}")
            # 【核心修正】创建备用Reflection对象时，提供所有必需的字段
            reflection = Reflection(
                step_id=step_to_reflect.step_id,
                critique="反思模块在尝试解析LLM输出时遇到内部错误。",
                is_success=step_to_reflect.is_success,  # 直接沿用上一步的成功状态
                confidence=0.0,  # 置信度为0
                suggestion="建议重新规划以尝试不同的方法。",
                is_finished=False  # 无法判断是否完成，默认为False
            )

        # 将新生成的反思对象添加到历史记录中
        agent_state.history.append(reflection)
        return agent_state