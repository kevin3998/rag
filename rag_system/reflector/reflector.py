# rag_system/reflector/reflector.py

import json
import re
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.output_parsers import OutputFixingParser

from rag_system.config import settings
from rag_system.reflector.prompt import PROMPT_TEMPLATE
from rag_system.state import AgentState, Reflection


class Reflector:
    def __init__(self):
        self.llm = ChatOllama(model=settings.LOCAL_LLM_MODEL_NAME, temperature=0.0)
        pydantic_parser = PydanticOutputParser(pydantic_object=Reflection)
        self.output_parser = OutputFixingParser.from_llm(
            parser=pydantic_parser,
            llm=self.llm,
            max_retries=3
        )
        self.prompt_template = PromptTemplate(
            template=PROMPT_TEMPLATE,
            input_variables=["context_str"],
            partial_variables={
                "format_instructions": self.output_parser.get_format_instructions()
            },
        )
        print("✅ Reflector initialized successfully.")

    def _extract_json_from_response(self, text: str) -> str:
        """
        从包含<think>标签和其他文本的LLM响应中，稳健地提取出JSON部分。
        """
        match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            return match.group(1)
        start_index = text.find('{')
        end_index = text.rfind('}')
        if start_index != -1 and end_index != -1 and end_index > start_index:
            return text[start_index: end_index + 1]
        raise ValueError("Response does not contain a valid JSON object.")

    def _format_context_for_prompt(self, agent_state: AgentState) -> str:
        if not agent_state.plan:
            return "错误：计划尚未生成。"
        context_parts = []
        context_parts.append(f"- goal: \"{agent_state.plan.goal}\"")
        executed_steps_info = []
        for step in agent_state.plan.steps:
            if step.result is not None or step.error_message is not None:
                step_info = {
                    "step_id": step.step_id,
                    "tool_name": step.tool_name,
                    "tool_input": step.tool_input,
                    "is_success": step.is_success,
                    "result": step.result,
                    "error_message": step.error_message
                }
                executed_steps_info.append(json.dumps(step_info, ensure_ascii=False, indent=2))
        context_parts.append("- executed_steps:")
        if executed_steps_info:
            context_parts.extend(executed_steps_info)
        else:
            context_parts.append("  (尚未执行任何步骤)")
        return "\n".join(context_parts)

    def reflect(self, agent_state: AgentState) -> AgentState:
        print("🤔 Reflector starting to reflect on the execution results...")
        context_str = self._format_context_for_prompt(agent_state)
        try:
            # 【核心修改】重构调用流程，手动提取JSON
            prompt_value = self.prompt_template.invoke({"context_str": context_str})
            raw_output = self.llm.invoke(prompt_value).content
            print("✅ Successfully received raw response from Reflector's LLM.")

            print("... Attempting to extract JSON from Reflector's response...")
            json_string = self._extract_json_from_response(raw_output)
            print("✅ Successfully extracted JSON string from Reflector's response.")

            reflection_result: Reflection = self.output_parser.parse(json_string)
            agent_state.history.append(reflection_result)
            print("✅ Reflection generated and parsed successfully.")

        except Exception as e:
            print(f"❌ Reflector failed: {e}")
            fallback_reflection = Reflection(
                critique="反思模块在解析自身输出时遇到内部错误。",
                suggestion="系统在进行自我评估时遇到内部错误，建议终止当前任务。",
                is_finished=True
            )
            agent_state.history.append(fallback_reflection)
        return agent_state