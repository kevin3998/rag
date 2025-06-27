# rag_system/decider/decider.py

import json
import re
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from rag_system.config import settings
from rag_system.state import AgentState, Action, Reflection
from rag_system.decider.prompt import PROMPT_TEMPLATE


class Decider:
    def __init__(self):
        self.llm = ChatOllama(model=settings.LOCAL_LLM_MODEL_NAME, temperature=0.0)
        self.output_parser = PydanticOutputParser(pydantic_object=Action)
        self.prompt_template = PromptTemplate(
            template=PROMPT_TEMPLATE,
            input_variables=["agent_state_str", "suggestion"],
            partial_variables={"format_instructions": self.output_parser.get_format_instructions()}
        )
        print("✅ Decider initialized successfully.")

    def _extract_json_from_response(self, text: str) -> str:
        # (这个函数与我们在其他模块中使用的完全相同)
        match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match: return match.group(1)
        start_index = text.find('{')
        end_index = text.rfind('}')
        if start_index != -1 and end_index != -1 and end_index > start_index:
            return text[start_index : end_index + 1]
        raise ValueError("Response does not contain a valid JSON object.")

    def _format_agent_state_for_prompt(self, agent_state: AgentState) -> str:
        state_dict = agent_state.model_dump(exclude={'history'})
        return json.dumps(state_dict, indent=2, ensure_ascii=False)

    def decide(self, agent_state: AgentState) -> Action:
        print("🤔 Decider starting to make a decision...")
        latest_reflection = next((item for item in reversed(agent_state.history) if isinstance(item, Reflection)), None)
        if not latest_reflection:
            raise ValueError("Cannot make a decision without a reflection.")
        agent_state_str = self._format_agent_state_for_prompt(agent_state)
        suggestion = latest_reflection.suggestion
        try:
            prompt_value = self.prompt_template.invoke({"agent_state_str": agent_state_str, "suggestion": suggestion})
            raw_output = self.llm.invoke(prompt_value).content
            json_string = self._extract_json_from_response(raw_output)
            action = self.output_parser.parse(json_string)
            print(f"✅ Decider made a decision: {action.action_type} - Reason: {action.reasoning}")
            return action
        except Exception as e:
            print(f"❌ Decider failed to make a valid decision: {e}")
            return Action(action_type='FINISH', reasoning="决策模块遇到内部错误，安全起见终止任务。")