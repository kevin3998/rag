import re
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.output_parsers import OutputFixingParser
import json

from rag_system.config import settings
from rag_system.planner.prompt import PROMPT_TEMPLATE
from rag_system.state import AgentState, Plan


class Planner:
    def __init__(self):
        self.llm = ChatOllama(model=settings.LOCAL_LLM_MODEL_NAME, temperature=0.0)
        pydantic_parser = PydanticOutputParser(pydantic_object=Plan)
        self.output_parser = OutputFixingParser.from_llm(
            parser=pydantic_parser,
            llm=self.llm,
            max_retries=3
        )

        # 我们回归到简单的PromptTemplate，它现在是100%安全的
        self.prompt_template = PromptTemplate.from_template(PROMPT_TEMPLATE)

        print("✅ Planner initialized successfully with the safe template.")

    def _extract_json_from_response(self, text: str) -> str:
        match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            return match.group(1)
        start_index = text.find('{')
        end_index = text.rfind('}')
        if start_index != -1 and end_index != -1 and end_index > start_index:
            return text[start_index: end_index + 1]
        raise ValueError("Response does not contain a valid JSON object.")

    def generate_plan(self, agent_state: AgentState) -> AgentState:
        print("🤖 Planner starting to generate a plan...")

        user_goal = agent_state.initial_query
        if not user_goal:
            raise ValueError("AgentState's initial_query cannot be empty.")

        try:
            # 步骤 1: 正常调用LLM
            prompt_value = self.prompt_template.invoke({"user_goal": user_goal})
            raw_output = self.llm.invoke(prompt_value).content

            # 步骤 2: 提取JSON
            json_string = self._extract_json_from_response(raw_output)

            # 步骤 3: 【核心修改】在解析前，进行手动的、安全的占位符替换
            print("... Performing safe placeholder replacement ...")
            safe_json_string = json_string.replace("__PREVIOUS_STEP_RESULT__", "{previous_step_result}")

            # 步骤 4: 解析这个最终干净且正确的JSON字符串
            generated_plan: Plan = self.output_parser.parse(safe_json_string)

            agent_state.plan = generated_plan
            if generated_plan.steps:
                agent_state.current_step_id = 1  # 确保从第一步开始
            print(f"✅ Plan generated and parsed successfully with {len(generated_plan.steps)} steps.")

        except Exception as e:
            print(f"❌ Planner failed: {e}")
            agent_state.plan = None

        return agent_state
