# rag_system/executor/executor.py
from typing import Dict, Any
from rag_system.state import AgentState
from rag_system.agent.tools.semantic_search import semantic_search_tool
from rag_system.agent.tools.paper_finder_tool import paper_finder_tool


class Executor:
    def __init__(self):
        self.tools: Dict[str, Any] = {
            semantic_search_tool.name: semantic_search_tool,
            paper_finder_tool.name: paper_finder_tool,
        }
        print("✅ Executor initialized with final toolset:", list(self.tools.keys()))

    def execute_step(self, agent_state: AgentState) -> AgentState:
        print(f"🤖 Executor starting to execute step ID: {agent_state.current_step_id}...")

        if not agent_state.plan or agent_state.current_step_id is None:
            print("⚠️ Executor skipped: No plan or current step ID found in state.")
            return agent_state

        step_to_execute = agent_state.get_step_by_id(agent_state.current_step_id)
        if not step_to_execute:
            print(f"❌ Executor failed: Step with ID {agent_state.current_step_id} not found.")
            return agent_state

        tool_to_call = self.tools.get(step_to_execute.tool_name)
        if not tool_to_call:
            error_msg = f"Tool '{step_to_execute.tool_name}' not found."
            print(f"❌ {error_msg}")
            agent_state.update_step_result(step_to_execute.step_id, error_msg, False, error_msg)
            return agent_state

        try:
            tool_input = step_to_execute.tool_input

            # --- 【核心修改】智能判断工具输入类型 ---
            # LangChain的@tool装饰器会自动处理参数传递
            # 我们只需要把整个输入字典作为关键字参数传递进去即可
            if isinstance(tool_input, dict):
                result = tool_to_call.invoke(tool_input)
            else:
                # 为了兼容可能存在的、只接收单一字符串的简单工具
                result = tool_to_call.invoke(str(tool_input))

            # 检查工具是否返回了表明错误的字符串
            is_success = True
            error_message = None
            if isinstance(result, str) and "出现错误:" in result:
                is_success = False
                error_message = result
                print(f"⚠️ Step {step_to_execute.step_id} executed, but the tool reported a failure.")
            else:
                print(f"✅ Step {step_to_execute.step_id} executed successfully.")

            agent_state.update_step_result(step_to_execute.step_id, str(result), is_success, error_message)

        except Exception as e:
            error_msg = f"Error executing tool '{step_to_execute.tool_name}': {e}"
            print(f"❌ {error_msg}")
            agent_state.update_step_result(step_to_execute.step_id, error_msg, False, error_msg)

        return agent_state