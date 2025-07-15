# rag_system/executor/executor.py

from typing import Dict, Any, Tuple, Optional, List

from sklearn.ensemble._hist_gradient_boosting import predictor

from rag_system.graph_state import GraphState, Step
from rag_system.agent.tools.prediction_tool import prediction_tool # <-- 导入新工具
from rag_system.agent.tools.semantic_search import semantic_search_tool
from rag_system.agent.tools.paper_finder_tool import paper_finder_tool

PREVIOUS_STEP_RESULT_PLACEHOLDER = "__PREVIOUS_STEP_RESULT__"


# ================== [ 关 键 修 复 ] ==================
# 重写占位符解析函数，使其能够处理嵌套的字典和列表（递归）
def _resolve_placeholders(data: Any, last_result: Any) -> Any:
    """
    递归地解析数据结构（字典或列表）中的占位符。
    """
    if isinstance(data, dict):
        # 如果是字典，递归地解析它的每一个值
        return {key: _resolve_placeholders(value, last_result) for key, value in data.items()}
    elif isinstance(data, list):
        # 如果是列表，递归地解析它的每一个元素
        return [_resolve_placeholders(item, last_result) for item in data]
    elif data == PREVIOUS_STEP_RESULT_PLACEHOLDER:
        # 如果元素本身就是占位符，替换它
        print(f"    👉 Resolving placeholder with previous step's result.")
        # 如果上一步结果是None（例如工具执行失败），返回一个安全的空值
        return last_result if last_result is not None else []
    else:
        # 如果是其他类型，直接返回
        return data


# =====================================================


class Executor:
    # ... (Executor类的 __init__ 和 run_tool 方法无需修改) ...
    def __init__(self):
        self.tools: Dict[str, Any] = {
            semantic_search_tool.name: semantic_search_tool,
            paper_finder_tool.name: paper_finder_tool,
            prediction_tool.name: prediction_tool,
        }
        print("✅ Executor initialized with toolset:", list(self.tools.keys()))

    def run_tool(self, tool_name: str, tool_input: dict) -> Tuple[Any, bool, Optional[str]]:
        tool_to_call = self.tools.get(tool_name)
        if not tool_to_call:
            error_msg = f"Tool '{tool_name}' not found."
            return None, False, error_msg
        try:
            result = tool_to_call.invoke(tool_input)
            if isinstance(result, str) and "出现错误:" in result:
                return result, False, result
            print(f"✅ Tool '{tool_name}' executed successfully.")
            return result, True, None
        except Exception as e:
            error_msg = f"Error executing tool '{tool_name}': {e}"
            return None, False, error_msg


def execute_node(state: GraphState, executor_instance: Executor) -> dict:
    print("--- [节点: Executor] ---")

    executed_steps_count = sum(1 for item in state['history'] if isinstance(item, Step))
    plan = state['plan']

    if not plan or executed_steps_count >= len(plan.steps):
        print("⚠️ Executor Warning: All planned steps have been executed or no plan exists.")
        return {}

    step_to_execute = plan.steps[executed_steps_count]
    tool_name = step_to_execute.tool_name

    # [修改] 调用新的、更强大的占位符解析函数
    last_successful_step_result = next(
        (step.result for step in reversed(state['history']) if isinstance(step, Step) and step.is_success),
        None
    )

    print(f"    Original input: {step_to_execute.tool_input}")
    resolved_tool_input = _resolve_placeholders(step_to_execute.tool_input, last_successful_step_result)
    print(f"    Resolved input: {resolved_tool_input}")

    print(f"🤖 Executing step {step_to_execute.step_id}: Tool={tool_name}")

    result, is_success, error_message = executor_instance.run_tool(tool_name, resolved_tool_input)

    step_result = Step(
        step_id=step_to_execute.step_id,
        tool_name=tool_name,
        tool_input=step_to_execute.tool_input,
        reasoning=step_to_execute.reasoning,
        result=result,
        is_success=is_success,
        error_message=error_message
    )

    history = state['history'] + [step_result]
    error_count = state.get('error_count', 0)
    if not is_success:
        error_count += 1

    return {"history": history, "error_count": error_count}
