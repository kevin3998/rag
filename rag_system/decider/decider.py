from rag_system.graph_state import GraphState, Reflection, Step
from rag_system.config import settings

MAX_ERROR_COUNT = 3

def should_continue(state: GraphState) -> str:
    """
    决策节点 (Conditional Edge Logic)
    """
    print("--- [决策节点] ---")

    plan = state.get('plan')
    if not plan or not plan.steps:
        print("🤔 决策: 计划为空或不存在。结束流程。")
        return "finish"  # ✅ 不再使用 END

    if state.get('error_count', 0) >= MAX_ERROR_COUNT:
        print(f"🚨 决策: 达到最大错误次数 ({MAX_ERROR_COUNT})。强制结束。")
        return "finish"  # ✅

    last_reflection = next((item for item in reversed(state['history']) if isinstance(item, Reflection)), None)
    executed_steps_count = sum(1 for item in state['history'] if isinstance(item, Step))

    if not last_reflection:
        if executed_steps_count >= len(plan.steps):
            print("✅ 决策: 所有步骤已执行完毕，且没有需要反思的内容。结束。")
            return "finish"  # ✅
        else:
            print("👍 决策: 没有反思记录，但仍有步骤待执行。继续。")
            return "continue_execute"

    if getattr(last_reflection, 'is_finished', False):
        print("✅ 决策: 反思表明任务已完成。结束。")
        return "finish"  # ✅

    if last_reflection.is_success:
        if last_reflection.confidence >= settings.REFLECTION_CONFIDENCE_THRESHOLD:
            if executed_steps_count >= len(plan.steps):
                print("✅ 决策: 所有计划步骤已成功执行。结束。")
                return "finish"  # ✅
            else:
                print("👍 决策: 上一步成功且任务未完，继续执行。")
                return "continue_execute"
        else:
            print(f"⚠️ 决策: 成功但置信度低 ({last_reflection.confidence:.2f})，需要重新规划。")
            return "replan"
    else:
        print("❌ 决策: 上一步失败，需要重新规划。")
        return "replan"
