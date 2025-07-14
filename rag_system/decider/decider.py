# rag_system/decider.py

from langgraph.graph import END

from rag_system.graph_state import GraphState, Reflection, Step
from rag_system.config import settings

MAX_ERROR_COUNT = 3


def should_continue(state: GraphState) -> str:
    """
    决策节点 (Conditional Edge Logic)
    在每次反思后，根据图的完整状态决定流程的下一步走向。
    """
    print("--- [决策节点] ---")

    # --- 规则 0: 检查计划是否有效 ---
    # [新增] 这是解决当前无限循环的关键
    plan = state.get('plan')
    if not plan or not plan.steps:
        # 如果从一开始就没有计划，或者计划是空的，那么直接结束。
        print("🤔 决策: 计划为空或不存在。结束流程。")
        return END

    # --- 规则 1: 检查是否应强制结束 ---
    if state.get('error_count', 0) >= MAX_ERROR_COUNT:
        print(f"🚨 决策: 达到最大错误次数 ({MAX_ERROR_COUNT})。强制结束。")
        return END

    # --- 规则 2: 分析最新的反思结果 ---
    last_reflection = next((item for item in reversed(state['history']) if isinstance(item, Reflection)), None)

    # [修改] 调整这里的逻辑
    # 只有在确实有步骤需要执行，但还没有反思的情况下，才应该继续
    executed_steps_count = sum(1 for item in state['history'] if isinstance(item, Step))
    if not last_reflection:
        # 如果没有反思，但已经执行完了所有步骤，说明应该结束了
        if executed_steps_count >= len(plan.steps):
            print("✅ 决策: 所有步骤已执行完毕，且没有需要反思的内容。结束。")
            return END
        else:
            # 这种情况理论上不应该发生，但作为保护，我们让它继续
            print("👍 决策: 没有反思记录，但仍有步骤待执行。继续。")
            return "continue_execute"

    # --- 规则 3: 根据反思做出决策 (逻辑保持不变) ---
    if getattr(last_reflection, 'is_finished', False):
        print("✅ 决策: 反思表明任务已完成。结束。")
        return END

    if last_reflection.is_success:
        if last_reflection.confidence >= settings.REFLECTION_CONFIDENCE_THRESHOLD:
            if executed_steps_count >= len(plan.steps):
                print("✅ 决策: 所有计划步骤已成功执行。结束。")
                return END
            else:
                print("👍 决策: 上一步成功且任务未完，继续执行。")
                return "continue_execute"
        else:
            print(f"⚠️ 决策: 成功但置信度低 ({last_reflection.confidence:.2f})，需要重新规划。")
            return "replan"
    else:
        print("❌ 决策: 上一步失败，需要重新规划。")
        return "replan"

