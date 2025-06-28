# rag_system/decider.py

from typing import Literal
from rag_system.state import Reflection  # 确认是从state导入
from rag_system.config import settings

Decision = Literal["PROCEED", "RETRY", "REPLAN", "FINISH"]


class Decider:
    def __init__(self):
        # 基于规则的Decider不需要LLM
        print("✅ Rule-based Decider initialized successfully.")

    # 【核心修正】确保方法的参数是 reflection，类型是 Reflection
    def decide(self, reflection: Reflection) -> Decision:
        """
        根据最新的反思结果，做出下一步的决定。
        """
        if not reflection:
            print("⚠️ Decider received no reflection. Defaulting to REPLAN.")
            return "REPLAN"

        if reflection.is_success:
            if reflection.confidence >= settings.REFLECTION_CONFIDENCE_THRESHOLD:
                # 在state.py的Reflection中没有is_finished, 我们在Decider中判断
                # 如果是最后一步成功了，也应该FINISH
                if getattr(reflection, 'is_finished', False):
                    return "FINISH"
                return "PROCEED"
            else:
                print(f"⚠️ Low confidence success (Confidence: {reflection.confidence:.2f}), triggering a replan...")
                return "REPLAN"
        else:
            print(f"❌ Execution failed, triggering a replan...")
            return "REPLAN"