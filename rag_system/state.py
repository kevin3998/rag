# rag_system/state.py

from typing import List, Optional, Dict, Any, Literal, Union
from pydantic import BaseModel, Field


class Step(BaseModel):
    step_id: int = Field(..., description="步骤的唯一标识符，从1开始计数。")
    tool_name: str = Field(..., description="该步骤需要调用的工具名称。")
    tool_input: Dict[str, Any] = Field(..., description="调用工具时需要传入的参数字典。")
    reasoning: str = Field(..., description="Planner对此步骤的规划理由。")
    result: Optional[str] = Field(None, description="此步骤执行后由工具返回的原始输出。")
    is_success: bool = Field(False, description="此步骤是否成功执行。")
    error_message: Optional[str] = Field(None, description="如果执行失败，记录错误信息。")


class Plan(BaseModel):
    goal: str = Field(..., description="用户的原始、高层次目标。")
    steps: List[Step] = Field(default_factory=list, description="构成计划的所有步骤列表。")


class ReflectionOutput(BaseModel):
    """
    定义了LLM在进行反思时需要输出的Pydantic模型。
    这只包含需要LLM生成的部分，不包含 step_id 等上下文信息。
    """
    critique: str = Field(..., description="对上一步执行结果的详细、批判性的评估。")
    is_success: bool = Field(..., description="根据评估，判断上一步是否算成功。")
    confidence: float = Field(..., description="对is_success判断的置信度，范围在0.0到1.0之间。")
    suggestion: str = Field(..., description="基于评估，为下一步行动提出的具体建议。")
    is_finished: bool = Field(..., description="判断整个任务是否已经完成，可以得出最终答案。")

class Reflection(BaseModel):
    """
    Represents the reflector's assessment of a single step's execution.
    """
    step_id: int = Field(..., description="The ID of the step being reflected upon.")
    critique: str = Field(..., description="The LLM's critique of the step's outcome.")
    is_success: bool = Field(..., description="Whether the Reflector deems the step successful.")
    confidence: float = Field(..., description="The confidence score (0.0 to 1.0) in the success assessment.")
    suggestion: str = Field(..., description="A suggestion for what to do next.")


class Action(BaseModel):
    action_type: Literal['PROCEED', 'RETRY', 'REPLAN', 'FINISH'] = Field(
        ...,
        description="决策类型：PROCEED (继续), RETRY (重试), REPLAN (重新规划), FINISH (完成/终止)。"
    )
    corrected_input: Optional[Dict[str, Any]] = Field(
        None,
        description="如果action_type是 'RETRY'，这里应包含为失败步骤修正后的新输入参数。"
    )
    reasoning: str = Field(..., description="做出此决策的理由。")


class AgentState(BaseModel):
    initial_query: str = Field(..., description="用户输入的原始问题。")
    plan: Optional[Plan] = Field(None, description="由Planner生成的当前任务计划。")

    history: List[Union[Step, Reflection]] = Field(default_factory=list,
                                                   description="记录每一次的执行步骤(Step)和对其的反思(Reflection)历史。")

    current_step_id: int = Field(1, description="指向当前正在执行或等待执行的步骤ID，从1开始。")
    final_answer: Optional[str] = Field(None, description="当任务完成时，生成的最终答案。")

    def get_step_by_id(self, step_id: int) -> Optional[Step]:
        if self.plan:
            for step in self.plan.steps:
                if step.step_id == step_id:
                    return step
        return None

    def update_step_result(self, step_id: int, result: str, is_success: bool, error_message: Optional[str] = None):
        step = self.get_step_by_id(step_id)
        if step:
            step.result = result
            step.is_success = is_success
            step.error_message = error_message
            self.history.append(step)

    def get_next_step(self) -> Optional[Step]:
        """获取当前计划中需要执行的下一步。"""
        if self.is_plan_completed():
            return None
        return self.get_step_by_id(self.current_step_id)

    def is_plan_completed(self) -> bool:
        """检查计划中的所有步骤是否都已执行。"""
        if not self.plan or not self.plan.steps:
            return True  # 没有计划就等于计划已完成
        return self.current_step_id > len(self.plan.steps)