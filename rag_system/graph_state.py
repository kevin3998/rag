from typing import List, Optional, Dict, Any, Union, TypedDict
from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field


# Step, Plan, Reflection这些Pydantic模型定义保持不变
class Step(BaseModel):
    step_id: int
    tool_name: str
    tool_input: Dict[str, Any]
    reasoning: str
    result: Any = None
    is_success: bool = False
    error_message: Optional[str] = None


class Plan(BaseModel):
    goal: str
    steps: List[Step] = Field(default_factory=list)


class Reflection(BaseModel):
    critique: str
    is_success: bool
    confidence: float
    suggestion: str
    is_finished: bool = False


# [核心修改] 我们在您原有的TypedDict基础上增加两个字段
class GraphState(TypedDict):
    """The central state of the graph, passed between all nodes."""
    initial_query: str
    plan: Optional[Plan]
    history: List[Union[Step, Reflection, str]]
    error_count: int
    final_answer: Optional[str]  # 您原有的字段
    chat_history: List[BaseMessage]

    # [新增] 用于存放最终生成答案的字段
    generation: Optional[str]
    # [新增] 用于存放路由决策结果的字段
    route_decision: Optional[str]