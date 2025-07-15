# rag_system/agent/tools/prediction_tool.py

from typing import Any, Optional
from langchain_core.tools import tool
from langchain_core.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama
from pydantic import BaseModel, Field

from rag_system.config import settings

# --- 全局组件初始化 ---
# [核心] 这里我们将调用一个专门为预测任务微调的新模型
# 我们将在settings.py中定义它的名字
try:
    print("--- Initializing Prediction Model ---")
    # 假设您在settings.py中定义了一个新变量 PREDICTION_MODEL_NAME
    # 如果没有定义，它会回退到使用默认的LLM模型
    prediction_model_name = getattr(settings, 'PREDICTION_MODEL_NAME', settings.LOCAL_LLM_MODEL_NAME)
    prediction_llm = ChatOllama(model=prediction_model_name, temperature=0.1)
    print(f"✅ Prediction tool will use model: {prediction_model_name}")
except AttributeError:
    print("⚠️ Warning: PREDICTION_MODEL_NAME not found in settings. Using default LLM.")
    prediction_llm = ChatOllama(model=settings.LOCAL_LLM_MODEL_NAME, temperature=0.1)


# --- 为预测任务设计的专属Prompt ---
PREDICTION_PROMPT = PromptTemplate.from_template(
    """# 角色
    你是一位顶尖的、具有前瞻性的首席科学家。你的任务不是总结已知信息，而是基于现有的【文献证据】，进行大胆而又严谨的【预测、假设或实验设计】。

    # 任务
    请仔细分析用户提出的【核心问题】和所有相关的【文献证据】，然后完成以下一项或多项任务：
    1.  **趋势预测**: 如果被问及未来趋势，请识别出现有研究的空白、热点和潜在突破方向。
    2.  **因果推断**: 如果被问及机理，请深入分析变量之间的因果关系，而不仅仅是相关性。
    3.  **实验设计**: 如果需要验证一个假设，请设计一个逻辑严谨、包含对照组和关键测量指标的实验方案。
    4.  **风险与机遇评估**: 评估某个技术方向的潜在风险和未来机遇。

    你的回答必须扎根于【文献证据】，但又要超越它们，展现出真正的洞察力。

    ---
    【文献证据】:
    {context}
    ---
    【核心问题】:
    {question}
    ---
    你的预测与深度分析报告:
    """
)

prediction_chain = PREDICTION_PROMPT | prediction_llm

class PredictionInput(BaseModel):
    question: str = Field(description="一个需要进行深度预测、因果推断或实验设计的核心问题。")
    context: Any = Field(description="由前序工具（如semantic_search_tool）提供的、包含所有相关文献信息的上下文。")

@tool(args_schema=PredictionInput)
def prediction_tool(question: str, context: Any) -> str:
    """
    一个用于深度分析、趋势预测和实验设计的专家工具。
    当用户的问题需要超越简单的信息总结，进行前瞻性思考或因果推断时，应该在收集完所有相关信息后，最后一步调用此工具。
    """
    print("--- [Tool: prediction_tool] 启动深度预测与推理 ---")
    if not isinstance(context, str):
        # 增加一步转换，确保传递给Prompt的context是字符串
        context = str(context)

    if not context.strip():
        return "错误：无法在没有任何上下文信息的情况下进行预测或推理。前序的检索步骤未能提供有效信息。"

    try:
        response = prediction_chain.invoke({"question": question, "context": context})
        return response.content
    except Exception as e:
        return f"执行预测与推理时出现严重错误: {e}"

