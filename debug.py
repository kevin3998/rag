# debug_rag_flow.py (最终修复版)

import functools
import sys
import os
import pprint
from typing import List

# --- 路径设置 ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# --- 导入所有必需的模块 ---
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama

# 导入您的核心模块
from rag_system.graph_state import GraphState, Step
from rag_system.config import settings
from rag_system.planner.planner import Planner, plan_node
from rag_system.executor.executor import Executor, execute_node
from rag_system.reflector.reflector import Reflector, reflect_node
from rag_system.decider.decider import should_continue

# 导入所有需要用到的工具
from rag_system.agent.tools.paper_finder_tool import paper_finder_tool
from rag_system.agent.tools.semantic_search import semantic_search_tool
from rag_system.agent.tools.prediction_tool import prediction_tool

# --- 1. 初始化核心组件 ---
print("--- [步骤1] 正在初始化所有Agent组件 ---")
try:
    tools = [paper_finder_tool, semantic_search_tool, prediction_tool]

    # 根据您的确认，只有Planner需要tools
    planner_instance = Planner(tools=tools)
    executor_instance = Executor()
    reflector_instance = Reflector()

    # 创建一个单独的LLM实例，用于最终答案生成
    answer_llm = ChatOllama(model=settings.PREDICTION_MODEL_NAME, temperature=0)

    print("✅ 所有Agent组件初始化成功。")
except Exception as e:
    print(f"❌ 初始化核心组件时发生致命错误: {e}")
    sys.exit(1)


# --- [新增] 最终答案生成节点函数 ---
# 我们在这里模拟app.py中的最终答案生成逻辑
def generate_final_answer_node(state: GraphState) -> dict:
    print("--- [节点: Generate Final Answer] ---")

    # 从历史记录中提取所有工具的执行结果作为上下文
    final_context_list = []
    for item in state['history']:
        if isinstance(item, Step) and item.result is not None:
            final_context_list.append(str(item.result))

    final_context = "\n\n---\n\n".join(final_context_list)

    if not final_context:
        final_context = "未能从执行历史中找到任何有效结果。"

    prompt = PromptTemplate.from_template(
        "# 角色: 严谨的科研助理\n"
        "# 任务: 基于【最终检索结果】和【原始问题】，生成最终答案。\n"
        "【原始问题】:{initial_query}\n"
        "【最终检索结果】:{final_context}\n"
        "你的最终答案:"
    )

    final_chain = prompt | answer_llm
    response = final_chain.invoke({
        "initial_query": state['initial_query'],
        "final_context": final_context
    })

    # 将最终答案存入state
    return {"final_answer": response.content}


# --- 2. 构建Graph ---
@functools.lru_cache(maxsize=None)
def build_agent_graph():
    print("--- [步骤2] 正在构建LangGraph工作流 ---")
    workflow = StateGraph(GraphState)

    workflow.add_node("planner", lambda state: plan_node(state, planner_instance))
    workflow.add_node("executor", lambda state: execute_node(state, executor_instance))
    workflow.add_node("reflector", lambda state: reflect_node(state, reflector_instance))

    # [核心修复] 添加最终答案生成节点
    workflow.add_node("final_answer_generator", generate_final_answer_node)

    workflow.set_entry_point("planner")
    workflow.add_edge("planner", "executor")
    workflow.add_edge("executor", "reflector")

    # [核心修复] 将'finish'信号正确地连接到最终答案生成节点
    workflow.add_conditional_edges(
        "reflector",
        should_continue,
        {
            "continue_execute": "executor",
            "replan": "planner",
            "finish": "final_answer_generator",  # <-- 不再是END，而是连接到下一个节点
        }
    )

    # [核心修复] 设置最终的终点
    workflow.add_edge("final_answer_generator", END)

    print("✅ LangGraph工作流编译完成。")
    return workflow.compile()


# --- 3. 运行调试流程 ---
def run_debug():
    print("\n--- [步骤3] 开始执行RAG流程调试 ---")

    user_query = "请详细介绍一下PVDF材料"
    print(f"模拟用户输入: '{user_query}'")

    agent_app = build_agent_graph()

    initial_state = {
        "initial_query": user_query,
        "history": [],
        "chat_history": []
    }

    print("\n--- [执行日志] ---")
    try:
        final_state = agent_app.invoke(initial_state)

        print("\n--- [调试成功] ---")
        print("✅ RAG工作流成功执行到终点！")
        print("\n最终生成的答案是:")
        print("=" * 50)
        print(final_state.get("final_answer", "未能获取到最终答案。"))
        print("=" * 50)

        # print("\n完整的最终状态 (Final State):")
        # pprint.pprint(final_state)

    except Exception as e:
        print("\n--- [调试失败] ---")
        print(f"❌ 在执行RAG工作流时发生错误: {e}")
        import traceback
        traceback.print_exc()


# --- 主程序入口 ---
if __name__ == "__main__":
    run_debug()