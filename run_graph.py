# run_graph.py

import functools
import sys
import os

# --- 动态添加项目根目录到Python搜索路径 ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from langgraph.graph import StateGraph, END

# --- 导入我们所有的模块 ---
from rag_system.graph_state import GraphState
from rag_system.planner.planner import Planner, plan_node
from rag_system.executor.executor import Executor, execute_node
from rag_system.reflector.reflector import Reflector, reflect_node
from rag_system.decider.decider import should_continue

from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from rag_system.config import settings

def generate_final_answer_node(state: GraphState) -> dict:
    """
    生成最终答案节点
    """
    print("--- [节点: Final Answer Generator] ---")
    llm = ChatOllama(model=settings.LOCAL_LLM_MODEL_NAME, temperature=0.1)
    prompt = PromptTemplate.from_template(
        "你是一个科研助理，你需要根据用户的原始问题和系统的执行历史，生成一个完整、清晰且友好的最终答案。\n\n"
        "【用户原始问题】:\n{initial_query}\n\n"
        "【系统执行历史摘要】:\n{history}\n\n"
        "请根据以上信息，直接给出最终答案:"
    )
    chain = prompt | llm | StrOutputParser()
    history_summary = "\n".join([str(item) for item in state['history']])
    final_answer = chain.invoke({
        "initial_query": state['initial_query'],
        "history": history_summary
    })
    return {"final_answer": final_answer}

def build_agent_graph() -> StateGraph:
    """
    构建并返回一个编译好的LangGraph智能代理。
    """
    planner = Planner()
    executor = Executor()
    reflector = Reflector()

    bound_plan_node = functools.partial(plan_node, planner_instance=planner)
    bound_execute_node = functools.partial(execute_node, executor_instance=executor)
    bound_reflect_node = functools.partial(reflect_node, reflector_instance=reflector)

    workflow = StateGraph(GraphState)

    workflow.add_node("planner", bound_plan_node)
    workflow.add_node("executor", bound_execute_node)
    workflow.add_node("reflector", bound_reflect_node)
    workflow.add_node("final_answer_generator", generate_final_answer_node)

    workflow.set_entry_point("planner")

    workflow.add_edge("planner", "executor")
    workflow.add_edge("executor", "reflector")
    workflow.add_edge("final_answer_generator", END)

    workflow.add_conditional_edges(
        "reflector",
        should_continue,
        {
            "replan": "planner",
            "continue_execute": "executor",
            END: "final_answer_generator"
        }
    )

    app = workflow.compile()
    print("✅ Agent graph compiled successfully!")
    return app

if __name__ == "__main__":
    agent_app = build_agent_graph()
    query = "TFN膜的结构和性能有什么关系？请结合近三年的论文进行总结。"
    initial_state = {
        "initial_query": query,
        "plan": None,
        "history": [],
        "error_count": 0,
        "final_answer": None,
    }

    print(f"\n🚀 Starting Agent with query: '{query}'\n" + "="*50)

    final_state = None
    # [修改] 我们只使用stream来运行，并在循环中保存最后的状态
    for step_output in agent_app.stream(initial_state, stream_mode="values"):
        node_name = list(step_output.keys())[0]
        node_output = list(step_output.values())[0]
        print(f"\n--- ✅ Executed Node: {node_name} ---")
        print(f"Node Output:\n{node_output}")
        print("-" * 50)
        # 保存最新的完整状态
        final_state = step_output

    print("\n🏁 Agent run has finished.")

    if final_state and final_state.get('final_answer'):
        print("\n\nFINAL ANSWER:")
        print(final_state['final_answer'])
    else:
        print("\n\nAgent finished without a final answer. Dumping final state:")
        print(final_state)

