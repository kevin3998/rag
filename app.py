# app.py

import streamlit as st
import functools
import sys
import os
from typing import List

# --- 路径设置 ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# --- LangGraph 和 LangChain 核心导入 ---
from langgraph.graph import StateGraph, END
from langgraph.pregel import Pregel
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

# --- 导入我们所有的模块和状态定义 ---
from rag_system.graph_state import GraphState
from rag_system.planner.planner import Planner, plan_node
from rag_system.executor.executor import Executor, execute_node
from rag_system.reflector.reflector import Reflector, reflect_node
from rag_system.decider.decider import should_continue
from rag_system.config import settings

# --- 最终答案生成节点 ---
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


def generate_final_answer_node(state: GraphState) -> dict:
    # ... (此函数保持不变) ...
    print("--- [节点: Final Answer Generator] ---")
    llm = ChatOllama(model=settings.LOCAL_LLM_MODEL_NAME, temperature=0.1)
    prompt = PromptTemplate.from_template(
        "你是一个科研助理，你需要根据用户的原始问题和系统的完整执行历史（包括之前的对话），生成一个完整、清晰且友好的最终答案。\n\n"
        "【历史对话】:\n{chat_history}\n\n"
        "【当前任务执行历史】:\n{history}\n\n"
        "【用户当前问题】:\n{initial_query}\n\n"
        "请根据以上所有信息，直接给出最终答案（不要包含任何思考过程或XML标签）:"
    )
    chain = prompt | llm | StrOutputParser()
    chat_history_str = "\n".join([f"{msg.type}: {msg.content}" for msg in state['chat_history']])
    history_summary = "\n".join([str(item) for item in state['history']])
    final_answer = chain.invoke({
        "initial_query": state['initial_query'],
        "history": history_summary,
        "chat_history": chat_history_str
    })
    return {"final_answer": final_answer}


# --- 图的组装 ---
@st.cache_resource
def build_agent_graph() -> Pregel:
    """
    构建并返回一个编译好的、可缓存的LangGraph智能代理。
    """
    print("--- Initializing Agent Graph for the first time ---")

    # 1. 先实例化需要共享工具的组件
    executor = Executor()
    tools = list(executor.tools.values())

    # 2. [修改] 将工具列表传递给Planner
    planner = Planner(tools=tools)
    reflector = Reflector()

    # 3. 绑定节点
    bound_plan_node = functools.partial(plan_node, planner_instance=planner)
    bound_execute_node = functools.partial(execute_node, executor_instance=executor)
    bound_reflect_node = functools.partial(reflect_node, reflector_instance=reflector)

    # 4. 定义工作流
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
        "reflector", should_continue,
        {"replan": "planner", "continue_execute": "executor", END: "final_answer_generator"}
    )

    app = workflow.compile()
    print("✅ Agent graph compiled successfully!")
    return app


# --- Streamlit 应用主逻辑 (保持不变) ---
st.set_page_config(page_title="🔬 LangGraph RAG Agent", layout="wide")
st.title("🔬 LangGraph RAG Agent")
st.markdown("一个具备规划、执行、反思和记忆能力的智能研究助手。由您微调的Qwen3驱动。")
with st.sidebar:
    st.header("关于")
    st.markdown("""
    这个应用展示了一个基于LangGraph构建的高级RAG Agent。它能够：
    - **理解对话历史**
    - **动态规划**任务步骤
    - **调用工具**检索知识
    - **反思结果**并**自我纠正**
    """)
    if st.button("清除对话历史"):
        st.session_state.messages = []
        st.rerun()
if "messages" not in st.session_state:
    st.session_state.messages: List[BaseMessage] = []
agent_app = build_agent_graph()
for message in st.session_state.messages:
    with st.chat_message(message.type):
        st.markdown(message.content)
if prompt := st.chat_input("请输入您的问题..."):
    st.session_state.messages.append(HumanMessage(content=prompt))
    with st.chat_message("user"):
        st.markdown(prompt)
    initial_state = {
        "initial_query": prompt,
        "plan": None,
        "history": [],
        "error_count": 0,
        "final_answer": None,
        "chat_history": st.session_state.messages[:-1]
    }
    with st.chat_message("assistant"):
        thinking_container = st.container()
        final_state = None
        for step_output in agent_app.stream(initial_state, stream_mode="values"):
            node_name = list(step_output.keys())[0]
            thinking_container.write(f"🧠 **思考中... [执行节点: {node_name}]**")
            final_state = step_output
        final_answer = final_state.get("final_answer", "抱歉，我无法得出结论。")
        st.markdown(final_answer)
    st.session_state.messages.append(AIMessage(content=final_answer))
