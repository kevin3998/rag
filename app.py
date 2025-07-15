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

# --- LangChain 核心导入 ---
from langgraph.graph import StateGraph, END
from langgraph.pregel import Pregel
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

# --- 导入我们所有的模块 ---
from rag_system.graph_state import GraphState, Step
from rag_system.planner.planner import Planner, plan_node
from rag_system.executor.executor import Executor, execute_node
from rag_system.reflector.reflector import Reflector, reflect_node
from rag_system.decider.decider import should_continue
from rag_system.config import settings

# --- 最终答案生成节点 ---
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import PromptTemplate


# [修改] 我们不再需要 StrOutputParser
# from langchain_core.output_parsers import StrOutputParser

def generate_final_answer_node(state: GraphState) -> dict:
    """
    生成最终答案节点
    """
    print("--- [节点: Final Answer Generator] ---")
    llm = ChatOllama(model=settings.LOCAL_LLM_MODEL_NAME, temperature=0.1)

    prompt = PromptTemplate.from_template(
        """# 角色
        你是一个严谨的科研助理。你的任务是根据【最终检索结果】和用户的【原始问题】，生成一个清晰、准确的最终答案。

        # 任务
        1.  **分析问题类型**: 判断用户的【原始问题】是要求一个【列表】，还是要求一个【分析总结】。
        2.  **忠实呈现**:
            - 如果问题要求一个列表（例如，“找出所有...的论文”），你的回答应该直接、完整地列出【最终检索结果】中的所有项目，并可以附上一句总结，例如“根据您的条件，共找到 N 篇论文如下：”。
            - 如果问题要求分析总结，请基于【最终检索结果】进行深入分析。
        3.  **不要编造**: 你的所有回答都必须严格基于【最终检索结果】。

        ---
        【原始问题】:
        {initial_query}
        ---
        【最终检索结果】:
        {final_context}
        ---
        你的最终答案:
        """
    )

    # ================== [ 关 键 修 复 ] ==================
    # 1. 我们的链现在只包含Prompt和LLM
    chain = prompt | llm

    # 2. 我们只将最后一步成功的结果作为最终上下文
    last_successful_result = next(
        (step.result for step in reversed(state['history']) if isinstance(step, Step) and step.is_success),
        "未能从执行历史中找到任何有效结果。"
    )

    # 3. 调用链，得到的是一个AIMessage对象，而不是字符串
    ai_message_result = chain.invoke({
        "initial_query": state['initial_query'],
        "final_context": str(last_successful_result)
    })

    # 4. 我们手动从AIMessage对象中提取内容字符串
    final_answer = ai_message_result.content
    # =====================================================

    return {"final_answer": final_answer}


# ... (build_agent_graph 和 Streamlit 的主逻辑保持不变) ...
@st.cache_resource
def build_agent_graph() -> Pregel:
    print("--- Initializing Agent Graph for the first time ---")
    executor = Executor()
    tools = list(executor.tools.values())
    planner = Planner(tools=tools)
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
        "reflector", should_continue,
        {"replan": "planner", "continue_execute": "executor", END: "final_answer_generator"}
    )
    app = workflow.compile()
    print("✅ Agent graph compiled successfully!")
    return app


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
