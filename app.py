# app.py (最终修复版)

import streamlit as st
import functools
import sys
import os
from typing import List

# --- 路径设置 ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# --- LangChain 及项目模块导入 ---
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama
from pydantic import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser

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

# --- 初始化核心组件 ---
try:
    print("--- 正在初始化所有Agent组件 ---")

    tools = [paper_finder_tool, semantic_search_tool, prediction_tool]
    general_llm = ChatOllama(model=settings.PREDICTION_MODEL_NAME, temperature=0)

    # [最终修复] 根据您的确认，精确地为每个组件提供其所需的参数
    planner_instance = Planner(tools=tools)  # Planner 需要 tools
    executor_instance = Executor()  # Executor 不需要参数
    reflector_instance = Reflector()  # Reflector 不需要参数

    print("✅ 所有Agent组件初始化成功。")
except Exception as e:
    print(f"❌ 初始化核心组件时发生致命错误: {e}")
    st.error(f"应用启动失败：无法初始化核心组件。错误信息: {e}")
    sys.exit(1)


# --- 路由和节点函数定义 (这部分逻辑已正确，无需改动) ---

class RouteQuery(BaseModel):
    datasource: str = Field(enum=["membrane_query", "general_chat", "out_of_domain_query"])


def get_router(llm):
    parser = JsonOutputParser(pydantic_object=RouteQuery)
    prompt = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
        你是一个高级问题分类专家。你的核心知识领域严格限定在“膜科学与技术”。请根据用户最新的问题，将其分类到以下三个类别之一，并以JSON格式返回。
        1. 'membrane_query': 与膜科学直接相关的问题。
        2. 'general_chat': 日常问候、关于你身份的闲聊。
        3. 'out_of_domain_query': 其他领域的专业问题。
        {format_instructions}
        用户最新的问题: {question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
        input_variables=["question"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    json_llm = llm.bind(format="json")
    return prompt | json_llm | parser


def classify_question_node(state: GraphState):
    question = state['initial_query']
    router_chain = get_router(general_llm)
    result = router_chain.invoke({"question": question})
    decision = result.get("datasource")
    return {"route_decision": decision}


def decide_next_step(state: GraphState) -> str:
    return state.get('route_decision') or "planner"


def general_chat_node(state: GraphState):
    question = state['initial_query']
    persona_prompt = f"你是 'RAG-Intelligent-Researcher' AI助手。请友好、简洁地回答用户的问题。\n用户问题: {question}"
    response = general_llm.invoke(persona_prompt)
    return {"generation": response.content}


def decline_answer_node(state: GraphState):
    generation = "非常抱歉，我的知识范围主要集中在“膜科学与技术”领域。您提出的问题超出了我的专业范畴，为了避免提供不准确的信息，我无法给出有效的回答。"
    return {"generation": generation}


def generate_final_answer_node(state: GraphState) -> dict:
    print("--- [节点: Generate Final Answer] ---")

    # 优先处理来自闲聊或拒绝节点的直接生成内容
    if generation := state.get("generation"):
        return {"final_answer": generation}

    print("    - [来源]: RAG Pipeline")

    # 获取最后一个执行步骤
    last_step = state['history'][-1]

    # 检查最后一个步骤是否是成功的工具调用，并且返回了结果
    if isinstance(last_step, Step) and last_step.is_success and last_step.result:
        # 如果最后一个工具是semantic_search或prediction，它们的结果本身就是一份完整的报告
        if last_step.tool_name in ["semantic_search_tool", "prediction_tool"]:
            print("    - [策略]: 直接使用最后一个工具的输出作为最终答案。")
            return {"final_answer": last_step.result}

    # 如果不满足上述条件，则执行原有的“兜底”总结逻辑
    print("    - [策略]: 对所有历史结果进行最终总结。")
    final_context_list = []
    for item in state['history']:
        if isinstance(item, Step) and item.result is not None:
            final_context_list.append(str(item.result))
    final_context = "\n\n---\n\n".join(final_context_list)

    if not final_context:
        final_context = "未能从执行历史中找到任何有效结果。"

    prompt = PromptTemplate.from_template(
        "# 角色: 严谨的科研助理...\n【原始问题】:{initial_query}\n【最终检索结果】:{final_context}\n你的最终答案:")
    final_chain = prompt | general_llm
    response = final_chain.invoke({"initial_query": state['initial_query'], "final_context": final_context})
    return {"final_answer": response.content}

# --- [核心修复] 构建最终的Graph ---
def build_agent_graph():
    workflow = StateGraph(GraphState)

    plan_node_partial = functools.partial(plan_node, planner_instance=planner_instance)
    execute_node_partial = functools.partial(execute_node, executor_instance=executor_instance)
    reflect_node_partial = functools.partial(reflect_node, reflector_instance=reflector_instance)

    workflow.add_node("classifier", classify_question_node)
    workflow.add_node("general_chat", general_chat_node)
    workflow.add_node("decline_node", decline_answer_node)
    workflow.add_node("planner", plan_node_partial)
    workflow.add_node("executor", execute_node_partial)
    workflow.add_node("reflector", reflect_node_partial)
    workflow.add_node("final_answer_generator", generate_final_answer_node)

    workflow.set_entry_point("classifier")

    workflow.add_conditional_edges("classifier", decide_next_step, {
        "membrane_query": "planner",
        "general_chat": "general_chat",
        "out_of_domain_query": "decline_node"
    })

    workflow.add_edge("general_chat", "final_answer_generator")
    workflow.add_edge("decline_node", "final_answer_generator")
    workflow.add_edge("planner", "executor")
    workflow.add_edge("executor", "reflector")

    # [核心修复] 将这里的键改回 'finish'，以匹配您的decider的实际输出
    workflow.add_conditional_edges(
        "reflector",
        should_continue,
        {
            "continue_execute": "executor",
            "replan": "planner",
            "finish": "final_answer_generator",  # <-- 已改回 'finish'
        }
    )

    workflow.add_edge("final_answer_generator", END)

    print("✅ LangGraph工作流编译完成。")
    return workflow.compile()


# --- Streamlit UI (保持不变) ---
st.title("🤖 RAG 智能科研助手")
st.markdown("一个具备规划、执行、反思和记忆能力的智能研究助手。由您微调的Qwen3驱动。")

if "agent_app" not in st.session_state:
    st.session_state.agent_app = build_agent_graph()

if "messages" not in st.session_state:
    st.session_state.messages = []

if not st.session_state.messages:
    with st.chat_message("ai"):
        st.markdown("您好！我是您的AI科研助手，请问有什么可以帮您的？")
else:
    for message in st.session_state.messages:
        with st.chat_message(message.type):
            st.markdown(message.content)

if prompt := st.chat_input("请输入您的问题..."):
    st.session_state.messages.append(HumanMessage(content=prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    initial_state = {
        "initial_query": prompt,
        "history": [],
        "chat_history": st.session_state.messages[:-1]
    }

    with st.chat_message("assistant"):
        with st.spinner("🧠 **思考中...**"):
            try:
                final_state = st.session_state.agent_app.invoke(initial_state)
                final_answer = final_state.get("final_answer", "抱歉，处理过程中出现问题，无法生成回答。")
            except Exception as e:
                final_answer = f"处理请求时发生错误: {e}"
                st.error(final_answer)

        st.markdown(final_answer)
        st.session_state.messages.append(AIMessage(content=final_answer))