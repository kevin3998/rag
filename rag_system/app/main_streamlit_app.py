import streamlit as st
import sys
from pathlib import Path

# --- 项目路径设置 ---
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# --- 导入核心组件 ---
from rag_system.agent.agent_executor import MaterialScienceAgent
from rag_system.config import settings

# --- 页面配置与组件加载 ---
st.set_page_config(page_title="材料科学AI助手", page_icon="🧪", layout="wide")


@st.cache_resource
def load_agent():
    """
    加载并缓存智能体实例。
    【关键修改】移除了错误的 st.set_option 调用。
    """
    return MaterialScienceAgent()


st.title("🧪 材料科学AI助手")
st.caption(f"由本地模型 '{settings.LOCAL_LLM_MODEL_NAME}' 驱动 (ReAct架构)")

agent = load_agent()

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 侧边栏 ---
with st.sidebar:
    st.header("关于系统")
    st.info("这是一个基于ReAct框架的AI助手，能够调用工具来回答专业问题。")
    if st.button("清除聊天记录", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# --- 主聊天界面 ---
# 1. 渲染所有历史消息
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant" and message.get("think_process"):
            with st.expander("查看AI思考过程", expanded=False):
                st.markdown(message["think_process"], unsafe_allow_html=True)
        st.markdown(message["content"])

# 2. 接收并处理新输入
if prompt := st.chat_input("请输入您的问题..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # 为实时渲染准备UI占位符
        answer_placeholder = st.empty()
        with st.expander("AI思考过程", expanded=True) as think_process_expander:
            log_placeholder = st.empty()

        full_response = ""
        think_process_log = ""

        try:
            stream = agent.run(prompt)
            for chunk in stream:
                if "log" in chunk:
                    think_process_log += chunk["log"].strip().replace('<', '&lt;').replace('>', '&gt;') + "\n\n"
                    log_placeholder.markdown(think_process_log)
                elif "output" in chunk:
                    full_response += chunk["output"]
                    answer_placeholder.markdown(full_response + "▌")
        except Exception as e:
            st.error(f"处理时出现错误: {e}", icon="🚨")
            full_response = "抱歉，处理您的问题时出现了错误。"

        # 流程结束后，最终更新UI
        answer_placeholder.markdown(full_response)
        # 检查思考日志是否存在，再更新其最终状态
        if think_process_log:
            log_placeholder.markdown(think_process_log)

    # 将本次交互的完整结果存入历史记录
    st.session_state.messages.append({
        "role": "assistant",
        "content": full_response,
        "think_process": think_process_log
    })
    # 使用rerun来确保UI状态正确刷新，并将新消息变为历史
    st.rerun()