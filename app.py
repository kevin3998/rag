import streamlit as st

# 【重要】由于app.py现在位于根目录，它可以直接看到rag_system这个包
# 因此，这个绝对路径导入现在是完全正确的，并且IDE也不会报错
from rag_system.main_controller import MainController

# --- Streamlit 页面配置 ---
st.set_page_config(
    page_title="膜科学智能研究助手",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🔬 膜科学智能研究助手 (v2.0 - Plan & Reflect)")

# --- 应用侧边栏 ---
with st.sidebar:
    st.header("关于")
    st.markdown("""
    这是一个基于 **Plan-Act-Reflect** 架构的AI研究助手。

    它能够将您的复杂问题分解为多个步骤，并利用以下工具进行处理：

    - **语义检索引擎**: 用于理解和总结文献内容。
    - **结构化数据库**: 用于精确查询论文中的实验数据。

    输入您在膜科学研究中遇到的问题，观察AI如何规划、执行并反思，最终给出答案。
    """)

    st.header("使用示例")
    st.info("""
    - **简单查询**: "什么是聚偏氟乙烯(PVDF)膜？"
    - **精确数据查询**: "找出所有溶剂是NMP，且水接触角小于70度的论文"
    - **复杂任务**: "找出2022年后发表的，关于TFN膜的所有论文，并总结它们与TFC膜相比的优势。"
    """)

# --- 会话状态管理 ---
if "messages" not in st.session_state:
    st.session_state.messages = []


# --- 缓存主控制器实例 ---
@st.cache_resource
def get_main_controller():
    """
    缓存MainController实例，避免每次重新加载。
    """
    print("--- Initializing MainController for the first time ---")
    return MainController(max_loops=5)


# 获取控制器实例
controller = get_main_controller()

# --- 聊天界面 ---
# 显示历史消息
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 接收用户输入
if prompt := st.chat_input("请输入您的问题..."):
    # 将用户消息添加到聊天记录和显示中
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 调用新的MainController并流式显示输出
    with st.chat_message("assistant"):
        response_generator = controller.run(prompt)
        full_response = st.write_stream(response_generator)

    # 将助手的完整响应存入聊天记录
    st.session_state.messages.append({"role": "assistant", "content": full_response})