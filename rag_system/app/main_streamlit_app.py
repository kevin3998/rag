import streamlit as st
import sys
from pathlib import Path

# 将项目根目录添加到Python路径中
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# 确保这里的类名与您 qa_chain.py 文件中的一致
from rag_system.generation.qa_chain import AdvancedQAChain
from rag_system.config import settings

# --- 页面配置 ---
st.set_page_config(
    page_title="膜材料科学RAG智能体",
    page_icon="🧪",
    layout="wide"
)

st.title("🧪 膜材料科学RAG智能体")
# 使用您在Ollama中为模型取的名字
st.caption(f"由本地模型 'qwen3-14b-f16:latest' 和 ChromaDB 提供支持")


# --- 初始化与状态管理 ---
@st.cache_resource
def load_qa_chain():
    """
    使用缓存加载QA链，避免每次页面重载时都重新初始化模型。
    """
    try:
        qa_chain_instance = AdvancedQAChain()
        return qa_chain_instance
    except Exception as e:
        st.error(f"加载RAG系统失败: {e}")
        st.stop()


qa_chain = load_qa_chain()

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 侧边栏 ---
with st.sidebar:
    st.header("关于")
    st.info("这是一个基于检索增强生成 (RAG) 的智能问答系统。")
    st.markdown("""
    **功能:**
    - **日常对话**: 可以进行通用聊天。
    - **专业问答**: 能基于本地“膜材料”文献库回答专业问题。
    - **领域识别**: 当问题超出膜材料领域时，会礼貌地拒绝回答。
    """)
    if st.button("清除聊天记录"):
        st.session_state.messages = []
        st.rerun()

# --- 聊天界面 ---
# 显示历史消息
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 接收新输入
if prompt := st.chat_input("请就文献内容进行提问，或随意聊聊..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # AI的回答区
    with st.chat_message("assistant"):
        # 【关键优化】
        # 1. 创建一个独立的、用于显示最终答案的占位符。
        answer_placeholder = st.empty()

        # 2. 使用 st.status 来独立显示“思考过程”。
        with st.status("AI 正在思考...", expanded=True) as status:
            full_response = ""

            try:
                # 3. 调用流式回答接口
                stream = qa_chain.stream_answer(prompt)

                for chunk in stream:
                    # 4. 判断流输出的类型
                    if isinstance(chunk, dict) and chunk.get("type") == "status":
                        # 如果是状态信息，更新 status 组件的标签
                        status.update(label=chunk["message"])
                    else:
                        # 如果是答案文本，累加并更新独立的答案占位符
                        full_response += chunk
                        answer_placeholder.markdown(full_response + "▌")

                # 5. 流结束后，更新最终状态并折叠 status 组件
                status.update(label="回答生成完毕！", state="complete", expanded=False)

            except Exception as e:
                status.update(label="处理时出现错误", state="error", expanded=False)
                st.error(f"抱歉，处理您的问题时出现错误: {e}")
                full_response = f"错误: {e}"

        # 6. 在所有操作完成后，最终更新一次答案占位符，移除光标
        answer_placeholder.markdown(full_response)

    # 7. 将完整的、干净的最终答案存入聊天记录
    st.session_state.messages.append({"role": "assistant", "content": full_response})