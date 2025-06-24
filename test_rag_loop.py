import os
# 必须先设置环境变量，再导入其他模块
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com" # 如果您的网络稳定，可以注释掉这行

# 这里的导入路径可能需要根据您的项目结构调整
# 假设 test_rag_tool.py 在项目根目录
from rag_system.generation.qa_chain import AdvancedQAChain

def test_rag_path():
    """
    这个函数只用于独立测试RAG链本身的功能。
    """
    print("--- 正在独立测试RAG问答链 (AdvancedQAChain) ---")

    # 为了进行最纯粹的测试，我们直接实例化这个链
    # 它内部会加载检索器和LLM
    qa_chain = AdvancedQAChain()

    # 使用一个明确应该能找到结果的问题
    test_query = "What are the properties of PVDF membrane?"

    print(f"正在用问题 '{test_query}' 进行测试...")

    full_response = ""

    # 我们需要临时修改qa_chain，让它跳过路由，直接走RAG路径
    # 最简单的方法是直接调用内部的rag_chain
    # 注意：这里的 .rag_chain 是基于我上一条回复中最终代码的假设
    if hasattr(qa_chain, 'rag_chain'):
        for chunk in qa_chain.rag_chain.stream(test_query):
            full_response += chunk
    else:
        print("错误：在AdvancedQAChain中未找到'rag_chain'组件，请检查您的qa_chain.py代码。")
        return

    print("\n--- RAG工具独立测试完成 ---")
    if "未能找到" in full_response or not full_response.strip():
        print("❌ 测试失败：RAG链未能从数据库中检索到有效信息。")
        print("   请检查: 1. `settings.py`中的VECTOR_DB_PATH是否正确？ 2. 数据库中是否有数据？ 3. `settings.py`中的RETRIEVER_K值是否过低？")
    else:
        print("✅ 测试成功！RAG链返回了以下内容：")
        print(full_response)

if __name__ == "__main__":
    test_rag_path()