# test_rag_loop.py

import asyncio
from rag_system.planner.planner import Planner


async def test_planner_raw_output():
    """
    这个测试函数将直接调用Planner，并打印出LLM返回的原始文本，
    以便我们诊断为什么计划会生成失败。
    """
    print("--- 初始化Planner... ---")
    try:
        planner = Planner()
    except Exception as e:
        print(f"❌ 初始化Planner时发生错误: {e}")
        return

    query = "什么是PVDF"
    print(f"\n--- 准备为以下问题生成计划: '{query}' ---")

    # --- 关键诊断步骤 ---
    # 我们将直接调用链的前半部分（prompt | llm），来捕获未经解析的原始输出
    # 这能让我们看到LLM到底返回了什么
    print("\n--- 1. 获取LLM的原始输出 (Raw Output) ---")
    try:
        # 构建部分链：仅包含提示和LLM
        partial_chain = planner.prompt_template | planner.llm
        raw_output = await partial_chain.ainvoke({"user_goal": query})
        print("✅ 成功获取到LLM的原始响应:")
        print("=" * 50)
        print(raw_output)
        print("=" * 50)

    except Exception as e:
        print(f"❌ 在调用LLM获取原始输出时发生严重错误: {e}")
        print(
            "   这通常意味着无法连接到Ollama服务或模型未加载。请检查Ollama是否正在运行，以及模型'qwen3-tuned:latest'是否已下载。")
        return

    # --- 正常的完整链调用，用于对比 ---
    print("\n--- 2. 尝试完整的计划生成 (Full Planning) ---")
    try:
        # 为了方便，我们这里用一个简单的状态对象
        from rag_system.state import AgentState
        agent_state = AgentState(initial_query=query)

        final_state = planner.generate_plan(agent_state)

        if final_state.plan:
            print("✅ 成功生成并解析了计划！")
            # Pydantic模型可以直接导出为字典
            print(final_state.plan.model_dump_json(indent=2))
        else:
            print("❌ 计划生成失败，但未抛出异常。这通常发生在解析器多次重试后仍然失败。")

    except Exception as e:
        print(f"❌ 在调用完整计划生成链时发生错误: {e}")


if __name__ == "__main__":
    # 使用asyncio.run来执行异步函数
    asyncio.run(test_planner_raw_output())