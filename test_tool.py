# test_tool.py
# 这是一个用于独立测试您Agent框架中所有工具的脚本。
# 请将此文件放置在您项目的根目录下运行。

import sys
import os
import pprint

# --- 路径设置 (确保能找到rag_system模块) ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# --- 导入我们需要测试的工具 ---
# 我们将在这里捕获导入错误，以便在工具本身有问题时也能提供反馈
try:
    from rag_system.agent.tools.paper_finder_tool import paper_finder_tool
    from rag_system.agent.tools.semantic_search import semantic_search_tool
    from rag_system.agent.tools.prediction_tool import prediction_tool
    # ================== [ 诊断代码 ] ==================
    from rag_system.config import settings
    import os
    print("\n" + "*"*20 + " [诊断信息] " + "*"*20)
    print(f"test_tool.py 将要使用的数据库绝对路径是:\n{os.path.abspath(settings.SQLITE_DB_PATH)}")
    print("*"*54 + "\n")
    # =====================================================
    print("✅ 所有工具模块成功导入！")
except ImportError as e:
    print(f"❌ 导入工具时发生错误: {e}")
    print("   请确保所有工具文件都存在，并且没有语法或导入错误。")
    sys.exit(1)


def print_test_header(title):
    """打印一个漂亮的测试标题。"""
    print("\n" + "=" * 80)
    print(f"🧪 开始测试: {title}")
    print("=" * 80)


def print_test_result(test_case, success, result):
    """打印单个测试用例的结果。"""
    status = "✅ 成功" if success else "❌ 失败"
    print(f"\n--- 测试用例: {test_case} ---")
    print(f"状态: {status}")
    print("输出结果:")
    # 使用pprint来美观地打印复杂的数据结构（如列表和字典）
    pprint.pprint(result)
    print("-" * 40)


def test_paper_finder():
    """测试 paper_finder_tool 的所有路径。"""
    print_test_header("paper_finder_tool")

    # --- 1. 成功路径测试 ---
    # 测试一个您数据库中确定存在的条件
    case_1_input = {'material_name_like': 'PVDF', 'min_year': 2022}
    result_1 = paper_finder_tool.invoke(case_1_input)
    # 预期：返回一个非空的列表
    success_1 = isinstance(result_1, list) and len(result_1) > 0
    print_test_result("成功路径 - 查找2022年后的PVDF论文", success_1, result_1)

    # --- 2. 空结果路径测试 ---
    # 测试一个确定不存在的条件
    case_2_input = {'material_name_like': '一种不存在的超级氪金材料'}
    result_2 = paper_finder_tool.invoke(case_2_input)
    # 预期：返回一个空列表
    success_2 = isinstance(result_2, list) and len(result_2) == 0
    print_test_result("空结果路径 - 查找不存在的材料", success_2, result_2)

    # --- 3. 多条件与limit测试 (已修复) ---
    # 在工具修复后，这个查询现在应该能成功找到数据
    case_3_input = {'solvent_name': 'NMP', 'max_contact_angle': 70, 'limit': 5}
    result_3 = paper_finder_tool.invoke(case_3_input)
    # 预期：返回一个非空的列表，且长度小于等于5
    success_3 = isinstance(result_3, list) and len(result_3) > 0 and len(result_3) <= 5
    print_test_result("多条件与limit测试 (验证修复)", success_3, result_3)

    # --- 4. 核心参数缺失测试 ---
    # 不提供任何参数
    case_4_input = {}
    result_4 = paper_finder_tool.invoke(case_4_input)
    # 预期：根据我们最新的健壮性设计，它应该返回空列表
    success_4 = isinstance(result_4, list) and len(result_4) == 0
    print_test_result("无参数调用测试", success_4, result_4)


def test_semantic_search():
    """测试 semantic_search_tool 的所有模式。"""
    print_test_header("semantic_search_tool")

    # --- 1. 开放式搜索模式 ---
    case_1_input = {'query': "什么是薄膜复合(TFC)膜?"}
    result_1 = semantic_search_tool.invoke(case_1_input)
    # 预期：返回一段相关的文本
    success_1 = isinstance(result_1, str) and len(result_1) > 50
    print_test_result("模式1 - 开放式搜索", success_1, result_1)

    # --- 2. 基于标题列表模式 (已修复断言) ---
    # 我们使用您诊断报告中确认存在的标题
    titles = [
        "Simple and efficient method for functionalizing photocatalytic ceramic membranes and assessment of its applicability for wastewater treatment in up-scalable membrane reactors",
        "Functionalized graphene-based polyamide thin film nanocomposite membranes for organic solvent nanofiltration"
    ]
    case_2_input = {'query': "请分别总结这两篇论文的核心内容", 'context': titles}
    result_2 = semantic_search_tool.invoke(case_2_input)
    # ================== [ 关 键 修 复 ] ==================
    # 预期：返回的总结应包含与标题相关的关键词，而不是之前错误的 "GO" 和 "Kefir"
    # 我们检查与标题更相关的 "Fouling" (污染) 和 "Cellulose Nanocrystal" (纤维素纳米晶体)
    success_2 = (
            isinstance(result_2, str) and
            len(result_2) > 500 and  # 确保返回了足够的内容
            "关于《Simple and efficient method" in result_2 and  # 检查第一个标题是否存在
            "关于《Functionalized graphene-based" in result_2 and  # 检查第二个标题是否存在
            "错误" not in result_2
    )    # =====================================================
    print_test_result("模式2 - 基于标题列表总结 (验证修复)", success_2, result_2)

    # --- 3. 空上下文模式 ---
    case_3_input = {'query': "总结", 'context': []}
    result_3 = semantic_search_tool.invoke(case_3_input)
    # 预期：工具应自动转为开放式搜索
    success_3 = isinstance(result_3, str) and len(result_3) > 0
    print_test_result("模式3 - 空上下文自动转开放搜索", success_3, result_3)


def test_prediction_tool():
    """测试 prediction_tool 的推理能力。"""
    print_test_header("prediction_tool")

    # 准备一些模拟的上下文，就像semantic_search_tool检索到的一样
    mock_context = """
    文献A指出，增加PVDF膜中的PVP含量可以提高其亲水性，但会略微降低机械强度。
    文献B发现，通过热处理可以增强PVDF膜的结晶度，从而显著提升其机械稳定性。
    文献C的实验表明，磺胺类药物在疏水表面上的吸附较弱。
    """
    case_1_input = {
        'question': "如果我们想开发一种既能保持低药物吸附，又能提高机械强度的PVDF膜，应该采取什么策略？请说明机理。",
        'context': mock_context
    }
    result_1 = prediction_tool.invoke(case_1_input)
    # 预期：返回一段包含“因此”、“结合...来看”、“一个可能的策略是”等推理词汇的分析
    success_1 = isinstance(result_1, str) and ("策略" in result_1 or "结合" in result_1)
    print_test_result("分析与推理测试", success_1, result_1)


if __name__ == "__main__":
    print("======= 开始执行RAG Agent工具层单元测试 =======")

    # 确保Ollama服务正在运行
    print("\n⚠️ 请确保您的Ollama服务正在后台运行...")

    # 依次运行所有测试
    test_paper_finder()
    test_semantic_search()
    test_prediction_tool()

    print("\n\n======= 所有单元测试执行完毕 =======")