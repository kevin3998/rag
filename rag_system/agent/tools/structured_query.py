from langchain.tools import tool
from langchain_community.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from rag_system.config import settings
from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate # <-- 新增导入

# --- 【新增】为Text-to-SQL任务设计的、非常严格的提示词模板 ---
SQL_PROMPT_TEMPLATE = """
你是一个SQLite专家。给定一个输入问题，首先创建一个语法正确的SQLite查询语句来运行，然后查看查询结果并返回答案。
除非用户在问题中指定了要获取的示例数量，否则只能查询最多5个结果。
你只能使用以下表格中存在的列来查询。你必须检查你的查询语句是否只包含这些表中的列名，并且没有错误。

【重要】你的回答中只能包含SQL查询语句，绝对不能包含任何其他文字、解释、思考过程、或者"<think>"标签。只输出SQL代码本身。

使用以下格式：

Question: "用户在这里提问"
SQLQuery: "在这里生成SQL查询"

这里有一些例子：

Question: 列出所有材料
SQLQuery: SELECT DISTINCT material_name FROM Materials LIMIT 5;

Question: 2020年后发表的论文有多少篇？
SQLQuery: SELECT count(*) FROM Papers WHERE year > 2020;

现在，轮到你了：
Question: {input}
SQLQuery:
"""

# 将模板字符串转换为LangChain的PromptTemplate对象
SQL_PROMPT = PromptTemplate.from_template(SQL_PROMPT_TEMPLATE)

# --- 后续代码修改 ---

db = SQLDatabase.from_uri(f"sqlite:///{settings.SQL_DB_PATH}")
llm = ChatOllama(model=settings.LOCAL_LLM_MODEL_NAME)

# 【修改】在创建db_chain时，注入我们自定义的提示词
db_chain = SQLDatabaseChain.from_llm(
    llm=llm,
    db=db,
    prompt=SQL_PROMPT, # <-- 注入自定义提示词
    verbose=True,
    use_query_checker=True
)

@tool
def structured_data_query_tool(query: str) -> str:
    """
    当需要查找精确的数据点、进行筛选、排序或聚合操作时，使用此工具。
    【重要】你的输入应该是一个完整的、用自然语言描述的问题，而不是简单的关键词或键值对。
    例如，你应该问“找出所有使用NMP作为溶剂的材料”，而不是“solvent:NMP”。
    """
    try:
        # 直接运行SQLDatabaseChain
        result = db_chain.invoke(query)
        # result 可能是一个字典，我们需要提取最终答案
        return result.get("result", "查询成功，但未能提取明确结果。")
    except Exception as e:
        return f"执行结构化数据查询时出现错误: {e}"