# rag_system/agent/tools/paper_finder_tool.py

from langchain.tools import tool
from langchain_community.utilities import SQLDatabase
from pydantic import BaseModel, Field
from typing import Optional, List

from rag_system.config import settings

# 初始化数据库连接，这个工具将独占使用它
db = SQLDatabase.from_uri(f"sqlite:///{settings.SQL_DB_PATH}")


# --- 为新工具定义一个结构化的输入模型 ---
class PaperFinderInput(BaseModel):
    material_name_like: Optional[str] = Field(None, description="要搜索的材料名称的关键词，例如 'TFN' 或 'PVDF'。")
    min_year: Optional[int] = Field(None, description="发表年份的下限（包含），例如 2022。")
    max_contact_angle: Optional[float] = Field(None, description="水接触角（contact_angle）的上限。")
    solvent_name: Optional[str] = Field(None, description="使用的溶剂名称，例如 'NMP'。")
    limit: int = Field(default=10, description="返回结果的最大数量。")


@tool(args_schema=PaperFinderInput)
def paper_finder_tool(material_name_like: Optional[str] = None,
                      min_year: Optional[int] = None,
                      max_contact_angle: Optional[float] = None,
                      solvent_name: Optional[str] = None,
                      limit: int = 10) -> str:
    """
    一个高级论文检索工具。它可以根据材料名称、发表年份、水接触角和溶剂等多个条件，
    在数据库中进行复杂的、跨表的关联查询，并返回符合条件的论文列表。
    """
    print("--- [Tool: paper_finder_tool] 启动高级论文检索 ---")

    # --- 由我们（开发者）来编写100%正确的SQL查询 ---
    base_query = """
                 SELECT DISTINCT p.title, \
                                 p.year, \
                                 m.material_name, \
                                 perf.contact_angle, \
                                 s.name as solvent_name
                 FROM Papers p \
                          LEFT JOIN \
                      Materials m ON p.id = m.paper_id \
                          LEFT JOIN \
                      Performances perf ON m.id = perf.material_id \
                          LEFT JOIN \
                      Solvents s ON m.id = s.material_id \
                 """

    conditions = []
    params = {}

    if material_name_like:
        conditions.append("m.material_name LIKE :material_name_like")
        params['material_name_like'] = f'%{material_name_like}%'

    if min_year:
        conditions.append("p.year > :min_year")
        params['min_year'] = min_year

    if max_contact_angle:
        conditions.append("perf.contact_angle < :max_contact_angle")
        params['max_contact_angle'] = max_contact_angle

    if solvent_name:
        conditions.append("s.name = :solvent_name")
        params['solvent_name'] = solvent_name

    if conditions:
        base_query += " WHERE " + " AND ".join(conditions)

    base_query += " LIMIT :limit"
    params['limit'] = limit

    try:
        result = db.run(base_query, parameters=params)
        return str(result) if result else "查询成功，但未找到符合条件的记录。"
    except Exception as e:
        return f"执行高级论文检索时出现错误: {e}"