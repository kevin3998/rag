# rag_system/agent/tools/paper_finder_tool.py

import sqlite3
from typing import List, Optional
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from rag_system.config import settings


# --- 为工具定义一个与您原始代码功能匹配的、完整的输入模型 ---
class PaperFinderInput(BaseModel):
    material_name_like: Optional[str] = Field(None, description="要搜索的材料名称的关键词，例如 'TFN' 或 'PVDF'。")
    min_year: Optional[int] = Field(None, description="发表年份的下限（包含），例如 2022。")
    max_contact_angle: Optional[float] = Field(None, description="水接触角（contact_angle）的上限。")
    solvent_name: Optional[str] = Field(None, description="使用的溶剂名称，例如 'NMP'。")


@tool(args_schema=PaperFinderInput)
def paper_finder_tool(
        material_name_like: Optional[str] = None,
        min_year: Optional[int] = None,
        max_contact_angle: Optional[float] = None,
        solvent_name: Optional[str] = None
) -> List[str]:
    """
    一个高级论文检索工具。它可以根据材料名称、发表年份、水接触角和溶剂等多个条件，
    在数据库中进行复杂的、跨表的关联查询，并返回符合条件的论文标题列表。
    """
    print("--- [Tool: paper_finder_tool] 启动高级论文检索 ---")
    conn = sqlite3.connect(settings.SQLITE_DB_PATH)
    cursor = conn.cursor()

    # ================== [ 关 键 修 复 ] ==================
    # 1. 完整地恢复了您原始代码中正确的、跨四张表的JOIN查询逻辑。
    # 2. 使用了我们通过诊断脚本确认的、100%正确的列名（例如 p.year）。
    base_query = """
        SELECT DISTINCT p.title
        FROM Papers p
        LEFT JOIN Materials m ON p.id = m.paper_id
        LEFT JOIN Performances perf ON m.id = perf.material_id
        LEFT JOIN Solvents s ON m.id = s.material_id
    """
    # =====================================================

    conditions = []
    params = []

    if material_name_like:
        conditions.append("m.material_name LIKE ?")
        params.append(f'%{material_name_like}%')

    if min_year:
        # 使用正确的列名 p.year
        conditions.append("p.year >= ?")
        params.append(min_year)

    if max_contact_angle:
        conditions.append("perf.contact_angle < ?")
        params.append(max_contact_angle)

    if solvent_name:
        conditions.append("s.name = ?")
        params.append(solvent_name)

    if conditions:
        base_query += " WHERE " + " AND ".join(conditions)

    try:
        cursor.execute(base_query, tuple(params))
        results = cursor.fetchall()

        if not results:
            print("    - [Tool Log] 查询成功，但未找到符合条件的记录。")
            return []

        paper_titles = [row[0] for row in results]
        print(f"    - [Tool Log] 成功找到 {len(paper_titles)} 篇论文。")
        return paper_titles

    except sqlite3.Error as e:
        print(f"    - [Tool Error] 数据库查询失败: {e}")
        return []
    finally:
        if conn:
            conn.close()
