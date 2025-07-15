# rag_system/agent/tools/paper_finder_tool.py

import sqlite3
from typing import List, Optional
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from rag_system.config import settings


class PaperFinderInput(BaseModel):
    material_name_like: Optional[str] = Field(None, description="要搜索的材料名称的关键词。")
    min_year: Optional[int] = Field(None, description="发表年份的下限（包含）。")
    max_contact_angle: Optional[float] = Field(None, description="水接触角的上限。")
    solvent_name: Optional[str] = Field(None, description="使用的溶剂名称。")
    limit: int = Field(default=20, description="限制返回结果的最大数量，以避免上下文过长。")


def _get_db_connection():
    """私有辅助函数，用于建立并返回数据库连接。"""
    return sqlite3.connect(settings.SQLITE_DB_PATH)


@tool(args_schema=PaperFinderInput)
def paper_finder_tool(
        material_name_like: Optional[str] = None,
        min_year: Optional[int] = None,
        max_contact_angle: Optional[float] = None,
        solvent_name: Optional[str] = None,
        limit: int = 20
) -> List[str]:
    """
    一个高级论文检索工具。它可以根据多个条件进行复杂的跨表查询，并返回一个论文标题列表。
    """
    print("--- [Tool: paper_finder_tool] 启动高级论文检索 ---")

    # ================== [ 关 键 修 复 ] ==================
    # 增加一个“前置守卫”，确保核心查询参数存在。
    # .strip()可以处理空字符串的情况。
    if not material_name_like or not material_name_like.strip():
        print("    - [Tool Log] 核心参数 'material_name_like' 未提供，查询终止。")
        return []
    # =====================================================

    conn = sqlite3.connect(settings.SQLITE_DB_PATH)
    cursor = conn.cursor()

    base_query = """
        SELECT DISTINCT p.title
        FROM Papers p
        LEFT JOIN Materials m ON p.id = m.paper_id
        LEFT JOIN Performances perf ON m.id = perf.material_id
        LEFT JOIN Solvents s ON m.id = s.material_id
    """

    # 我们总是以 material_name_like 作为查询的基础
    conditions = ["m.material_name LIKE ?"]
    params = [f'%{material_name_like}%']

    if min_year is not None:
        conditions.append("p.year >= ?")
        params.append(min_year)
    if max_contact_angle is not None:
        conditions.append("perf.contact_angle < ?")
        params.append(max_contact_angle)
    if solvent_name:
        conditions.append("s.name = ?")
        params.append(solvent_name)

    base_query += " WHERE " + " AND ".join(conditions)
    base_query += " LIMIT ?"
    params.append(limit)

    try:
        cursor.execute(base_query, tuple(params))
        results = cursor.fetchall()

        if not results:
            print("    - [Tool Log] 查询成功，但未找到符合条件的记录。")
            return []

        paper_titles = [row[0] for row in results]
        print(f"    - [Tool Log] 成功找到 {len(paper_titles)} 篇论文 (受限于 limit={limit})。")
        return paper_titles

    except sqlite3.Error as e:
        print(f"    - [Tool Error] 数据库查询失败: {e}")
        return []
    finally:
        if conn:
            conn.close()

