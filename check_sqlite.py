# check_db.py
# 这是一个独立的诊断脚本，用于检查您的SQLite数据库。

import sqlite3
import os
from pathlib import Path

# --- 请在这里配置您的数据库路径 ---
# 确保这个路径与您 settings.py 中的完全一致
try:
    from rag_system.config import settings
    DB_PATH = settings.SQLITE_DB_PATH
except (ImportError, AttributeError):
    print("无法从settings导入配置，将使用默认路径。")
    # 如果无法导入，请手动设置路径
    PROJECT_ROOT = Path(__file__).parent
    DB_PATH = PROJECT_ROOT / "data" / "database" / "literature_materials.db"

def check_database():
    """连接数据库并执行诊断查询。"""
    if not os.path.exists(DB_PATH):
        print(f"❌ 错误: 数据库文件不存在于路径: {DB_PATH}")
        return

    print(f"✅ 成功找到数据库文件: {DB_PATH}")
    conn = None
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        # 诊断1: 获取 'papers' 表的所有列名
        print("\n--- 诊断1: 'papers' 表的结构 ---")
        cursor.execute("PRAGMA table_info(papers);")
        columns = cursor.fetchall()
        if not columns:
            print("❌ 错误: 'papers' 表不存在或为空。")
        else:
            print("'papers' 表包含以下列名:")
            for col in columns:
                # col 的结构是 (id, name, type, notnull, default_value, pk)
                print(f"  - {col[1]} (类型: {col[2]})")

        # 诊断2: 查询前5条记录，看看数据长什么样
        print("\n--- 诊断2: 'papers' 表的前5条数据示例 ---")
        # 我们假设列名是 'title' 和 'year'，如果上一步诊断出不同，请在这里修改
        # 使用 try-except 来捕获可能的列名错误
        try:
            cursor.execute("SELECT title, year FROM papers LIMIT 5;")
            rows = cursor.fetchall()
            if not rows:
                print("'papers' 表中没有任何数据。")
            else:
                for row in rows:
                    print(f"  - Title: {row[0]}, Year: {row[1]}")
        except sqlite3.OperationalError as e:
            print(f"❌ 查询数据时出错: {e}")
            print("   这通常意味着 'title' 或 'year' 不是正确的列名。请参考上面的列名列表。")

    except sqlite3.Error as e:
        print(f"❌ 连接或查询数据库时发生错误: {e}")
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    check_database()
