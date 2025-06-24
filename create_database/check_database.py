import sqlite3
import pandas as pd  # pandas库能极大地美化表格输出

# --- 配置 ---
DB_PATH = "../data/database/literature_materials.db"


def run_query(conn, query):
    """ 执行一个SQL查询并使用pandas打印结果 """
    try:
        # 使用pandas的read_sql_query函数，可以直接将查询结果转换为一个漂亮的DataFrame
        df = pd.read_sql_query(query, conn)
        if df.empty:
            print("查询结果为空。")
        else:
            print(df.to_string())  # .to_string()可以确保所有列和行都被完整显示
    except Exception as e:
        print(f"执行查询时出错: {e}")


def main():
    """ 连接到数据库并执行一系列检查查询 """
    try:
        conn = sqlite3.connect(DB_PATH)
        print(f"成功连接到数据库: {DB_PATH}\n")

        # --- 查询1: 检查每个表有多少条记录 ---
        print("--- 1. 各表记录总数检查 ---")
        tables_to_check = ["Papers", "Materials", "BasePolymers", "Solvents", "Performances", "Applications"]
        for table in tables_to_check:
            cursor = conn.cursor()
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            count = cursor.fetchone()[0]
            print(f"表 '{table}': {count} 条记录")

        print("\n" + "=" * 40 + "\n")

        # --- 查询2: 查看前5篇论文的基本信息 ---
        print("--- 2. 查看前5篇论文信息 ---")
        query_papers = "SELECT doi, title, year, journal FROM Papers LIMIT 5;"
        run_query(conn, query_papers)

        print("\n" + "=" * 40 + "\n")

        # --- 查询3: 复杂关联查询 - 查找一篇特定论文下的所有材料及其关键性能 ---
        print("--- 3. 复杂关联查询示例 ---")
        # 您可以从上面的查询结果中，任选一个DOI填入下方
        target_doi = "local.1747324669.1125507.e11dae19"  # 这是一个示例DOI，请替换

        query_join = f"""
        SELECT
            p.title,
            m.material_name,
            perf.water_permeability,
            perf.nacl_rejection,
            perf.tensile_strength
        FROM Papers p
        JOIN Materials m ON p.id = m.paper_id
        LEFT JOIN Performances perf ON m.id = perf.material_id
        WHERE p.doi = '{target_doi}';
        """
        print(f"正在查询DOI为 '{target_doi}' 的论文下的所有材料性能...")
        run_query(conn, query_join)

    except sqlite3.Error as e:
        print(f"数据库错误: {e}")
    finally:
        if conn:
            conn.close()
            print(f"\n数据库连接已关闭。")


if __name__ == '__main__':
    # 确保您已安装pandas: pip install pandas
    main()