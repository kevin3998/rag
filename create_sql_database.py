# create_sql_database.py
import json
import sqlite3
import os
import logging
import re
from typing import Dict, Any, Iterator, List, Tuple, Optional
from tqdm import tqdm

# --- 日志设置 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# =================================================================================
# === 1. 参数配置区 ===
# =================================================================================
class Config:
    # --- 输入文件路径 ---
    # !! 请确保这个路径指向您最终的、干净的结构化JSON文件 !!
    JSON_INPUT_PATH = "data/extracted_json/structured_info.json"

    # --- 输出文件路径 ---
    # 将要创建的SQLite数据库文件名
    SQLITE_DB_PATH = "membrane_database.db"

    # 数据库中的表名
    TABLE_NAME = "materials"

    # 需要创建索引的关键字段
    INDEXED_COLUMNS = ["doi", "material_name"]


# =================================================================================

def flatten_json(data: Dict[str, Any], prefix: str = '') -> Iterator[Tuple[str, Any]]:
    """
    递归地将嵌套的JSON字典“压平”为单一层级的键值对。
    例如：{'a': {'b': 1}} -> 'a_b': 1
    """
    for key, value in data.items():
        new_key = f"{prefix}_{key}" if prefix else key
        # 将键名转换为适合SQL列名的格式 (小写，去除特殊字符)
        safe_key = re.sub(r'[^a-zA-Z0-9_]', '', new_key).lower()
        if isinstance(value, dict):
            yield from flatten_json(value, safe_key)
        else:
            yield safe_key, value


def extract_numeric_value(text_value: Any) -> Optional[float]:
    """
    尝试从一个字符串中提取第一个出现的数值（整数或浮点数）。
    """
    if not isinstance(text_value, str):
        return None

    # 这个正则表达式会寻找整数、浮点数、科学记数法表示的数
    match = re.search(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', text_value)
    if match:
        try:
            return float(match.group(0))
        except (ValueError, TypeError):
            return None
    return None


def create_database_and_insert_data(config: Config):
    """
    主函数：读取JSON，创建数据库和表，并插入数据。
    """
    # --- Step 1: 加载JSON数据 ---
    try:
        logging.info(f"正在从 '{config.JSON_INPUT_PATH}' 加载数据...")
        with open(config.JSON_INPUT_PATH, 'r', encoding='utf-8') as f:
            all_entries = json.load(f)
        if not all_entries or not isinstance(all_entries, list):
            logging.error("输入文件为空或格式不正确（不是一个列表）。")
            return
        logging.info(f"成功加载 {len(all_entries)} 条记录。")
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logging.error(f"加载或解析JSON文件时出错: {e}")
        return

    # --- Step 2: 收集所有可能的列名 ---
    logging.info("正在分析数据结构以确定所有可能的列名...")
    all_columns = set(["doi", "title", "journal", "year", "material_name"])

    for entry in all_entries:
        details = entry.get("extracted_material_data", {}).get("Details", {})
        for flat_key, _ in flatten_json(details):
            all_columns.add(f"{flat_key}_text")  # 存储原始文本的列
            all_columns.add(f"{flat_key}_value")  # 存储提取出的数值的列

    # 确保列名顺序一致，方便后续操作
    sorted_columns = sorted(list(all_columns))
    logging.info(f"分析完成，将创建包含 {len(sorted_columns)} 个字段的表。")

    # --- Step 3: 连接数据库并创建表 ---
    # 如果已存在同名数据库，先删除以确保从干净状态开始
    if os.path.exists(config.SQLITE_DB_PATH):
        logging.warning(f"数据库 '{config.SQLITE_DB_PATH}' 已存在，将删除并重建。")
        os.remove(config.SQLITE_DB_PATH)

    conn = sqlite3.connect(config.SQLITE_DB_PATH)
    cursor = conn.cursor()

    # 动态构建 CREATE TABLE 语句
    # 所有数值列都使用 REAL 类型，文本列使用 TEXT 类型
    column_definitions = "id INTEGER PRIMARY KEY AUTOINCREMENT"
    for col in sorted_columns:
        # 根据列名后缀判断数据类型
        col_type = "REAL" if col.endswith("_value") else "TEXT"
        column_definitions += f", `{col}` {col_type}"  # 使用反引号处理可能包含特殊字符的列名

    create_table_sql = f"CREATE TABLE IF NOT EXISTS {config.TABLE_NAME} ({column_definitions});"
    logging.info("正在创建数据库表...")
    cursor.execute(create_table_sql)

    # --- Step 4: 插入数据 ---
    logging.info("正在将数据插入数据库...")
    for entry in tqdm(all_entries, desc="Inserting data"):
        # 准备每一行的数据
        row_data = {}
        # 提取元数据
        meta = entry.get("meta_source_paper", {})
        row_data['doi'] = meta.get('doi')
        row_data['title'] = meta.get('title')
        row_data['journal'] = meta.get('journal')
        row_data['year'] = meta.get('year')

        extracted_data = entry.get("extracted_material_data", {})
        row_data['material_name'] = extracted_data.get('MaterialName')

        # 压平并填充细节数据
        details = extracted_data.get("Details", {})
        for flat_key, text_value in flatten_json(details):
            row_data[f"{flat_key}_text"] = str(text_value)
            row_data[f"{flat_key}_value"] = extract_numeric_value(text_value)

        # 构建 INSERT 语句
        columns_to_insert = [col for col in sorted_columns if col in row_data]
        values_to_insert = [row_data.get(col) for col in columns_to_insert]

        placeholders = ", ".join(["?"] * len(columns_to_insert))
        insert_sql = f"INSERT INTO {config.TABLE_NAME} (`{'`, `'.join(columns_to_insert)}`) VALUES ({placeholders});"

        cursor.execute(insert_sql, values_to_insert)

    # --- Step 5: 创建索引 ---
    logging.info("正在为关键字段创建索引以优化查询速度...")
    for col in config.INDEXED_COLUMNS:
        if col in sorted_columns:
            index_name = f"idx_{config.TABLE_NAME}_{col}"
            create_index_sql = f"CREATE INDEX {index_name} ON {config.TABLE_NAME} (`{col}`);"
            cursor.execute(create_index_sql)
            logging.info(f"为字段 '{col}' 创建索引 '{index_name}' 成功。")

    # --- Step 6: 提交并关闭 ---
    conn.commit()
    conn.close()
    logging.info(f"✅ 数据库构建完成！数据已存入 '{config.SQLITE_DB_PATH}'。")


if __name__ == "__main__":
    main_config = Config()
    create_database_and_insert_data(main_config)