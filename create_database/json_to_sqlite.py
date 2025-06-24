import json
import sqlite3
from pathlib import Path
from tqdm import tqdm
import logging

# --- 配置 ---
# 输入的JSON文件路径 (请确保文件名正确)
SOURCE_JSON_PATH = "../data/extracted_json/structured_info.json"
# 输出的SQLite数据库路径
DB_PATH = "../data/database/literature_materials.db"

# 日志配置
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# --- 数据库操作函数 ---

def create_connection(db_file):
    """ 创建数据库连接 """
    conn = None
    try:
        Path(db_file).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(db_file)
        # 开启外键约束支持
        conn.execute("PRAGMA foreign_keys = ON;")
        logging.info(f"成功连接到SQLite数据库: {db_file}")
    except sqlite3.Error as e:
        logging.error(f"连接数据库时发生错误: {e}")
    return conn


def create_tables(conn):
    """ 根据新的、详细的模式创建数据库表 """
    cursor = conn.cursor()
    logging.info("正在创建数据库表...")

    # 论文表 (顶层信息)
    cursor.execute("""
                   CREATE TABLE IF NOT EXISTS Papers
                   (
                       id
                       INTEGER
                       PRIMARY
                       KEY
                       AUTOINCREMENT,
                       doi
                       TEXT
                       UNIQUE
                       NOT
                       NULL,
                       title
                       TEXT,
                       journal
                       TEXT,
                       year
                       INTEGER,
                       original_filename
                       TEXT,
                       local_path
                       TEXT
                   );
                   """)

    # 材料表 (核心实体，关联到论文)
    cursor.execute("""
                   CREATE TABLE IF NOT EXISTS Materials
                   (
                       id
                       INTEGER
                       PRIMARY
                       KEY
                       AUTOINCREMENT,
                       paper_id
                       INTEGER
                       NOT
                       NULL,
                       material_name
                       TEXT
                       NOT
                       NULL,
                       fabrication_method
                       TEXT,
                       film_thickness
                       TEXT,
                       FOREIGN
                       KEY
                   (
                       paper_id
                   ) REFERENCES Papers
                   (
                       id
                   ) ON DELETE CASCADE
                       );
                   """)

    # 基底聚合物表 (与材料关联)
    cursor.execute("""
                   CREATE TABLE IF NOT EXISTS BasePolymers
                   (
                       id
                       INTEGER
                       PRIMARY
                       KEY
                       AUTOINCREMENT,
                       material_id
                       INTEGER
                       NOT
                       NULL,
                       name
                       TEXT
                       NOT
                       NULL,
                       concentration
                       TEXT,
                       FOREIGN
                       KEY
                   (
                       material_id
                   ) REFERENCES Materials
                   (
                       id
                   ) ON DELETE CASCADE
                       );
                   """)

    # 溶剂表 (与材料关联)
    cursor.execute("""
                   CREATE TABLE IF NOT EXISTS Solvents
                   (
                       id
                       INTEGER
                       PRIMARY
                       KEY
                       AUTOINCREMENT,
                       material_id
                       INTEGER
                       NOT
                       NULL,
                       name
                       TEXT
                       NOT
                       NULL,
                       FOREIGN
                       KEY
                   (
                       material_id
                   ) REFERENCES Materials
                   (
                       id
                   ) ON DELETE CASCADE
                       );
                   """)

    # 性能表 (与材料关联)
    cursor.execute("""
                   CREATE TABLE IF NOT EXISTS Performances
                   (
                       id
                       INTEGER
                       PRIMARY
                       KEY
                       AUTOINCREMENT,
                       material_id
                       INTEGER
                       UNIQUE
                       NOT
                       NULL, -- 一个材料对应一组性能数据
                       porosity
                       TEXT,
                       contact_angle
                       TEXT,
                       water_permeability
                       TEXT,
                       nacl_rejection
                       TEXT,
                       tensile_strength
                       TEXT,
                       elongation_at_break
                       TEXT,
                       FOREIGN
                       KEY
                   (
                       material_id
                   ) REFERENCES Materials
                   (
                       id
                   ) ON DELETE CASCADE
                       );
                   """)

    # 应用表 (与材料关联)
    cursor.execute("""
                   CREATE TABLE IF NOT EXISTS Applications
                   (
                       id
                       INTEGER
                       PRIMARY
                       KEY
                       AUTOINCREMENT,
                       material_id
                       INTEGER
                       UNIQUE
                       NOT
                       NULL, -- 一个材料对应一组应用数据
                       scenario
                       TEXT,
                       achieved_performance
                       TEXT,
                       operating_temperature
                       TEXT,
                       FOREIGN
                       KEY
                   (
                       material_id
                   ) REFERENCES Materials
                   (
                       id
                   ) ON DELETE CASCADE
                       );
                   """)

    conn.commit()
    logging.info("数据库表创建成功或已存在。")


def insert_structured_data(conn, data):
    """ 将新的结构化JSON数据插入到数据库中，并正确处理多材料条目 """
    cursor = conn.cursor()

    for entry in tqdm(data, desc="正在将数据存入数据库"):
        paper_meta = entry.get("meta_source_paper", {})
        doi = paper_meta.get("doi")

        if not doi:
            logging.warning(f"发现一条记录缺少DOI，跳过。文件名: {paper_meta.get('original_filename')}")
            continue

        try:
            # --- 1. 插入或获取论文ID ---
            cursor.execute("SELECT id FROM Papers WHERE doi = ?", (doi,))
            result = cursor.fetchone()
            if result:
                paper_id = result[0]
            else:
                cursor.execute(
                    "INSERT INTO Papers (doi, title, journal, year, original_filename, local_path) VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        doi, paper_meta.get("title"), paper_meta.get("journal"),
                        paper_meta.get("year"), paper_meta.get("original_filename"),
                        paper_meta.get("local_path")
                    )
                )
                paper_id = cursor.lastrowid

            # --- 2. 【逻辑修正】无论论文是否已存在，都继续处理材料信息 ---
            material_data = entry.get("extracted_material_data", {})
            material_name = material_data.get("MaterialName")

            if not material_name:
                logging.warning(f"论文 {doi} 中有一条记录缺少MaterialName，跳过此材料。")
                continue

            # 检查该材料是否已为该论文添加过
            cursor.execute("SELECT id FROM Materials WHERE paper_id = ? AND material_name = ?",
                           (paper_id, material_name))
            if cursor.fetchone():
                logging.info(f"材料 '{material_name}' 已存在于论文 '{doi}' 中，跳过。")
                continue  # 跳到下一个JSON条目

            logging.info(f"为论文 '{doi}' 添加新材料 '{material_name}'...")

            details = material_data.get("Details", {})
            fab = details.get("Fabrication", {})

            cursor.execute(
                "INSERT INTO Materials (paper_id, material_name, fabrication_method, film_thickness) VALUES (?, ?, ?, ?)",
                (paper_id, material_name, fab.get("FabricationMethod"), fab.get("FilmThicknessText"))
            )
            material_id = cursor.lastrowid

            # --- 3. 插入设计细节 (聚合物, 溶剂等) ---
            design = details.get("Design", {})
            for polymer in design.get("BasePolymer", []):
                cursor.execute(
                    "INSERT INTO BasePolymers (material_id, name, concentration) VALUES (?, ?, ?)",
                    (material_id, polymer.get("Name"), polymer.get("ConcentrationText"))
                )
            for solvent in design.get("Solvents", []):
                cursor.execute(
                    "INSERT INTO Solvents (material_id, name) VALUES (?, ?)",
                    (material_id, solvent.get("Name"))
                )

            # --- 4. 插入性能数据 ---
            perf = details.get("Performance", {})
            struct_props = perf.get("StructuralPhysicalProperties", {})
            liq_props = perf.get("LiquidTransportProperties", {})
            mech_props = perf.get("MechanicalProperties", {})
            cursor.execute(
                """INSERT INTO Performances (material_id, porosity, contact_angle, water_permeability, nacl_rejection,
                                             tensile_strength, elongation_at_break)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    material_id, struct_props.get("Porosity"), struct_props.get("ContactAngleText"),
                    liq_props.get("WaterPermeability"), liq_props.get("Rejections", {}).get("NaCl"),
                    mech_props.get("TensileStrength"), mech_props.get("ElongationAtBreak")
                )
            )

            # --- 5. 插入应用数据 ---
            app = details.get("Application", {})
            cursor.execute(
                "INSERT INTO Applications (material_id, scenario, achieved_performance, operating_temperature) VALUES (?, ?, ?, ?)",
                (
                    material_id, app.get("ApplicationScenario"),
                    app.get("AchievedPerformanceInApplication"), app.get("OperatingTemperature")
                )
            )

        except sqlite3.Error as e:
            logging.error(f"处理论文 {doi} 时发生数据库错误: {e}")
            conn.rollback()
        else:
            conn.commit()


def main():
    """ 主函数 """
    try:
        with open(SOURCE_JSON_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logging.info(f"成功从 {SOURCE_JSON_PATH} 加载了 {len(data)} 条记录。")
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logging.error(f"加载或解析JSON文件时出错: {e}")
        return

    conn = create_connection(DB_PATH)
    if conn:
        create_tables(conn)
        insert_structured_data(conn, data)
        conn.close()
        logging.info("数据库操作完成，连接已关闭。")


if __name__ == '__main__':
    main()