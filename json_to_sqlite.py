import json
import sqlite3
from pathlib import Path
from tqdm import tqdm
import logging

# --- 配置 ---
# 输入的JSON文件路径
SOURCE_JSON_PATH = "data/processed_text/structured_info.json"
# 输出的SQLite数据库路径
DB_PATH = "data/database/literature.db"

# 日志配置
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# --- 数据库操作函数 ---

def create_connection(db_file):
    """ 创建一个数据库连接到SQLite数据库 """
    conn = None
    try:
        # 确保数据库文件的父目录存在
        Path(db_file).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(db_file)
        logging.info(f"成功连接到SQLite数据库: {db_file}")
    except sqlite3.Error as e:
        logging.error(f"连接数据库时发生错误: {e}")
    return conn


def create_tables(conn):
    """ 根据我们的设计创建数据库表 """
    # SQL建表语句
    sql_create_papers_table = """
                              CREATE TABLE IF NOT EXISTS papers \
                              ( \
                                  id \
                                  INTEGER \
                                  PRIMARY \
                                  KEY \
                                  AUTOINCREMENT, \
                                  doi \
                                  TEXT \
                                  UNIQUE \
                                  NOT \
                                  NULL, \
                                  title \
                                  TEXT, \
                                  year \
                                  INTEGER, \
                                  journal \
                                  TEXT, \
                                  abstract \
                                  TEXT, \
                                  local_path \
                                  TEXT
                              ); \
                              """
    sql_create_authors_table = """
                               CREATE TABLE IF NOT EXISTS authors \
                               ( \
                                   id \
                                   INTEGER \
                                   PRIMARY \
                                   KEY \
                                   AUTOINCREMENT, \
                                   name \
                                   TEXT \
                                   UNIQUE \
                                   NOT \
                                   NULL
                               ); \
                               """
    sql_create_paper_authors_table = """
                                     CREATE TABLE IF NOT EXISTS paper_authors \
                                     ( \
                                         paper_id \
                                         INTEGER, \
                                         author_id \
                                         INTEGER, \
                                         PRIMARY \
                                         KEY \
                                     ( \
                                         paper_id, \
                                         author_id \
                                     ),
                                         FOREIGN KEY \
                                     ( \
                                         paper_id \
                                     ) REFERENCES papers \
                                     ( \
                                         id \
                                     ) ON DELETE CASCADE,
                                         FOREIGN KEY \
                                     ( \
                                         author_id \
                                     ) REFERENCES authors \
                                     ( \
                                         id \
                                     ) \
                                       ON DELETE CASCADE
                                         ); \
                                     """
    sql_create_keywords_table = """
                                CREATE TABLE IF NOT EXISTS keywords \
                                ( \
                                    id \
                                    INTEGER \
                                    PRIMARY \
                                    KEY \
                                    AUTOINCREMENT, \
                                    keyword \
                                    TEXT \
                                    UNIQUE \
                                    NOT \
                                    NULL
                                ); \
                                """
    sql_create_paper_keywords_table = """
                                      CREATE TABLE IF NOT EXISTS paper_keywords \
                                      ( \
                                          paper_id \
                                          INTEGER, \
                                          keyword_id \
                                          INTEGER, \
                                          PRIMARY \
                                          KEY \
                                      ( \
                                          paper_id, \
                                          keyword_id \
                                      ),
                                          FOREIGN KEY \
                                      ( \
                                          paper_id \
                                      ) REFERENCES papers \
                                      ( \
                                          id \
                                      ) ON DELETE CASCADE,
                                          FOREIGN KEY \
                                      ( \
                                          keyword_id \
                                      ) REFERENCES keywords \
                                      ( \
                                          id \
                                      ) \
                                        ON DELETE CASCADE
                                          ); \
                                      """
    try:
        cursor = conn.cursor()
        logging.info("正在创建数据库表...")
        cursor.execute(sql_create_papers_table)
        cursor.execute(sql_create_authors_table)
        cursor.execute(sql_create_paper_authors_table)
        cursor.execute(sql_create_keywords_table)
        cursor.execute(sql_create_paper_keywords_table)
        conn.commit()
        logging.info("数据库表创建成功或已存在。")
    except sqlite3.Error as e:
        logging.error(f"创建表时发生错误: {e}")


def insert_data(conn, papers_data):
    """ 将从JSON读取的数据插入到数据库中 """
    cursor = conn.cursor()

    # 使用tqdm来显示处理进度
    for paper in tqdm(papers_data, desc="正在将数据存入数据库"):
        try:
            # --- 插入论文核心信息 ---
            # 首先检查论文是否已存在（基于DOI）
            cursor.execute("SELECT id FROM papers WHERE doi = ?", (paper.get("doi"),))
            result = cursor.fetchone()

            if result:
                logging.warning(f"DOI为 {paper.get('doi')} 的论文已存在，跳过。")
                continue

            sql_paper = '''INSERT INTO papers(doi, title, year, journal, abstract, local_path)
                           VALUES (?, ?, ?, ?, ?, ?)'''
            paper_values = (
                paper.get("doi"),
                paper.get("retrieved_title"),
                paper.get("retrieved_year"),
                paper.get("retrieved_journal"),
                paper.get("extracted_abstract_cleaned"),
                paper.get("local_path")
            )
            cursor.execute(sql_paper, paper_values)
            paper_id = cursor.lastrowid  # 获取刚刚插入的论文的ID

            # --- 插入作者信息并建立关联 ---
            author_names = paper.get("retrieved_authors", [])
            for name in author_names:
                # 插入作者，如果已存在则忽略
                cursor.execute("INSERT OR IGNORE INTO authors(name) VALUES(?)", (name,))
                # 获取作者ID
                cursor.execute("SELECT id FROM authors WHERE name = ?", (name,))
                author_id = cursor.fetchone()[0]
                # 插入到关联表
                cursor.execute("INSERT OR IGNORE INTO paper_authors(paper_id, author_id) VALUES(?,?)",
                               (paper_id, author_id))

            # --- 插入关键词信息并建立关联 ---
            keywords = paper.get("extracted_keywords", [])
            for keyword in keywords:
                # 插入关键词，如果已存在则忽略
                cursor.execute("INSERT OR IGNORE INTO keywords(keyword) VALUES(?)", (keyword,))
                # 获取关键词ID
                cursor.execute("SELECT id FROM keywords WHERE keyword = ?", (keyword,))
                keyword_id = cursor.fetchone()[0]
                # 插入到关联表
                cursor.execute("INSERT OR IGNORE INTO paper_keywords(paper_id, keyword_id) VALUES(?,?)",
                               (paper_id, keyword_id))

        except sqlite3.IntegrityError as e:
            logging.error(f"插入数据时发生完整性错误 (可能是重复的唯一键): {e} - 论文信息: {paper.get('doi')}")
        except Exception as e:
            logging.error(f"处理论文 {paper.get('doi')} 时发生未知错误: {e}")

    # 所有数据处理完毕后，提交事务
    conn.commit()
    logging.info("数据插入完成，事务已提交。")


# --- 主执行函数 ---

def main():
    """ 主函数，协调整个流程 """
    # 1. 读取JSON文件
    try:
        with open(SOURCE_JSON_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logging.info(f"成功从 {SOURCE_JSON_PATH} 加载了 {len(data)} 条记录。")
    except FileNotFoundError:
        logging.error(f"错误: JSON源文件未找到 -> {SOURCE_JSON_PATH}")
        return
    except json.JSONDecodeError:
        logging.error(f"错误: JSON文件格式无效 -> {SOURCE_JSON_PATH}")
        return

    # 2. 创建数据库连接和表结构
    conn = create_connection(DB_PATH)

    if conn is not None:
        create_tables(conn)

        # 3. 插入数据
        insert_data(conn, data)

        # 4. 关闭连接
        conn.close()
        logging.info("数据库连接已关闭。")
    else:
        logging.error("无法创建数据库连接，程序退出。")


if __name__ == '__main__':
    main()