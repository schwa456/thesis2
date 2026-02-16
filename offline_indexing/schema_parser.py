import sqlite3
import os
from typing import Dict, List, Any

class SQLiteSchemaParser:
    """
    SQLite 데이터베이스 파일에서 테이블, 컬럼, Primary Key, Foreign Key 정보를
    추출하여 그래프 생성(GAT) 및 LLM Verbalization을 위한 구조화된 딕셔너리로 반환합니다.
    """
    def __init__(self, db_path: str):
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"Database file not found at: {db_path}")
        self.db_path = db_path
    
    def _get_tables(self, cursor: sqlite3.Cursor) -> List[str]:
        """데이터베이스 내의 모든 테이블 이름을 추출합니다."""
        cursor.execute("SELECT name from sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall() if row[0] != 'sqlite_sequence']
        return tables
    
    def parse_schema(self) -> Dict[str, Any]:
        """
        전체 스키마 정보를 파싱하여 Dictionary 형태로 반환합니다.
        반환 구조는 Node(Table, Column)와 Edge(belongs_to, PK-FK) 생성에 최적화되어 있습니다.
        """
        schema_info = {
            "tables": [],
            "columns": {},
            "primary_keys": {},
            "foreign_keys": []
        }

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            tables = self._get_tables(cursor)
            schema_info["tables"] = tables

            for table in tables:
                # 1. 컬럼 및 PK 정보 추출 (PRAGMA table_info 사용)
                cursor.execute(f"PRAGMA table_info('{table}');")
                columns_info = cursor.fetchall()

                col_list = []
                pk_list = []
                for col in columns_info:
                    col_id, col_name, col_type, notnull, default_val, is_pk = col
                    col_list.append({"name": col_name, "type": col_type.lower()})
                    if is_pk:
                        pk_list.append(col_name)
                
                schema_info["columns"][table] = col_list
                schema_info["primary_keys"][table] = pk_list

                # 2. FK 정보 추출 (PRAGMA foreign_key_list 사용)
                cursor.execute(f"PRAGMA foreign_key_list('{table}');")
                fks_info = cursor.fetchall()

                for fk in fks_info:
                    fk_id, seq, to_table, from_col, to_col, on_update, on_delete, match = fk
                    schema_info["foreign_keys"].append({
                        "from_table": table,
                        "from_column": from_col,
                        "to_table": to_table,
                        "to_column": to_col
                    })
        
        except Exception as e:
            print(f"Error parsing schema for {self.db_path}: {e}")
        
        finally:
            cursor.close()
            conn.close()
        
        return schema_info