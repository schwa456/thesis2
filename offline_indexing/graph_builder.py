import torch
from torch_geometric.data import HeteroData
from sentence_transformers import SentenceTransformer
from typing import Dict, Any

class SchemaGraphBuilder:
    """
    파싱된 스키마 정보와 LLM이 생성한 FK Description을 결합하여,
    PyTorch Geometric의 HeteroData(이종 그래프) 객체를 생성합니다.
    (핵심 기여: FK 관계를 독립적인 'fk_node'로 승격시키는 Line Graph Transformation 적용)
    """
    def __init__(self, plm_model_name: str = 'all-MiniLM-L6-v2'):
        print(f"Loading PLM for initial node features: {plm_model_name}...")
        # GAT에 들어가기 전, 텍스트를 초기 Dense Vector로 변환해 줄 PLM
        self.encoder = SentenceTransformer(plm_model_name)

    def build_graph(self, schema_info: Dict[str, Any], fk_descriptions: Dict[str, str]) -> HeteroData:
        data = HeteroData()

        # 1. ID 매핑 딕셔너리 초기화 (문자열 고유 ID -> 정수 인덱스)
        table_to_id = {}
        col_to_id = {}
        fk_to_id = {}

        # 노드의 초기 텍스트 피처를 담을 리스트
        table_texts = []
        col_texts = []
        fk_texts = []

        # ---------------------------------------------------------
        # Step 1: Node 생성 및 초기 텍스트 수집
        # ---------------------------------------------------------
        # Tables
        for idx, table_name in enumerate(schema_info.get("tables", [])):
            table_to_id[table_name] = idx
            table_texts.append(f"Table: {table_name}")

        # Columns
        col_idx = 0
        for table, cols in schema_info.get("columns", {}).items():
            for col in cols:
                col_name = col["name"]
                col_type = col["type"]
                col_full_name = f"{table}.{col_name}"
                col_to_id[col_full_name] = col_idx
                col_texts.append(f"Column: {col_name} in table {table}, type {col_type}")
                col_idx += 1

        # FK (Edge-as-Node)
        for idx, (edge_id, desc) in enumerate(fk_descriptions.items()):
            fk_to_id[edge_id] = idx
            fk_texts.append(desc) # LLM이 생성한 자연어 Description 그대로 사용!

        # ---------------------------------------------------------
        # Step 2: PLM을 이용한 Node Feature Tensor 생성 (초기 임베딩)
        # ---------------------------------------------------------
        print("Encoding node features...")
        # (N, D) 형태의 Float Tensor로 변환됨
        data['table'].x = self.encoder.encode(table_texts, convert_to_tensor=True)
        data['column'].x = self.encoder.encode(col_texts, convert_to_tensor=True)
        if fk_texts: # FK가 없는 DB도 있을 수 있으므로 방어 로직
            data['fk_node'].x = self.encoder.encode(fk_texts, convert_to_tensor=True)
        else:
            data['fk_node'].x = torch.empty((0, self.encoder.get_sentence_embedding_dimension()))

        # ---------------------------------------------------------
        # Step 3: Edge Index (연결 구조) 생성
        # ---------------------------------------------------------
        # PyG의 edge_index는 [2, num_edges] 형태의 Long Tensor여야 합니다.
        has_col_src, has_col_dst = [], []
        fk_src, fk_dst = [], []
        fk_rev_src, fk_rev_dst = [], []

        # A. Table <-> Column (belongs_to 관계)
        for table, cols in schema_info.get("columns", {}).items():
            t_id = table_to_id[table]
            for col in cols:
                c_id = col_to_id[f"{table}.{col['name']}"]
                has_col_src.append(t_id)
                has_col_dst.append(c_id)

        # B. Column <-> FK Node <-> Column (관계의 승격)
        for fk in schema_info.get("foreign_keys", []):
            edge_id = f"{fk['from_table']}.{fk['from_column']}->{fk['to_table']}.{fk['to_column']}"
            if edge_id not in fk_to_id:
                continue
            
            f_id = fk_to_id[edge_id]
            from_c_id = col_to_id[f"{fk['from_table']}.{fk['from_column']}"]
            to_c_id = col_to_id[f"{fk['to_table']}.{fk['to_column']}"]

            # Source Column -> FK Node 연결
            fk_src.append(from_c_id)
            fk_dst.append(f_id)

            # FK Node -> Target Column 연결
            fk_rev_src.append(f_id)
            fk_rev_dst.append(to_c_id)

        # Edge Tensor 할당
        data['table', 'has_column', 'column'].edge_index = torch.tensor([has_col_src, has_col_dst], dtype=torch.long)
        # GAT는 양방향 메시지 패싱이 유리하므로 역방향도 추가해 줍니다.
        data['column', 'belongs_to', 'table'].edge_index = torch.tensor([has_col_dst, has_col_src], dtype=torch.long)
        
        data['column', 'is_source_of', 'fk_node'].edge_index = torch.tensor([fk_src, fk_dst], dtype=torch.long)
        data['fk_node', 'points_to', 'column'].edge_index = torch.tensor([fk_rev_src, fk_rev_dst], dtype=torch.long)

        # 디버깅 및 추적을 위해 매핑 딕셔너리를 객체에 저장해 둡니다.
        data.metadata_mapping = {
            'table_to_id': table_to_id,
            'col_to_id': col_to_id,
            'fk_to_id': fk_to_id
        }

        return data

# --- 테스트 코드 (Unit Test) ---
if __name__ == "__main__":
    # 이전 모듈들에서 파싱 및 생성되었다고 가정한 더미 데이터
    mock_schema = {
        "tables": ["department", "employee"],
        "columns": {
            "department": [{"name": "dept_id", "type": "integer"}, {"name": "dept_name", "type": "text"}],
            "employee": [{"name": "emp_id", "type": "integer"}, {"name": "dept_id", "type": "integer"}]
        },
        "foreign_keys": [
            {"from_table": "employee", "from_column": "dept_id", "to_table": "department", "to_column": "dept_id"}
        ]
    }
    
    mock_fk_desc = {
        "employee.dept_id->department.dept_id": "Links an employee to the department they work in, allowing queries about staff locations."
    }

    builder = SchemaGraphBuilder()
    graph_data = builder.build_graph(mock_schema, mock_fk_desc)

    print("\n--- Generated PyG HeteroData Object ---")
    print(graph_data)
    
    print("\n[Node Feature Shapes]")
    print(f"Table Features: {graph_data['table'].x.shape}")
    print(f"Column Features: {graph_data['column'].x.shape}")
    print(f"FK Node Features: {graph_data['fk_node'].x.shape}")
    
    print("\n[Edge Index Shapes]")
    print(f"Table -> Column: {graph_data['table', 'has_column', 'column'].edge_index.shape}")
    print(f"Column -> FK Node: {graph_data['column', 'is_source_of', 'fk_node'].edge_index.shape}")