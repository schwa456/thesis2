import numpy as np
import torch
import pcst_fast
from typing import List, Dict, Tuple, Any

class PCSTSubgraphRouter:
    """
    MaxSim으로 얻은 Initial Node Prize와 동적으로 계산된 Edge Cost를 활용하여,
    PCST (Prize-Collecting Steiner Tree) 알고리즘을 통해 최소한의 노드와 엣지로
    이루어진 최적의 서브그래프를 추출합니다.
    """
    def __init__(self, base_cost: float = 1.0, alpha: float = 0.5, belongs_to_cost: float = 0.01):
        self.base_cost = base_cost              # PK-FK 연결의 기본 비용
        self.alpha = alpha                      # 의미적 유사도(Semantic Similarity)에 따른 비용 할인율
        self.belongs_to_cost = belongs_to_cost  # Table-Column 간의 종속 관계 비용 (매우 낮게 설정)

    def _compute_dynamic_cost(self, 
                              edge_type: str, 
                              query_embs: torch.Tensor, 
                              edge_emb: torch.Tensor = None) -> float:
        """
        [핵심 논문 기여점] 질문 문맥(Query)에 따라 엣지 비용을 동적으로 계산합니다.
        c = c_base - alpha * max_sim(Q, e)
        """
        if edge_type == 'belongs_to':
            # 테이블과 소속 컬럼 간의 연결은 무조건 유지해야 하므로 비용을 거의 0으로 둡니다.
            return self.belongs_to_cost
            
        elif edge_type == 'pk_fk':
            if edge_emb is None:
                return self.base_cost
            
            # Query Tokens (N, D)와 특정 Edge (1, D) 간의 유사도 행렬
            # alignment_layer에서 L2 정규화가 완료되었다고 가정하므로 내적 == 코사인 유사도
            sim_scores = torch.matmul(query_embs, edge_emb.t()) # (N, 1)
            max_sim = sim_scores.max().item()                   # 스칼라 값 (가장 강력한 매칭 점수)
            
            # 유사도가 높을수록 비용을 깎아줌 (할인)
            cost = self.base_cost - (self.alpha * max_sim)
            
            # PCST 알고리즘은 음수 비용을 허용하지 않으므로 최소값을 보장
            return max(cost, 0.01)
            
        return self.base_cost

    def route(self, 
              node_prizes: List[float], 
              edges: List[Tuple[int, int]], 
              edge_types: List[str],
              query_embs: torch.Tensor,
              edge_embs_dict: Dict[int, torch.Tensor],
              root: int = -1) -> Tuple[List[int], List[int]]:
        """
        주어진 그래프와 Prize/Cost를 바탕으로 GW(Goemans-Williamson) 기반의 
        PCST 알고리즘을 실행하여 선택된 노드와 엣지 인덱스를 반환합니다.
        """
        num_nodes = len(node_prizes)
        num_edges = len(edges)
        
        # 1. 텐서를 NumPy 배열로 변환 (pcst_fast 입력용)
        prizes = np.array(node_prizes, dtype=np.float64)
        edges_arr = np.array(edges, dtype=np.int64)
        costs = np.zeros(num_edges, dtype=np.float64)
        
        # 2. Edge Cost 동적 계산
        for i, (u, v) in enumerate(edges):
            e_type = edge_types[i]
            # 해당 엣지의 임베딩이 캐시에 있다면 가져옴 (Targeted Lookup)
            e_emb = edge_embs_dict.get(i, None) 
            
            costs[i] = self._compute_dynamic_cost(
                edge_type=e_type, 
                query_embs=query_embs, 
                edge_emb=e_emb
            )

        print(f"Running PCST on {num_nodes} nodes and {num_edges} edges...")
        
        # 3. PCST-Fast 실행
        # num_clusters=1: 하나의 연결된 트리(Sub-graph) 생성
        # pruning='gw': 강한 가지치기(Strong Pruning) 적용으로 불필요한 리프 노드 제거
        selected_nodes, selected_edges = pcst_fast.pcst_fast(
            edges_arr, prizes, costs, root, num_clusters=1, pruning='gw', verbosity_level=0
        )
        
        return selected_nodes.tolist(), selected_edges.tolist()

    def extract_subgraph_schema(self, 
                                selected_nodes: List[int], 
                                node_id_to_name: Dict[int, str]) -> Dict[str, List[str]]:
        """
        PCST 결과(인덱스)를 Agentic Workflow에 던져줄 수 있는 
        읽기 쉬운 형태(Dictionary)로 재구성합니다.
        """
        schema_dict = {}
        for n_id in selected_nodes:
            name = node_id_to_name[n_id]
            if "." in name: # 컬럼 (예: employee.salary)
                table, col = name.split(".")
                if table not in schema_dict:
                    schema_dict[table] = []
                schema_dict[table].append(col)
            else:           # 테이블 (예: employee)
                if name not in schema_dict:
                    schema_dict[name] = []
                    
        return schema_dict

# --- 테스트 코드 (Unit Test) ---
if __name__ == "__main__":
    # 1. 가상의 환경 세팅
    # 노드 0: employee (Table), 노드 1: employee.id (Col), 노드 2: employee.dept_id (Col)
    # 노드 3: department (Table), 노드 4: department.id (Col), 노드 5: department.name (Col)
    node_id_to_name = {
        0: "employee", 1: "employee.id", 2: "employee.dept_id",
        3: "department", 4: "department.id", 5: "department.name"
    }
    
    # 2. Alignment Layer에서 나온 가상의 Prize (MaxSim 점수)
    # 질문: "What is the department name of employee?"
    # -> employee(0.8), department.name(0.9) 점수가 높게 잡혔다고 가정
    prizes = [0.8, 0.1, 0.0, 0.5, 0.0, 0.9] 
    
    # 3. 스키마 구조 (엣지 리스트)
    edges = [
        (0, 1), (0, 2),       # employee belongs_to id, dept_id
        (3, 4), (3, 5),       # department belongs_to id, name
        (2, 4)                # PK-FK 연결 (employee.dept_id -> department.id)
    ]
    edge_types = ['belongs_to', 'belongs_to', 'belongs_to', 'belongs_to', 'pk_fk']
    
    # 4. 동적 비용 계산을 위한 가상 임베딩
    # 질문에 'department'라는 단어가 있어서 PK-FK 엣지(인덱스 4)와 매칭 점수가 높다고 가정
    mock_query_embs = torch.randn((5, 256)) 
    mock_edge_embs = {4: torch.randn((1, 256))} 
    
    # 5. PCST 라우팅 실행
    router = PCSTSubgraphRouter(base_cost=1.0, alpha=0.5)
    sel_nodes, sel_edges = router.route(
        node_prizes=prizes, 
        edges=edges, 
        edge_types=edge_types, 
        query_embs=mock_query_embs, 
        edge_embs_dict=mock_edge_embs
    )
    
    print("\n--- PCST Selected Indices ---")
    print(f"Nodes: {sel_nodes}")
    print(f"Edges: {sel_edges}")
    
    # 6. 최종 서브그래프 추출
    final_schema = router.extract_subgraph_schema(sel_nodes, node_id_to_name)
    print("\n--- Refined Sub-graph for Agent ---")
    for tbl, cols in final_schema.items():
        print(f"Table: {tbl}")
        if cols:
            print(f"  Columns: {', '.join(cols)}")