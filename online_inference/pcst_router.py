import numpy as np
import torch
import pcst_fast
from typing import List, Dict, Tuple, Any
from utils.logger import exp_logger

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
            edge_emb_device = edge_emb.to(query_embs.device)
            sim_scores = torch.matmul(query_embs, edge_emb_device.t()) # (N, 1)
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
        prizes = np.maximum(prizes, 0.0)
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

        exp_logger.debug(f"Running PCST on {num_nodes} nodes and {num_edges} edges...")
        
        # 3. PCST-Fast 실행
        # num_clusters=1: 하나의 연결된 트리(Sub-graph) 생성
        # pruning='gw': 강한 가지치기(Strong Pruning) 적용으로 불필요한 리프 노드 제거
        selected_nodes, selected_edges = pcst_fast.pcst_fast(
            edges_arr, prizes, costs, root, 1, 'gw', 0
        )

        exp_logger.debug(f"[Selected Nodes] {selected_nodes}")
        exp_logger.debug(f"[Selected Edges] {selected_edges}")
        
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
        
        exp_logger.debug(f"[Subgraph Schema] {schema_dict}")
        
        return schema_dict