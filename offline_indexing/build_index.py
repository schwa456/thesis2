import os
import faiss
import torch
import pickle
import numpy as np
from typing import Dict, Any

class FAISSIndexBuilder:
    """
    Offline 단계에서 계산된 DB 스키마 노드(Table, Column)의 최종 임베딩을 FAISS에 적재하고,
    PCST 라우팅 시 필요한 엣지(Edge) 임베딩을 별도의 딕셔너리로 저장합니다.
    """
    def __init__(self, vector_dim: int = 256, save_dir: str = "./data/processed"):
        self.vector_dim = vector_dim
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Alignment Layer에서 L2 정규화를 거쳤으므로, 내적(Inner Product)이 곧 코사인 유사도입니다.
        self.index = faiss.IndexFlatIP(self.vector_dim)
        
        self.node_metadata = {} # FAISS ID -> Node Name 매핑
        self.edge_embs_dict = {} # Edge ID -> Edge Embedding 매핑

    def build_and_save(self, 
                       node_embs: Dict[str, torch.Tensor], 
                       edge_embs: torch.Tensor,
                       metadata_mapping: Dict[str, dict]):
        """
        node_embs: {'table': Tensor, 'column': Tensor} (Alignment Layer 통과 후)
        edge_embs: Tensor (fk_node 임베딩)
        metadata_mapping: graph_builder에서 만든 ID 매핑 정보
        """
        print(f"Building FAISS Index (Dimension: {self.vector_dim})...")
        
        global_idx = 0
        all_vectors = []
        
        # 1. Table Node 적재
        table_to_id = metadata_mapping['table_to_id']
        id_to_table = {v: k for k, v in table_to_id.items()}
        for i, emb in enumerate(node_embs['table']):
            all_vectors.append(emb.detach().cpu().numpy())
            self.node_metadata[global_idx] = id_to_table[i]
            global_idx += 1
            
        # 2. Column Node 적재
        col_to_id = metadata_mapping['col_to_id']
        id_to_col = {v: k for k, v in col_to_id.items()}
        for i, emb in enumerate(node_embs['column']):
            all_vectors.append(emb.detach().cpu().numpy())
            self.node_metadata[global_idx] = id_to_col[i]
            global_idx += 1
            
        # FAISS에 벡터 일괄 추가
        all_vectors_np = np.vstack(all_vectors).astype('float32')
        self.index.add(all_vectors_np)
        print(f"Successfully added {self.index.ntotal} nodes to FAISS.")

        # 3. Edge Embedding은 PCST 조회를 위해 별도 저장 (Targeted Lookup 용도)
        # edge_index_dict의 fk_node 순서와 매핑
        fk_to_id = metadata_mapping['fk_to_id']
        for edge_name, e_id in fk_to_id.items():
            self.edge_embs_dict[edge_name] = edge_embs[e_id].detach().cpu()

        # 4. 디스크에 저장 (영구 보관)
        index_path = os.path.join(self.save_dir, "schema_nodes.index")
        faiss.write_index(self.index, index_path)
        
        meta_path = os.path.join(self.save_dir, "metadata.pkl")
        with open(meta_path, 'wb') as f:
            pickle.dump({
                "node_metadata": self.node_metadata,
                "edge_embs_dict": self.edge_embs_dict,
                "vector_dim": self.vector_dim
            }, f)
            
        print(f"Index and Metadata saved to {self.save_dir}")

# --- 단위 테스트 ---
if __name__ == "__main__":
    # 가상의 임베딩 데이터 (GAT -> Alignment Layer 통과 후라고 가정)
    mock_node_embs = {
        'table': torch.randn(2, 256),
        'column': torch.randn(5, 256)
    }
    # L2 정규화 시뮬레이션
    mock_node_embs['table'] = torch.nn.functional.normalize(mock_node_embs['table'], p=2, dim=1)
    mock_node_embs['column'] = torch.nn.functional.normalize(mock_node_embs['column'], p=2, dim=1)
    
    mock_edge_embs = torch.nn.functional.normalize(torch.randn(1, 256), p=2, dim=1)
    
    mock_mapping = {
        'table_to_id': {'department': 0, 'employee': 1},
        'col_to_id': {'department.id': 0, 'department.name': 1, 'employee.id': 2, 'employee.dept_id': 3, 'employee.salary': 4},
        'fk_to_id': {'employee.dept_id->department.id': 0}
    }
    
    builder = FAISSIndexBuilder(vector_dim=256)
    builder.build_and_save(mock_node_embs, mock_edge_embs, mock_mapping)