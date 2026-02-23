import os
import faiss
import torch
import pickle
import numpy as np
from typing import Dict, Any
from utils.logger import data_logger

class FAISSIndexBuilder:
    def __init__(self, vector_dim: int = 256, save_dir: str = "./data/processed"):
        self.vector_dim = vector_dim
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        
    def build_and_save(self, 
                       node_embs: Dict[str, torch.Tensor], 
                       edge_embs: torch.Tensor,
                       metadata_mapping: Dict[str, dict],
                       save_name: str):
        
        data_logger.debug(f"Building FAISS Index for [{save_name}] (Dim: {self.vector_dim})...")
        
        self.index = faiss.IndexFlatIP(self.vector_dim)
        self.node_metadata = {}
        
        global_idx = 0
        all_vectors = []
        
        # [핵심 1] 로컬 ID(테이블별/컬럼별 인덱스)를 FAISS의 글로벌 ID로 변환하기 위한 맵
        local_to_global_table = {}
        local_to_global_col = {}
        
        # 1. Table Node 적재
        table_to_id = metadata_mapping['table_to_id']
        id_to_table = {v: k for k, v in table_to_id.items()}
        for i, emb in enumerate(node_embs['table']):
            all_vectors.append(emb.detach().cpu().numpy())
            self.node_metadata[global_idx] = id_to_table[i]
            local_to_global_table[i] = global_idx
            global_idx += 1
            
        # 2. Column Node 적재
        col_to_id = metadata_mapping['col_to_id']
        id_to_col = {v: k for k, v in col_to_id.items()}
        for i, emb in enumerate(node_embs['column']):
            all_vectors.append(emb.detach().cpu().numpy())
            self.node_metadata[global_idx] = id_to_col[i]
            local_to_global_col[i] = global_idx
            global_idx += 1
            
        if all_vectors:
            all_vectors_np = np.vstack(all_vectors).astype('float32')
            self.index.add(all_vectors_np)

        # ---------------------------------------------------------
        # [핵심 2] PCST를 위한 그래프 토폴로지(Edges, Edge Types) 생성
        # ---------------------------------------------------------
        edges = []
        edge_types = []
        pcst_edge_embs_dict = {} # PCST 배열 인덱스에 맞춘 엣지 임베딩
        
        # A. Table <-> Column 엣지 복원 (belongs_to)
        for col_name, local_c_id in col_to_id.items():
            table_name = col_name.split('.')[0]
            if table_name in table_to_id:
                local_t_id = table_to_id[table_name]
                global_t_id = local_to_global_table[local_t_id]
                global_c_id = local_to_global_col[local_c_id]
                
                edges.append((global_t_id, global_c_id))
                edge_types.append('belongs_to')
                
        # B. Column <-> Column 외래키 엣지 복원 (pk_fk)
        fk_to_id = metadata_mapping.get('fk_to_id', {})
        for edge_name, e_id in fk_to_id.items():
            try:
                from_col, to_col = edge_name.split('->')
                if from_col in col_to_id and to_col in col_to_id:
                    global_from_id = local_to_global_col[col_to_id[from_col]]
                    global_to_id = local_to_global_col[col_to_id[to_col]]
                    
                    curr_edge_idx = len(edges)
                    edges.append((global_from_id, global_to_id))
                    edge_types.append('pk_fk')
                    
                    # PCST 계산 시 사용할 텐서 매핑
                    if e_id < edge_embs.size(0):
                        pcst_edge_embs_dict[curr_edge_idx] = edge_embs[e_id].detach().cpu()
            except ValueError:
                continue # 파싱 오류 방어

        # 3. 디스크에 저장
        index_path = os.path.join(self.save_dir, f"{save_name}.faiss")
        faiss.write_index(self.index, index_path)
        
        meta_path = os.path.join(self.save_dir, f"{save_name}_metadata.pkl")
        with open(meta_path, 'wb') as f:
            pickle.dump({
                "node_metadata": self.node_metadata,
                "edges": edges,                   # 새로 추가됨!
                "edge_types": edge_types,         # 새로 추가됨!
                "edge_embs_dict": pcst_edge_embs_dict,
                "vector_dim": self.vector_dim
            }, f)
            
        data_logger.debug(f"Index and Metadata saved to {self.save_dir}")