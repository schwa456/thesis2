import os
import faiss
import torch
import pickle
import numpy as np
from typing import List, Dict, Tuple
from utils.logger import exp_logger

class InitialNodeRetriever:
    """
    사용자 질의 토큰 벡터를 해당 DB의 FAISS 인덱스에서 조회하여 Initial Seed Node를 추출합니다.
    (BIRD 대응: db_id에 따라 실시간으로 다른 인덱스를 로드합니다.)
    """
    def __init__(self, db_id: str, config: dict):
        # 1. 설정값 로드
        self.db_id = db_id
        self.load_dir = config['paths']['processed_data_dir']
        self.threshold = config.get('retrieval', {}).get('threshold', 0.6)
        
        # 2. DB별 고유 파일 경로 설정 (build_index.py에서 저장한 규칙과 일치해야 함)
        index_path = os.path.join(self.load_dir, f"{db_id}_index.faiss")
        meta_path = os.path.join(self.load_dir, f"{db_id}_index_metadata.pkl")
        
        # 3. 파일 존재 여부 확인 및 로드
        if not os.path.exists(index_path) or not os.path.exists(meta_path):
            exp_logger.error(f"Index or metadata for DB '{db_id}' not found at {self.load_dir}")
            raise FileNotFoundError(f"Missing FAISS artifacts for {db_id}. Run run_offline.py first.")
            
        self.index = faiss.read_index(index_path)
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
            self.node_metadata = meta["node_metadata"]
            self.edge_embs_dict = meta["edge_embs_dict"]
            self.edges = meta.get("edges", [])
            self.edge_types = meta.get("edge_types", [])
            
        exp_logger.debug(f"[{db_id}] Loaded FAISS Index with {self.index.ntotal} nodes.")

    def retrieve_seed_nodes(self, query_token_embs: torch.Tensor, top_k: int = 3) -> Tuple[List[Dict], List[float]]:
        """
        query_token_embs: (Num_Tokens, Joint_Dim) 형태의 질문 토큰 임베딩
        각 토큰마다 Top-K 노드를 찾은 뒤, 임계값(Threshold)을 넘는 녀석들만 필터링하여 반환.
        """
        if self.index.ntotal == 0:
            return [], [0.0] * 0

        # FAISS 검색을 위해 NumPy로 변환
        q_np = query_token_embs.detach().cpu().numpy().astype('float32')
        
        # 각 토큰에 대해 Top-K 검색
        distances, indices = self.index.search(q_np, top_k)

        exp_logger.debug(f"[Retriever] Max Similarity Score for this query: {distances.max():.4f} (Threshold: {self.threshold})")
        
        selected_nodes = {} # {Node_ID: Max_Score}
        
        for token_idx in range(len(q_np)):
            for rank in range(top_k):
                score = distances[token_idx][rank]
                node_id = indices[token_idx][rank]
                
                if node_id == -1: continue # 검색 결과가 부족할 경우 방어

                # 임계값(Threshold) 기반 필터링
                if score >= self.threshold or len(selected_nodes) < 5:
                    if node_id not in selected_nodes or score > selected_nodes[node_id]:
                        selected_nodes[node_id] = float(score)

        # 결과 포맷팅
        seed_nodes_info = []
        node_prizes = [0.0] * self.index.ntotal 
        
        for n_id, max_score in selected_nodes.items():
            if n_id in self.node_metadata:
                node_name = self.node_metadata[n_id]
                seed_nodes_info.append({"node_id": n_id, "name": node_name, "score": max_score})
                node_prizes[n_id] = max_score

        seed_nodes_info = sorted(seed_nodes_info, key=lambda x: x["score"], reverse=True)
        
        exp_logger.debug(f"[Retrieval Result] {seed_nodes_info}")
        exp_logger.debug(f"[Prizes] {node_prizes}")
        
        return seed_nodes_info, node_prizes