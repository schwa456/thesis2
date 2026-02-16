import os
import faiss
import torch
import pickle
import numpy as np
from typing import List, Dict, Tuple

class InitialNodeRetriever:
    """
    사용자 질의 토큰 벡터를 FAISS에 쿼리하여 Initial Seed Node를 추출합니다.
    (논문 기여점: 단순 Top-K가 아닌 Threshold(임계값) 기반 필터링으로 노이즈 원천 차단)
    """
    def __init__(self, load_dir: str = "./data/processed", threshold: float = 0.6):
        self.load_dir = load_dir
        self.threshold = threshold # 유사도 임계값 (tau)
        
        # 저장된 FAISS 인덱스와 메타데이터 로드
        index_path = os.path.join(self.load_dir, "schema_nodes.index")
        meta_path = os.path.join(self.load_dir, "metadata.pkl")
        
        if not os.path.exists(index_path) or not os.path.exists(meta_path):
            raise FileNotFoundError("FAISS index or metadata not found. Run build_index.py first.")
            
        self.index = faiss.read_index(index_path)
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
            self.node_metadata = meta["node_metadata"]
            self.edge_embs_dict = meta["edge_embs_dict"]
            
        print(f"Loaded FAISS Index with {self.index.ntotal} nodes.")

    def retrieve_seed_nodes(self, query_token_embs: torch.Tensor, top_k: int = 3) -> Tuple[List[Dict], List[float]]:
        """
        query_token_embs: (Num_Tokens, 256) 형태의 질문 토큰 임베딩 (Alignment Layer 통과 후)
        각 토큰마다 Top-K 노드를 찾은 뒤, 임계값(Threshold)을 넘는 녀석들만 모아서 반환.
        """
        # FAISS 검색을 위해 NumPy로 변환
        q_np = query_token_embs.detach().cpu().numpy().astype('float32')
        
        # 각 토큰에 대해 Top-K 검색 (Inner Product = Cosine Similarity)
        # distances: (Num_Tokens, Top_K), indices: (Num_Tokens, Top_K)
        distances, indices = self.index.search(q_np, top_k)
        
        selected_nodes = {} # 중복 제거를 위한 Dict (Node_ID -> Max Score)
        
        for token_idx in range(len(q_np)):
            for rank in range(top_k):
                score = distances[token_idx][rank]
                node_id = indices[token_idx][rank]
                
                # [핵심] 임계값(Threshold)을 넘지 못하면 과감히 버림 (Uncertainty 연계)
                if score >= self.threshold:
                    # 한 노드가 여러 토큰에 매칭되었다면, 가장 높은 점수(MaxSim)로 덮어씀
                    if node_id not in selected_nodes or score > selected_nodes[node_id]:
                        selected_nodes[node_id] = float(score)

        # 결과 포맷팅 (PCST 라우터에 넘겨줄 형태)
        seed_nodes_info = []
        node_prizes = [0.0] * self.index.ntotal # 전체 노드 수만큼 0.0으로 초기화
        
        for n_id, max_score in selected_nodes.items():
            node_name = self.node_metadata[n_id]
            seed_nodes_info.append({"node_id": n_id, "name": node_name, "score": max_score})
            # PCST를 위한 Prize 배열 업데이트
            node_prizes[n_id] = max_score

        # 유사도 점수 기준 내림차순 정렬
        seed_nodes_info = sorted(seed_nodes_info, key=lambda x: x["score"], reverse=True)
        
        return seed_nodes_info, node_prizes

# --- 단위 테스트 ---
if __name__ == "__main__":
    retriever = InitialNodeRetriever(threshold=0.5)
    
    # 가상의 질문 토큰 임베딩 ("salary", "employee" 토큰이라고 가정)
    mock_query_embs = torch.randn(2, 256)
    mock_query_embs = torch.nn.functional.normalize(mock_query_embs, p=2, dim=1)
    
    seed_info, prizes = retriever.retrieve_seed_nodes(mock_query_embs, top_k=2)
    
    print("\n--- Retrieved Initial Seed Nodes ---")
    for info in seed_info:
        print(f"Node: {info['name']}, Score: {info['score']:.4f}")