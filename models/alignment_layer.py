import torch
import torch.nn as nn
import torch.nn.functional as F

class DualTowerAlignment(nn.Module):
    """
    Symmetric Dual-Tower 구조를 통해 텍스트(Token/Edge Desc)와 그래프(Node) 임베딩을
    동일한 공유 공간(Shared Latent Space)으로 투영합니다.
    (논문 기여점: L2 정규화가 적용된 투영과 MaxSim 기반의 스코어링 결합)
    """
    def __init__(self, text_dim: int = 384, graph_dim: int = 256, joint_dim: int = 256):
        super(DualTowerAlignment, self).__init__()
        
        # 1. Text Projection Head (NLQ Token 및 LLM Edge Description 용)
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, joint_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(joint_dim, joint_dim)
        )
        
        # 2. Graph Projection Head (GAT의 출력 노드 용)
        self.graph_proj = nn.Sequential(
            nn.Linear(graph_dim, joint_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(joint_dim, joint_dim)
        )
        
        # 대조 학습(Contrastive Learning)을 위한 온도(Temperature) 파라미터.
        # 학습을 통해 최적화되도록 Parameter로 설정 (초기값 0.07은 CLIP 논문 기준)
        self.logit_scale = nn.Parameter(torch.ones([]) * (1 / 0.07).log())

    def forward(self, text_embs: torch.Tensor, graph_embs: torch.Tensor):
        """
        두 모달리티의 벡터를 투영하고, 코사인 유사도 연산을 위해 L2 정규화를 수행합니다.
        """
        # 투영 (Projection)
        z_text = self.text_proj(text_embs)
        z_graph = self.graph_proj(graph_embs)
        
        # L2 정규화 (Unit Hypersphere 상에 매핑)
        z_text = F.normalize(z_text, p=2, dim=-1)
        z_graph = F.normalize(z_graph, p=2, dim=-1)
        
        return z_text, z_graph

    def compute_contrastive_loss(self, z_text: torch.Tensor, z_graph: torch.Tensor) -> torch.Tensor:
        """
        In-batch Negative를 활용한 InfoNCE(대조 학습) 손실 함수 계산.
        학습 시 사용됩니다 (Offline).
        (z_text와 z_graph는 1:1로 매칭되는 정답 쌍(Positive Pair) 형태로 들어와야 함)
        """
        # Temperature 스케일링
        logit_scale = self.logit_scale.exp()
        
        # 유사도 행렬 계산 (Batch Size x Batch Size)
        # 내적(Dot Product)이 곧 코사인 유사도(Cosine Similarity)가 됨 (L2 정규화 덕분)
        sim_matrix = logit_scale * torch.matmul(z_text, z_graph.t())
        
        # 대각선(Diagonal) 요소들이 정답(Positive)
        labels = torch.arange(sim_matrix.size(0), device=sim_matrix.device, dtype=torch.long)
        
        # 양방향 Cross Entropy (Text -> Graph, Graph -> Text)
        loss_t2g = F.cross_entropy(sim_matrix, labels)
        loss_g2t = F.cross_entropy(sim_matrix.t(), labels)
        
        return (loss_t2g + loss_g2t) / 2

    def compute_maxsim_scores(self, z_query_tokens: torch.Tensor, z_schema_nodes: torch.Tensor) -> torch.Tensor:
        """
        [핵심 알고리즘] Online Inference 단계에서 초기 노드(Initial Seed Nodes)를 찾기 위한 함수.
        N개의 질의 토큰과 M개의 스키마 노드 간의 최대 유사도(MaxSim)를 계산합니다.
        """
        # (N, M) 크기의 유사도 행렬 생성
        # z_query_tokens: (N, D), z_schema_nodes: (M, D)
        sim_matrix = torch.matmul(z_query_tokens, z_schema_nodes.t()) 
        
        # 각 스키마 노드 입장에서, 질의 토큰들 중 가장 높은 유사도를 가진 값을 추출 (Max Pooling)
        # max_sim_scores: (M,) 크기의 벡터 (각 노드별 획득 점수 == Prize)
        max_sim_scores, _ = sim_matrix.max(dim=0)
        
        return max_sim_scores

# --- 테스트 코드 (Unit Test) ---
if __name__ == "__main__":
    # 1. 하이퍼파라미터 세팅
    num_query_tokens = 6  # "Show", "me", "the", "salary", "of", "employees" (불용어 제거 전)
    num_schema_nodes = 10 # 전체 DB의 Table/Column 총합 (테스트용)
    
    text_dim = 384   # PLM 출력 차원
    graph_dim = 256  # GAT 출력 차원
    joint_dim = 256  # 매핑될 공통 차원
    
    # 2. 가상의 텐서 생성
    mock_text_embs = torch.randn((num_query_tokens, text_dim))
    mock_graph_embs = torch.randn((num_schema_nodes, graph_dim))
    
    # 3. 모듈 초기화
    alignment_layer = DualTowerAlignment(text_dim=text_dim, graph_dim=graph_dim, joint_dim=joint_dim)
    
    # 4. Forward Pass (투영 및 정규화)
    z_text, z_graph = alignment_layer(mock_text_embs, mock_graph_embs)
    print(f"Projected Text Shape:  {z_text.shape} (L2 Norm: {torch.norm(z_text[0]).item():.4f})")
    print(f"Projected Graph Shape: {z_graph.shape} (L2 Norm: {torch.norm(z_graph[0]).item():.4f})")
    
    # 5. [Online Inference] MaxSim 점수 계산
    maxsim_scores = alignment_layer.compute_maxsim_scores(z_text, z_graph)
    print(f"\nMaxSim Scores for Schema Nodes: \n{maxsim_scores}")
    print(f"Score Shape: {maxsim_scores.shape} (1점당 1개의 스키마 노드 매칭)")
    
    # 6. [Offline Training] Contrastive Loss 계산 테스트 (배치 사이즈 4 가정)
    mock_anchor_text = torch.randn((4, text_dim))
    mock_positive_graph = torch.randn((4, graph_dim))
    z_a, z_p = alignment_layer(mock_anchor_text, mock_positive_graph)
    loss = alignment_layer.compute_contrastive_loss(z_a, z_p)
    print(f"\nContrastive Loss (InfoNCE): {loss.item():.4f}")