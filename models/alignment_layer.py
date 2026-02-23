import math
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
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / 0.07))

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