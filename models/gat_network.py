import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GATv2Conv, Linear
from torch_geometric.data import HeteroData

class SchemaHeteroGAT(nn.Module):
    """
    이종 그래프(Heterogeneous Graph) 구조의 DB 스키마를 학습하는 모델.
    Table, Column, FK_Node가 서로 Attention 기반의 메시지 패싱을 수행하여,
    텍스트 의미(Semantic)와 구조적 위치(Structure)가 융합된 임베딩을 출력합니다.
    """
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, num_layers: int = 2, heads: int = 4):
        super(SchemaHeteroGAT, self).__init__()
        self.num_layers = num_layers
        
        # 1. 초기 차원 축소 및 통일 (PLM의 384 차원 -> GAT의 256 차원 등)
        self.lin_dict = nn.ModuleDict({
            'table': Linear(in_channels, hidden_channels),
            'column': Linear(in_channels, hidden_channels),
            'fk_node': Linear(in_channels, hidden_channels)
        })

        # 2. Heterogeneous GAT 레이어 정의
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            # HeteroConv는 각 엣지 타입별로 서로 다른 GATConv를 적용한 후 결과를 합칩니다.
            conv = HeteroConv({
                # A. Table <-> Column 상호작용
                ('table', 'has_column', 'column'): GATv2Conv(-1, hidden_channels, heads=heads, add_self_loops=False),
                ('column', 'belongs_to', 'table'): GATv2Conv(-1, hidden_channels, heads=heads, add_self_loops=False),
                
                # B. Column <-> FK_Node 상호작용 (논문의 핵심 기여 포인트)
                ('column', 'is_source_of', 'fk_node'): GATv2Conv(-1, hidden_channels, heads=heads, add_self_loops=False),
                ('fk_node', 'points_to', 'column'): GATv2Conv(-1, hidden_channels, heads=heads, add_self_loops=False)
            }, aggr='mean') # 여러 엣지에서 들어오는 정보는 평균(mean)으로 병합
            
            self.convs.append(conv)

        # 3. 최종 출력 차원 변환 (Alignment Layer와 연결될 차원)
        self.out_lin_dict = nn.ModuleDict({
            'table': Linear(hidden_channels * heads, out_channels),
            'column': Linear(hidden_channels * heads, out_channels),
            'fk_node': Linear(hidden_channels * heads, out_channels)
        })

        self.skip_dict = nn.ModuleDict({
            'table': Linear(in_channels, out_channels),
            'column': Linear(in_channels, out_channels),
            'fk_node': Linear(in_channels, out_channels)
        })

    def forward(self, x_dict: dict, edge_index_dict: dict) -> dict:
        """
        x_dict: {'table': Tensor, 'column': Tensor, 'fk_node': Tensor}
        edge_index_dict: {('table', 'has_column', 'column'): Tensor, ...}
        """
        # Step 1: Input Projection (Linear 통과 및 Activation)
        out_dict = {}
        for node_type, x in x_dict.items():
            out_dict[node_type] = F.leaky_relu(self.lin_dict[node_type](x))

        # Step 2: Message Passing (GAT Layers)
        for i in range(self.num_layers):
            out_dict = self.convs[i](out_dict, edge_index_dict)
            
            # 레이어 사이에 비선형 활성화 함수 적용
            out_dict = {node_type: F.elu(x) for node_type, x in out_dict.items()}

        # Step 3: Output Projection
        final_dict = {}
        for node_type, x in out_dict.items():
            final_dict[node_type] = self.out_lin_dict[node_type](x) + self.skip_dict[node_type](x_dict[node_type])

        # 최종 반환 형태: 각 노드 타입별 업데이트된 임베딩 텐서
        return final_dict