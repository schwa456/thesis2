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
            final_dict[node_type] = self.out_lin_dict[node_type](x)

        # 최종 반환 형태: 각 노드 타입별 업데이트된 임베딩 텐서
        return final_dict

# --- 테스트 코드 (Unit Test) ---
if __name__ == "__main__":
    # graph_builder.py에서 만든 것과 동일한 형태의 가짜 데이터 생성
    # 가정: PLM(all-MiniLM)이 384차원의 벡터를 뱉었다고 가정
    num_tables, num_cols, num_fks = 2, 5, 1
    in_dim = 384

    mock_x_dict = {
        'table': torch.randn((num_tables, in_dim)),
        'column': torch.randn((num_cols, in_dim)),
        'fk_node': torch.randn((num_fks, in_dim))
    }

    mock_edge_index_dict = {
        ('table', 'has_column', 'column'): torch.tensor([[0, 0, 1, 1, 1], [0, 1, 2, 3, 4]], dtype=torch.long),
        ('column', 'belongs_to', 'table'): torch.tensor([[0, 1, 2, 3, 4], [0, 0, 1, 1, 1]], dtype=torch.long),
        ('column', 'is_source_of', 'fk_node'): torch.tensor([[1], [0]], dtype=torch.long),
        ('fk_node', 'points_to', 'column'): torch.tensor([[0], [2]], dtype=torch.long)
    }

    # 2. 모델 초기화 (Input 384 -> Hidden 128 -> Output 256)
    model = SchemaHeteroGAT(in_channels=384, hidden_channels=128, out_channels=256, num_layers=2, heads=4)
    
    # 3. 모델 Forward Pass (Inference)
    output_dict = model(mock_x_dict, mock_edge_index_dict)

    print("\n--- GAT Output Embeddings ---")
    for node_type, tensor in output_dict.items():
        print(f"Updated {node_type.capitalize()} Node Shape: {tensor.shape}")