import os
import torch
import pickle
from offline_indexing.schema_parser import SQLiteSchemaParser
from offline_indexing.llm_verbalizer import SchemaVerbalizer
from offline_indexing.graph_builder import SchemaGraphBuilder
from models.gat_network import SchemaHeteroGAT
from models.alignment_layer import DualTowerAlignment
from offline_indexing.build_index import FAISSIndexBuilder

def run_offline_pipeline(db_path: str):
    print(f"========== [Offline Pipeline Started] ==========")
    print(f"Target Database: {db_path}")

    # 1. DB Schema 파싱
    print("\n[1/6] Parsing SQLite Schema...")
    parser = SQLiteSchemaParser(db_path)
    schema_info = parser.parse_schema()

    # 2. LLM을 이용한 Edge Verbalization (vLLM 연동)
    print("\n[2/6] Verbalizing Foreign Keys using LLM...")
    verbalizer = SchemaVerbalizer() # vLLM 로컬 포트 세팅됨
    fk_descriptions = verbalizer.process_all_fks(schema_info)

    # 3. PyG 이종 그래프(HeteroData) 구축
    print("\n[3/6] Building Heterogeneous Graph...")
    graph_builder = SchemaGraphBuilder(plm_model_name='sentence-transformers/all-MiniLM-L6-v2')
    graph_data = graph_builder.build_graph(schema_info, fk_descriptions)
    
    # 모델 학습을 위한 Edge 리스트 및 타입 캐싱 (나중에 PCST에서 사용)
    topology_info = {
        'edges': [], 'edge_types': [], 'metadata_mapping': graph_data.metadata_mapping
    }
    # PCST를 위한 Graph Topology 재구성 (단순화된 예시)
    table_has_col = graph_data['table', 'has_column', 'column'].edge_index.t().tolist()
    fk_points_to = graph_data['fk_node', 'points_to', 'column'].edge_index.t().tolist()
    # 실제 구현 시에는 table, col, fk_node의 global index를 맞춰 topology_info에 저장해야 합니다.
    # 여기서는 데모를 위해 메타데이터만 저장합니다.
    
    with open("./data/processed/topology_info.pkl", "wb") as f:
        pickle.dump(topology_info, f)

    # 4. GAT 모델 통과 (구조적 임베딩 학습/추론)
    print("\n[4/6] Running GAT for Structural Embeddings...")
    gat_model = SchemaHeteroGAT(in_channels=384, hidden_channels=128, out_channels=256)
    gat_model.eval() # 추론 모드
    
    with torch.no_grad():
        gat_embeddings = gat_model(graph_data.x_dict, graph_data.edge_index_dict)

    # 5. Dual-Tower Alignment 통과 (텍스트/그래프 Joint Space 투영)
    print("\n[5/6] Projecting to Shared Latent Space...")
    alignment_layer = DualTowerAlignment(text_dim=384, graph_dim=256, joint_dim=256)
    alignment_layer.eval()
    
    with torch.no_grad():
        # Text 텐서는 임시로 GAT의 출력을 크기만 맞춰서 통과 (실제론 PLM 출력이어야 함)
        _, z_graph_table = alignment_layer(torch.zeros(1,384), gat_embeddings['table'])
        _, z_graph_col = alignment_layer(torch.zeros(1,384), gat_embeddings['column'])
        _, z_graph_fk = alignment_layer(torch.zeros(1,384), gat_embeddings['fk_node'])
        
        final_node_embs = {'table': z_graph_table, 'column': z_graph_col}
        final_edge_embs = z_graph_fk

    # 6. FAISS 인덱스 빌드 및 저장
    print("\n[6/6] Building FAISS Index and KV Store...")
    faiss_builder = FAISSIndexBuilder(vector_dim=256, save_dir="./data/processed")
    faiss_builder.build_and_save(
        node_embs=final_node_embs, 
        edge_embs=final_edge_embs, 
        metadata_mapping=graph_data.metadata_mapping
    )

    print("\n========== [Offline Pipeline Completed] ==========")
    print("System is ready for Online Inference!")

if __name__ == "__main__":
    # 실행 예시 (로컬에 더미 sqlite 파일이 있다고 가정)
    test_db = "dummy_bird.sqlite"
    # sqlite3 셋업 생략 (schema_parser 테스트 코드 참고)
    try:
        run_offline_pipeline(test_db)
    except FileNotFoundError:
        print(f"Please provide a valid SQLite database path. (Looking for: {test_db})")