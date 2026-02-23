import os
import torch
import pickle
import argparse
from offline_indexing.schema_parser import SQLiteSchemaParser
from offline_indexing.llm_verbalizer import SchemaVerbalizer
from offline_indexing.graph_builder import SchemaGraphBuilder
from models.gat_network import SchemaHeteroGAT
from models.alignment_layer import DualTowerAlignment
from offline_indexing.build_index import FAISSIndexBuilder
from utils.config_loader import load_config
from utils.logger import data_logger

def run_offline_pipeline(db_path: str):
    config = load_config()

    data_logger.info(f"========== [Offline Pipeline Started] ==========")
    data_logger.info(f"Target Database: {db_path}")
    
    # 디바이스 단 1번만 명확히 초기화
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_logger.info(f"Using device: {device}")

    # 1. DB Schema 파싱
    data_logger.info("\n[1/6] Parsing SQLite Schema...")
    parser = SQLiteSchemaParser(db_path)
    schema_info = parser.parse_schema()

    # 2. LLM을 이용한 Edge Verbalization (vLLM 연동)
    data_logger.info("\n[2/6] Verbalizing Foreign Keys using LLM...")
    verbalizer = SchemaVerbalizer(
        model_name=config['llm']['model_name'],
        api_base=config['llm']['api_base'],
        api_key=config['llm']['api_key']
    )
    fk_descriptions = verbalizer.process_all_fks(schema_info)

    # 3. PyG 이종 그래프(HeteroData) 구축
    data_logger.info("\n[3/6] Building Heterogeneous Graph...")
    graph_builder = SchemaGraphBuilder(plm_model_name=config['models']['plm_model_name'])
    graph_data = graph_builder.build_graph(schema_info, fk_descriptions)
    
    # 모델 학습을 위한 Edge 리스트 및 타입 캐싱 (나중에 PCST에서 사용)
    topology_info = {
        'edges': [], 'edge_types': [], 'metadata_mapping': graph_data.metadata_mapping
    }
    # PCST를 위한 Graph Topology 재구성 (단순화된 예시)
    table_has_col = graph_data['table', 'has_column', 'column'].edge_index.t().tolist()
    fk_points_to = graph_data['fk_node', 'points_to', 'column'].edge_index.t().tolist()
    
    with open("./data/processed/topology_info.pkl", "wb") as f:
        pickle.dump(topology_info, f)

    # 그래프 데이터를 GPU로 이동!
    graph_data = graph_data.to(device)

    # 4. GAT 모델 초기화 및 가중치 로드
    data_logger.info("\n[4/6] Running GAT for Structural Embeddings...")
    gat_model = SchemaHeteroGAT(
        in_channels=config['models']['dimensions']['plm_out'], 
        hidden_channels=config['models']['dimensions']['gat_hidden'], 
        out_channels=config['models']['dimensions']['gat_out'],
        num_layers=config['models']['gat_params']['num_layers'],
        heads=config['models']['gat_params']['heads']
    ).to(device)

    # 추론 전에 학습된 가중치 불러오기 및 eval 모드 전환
    if os.path.exists("./models/saved/gat_best.pt"):
        data_logger.debug(" -> Loading trained weights for GAT...")
        gat_model.load_state_dict(torch.load("./models/saved/gat_best.pt", map_location=device))
    gat_model.eval()

    with torch.no_grad():
        gat_embeddings = gat_model(graph_data.x_dict, graph_data.edge_index_dict)

    # 5. Dual-Tower Alignment 통과 (텍스트/그래프 Joint Space 투영)
    data_logger.info("\n[5/6] Projecting to Shared Latent Space...")
    alignment_layer = DualTowerAlignment(
        text_dim=config['models']['dimensions']['plm_out'], 
        graph_dim=config['models']['dimensions']['gat_out'], 
        joint_dim=config['models']['dimensions']['joint_space']
    ).to(device)

    # 추론 전에 학습된 가중치 불러오기 및 eval 모드 전환
    if os.path.exists("./models/saved/alignment_best.pt"):
        data_logger.debug(" -> Loading trained weights for Alignment Layer...")
        alignment_layer.load_state_dict(torch.load("./models/saved/alignment_best.pt", map_location=device))
    alignment_layer.eval()
    
    with torch.no_grad():
        dummy_text = torch.zeros(1, config['models']['dimensions']['plm_out']).to(device)
        
        _, z_graph_table = alignment_layer(dummy_text, gat_embeddings['table'])
        _, z_graph_col = alignment_layer(dummy_text, gat_embeddings['column'])
        
        if 'fk_node' in gat_embeddings and gat_embeddings['fk_node'].shape[0] > 0:
            _, z_graph_fk = alignment_layer(dummy_text, gat_embeddings['fk_node'])
        else:
            z_graph_fk = torch.empty((0, config['models']['dimensions']['joint_space'])).to(device)

    # FAISS에 넘겨줄 최종 딕셔너리 변수 정의
    final_node_embs = {'table': z_graph_table, 'column': z_graph_col}
    final_edge_embs = z_graph_fk

    # 6. FAISS 인덱스 빌드 및 저장
    data_logger.info("\n[6/6] Building FAISS Index and KV Store...")
    faiss_builder = FAISSIndexBuilder(
        vector_dim=config['models']['dimensions']['joint_space'], 
        save_dir=config['paths']['processed_data_dir']
    )
    db_id = os.path.basename(db_path).replace(".sqlite", "") # DB 이름 추출

    faiss_builder.build_and_save(
        node_embs=final_node_embs, 
        edge_embs=final_edge_embs, 
        metadata_mapping=graph_data.metadata_mapping,
        save_name=f"{db_id}_index"
    )

    data_logger.info("\n========== [Offline Pipeline Completed] ==========")
    data_logger.info("System is ready for Online Inference!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Offline Indexing Pipeline")
    parser.add_argument("--db_path", type=str, default="dummy_bird.sqlite", help="Path to the SQLite database")
    args = parser.parse_args()

    try:
        run_offline_pipeline(args.db_path)
    except FileNotFoundError:
        data_logger.error(f"Please provide a valid SQLite database path. (Looking for: {args.db_path})")