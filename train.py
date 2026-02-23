import os
import json
import torch
import torch.optim as optim
import wandb
import pickle
import math
import re
from collections import defaultdict
from tqdm import tqdm

from utils.config_loader import load_config
from models.plm_encoder import PLMEncoder
from models.gat_network import SchemaHeteroGAT
from models.alignment_layer import DualTowerAlignment
from offline_indexing.schema_parser import SQLiteSchemaParser
from offline_indexing.llm_verbalizer import SchemaVerbalizer
from offline_indexing.graph_builder import SchemaGraphBuilder
from utils.logger import train_logger

# ==========================================
# 1. SQL에서 Ground Truth Node 추출 헬퍼 함수
# ==========================================
def extract_gt_nodes_from_sql(sql: str, metadata_mapping: dict) -> list:
    gt_nodes = []
    sql_lower = sql.lower()
    words = set(re.findall(r'\b\w+\b', sql_lower))

    for tbl in metadata_mapping['table_to_id'].keys():
        if tbl.lower() in words:
            gt_nodes.append(tbl)
    
    for col_full in metadata_mapping['col_to_id'].keys():
        tbl, col = col_full.split('.')
        if col.lower() in words:
            gt_nodes.append(col_full)

    if not gt_nodes and metadata_mapping['table_to_id']:
        gt_nodes.append(list(metadata_mapping['table_to_id'].keys())[0])
    
    return list(set(gt_nodes))


# ==========================================
# 2. Main Training Loop
# ==========================================

def train():
    # --- 0. 설정 및 wandb 초기화 ---
    config = load_config()
    train_cfg = config.get('training', {})
    paths = config.get('paths', {})

    wandb.init(
        project=train_cfg.get('project_name', 'Text-to-SQL-Alignment'),
        config=config,
        name="BIRD-Joint-Training"
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_logger.info(f"🚀 Training started on device: {device}")

    # --- 1. 데이터 로드 및 db_id 별 그룹화 ---
    with open(paths['train_json'], 'r', encoding='utf-8') as f:
        train_data = json.load(f)
        
    # db_id 기준으로 질문들을 묶음: { "california_schools": [ (질문, SQL), ... ], ... }
    db_groups = defaultdict(list)
    for item in train_data:
        # BIRD 데이터셋의 키 구조 반영
        q = item.get('question', '')
        sql = item.get('SQL', item.get('query', ''))
        db_id = item.get('db_id', '')
        if q and sql and db_id:
            db_groups[db_id].append((q, sql))
            
    train_logger.info(f"Loaded {len(train_data)} queries across {len(db_groups)} databases.")

    # --- 2. 모델 및 모듈 초기화 ---
    plm_encoder = PLMEncoder(model_name=config['models']['plm_model_name']).to(device)
    gat_model = SchemaHeteroGAT(
        in_channels=config['models']['dimensions']['plm_out'], 
        hidden_channels=config['models']['dimensions']['gat_hidden'], 
        out_channels=config['models']['dimensions']['gat_out'],
        num_layers=config['models']['gat_params']['num_layers'],
        heads=config['models']['gat_params']['heads']
    ).to(device)
    
    alignment_layer = DualTowerAlignment(
        text_dim=config['models']['dimensions']['plm_out'], 
        graph_dim=config['models']['dimensions']['gat_out'], 
        joint_dim=config['models']['dimensions']['joint_space']
    ).to(device)

    optimizer = optim.AdamW(
        list(gat_model.parameters()) + list(alignment_layer.parameters()),
        lr=float(train_cfg.get('learning_rate', 1e-4)),
        weight_decay=float(train_cfg.get('weight_decay', 1e-5))
    )

    # 그래프 생성을 위한 헬퍼 클래스들 (vLLM 서버가 켜져 있어야 함)
    verbalizer = SchemaVerbalizer(
        model_name=config['llm']['model_name'],
        api_base=config['llm']['api_base'],
        api_key=config['llm']['api_key']
    )
    graph_builder = SchemaGraphBuilder(plm_model_name=config['models']['plm_model_name'])
    
    os.makedirs(paths['processed_data_dir'], exist_ok=True)

    # --- 3. 학습 루프 ---
    epochs = train_cfg.get('num_epochs', 50)
    save_dir = train_cfg.get('save_dir', './models/saved')
    os.makedirs(save_dir, exist_ok=True)
    batch_size = train_cfg.get('batch_size', 16)

    gat_model.train()
    alignment_layer.train()

    best_loss = float('inf')

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        batch_count = 0
        
        # DB(db_id) 단위로 순회
        pbar_db = tqdm(db_groups.items(), desc=f"Epoch {epoch}/{epochs}")
        for db_id, q_sql_pairs in pbar_db:
            
            # 3.1. 동적 그래프 로드 및 캐싱
            cache_path = os.path.join(paths['processed_data_dir'], f"{db_id}_graph.pkl")
            if os.path.exists(cache_path):
                with open(cache_path, 'rb') as f:
                    graph_data, metadata_mapping = pickle.load(f)
            else:
                pbar_db.set_postfix_str(f"Building graph for {db_id}...")
                # BIRD 구조: train_databases/db_id/db_id.sqlite
                db_path = os.path.join(paths['train_db_dir'], db_id, f"{db_id}.sqlite")
                if not os.path.exists(db_path):
                    continue
                
                parser = SQLiteSchemaParser(db_path)
                schema_info = parser.parse_schema()
                fk_desc = verbalizer.process_all_fks(schema_info)
                graph_data = graph_builder.build_graph(schema_info, fk_desc)
                metadata_mapping = graph_data.metadata_mapping
                
                with open(cache_path, 'wb') as f:
                    pickle.dump((graph_data, metadata_mapping), f)
            
            # 그래프를 GPU로 이동
            graph_data = graph_data.to(device)
            
            # Inference Tensor 속성을 끊어내고 일반 텐서로 변환
            for node_type in graph_data.node_types:
                if 'x' in graph_data[node_type]:
                    graph_data[node_type].x = graph_data[node_type].x.clone().detach().requires_grad_(True)
            
            # 3.2. 해당 DB의 질문들을 Batch 단위로 학습
            for i in range(0, len(q_sql_pairs), batch_size):
                batch_pairs = q_sql_pairs[i:i+batch_size]
                queries = [p[0] for p in batch_pairs]
                sqls = [p[1] for p in batch_pairs]
                
                optimizer.zero_grad()

                # GAT 통과 (현재 DB에 대해 1번만 연산하여 재사용)
                gat_embs = gat_model(graph_data.x_dict, graph_data.edge_index_dict)

                # Text Encoding
                token_embs, _ = plm_encoder(queries) 
                query_embs = token_embs.mean(dim=1) # (Batch, 384)

                # Ground Truth Graph Embeddings 수집
                target_graph_embs = []
                for sql in sqls:
                    gt_nodes = extract_gt_nodes_from_sql(sql, metadata_mapping)
                    
                    # 해당 질문의 정답 노드들의 벡터 평균을 구함 (다중 정답 지원)
                    node_vecs = []
                    for name in gt_nodes:
                        if name in metadata_mapping['col_to_id']:
                            idx = metadata_mapping['col_to_id'][name]
                            node_vecs.append(gat_embs['column'][idx])
                        elif name in metadata_mapping['table_to_id']:
                            idx = metadata_mapping['table_to_id'][name]
                            node_vecs.append(gat_embs['table'][idx])
                            
                    if node_vecs:
                        target_graph_embs.append(torch.stack(node_vecs).mean(dim=0))
                    else:
                        target_graph_embs.append(torch.zeros(config['models']['dimensions']['gat_out']).to(device))
                        
                target_graph_embs = torch.stack(target_graph_embs) # (Batch, 256)

                # Alignment Layer 투영 및 InfoNCE Loss 계산
                z_query, z_graph = alignment_layer(query_embs, target_graph_embs)
                loss = alignment_layer.compute_contrastive_loss(z_query, z_graph)

                # 역전파
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                batch_count += 1
                
            pbar_db.set_postfix({"loss": f"{loss.item():.4f}"})

        # Epoch 종료 후 Wandb 로깅 및 모델 저장
        if batch_count > 0:
            avg_loss = total_loss / batch_count
            wandb.log({
                "epoch": epoch, 
                "train_loss": avg_loss,
                "temperature_scale": alignment_layer.logit_scale.exp().item()
            })

            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(gat_model.state_dict(), os.path.join(save_dir, "gat_best.pt"))
                torch.save(alignment_layer.state_dict(), os.path.join(save_dir, "alignment_best.pt"))
                train_logger.info(f"🌟 Best model saved with loss: {best_loss:.4f} at epoch {epoch}")

    train_logger.info("🎉 BIRD Training completed!")
    wandb.finish()

if __name__ == "__main__":
    train()