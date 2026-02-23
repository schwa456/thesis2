import os
import json
import torch
import asyncio
import argparse
from tqdm import tqdm

from utils.config_loader import load_config
from utils.logger import exp_logger
from models.gat_network import SchemaHeteroGAT
from models.alignment_layer import DualTowerAlignment
from online_inference.retriever import InitialNodeRetriever
from online_inference.pcst_router import PCSTSubgraphRouter
from online_inference.agent_workflow import AdaptiveAgentWorkflow
from models.plm_encoder import PLMEncoder

async def run_evaluation():
    config = load_config()
    paths = config['paths']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    exp_logger.info("========== [BIRD Evaluation Started] ==========")

    # 1. 학습된 모델 로드
    exp_logger.info("1. Model Loading Started")
    exp_logger.info("GAT Model Loading...")
    gat_model = SchemaHeteroGAT(
        in_channels=config['models']['dimensions']['plm_out'], 
        hidden_channels=config['models']['dimensions']['gat_hidden'], 
        out_channels=config['models']['dimensions']['gat_out']
    ).to(device)
    exp_logger.info("GAT Model Loading Completed.")

    exp_logger.info("Alignment Model Loading...")
    alignment_layer = DualTowerAlignment(
        text_dim=config['models']['dimensions']['plm_out'], 
        graph_dim=config['models']['dimensions']['gat_out'], 
        joint_dim=config['models']['dimensions']['joint_space']
    ).to(device)
    exp_logger.info("Alignment Model Loading Completed")

    # 가중치 불러오기
    if os.path.exists("./models/saved/gat_best.pt"):
        gat_model.load_state_dict(torch.load("./models/saved/gat_best.pt", map_location=device))
        alignment_layer.load_state_dict(torch.load("./models/saved/alignment_best.pt", map_location=device))
        exp_logger.info("Successfully loaded trained weights.")

    gat_model.eval()
    alignment_layer.eval()

    exp_logger.info("PLM Encoder Loading...")
    plm_encoder = PLMEncoder(model_name=config['models']['plm_model_name']).to(device)
    plm_encoder.eval()
    exp_logger.info("PLM Encoder Loading Completed.")

    # 2. 평가 데이터 로드 (BIRD dev.json)
    dev_json_path = "/home/hyeonjin/thesis2/data/raw/BIRD_dev/dev.json"
    with open(dev_json_path, 'r') as f:
        dev_data = json.load(f)

    results = []

    # 3. 평가 루프
    for item in tqdm(dev_data, desc="Evaluating BIRD Dev"):
        query = item['question']
        db_id = item['db_id']
        
        exp_logger.debug("===============================================================================")
        exp_logger.debug(f"[{db_id}] Evaluation for Query {query}")
        try:
            # A. 해당 DB의 인덱스 로드 (run_offline.py에서 생성한 결과)
            # db_id별로 인덱스가 저장되어 있어야 함
            db_index_path = os.path.join(paths['processed_data_dir'], f"{db_id}_index.faiss")
            
            if not os.path.exists(db_index_path):
                exp_logger.warning(f"Index for {db_id} not found. Skipping...")
                continue

            retriever = InitialNodeRetriever(db_id=db_id, config=config)
            pcst_router = PCSTSubgraphRouter(base_cost=config['pcst']['base_cost'], alpha=config['pcst']['alpha'], belongs_to_cost=config['pcst']['belongs_to_cost'])
            agent_workflow = AdaptiveAgentWorkflow(model_name=config['llm']['model_name'], api_base=config['llm']['api_base'], api_key=config['llm']['api_key'], uncertainty_threshold=config['agent']['uncertainty_threshold'])

            # B. Retrieval & PCST
            with torch.no_grad():
                exp_logger.debug("[PLM Encoder] Token Embedding by PLM Encoder...")
                token_embs, _ = plm_encoder([query])
                query_emb = token_embs.mean(dim=1) # (1, 384)
                
                # 2) Alignment Layer를 통해 Joint Space(256차원)로 투영
                exp_logger.debug("Projection to Joint Space by Alignment Layer")
                dummy_graph = torch.zeros(1, config['models']['dimensions']['gat_out']).to(device)
                z_query, _ = alignment_layer(query_emb, dummy_graph) # z_query: (1, 256)

            # 3) 변환된 텐서를 Retriever에 전달
            exp_logger.debug("[Retriever] Retrieval Conducting...")
            seed_nodes_info, node_prizes = retriever.retrieve_seed_nodes(z_query, top_k=300)
            exp_logger.debug(f"[Retriever] Seed Nodes: {seed_nodes_info}")
            exp_logger.debug(f"[Retriever] Node Prizes: {node_prizes}")
            
            # 4) PCST 알고리즘 라우팅            
            exp_logger.debug("[PCST] PCST Conducting...")
            selected_nodes_idx, _ = pcst_router.route(
                node_prizes=node_prizes,
                edges=retriever.edges,
                edge_types=retriever.edge_types,
                query_embs=z_query,
                edge_embs_dict=retriever.edge_embs_dict
            )
            exp_logger.debug(f"[PCST] Selected Node Ids: {selected_nodes_idx}")

            # 5) PCST로 선택된 정수 인덱스들을 에이전트가 읽을 수 있는 스키마(Dict)로 변환
            selected_schema_dict = pcst_router.extract_subgraph_schema(
                selected_nodes=selected_nodes_idx, 
                node_id_to_name=retriever.node_metadata
            )
            exp_logger.debug(f"[PCST] Selected Schema Dict: {selected_schema_dict}")

            # C. Multi-Agent Consensus 시작
            final_decision = await agent_workflow.run_workflow(query, selected_schema_dict)
            exp_logger.debug(f"[Agent] Multi-Agent Consensus: {final_decision}")

            # D. 결과 저장 (BIRD 제출 포맷에 맞춤)
            results.append({
                "question_id": item.get("question_id", 0),
                "db_id": db_id,
                "query": item['question'],
                "predicted_nodes": final_decision['final_nodes'],
                "uncertainty": final_decision['uncertainty'],
                "status": final_decision['status']
            })

        except Exception as e:
            exp_logger.exception(f"Error processing query: {query}")

        exp_logger.debug("===============================================================================")

    # 4. 최종 결과 저장
    output_path = "./output/predictions.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    exp_logger.info(f"Evaluation finished. Results saved to {output_path}")

if __name__ == "__main__":
    asyncio.run(run_evaluation())