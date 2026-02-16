import asyncio
import torch
import pickle
from online_inference.query_processor import QueryProcessor
from models.plm_encoder import PLMEncoder
from models.alignment_layer import DualTowerAlignment
from online_inference.retriever import InitialNodeRetriever
from online_inference.pcst_router import PCSTSubgraphRouter
from online_inference.agent_workflow import AdaptiveAgentWorkflow

async def run_online_pipeline(nl_query: str):
    print(f"========== [Online Inference Started] ==========")
    print(f"User Query: '{nl_query}'\n")

    # 1. 자연어 질의 전처리 및 토큰 임베딩 추출
    print("[1/5] Extracting Keywords & Token Embeddings...")
    processor = QueryProcessor()
    keywords = processor.extract_keywords(nl_query)
    
    plm_encoder = PLMEncoder()
    # 전체 쿼리를 넣고 (1, Seq_Len, 384) 추출
    token_embs, tokens = plm_encoder([nl_query]) 
    token_embs = token_embs.squeeze(0) # (Seq_Len, 384)
    tokens = tokens[0]
    
    # 불용어 마스킹
    masked_embs, valid_tokens = processor.mask_embeddings(token_embs, tokens)
    print(f"  -> Masked Tokens: {valid_tokens}")

    # 2. Alignment Layer 통과 (Joint Space 투영)
    print("[2/5] Projecting Query to Joint Space...")
    alignment_layer = DualTowerAlignment(text_dim=384, graph_dim=256, joint_dim=256)
    alignment_layer.eval()
    
    with torch.no_grad():
        # z_graph는 여기선 필요 없으므로 더미 텐서 전달
        z_query_tokens, _ = alignment_layer(masked_embs, torch.zeros(1, 256))

    # 3. FAISS Retrieval (초기 시드 노드 및 Prize 계산)
    print("[3/5] Retrieving Initial Seed Nodes (MaxSim)...")
    retriever = InitialNodeRetriever(threshold=0.6)
    seed_info, node_prizes = retriever.retrieve_seed_nodes(z_query_tokens, top_k=3)
    
    print(f"  -> Found {len(seed_info)} relevant nodes above threshold.")

    # 4. PCST 라우팅 (동적 엣지 비용 기반 서브그래프 추출)
    print("[4/5] Running PCST Algorithm for Sub-graph Routing...")
    # 오프라인 단계에서 저장해둔 위상 정보(Topology) 로드
    with open("./data/processed/topology_info.pkl", "rb") as f:
        topology_info = pickle.load(f)
    
    # (데모를 위한 가상의 엣지 리스트 - 실제론 topology_info에서 가져와야 함)
    mock_edges = [(0, 1), (0, 2), (2, 3)] 
    mock_edge_types = ['belongs_to', 'belongs_to', 'pk_fk']
    
    # Retriever의 node_metadata 활용 (id -> name 변환용)
    id_to_name = retriever.node_metadata
    
    pcst_router = PCSTSubgraphRouter(base_cost=1.0, alpha=0.5)
    sel_nodes, sel_edges = pcst_router.route(
        node_prizes=node_prizes,
        edges=mock_edges,
        edge_types=mock_edge_types,
        query_embs=z_query_tokens, # 동적 비용 할인을 위한 Query Vector
        edge_embs_dict=retriever.edge_embs_dict
    )
    
    subgraph = pcst_router.extract_subgraph_schema(sel_nodes, id_to_name)
    print(f"  -> Condensed Sub-graph: {list(subgraph.keys())}")

    # 5. 다중 에이전트 합의 및 불확실성 산출
    print("\n[5/5] Launching Adaptive Agentic Workflow...")
    agent_workflow = AdaptiveAgentWorkflow(uncertainty_threshold=0.3)
    
    # 비동기로 LLM 호출 실행
    final_result = await agent_workflow.run_workflow(nl_query, subgraph)

    print("\n========== [Final Decision] ==========")
    print(f"Status: {final_result['status']}")
    print(f"Uncertainty Score: {final_result['uncertainty']:.2f}")
    if final_result['status'] == "Answerable":
        print(f"Final Schema Nodes: {final_result['final_nodes']}")
    print(f"Reasoning: {final_result['reasoning']}")

if __name__ == "__main__":
    test_query = "List the names of employees in the IT department who have a salary greater than 50000."
    asyncio.run(run_online_pipeline(test_query))