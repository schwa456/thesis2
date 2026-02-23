import faiss
import pickle
import numpy as np
from collections import defaultdict

def inspect_all_embeddings(db_id="california_schools", processed_dir="/home/hyeonjin/thesis2/data/processed"):
    index_path = f"{processed_dir}/{db_id}_index.faiss"
    meta_path = f"{processed_dir}/{db_id}_index_metadata.pkl"

    try:
        index = faiss.read_index(index_path)
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
    except Exception as e:
        print(f"Error loading files: {e}")
        return

    node_metadata = meta["node_metadata"]
    
    print(f"🔍 Inspecting ALL Vectors for DB: [{db_id}]")
    print("=" * 80)
    
    # 테이블별로 노드 정보를 모아둘 딕셔너리
    grouped_nodes = defaultdict(list)
    
    for i in range(index.ntotal):
        name = node_metadata.get(i, "Unknown")
        vec = index.reconstruct(i) 
        
        vec_preview = ", ".join([f"{x:.4f}" for x in vec[:5]])
        vec_sum = vec.sum()
        
        # 이름에 '.'이 있으면 컬럼이므로 앞부분(테이블명)만 추출, 아니면 그 자체가 테이블명
        table_name = name.split(".")[0] if "." in name else name
        
        grouped_nodes[table_name].append((i, name, vec_preview, vec_sum))

    # 테이블 이름 알파벳 순으로 정렬하여 출력
    for table_name in sorted(grouped_nodes.keys()):
        print(f"\n📁 Table: {table_name}")
        print("-" * 80)
        # 테이블 내에서도 이름 순으로 정렬
        for node_id, name, preview, v_sum in sorted(grouped_nodes[table_name], key=lambda x: x[1]):
            print(f"[{node_id:3d}] {name:<25} | Preview: [{preview}...] | Sum: {v_sum:.6f}")

    print("=" * 80)
    print(f"Total nodes inspected: {index.ntotal}")

if __name__ == "__main__":
    inspect_all_embeddings(db_id="california_schools")