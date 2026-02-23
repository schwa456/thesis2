import pandas as pd
import json
import sqlglot
import sqlglot.expressions as exp
from utils.metrics import EvaluatorMetrics

def parse_ground_truth_sql(sql: str) -> list:
    """정답 SQL을 파싱하여 테이블/컬럼 노드를 추출합니다."""
    if pd.isna(sql) or not str(sql).strip():
        return []
    try:
        parsed = sqlglot.parse_one(sql, read='sqlite')
        alias_to_table = {}
        for table in parsed.find_all(exp.Table):
            table_name = table.name.lower()
            alias = table.alias.lower() if table.alias else table_name
            alias_to_table[alias] = table_name

        nodes = set()
        for t_name in alias_to_table.values():
            nodes.add(t_name)
            
        for column in parsed.find_all(exp.Column):
            col_name = column.name.lower()
            col_alias = column.table.lower() if column.table else None
            
            if col_alias and col_alias in alias_to_table:
                actual_table = alias_to_table[col_alias]
                nodes.add(f"{actual_table}.{col_name}")
            elif len(alias_to_table) == 1:
                actual_table = list(alias_to_table.values())[0]
                nodes.add(f"{actual_table}.{col_name}")
        return list(nodes)
    except Exception:
        return []

def evaluate_pipeline(df_merged: pd.DataFrame) -> pd.DataFrame:
    """합쳐진 데이터프레임을 기반으로 Metrics를 계산합니다."""
    # 1. Rejection Metrics
    if 'gt_status' not in df_merged.columns:
        df_merged['gt_status'] = 'Answerable' 
        
    predictions = df_merged['status'].tolist()
    ground_truths = df_merged['gt_status'].tolist()
    rejection_results = EvaluatorMetrics.rejection_metrics(predictions, ground_truths)
    
    print("\n========== [ Rejection Metrics ] ==========")
    for k, v in rejection_results.items():
        print(f"{k}: {v}")
    
    # 2. Schema Linking F1
    precisions, recalls, f1_scores = [], [], []
    for idx, row in df_merged.iterrows():
        preds = row.get('predicted_nodes', [])
        # LLM이 이상하게 뱉은 값 방어 (Dict 등)
        if isinstance(preds, dict) and 'Unanswerable' in preds:
            preds = []
        elif not isinstance(preds, list):
            preds = []
            
        preds = [str(p).lower() for p in preds]
        gts = row.get('ground_truth_nodes', [])
        
        metrics = EvaluatorMetrics.schema_linking_f1(preds, gts)
        precisions.append(metrics['Precision'])
        recalls.append(metrics['Recall'])
        f1_scores.append(metrics['F1_Score'])
        
    df_merged['Precision'] = precisions
    df_merged['Recall'] = recalls
    df_merged['F1_Score'] = f1_scores
    
    macro_precision = sum(precisions) / len(precisions) if precisions else 0.0
    macro_recall = sum(recalls) / len(recalls) if recalls else 0.0
    macro_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
    
    print("\n========== [ Schema Linking Metrics (Macro Avg) ] ==========")
    print(f"Average Precision : {macro_precision:.4f}")
    print(f"Average Recall    : {macro_recall:.4f}")
    print(f"Average F1-Score  : {macro_f1:.4f}")
    
    return df_merged

def main():
    print("Loading data...")
    # 1. 원본 BIRD 정답지 로드
    dev_path = "/home/hyeonjin/thesis2/data/raw/BIRD_dev/dev.json" 
    with open(dev_path, 'r', encoding='utf-8') as f:
        df_dev = pd.DataFrame(json.load(f))
        
    # 2. run_online.py 에서 생성한 예측 결과 로드
    pred_path = "/home/hyeonjin/thesis2/output/predictions.json"
    with open(pred_path, 'r', encoding='utf-8') as f:
        df_pred = pd.DataFrame(json.load(f))

    print("Merging dataframes...")
    # 3. 데이터프레임 병합 (question_id 기준)
    df_merged = pd.merge(df_dev, df_pred, on='question_id', how='left')
    
    # 중복되는 db_id 정리
    if 'db_id_y' in df_merged.columns:
        df_merged = df_merged.drop(columns=['db_id_y']).rename(columns={'db_id_x': 'db_id'})

    print("Parsing Ground Truth SQL...")
    # 4. 정답 파싱
    df_merged['ground_truth_nodes'] = df_merged['SQL'].apply(parse_ground_truth_sql)

    print("Evaluating metrics...")
    # 5. 성능 평가 실행
    df_final = evaluate_pipeline(df_merged)

    # 6. 최종 결과를 프로젝트 루트의 output.csv 로 저장
    df_final.to_csv("./output/output.csv", index=False, encoding='utf-8-sig')
    print("\n✅ Evaluation complete! Results saved to output.csv")

if __name__ == "__main__":
    main()