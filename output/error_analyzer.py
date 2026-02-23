import pandas as pd
import ast
import numpy as np

def safe_literal_eval(val):
    try:
        return ast.literal_eval(val) if pd.notna(val) else []
    except (ValueError, SyntaxError):
        return []

def analyze_pipeline_errors(csv_path="output.csv"):
    print("Loading and parsing output data...")
    df = pd.read_csv(csv_path)
    
    # 1. 문자열로 저장된 리스트 데이터를 실제 Python 리스트로 변환
    df['predicted_nodes'] = df['predicted_nodes'].apply(safe_literal_eval)
    df['ground_truth_nodes'] = df['ground_truth_nodes'].apply(safe_literal_eval)
    
    # 2. 에러 유형 분류 (Error Categorization)
    conditions = [
        (df['status'] == 'Unanswerable') & (df['gt_status'] == 'Answerable'), # False Rejection (쫄보 에이전트)
        (df['status'] == 'Answerable') & (df['F1_Score'] == 1.0),             # Exact Match (완벽)
        (df['status'] == 'Answerable') & (df['F1_Score'] > 0) & (df['F1_Score'] < 1.0), # Partial Match (PCST 가지치기 과도함 의심)
        (df['status'] == 'Answerable') & (df['F1_Score'] == 0.0)              # Total Failure (Retriever 아예 빗나감)
    ]
    choices = ['False Rejection', 'Exact Match', 'Partial Match', 'Total Failure']
    df['error_type'] = np.select(conditions, choices, default='Other')

    # 3. 전반적인 요약 통계
    print("\n" + "="*50)
    print("📈 Pipeline Performance Summary")
    print("="*50)
    print(f"Total Queries: {len(df)}")
    print(f"Average Precision: {df['Precision'].mean():.4f}")
    print(f"Average Recall: {df['Recall'].mean():.4f}")
    print(f"Average F1-Score: {df['F1_Score'].mean():.4f}")
    
    print("\n" + "="*50)
    print("🔍 Error Type Distribution")
    print("="*50)
    error_counts = df['error_type'].value_counts()
    for err_type, count in error_counts.items():
        print(f"- {err_type}: {count} ({count/len(df)*100:.1f}%)")

    # 4. Agent Uncertainty 분석 (Threshold 조정을 위한 인사이트)
    print("\n" + "="*50)
    print("🤖 Agent Uncertainty Analysis (Threshold: 0.6?)")
    print("="*50)
    uncertainty_stats = df.groupby('error_type')['uncertainty'].agg(['mean', 'median', 'max', 'min'])
    print(uncertainty_stats)
    
    # 5. Partial Match 깊은 분석 (Recall이 왜 낮을까?)
    partial_df = df[df['error_type'] == 'Partial Match'].copy()
    if not partial_df.empty:
        # 놓친 노드(False Negatives)와 잘못 잡은 노드(False Positives) 계산
        partial_df['missed_nodes'] = partial_df.apply(
            lambda row: list(set(row['ground_truth_nodes']) - set(row['predicted_nodes'])), axis=1
        )
        partial_df['hallucinated_nodes'] = partial_df.apply(
            lambda row: list(set(row['predicted_nodes']) - set(row['ground_truth_nodes'])), axis=1
        )
        
        print("\n" + "="*50)
        print("⚠️ Partial Match Insights (Low Recall Issue)")
        print("="*50)
        print(f"Average missed nodes per partial match: {partial_df['missed_nodes'].apply(len).mean():.2f}")
        print("Saving partial matches and missed nodes to 'partial_matches_analysis.csv' for qualitative review...")
        partial_df[['question_id', 'question', 'missed_nodes', 'hallucinated_nodes', 'Precision', 'Recall']].to_csv('partial_matches_analysis.csv', index=False)

    # 6. False Rejection 케이스 추출 (정성 평가용)
    false_rejections = df[df['error_type'] == 'False Rejection']
    if not false_rejections.empty:
        print(f"\nSaving {len(false_rejections)} False Rejection cases to 'false_rejections.csv'...")
        false_rejections.to_csv('false_rejections.csv', index=False)

if __name__ == "__main__":
    analyze_pipeline_errors("./output.csv")