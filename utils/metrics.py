from typing import List, Set

class EvaluatorMetrics:
    """
    Text-to-SQL 파이프라인의 성능을 학술적으로 평가하기 위한 지표들을 계산합니다.
    """
    
    @staticmethod
    def schema_linking_f1(predicted_nodes: List[str], ground_truth_nodes: List[str]) -> dict:
        """
        초기 노드 탐색 및 에이전트 합의를 거친 최종 스키마 노드들의 정확도를 평가합니다.
        Precision, Recall, F1-Score를 반환합니다.
        """
        pred_set = set(predicted_nodes)
        gt_set = set(ground_truth_nodes)
        
        true_positives = len(pred_set.intersection(gt_set))
        false_positives = len(pred_set - gt_set)
        false_negatives = len(gt_set - pred_set)
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        
        # F1 계산식: $F1 = 2 * (Precision * Recall) / (Precision + Recall)$
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            "Precision": round(precision, 4),
            "Recall": round(recall, 4),
            "F1_Score": round(f1_score, 4)
        }

    @staticmethod
    def rejection_metrics(predictions: List[str], ground_truths: List[str]) -> dict:
        """
        Unanswerable(거절해야 할 질문)을 얼마나 잘 걸러냈는지 혼동 행렬(Confusion Matrix) 기반으로 평가합니다.
        - predictions: 모델의 예측 상태 리스트 (e.g., ["Answerable", "Unanswerable", ...])
        - ground_truths: 실제 정답 상태 리스트
        """
        TP = sum(1 for p, g in zip(predictions, ground_truths) if p == "Unanswerable" and g == "Unanswerable")
        TN = sum(1 for p, g in zip(predictions, ground_truths) if p == "Answerable" and g == "Answerable")
        FP = sum(1 for p, g in zip(predictions, ground_truths) if p == "Unanswerable" and g == "Answerable")
        FN = sum(1 for p, g in zip(predictions, ground_truths) if p == "Answerable" and g == "Unanswerable")
        
        total = len(predictions)
        
        # 거절해야 할 것을 정확히 거절한 비율 (True Negative Rate / Specificity)
        rejection_accuracy = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        # 대답해야 할 것을 잘못 거절한 비율 (False Omission Rate 방어)
        over_rejection_rate = FP / (FP + TN) if (FP + TN) > 0 else 0.0
        
        return {
            "Total_Queries": total,
            "Correct_Rejections (TP)": TP,
            "False_Rejections (FP)": FP,
            "Rejection_Accuracy": round(rejection_accuracy, 4),
            "Over_Rejection_Rate": round(over_rejection_rate, 4)
        }