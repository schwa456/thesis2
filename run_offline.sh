#!/bin/bash

# ==============================================================================
# Text-to-SQL Architecture Training & Evaluation Pipeline
# ==============================================================================

# 1. 에러 발생 시 즉시 실행 중단
set -e

cleanup(){
    echo "=========================================================="
    echo "[Cleanup] Shutting down vLLM Server (PID: $VLLM_PID) safely..."
    kill $VLLM_PID 2>/dev/null || true
    echo "✅ Cleanup completed."
    echo "=========================================================="
}
trap cleanup EXIT

# 2. Wandb API Key 설정 (본인의 키로 변경하거나 터미널에 미리 설정하세요)
export WANDB_API_KEY="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"

echo "=========================================================="
echo "🚀 Starting Text-to-SQL Pipeline"
echo "=========================================================="

# ------------------------------------------------------------------------------
# Phase 1: vLLM Server Background 실행 (Verbalizer 및 에이전트 용도)
# ------------------------------------------------------------------------------
mkdir -p ./logs
TIMESTAMP=$(date +"%Y%m%d_%H%M")

echo "[Phase 1] Starting vLLM Server in background..."
export CUDA_VISIBLE_DEVICES=0,1

nohup python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --port 8000 \
    --max-model-len 8192 \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.8 > "./logs/server/vllm_server_${TIMESTAMP}.log" 2>&1 &
VLLM_PID=$!

echo "Waiting for vLLM server to start..."
sleep 60

# 이후 파이프라인(train.py 등)은 GPU 3번을 쓰도록 변경
export CUDA_VISIBLE_DEVICES=2,3

# ------------------------------------------------------------------------------
# Phase 2: Offline Graph Construction (토폴로지 데이터 생성)
# ------------------------------------------------------------------------------
echo "[Phase 2] Running Offline Indexing (Graph Construction)..."
# DB 파일 경로를 인자로 전달 (예: BIRD dev set)
python run_offline.py --db_path ./dummy_bird.sqlite

# ------------------------------------------------------------------------------
# Phase 3: Model Training (GAT + Dual-Tower Alignment)
# ------------------------------------------------------------------------------
echo "[Phase 3] Starting Model Training..."
python train.py

# ------------------------------------------------------------------------------
# Phase 4: Re-build FAISS Index with Trained Weights
# ------------------------------------------------------------------------------
# 주의: run_offline.py 내부에 학습된 가중치(gat_best.pt 등)를 불러오는 코드가 
# 추가되어야 진정한 효과를 발휘합니다.
echo "[Phase 4] Re-building FAISS Index with trained weights..."
python run_offline.py --db_path ./dummy_bird.sqlite

# ------------------------------------------------------------------------------
# Phase 5: Clean up
# ------------------------------------------------------------------------------
echo "[Phase 5] Shutting down vLLM Server (PID: $VLLM_PID)..."
kill $VLLM_PID

echo "=========================================================="
echo "✅ Pipeline completely successfully!"
echo "=========================================================="