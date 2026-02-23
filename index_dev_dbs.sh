#!/bin/bash

# 1. 경로 설정 (연구자님의 환경에 맞게 확인 완료)
DEV_DB_ROOT="/home/hyeonjin/thesis2/data/raw/BIRD_dev/dev_databases"
LOG_FILE="./logs/dev_indexing.log"

mkdir -p ./logs

echo "=========================================================="
echo "🚀 Starting Offline Indexing for 11 Dev Databases"
echo "Using trained weights from ./models/saved/"
echo "=========================================================="

# 2. 11개 DB 폴더를 순회
for db_id in $(ls $DEV_DB_ROOT); do
    # DB 파일 경로 구성 (예: .../california_schools/california_schools.sqlite)
    DB_PATH="$DEV_DB_ROOT/$db_id/$db_id.sqlite"
    
    if [ -f "$DB_PATH" ]; then
        echo " -> Processing Database: [$db_id]"
        
        # run_offline.py 실행 (학습된 가중치를 로드하여 FAISS 인덱스 생성)
        python run_offline.py --db_path "$DB_PATH" >> "$LOG_FILE" 2>&1
        
        if [ $? -eq 0 ]; then
            echo " ✅ [$db_id] Indexing Complete."
        else
            echo " ❌ [$db_id] Indexing Failed. Check $LOG_FILE"
        fi
    else
        echo " ⚠️  Skip: $db_id (SQLite file not found at $DB_PATH)"
    fi
done

echo "=========================================================="
echo "🎉 All 11 Dev Databases have been indexed!"
echo "Check ./data/processed/ for .faiss and .pkl files."
echo "=========================================================="