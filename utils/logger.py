import logging
import os
from datetime import datetime, timedelta, timezone

def setup_logger(logger_name: str, file_prefix: str, log_dir: str = "./logs"):
    """
    용도별로 분리된 로거를 생성하고, 터미널과 파일에 동시에 기록합니다.
    에러(ERROR) 레벨 이상의 로그는 별도의 error.log 파일에도 자동으로 수집됩니다.
    """
    os.makedirs(f"{log_dir}/{file_prefix}", exist_ok=True)
    
    # 시간 기반의 고유 로그 파일명 생성 (실험 단위로 묶기 위해 일/시/분까지만 사용)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    
    # 일반 로그 파일과 에러 전용 로그 파일 경로
    log_file = os.path.join(log_dir, f"{file_prefix}/{file_prefix}_{timestamp}.log")
    error_file = os.path.join(log_dir, f"error/error_{timestamp}.log")
    
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)

    def kst_converter(*args):
        utc_dt = datetime.now(timezone.utc)
        kst_dt = utc_dt + timedelta(hours=9)
        return kst_dt.timetuple()

    # 3. Format 설정
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    formatter.converter = kst_converter
    
    # 중복 핸들러 방지
    if not logger.handlers:
        # 1. 파일 핸들러 (상세 기록용)
        file_handler = logging.FileHandler(log_file, encoding='utf-8', delay=True)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        
        # 2. 콘솔 핸들러 (실시간 모니터링용, Info 이상만)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_format = logging.Formatter('[%(levelname)s] %(message)s')
        console_handler.setFormatter(console_format)
        
        # 3. 에러 전용 파일 핸들러 (ERROR 이상만 수집, 모든 로거 공통)
        error_handler = logging.FileHandler(error_file, encoding='utf-8', delay=True)
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        logger.addHandler(error_handler)
        
    return logger

# =====================================================================
# 미리 정의된 로거 인스턴스 (다른 파일에서 import 해서 바로 사용 가능)
# =====================================================================
train_logger = setup_logger("Train", "train")
data_logger  = setup_logger("DataBuild", "data_build")
exp_logger   = setup_logger("OnlineEval", "online_eval") # 기존 코드 유지
agent_logger = setup_logger("Agnet", "agent")