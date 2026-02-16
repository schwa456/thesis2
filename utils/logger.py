import logging
import os
from datetime import datetime

def setup_experiment_logger(log_dir: str = "./logs", experiment_name: str = "text2sql_eval"):
    """
    실험 결과를 터미널과 파일에 동시에 기록하는 커스텀 로거를 설정합니다.
    """
    os.makedirs(log_dir, exist_ok=True)
    
    # 시간 기반의 고유 로그 파일명 생성
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{experiment_name}_{timestamp}.log")
    
    logger = logging.getLogger("Text2SQL_Agent")
    logger.setLevel(logging.DEBUG)
    
    # 중복 핸들러 방지
    if not logger.handlers:
        # 1. 파일 핸들러 (상세 기록용)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter('%(asctime)s - [%(levelname)s] - %(message)s')
        file_handler.setFormatter(file_format)
        
        # 2. 콘솔 핸들러 (실시간 모니터링용, Info 이상만)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_format = logging.Formatter('[%(levelname)s] %(message)s')
        console_handler.setFormatter(console_format)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
    return logger

# 초기화 (다른 모듈에서 from utils.logger import exp_logger 로 바로 사용 가능)
exp_logger = setup_experiment_logger()