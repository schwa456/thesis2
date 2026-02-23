import os
import yaml
from typing import Dict, Any
from utils.logger import exp_logger

def load_config(config_path: str = "config/exp_config.yaml") -> Dict[str, Any]:
    if not os.path.isabs(config_path):
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(root_dir, config_path)

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration File Not Found At: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        try:
            config = yaml.safe_load(f)
            exp_logger.info(f"✅ Successfully loaded configuration from {config_path}")
            return config
        except yaml.YAMLError as exc:
            exp_logger.error(f"🚨 Error parsing YAML file: {exc}")
            raise