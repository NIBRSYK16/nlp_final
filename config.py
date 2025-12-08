"""
配置文件 - 存储所有配置参数
"""
import os
from typing import Dict, List

# ====== 路径配置 ======
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = "./models/Qwen2.5-Coder-0.5B-Instruct"
TRAINING_DATA_DIR = "./evolution_training_data"
CHECKPOINT_DIR = "./model_checkpoints"

# ====== API配置 ======
API_CONFIG = {
    "qwen_70b_api_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "qwen_14b_api_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "api_key": "sk-1d1d9ecf1f1b446588871b3e6d5d3a30",
}

# ====== 自我演化配置 ======
EVOLUTION_CONFIG = {
    "enable_self_evolution": True,
    "evolution_keywords": ["自我演化", "自我进化", "self-evolve", "self-evolution"],
    "max_fine_tune_steps": 100,
    "learning_rate": 5e-5,
    "evolution_batch_size": 3,
}

# ====== 生成配置 ======
GENERATION_CONFIG = {
    "default_max_tokens": 512,
    "default_temperature": 0.7,
    "default_top_p": 0.9,
    "default_system_prompt": "你是一个专业的编程助手，擅长编写和解释代码。",
}

# ====== 验证配置 ======
VALIDATION_CONFIG = {
    "max_retries": 3,
    "timeout_seconds": 60,
    "test_timeout": 5,
}

# 创建必要目录
for directory in [TRAINING_DATA_DIR, CHECKPOINT_DIR]:
    os.makedirs(directory, exist_ok=True)