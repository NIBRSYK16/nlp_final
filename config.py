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
    "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",  # base_url，不包含 /chat/completions
    "api_key": "sk-1d1d9ecf1f1b446588871b3e6d5d3a30", #_client["api_key"],
    "region": "beijing",  # 可选 "beijing" 或 "singapore"
    "qwen_32b_api_url": "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
    "qwen_14b_api_url": "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
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

# ====== 评估配置 ======
EVALUATION_CONFIG = {
    "default_max_tasks": 10,
    "default_max_tokens": 512,
    "default_temperature": 0.7,
    "default_top_p": 0.9,
    "dataset_path": "./datasets/human-eval-v2-20210705.jsonl",
}

# ====== 微调配置 ======
FINE_TUNE_CONFIG = {
    "default_output_dir": "./fine_tuned_model",
    "default_num_epochs": 3,
    "default_batch_size": 1,
    "default_learning_rate": 5e-5,
}

# 创建必要目录
for directory in [TRAINING_DATA_DIR, CHECKPOINT_DIR]:
    os.makedirs(directory, exist_ok=True)