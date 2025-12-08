"""
训练数据处理模块
"""
import os
import json
from datetime import datetime
from typing import Dict, List
from config import TRAINING_DATA_DIR
from utils.text_utils import classify_problem


def save_training_example(instruct: str, code: str, validation_result: str) -> str:
    """
    保存训练数据到文件
    """
    os.makedirs(TRAINING_DATA_DIR, exist_ok=True)
    
    training_example = {
        "instruction": instruct,
        "code": code,
        "validation_result": validation_result,
        "timestamp": datetime.now().isoformat(),
        "metadata": {
            "source": "self_evolution",
            "problem_type": classify_problem(instruct)
        }
    }
    
    # 生成文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    problem_hash = hash(instruct) % 10000
    filename = f"{TRAINING_DATA_DIR}/example_{timestamp}_{problem_hash}.json"
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(training_example, f, ensure_ascii=False, indent=2)
    
    return filename


def list_training_data(limit: int = 20) -> str:
    """列出训练数据"""
    if not os.path.exists(TRAINING_DATA_DIR):
        return "暂无训练数据"
    
    files = [f for f in os.listdir(TRAINING_DATA_DIR) if f.endswith('.json')]
    if not files:
        return "暂无训练数据"
    
    files.sort(reverse=True)
    files = files[:limit]
    
    result = f"📚 最近 {len(files)} 个训练样本：\n\n"
    
    for i, file in enumerate(files, 1):
        file_path = os.path.join(TRAINING_DATA_DIR, file)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                instruction = data.get('instruction', '未知指令')
                timestamp = data.get('timestamp', '未知时间')
                
                result += f"{i}. {file}\n"
                result += f"   指令: {instruction[:80]}...\n"
                result += f"   时间: {timestamp}\n"
                result += f"   类型: {data.get('metadata', {}).get('problem_type', 'general')}\n"
                result += "   ---\n"
        except:
            result += f"{i}. {file} (读取失败)\n"
    
    return result