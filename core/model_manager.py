"""
模型管理模块
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
from config import DEFAULT_MODEL_PATH

# 全局变量
model = None
tokenizer = None
device = None


def load_model(model_path=None):
    """加载模型和分词器"""
    global model, tokenizer, device
    
    if model_path is None or model_path.strip() == "":
        model_path = DEFAULT_MODEL_PATH
    
    if not os.path.exists(model_path):
        return f"错误：模型路径不存在: {model_path}"
    
    try:
        print(f"正在从本地路径加载模型: {model_path}")
        
        # 加载分词器
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        
        # 确定设备
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"使用设备: {device}")
        
        # 加载模型
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=True,
            dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        model = model.to(device)
        model.eval()
        
        return f"✅ 模型加载完成！\n模型路径: {model_path}\n使用设备: {device}"
        
    except Exception as e:
        return f"❌ 加载模型时出错：{str(e)}"


def get_model():
    """获取模型实例"""
    global model, tokenizer, device
    return model, tokenizer, device