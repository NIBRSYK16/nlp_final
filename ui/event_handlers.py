"""
事件处理函数
"""
from core.model_manager import load_model, get_model
from core.model_evaluation import evaluate_model_on_humaneval, get_evaluation_help
from core.fine_tune_manager import fine_tune_model_with_data, get_training_data_from_files, get_fine_tune_status, get_fine_tune_help
from utils.text_utils import detect_evolution_mode, extract_problems_from_text
from config import API_CONFIG, EVOLUTION_CONFIG, GENERATION_CONFIG, EVALUATION_CONFIG


def update_api_config(api_key, api_32b, api_14b):
    """更新API配置"""
    API_CONFIG["api_key"] = api_key
    API_CONFIG["qwen_32b_api_url"] = api_32b
    API_CONFIG["qwen_14b_api_url"] = api_14b
    return "API配置已更新"


def update_evolution_config(enable, keywords, batch, lr):
    """更新自我演化配置"""
    EVOLUTION_CONFIG["enable_self_evolution"] = enable
    EVOLUTION_CONFIG["evolution_keywords"] = [k.strip() for k in keywords.split(",") if k.strip()]
    EVOLUTION_CONFIG["evolution_batch_size"] = batch
    EVOLUTION_CONFIG["learning_rate"] = lr
    return "自我演化配置已更新"


def detect_mode(prompt):
    """检测当前模式"""
    if not prompt:
        return "**当前模式：** 等待输入..."
    
    should_evolve, problems = detect_evolution_mode(prompt)
    
    if should_evolve:
        if problems:
            return f"**当前模式：** 批量自我演化模式（检测到{len(problems)}个问题）"
        else:
            return "**当前模式：** 单问题自我演化模式"
    
    return "**当前模式：** 普通代码生成模式"


def test_problem_extraction(prompt):
    """测试问题提取"""
    should_evolve, problems = detect_evolution_mode(prompt)
    
    if not should_evolve:
        return "未检测到自我演化关键词。"
    
    if not problems:
        return "检测到自我演化关键词，但没有提取到问题。"
    
    result = f"检测到自我演化模式\n"
    result += f"提取到 {len(problems)} 个问题：\n\n"
    
    for i, problem in enumerate(problems, 1):
        result += f"{i}. {problem}\n"
    
    result += f"\n提示：点击'执行自我演化'按钮开始批量训练。"
    return result


def evaluate_model_wrapper(max_tasks, eval_all, max_tokens, temperature, top_p):
    """评估模型的包装函数，支持流式输出"""
    model, tokenizer, device = get_model()
    
    if model is None or tokenizer is None:
        yield "错误：模型尚未加载，请先点击'加载模型'按钮。"
        return
    
    if eval_all:
        max_tasks_val = None  # 评估全部任务
    else:
        max_tasks_val = int(max_tasks) if max_tasks and max_tasks > 0 else EVALUATION_CONFIG["default_max_tasks"]
    
    for result in evaluate_model_on_humaneval(model, tokenizer, device, max_tasks_val, max_tokens, temperature, top_p):
        yield result


def fine_tune_model_wrapper(output_dir, epochs, batch_size, learning_rate):
    """微调模型的包装函数"""
    model, tokenizer, device = get_model()
    
    if model is None or tokenizer is None:
        return "错误：模型尚未加载"
    
    # 从文件加载训练数据
    training_data = get_training_data_from_files()
    
    if not training_data:
        return "错误：没有找到训练数据。请先使用自我演化功能收集数据。"
    
    # 执行微调
    success, message = fine_tune_model_with_data(
        model, tokenizer, device,
        training_data,
        output_dir=output_dir or "./fine_tuned_model",
        num_epochs=int(epochs),
        batch_size=int(batch_size),
        learning_rate=float(learning_rate)
    )
    
    return message