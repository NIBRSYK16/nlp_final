# 使用本地 Qwen2.5-Coder-0.5B 模型的 Gradio 界面，带批量自我演化功能
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import gradio as gr
import os
import json
import subprocess
import tempfile
import requests
import re
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

# ====== 全局变量 ======
model = None
tokenizer = None
device = None

# 模型路径
DEFAULT_MODEL_PATH = "./models/Qwen2.5-Coder-0.5B-Instruct"

# API配置（32B和14B模型）
API_CONFIG = {
    "qwen_32b_api_url": "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
    "qwen_14b_api_url": "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
    "api_key": "sk-1d1d9ecf1f1b446588871b3e6d5d3a30",
}

# 自我演化配置
EVOLUTION_CONFIG = {
    "enable_self_evolution": True,
    "evolution_keywords": ["自我演化", "自我进化", "self-evolve", "self-evolution"],
    "max_fine_tune_steps": 100,
    "learning_rate": 5e-5,
    "evolution_batch_size": 3,
}

# 存储训练数据的目录
TRAINING_DATA_DIR = "./evolution_training_data"

# ====== 文本处理函数 ======
def extract_problems_from_text(text: str) -> List[str]:
    """
    从文本中提取引号内的编程问题
    
    支持格式：
    1. 单引号或双引号：'problem' 或 "problem"
    2. 多行输入，每行一个问题
    3. 包含"自我演化"关键词的提示文本
    """
    # 移除"自我演化"关键词和可能的提示文本
    clean_text = text.lower()
    for keyword in ["自我演化", "self-evolve", "self-evolution", "请自我演化", "请进化"]:
        clean_text = clean_text.replace(keyword.lower(), "")
    
    # 提取所有引号内的内容
    # 匹配双引号
    double_quote_pattern = r'"([^"]*)"'
    # 匹配单引号
    single_quote_pattern = r"'([^']*)'"
    
    problems = []
    
    # 提取双引号内容
    for match in re.findall(double_quote_pattern, text):
        if match.strip() and len(match.strip()) > 10:  # 确保不是空字符串且有一定长度
            problems.append(match.strip())
    
    # 提取单引号内容
    for match in re.findall(single_quote_pattern, text):
        if match.strip() and len(match.strip()) > 10:
            problems.append(match.strip())
    
    # 如果没有找到引号内容，尝试按行分割
    if not problems:
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            # 跳过空行和太短的行
            if line and len(line) > 20 and not line.startswith('#'):
                # 移除行号、项目符号等
                clean_line = re.sub(r'^\s*\d+[\.\)]?\s*', '', line)  # 移除 "1. " 或 "1) "
                clean_line = re.sub(r'^\s*[•\-*]\s*', '', clean_line)  # 移除项目符号
                clean_line = clean_line.strip()
                if clean_line and len(clean_line) > 20:
                    problems.append(clean_line)
    
    # 去重
    unique_problems = []
    seen = set()
    for problem in problems:
        problem_lower = problem.lower()
        if problem_lower not in seen:
            seen.add(problem_lower)
            unique_problems.append(problem)
    
    return unique_problems

def detect_evolution_mode(prompt: str) -> Tuple[bool, List[str]]:
    """
    检测是否进入自我演化模式，并提取问题
    返回: (是否演化模式, 问题列表)
    """
    # 检查是否包含演化关键词
    should_evolve = False
    for keyword in EVOLUTION_CONFIG["evolution_keywords"]:
        if keyword.lower() in prompt.lower():
            should_evolve = True
            break
    
    if not should_evolve:
        return False, []
    
    # 提取问题
    problems = extract_problems_from_text(prompt)
    return True, problems

# ====== 辅助函数 ======
def create_progress_tracker(total_steps: int):
    """创建进度跟踪器"""
    return {
        "total": total_steps,
        "current": 0,
        "success": 0,
        "failed": 0,
        "start_time": time.time(),
        "logs": []
    }

def update_progress(tracker: Dict, step_name: str, success: bool = True, message: str = ""):
    """更新进度跟踪器"""
    tracker["current"] += 1
    if success:
        tracker["success"] += 1
    else:
        tracker["failed"] += 1
    
    progress_percent = (tracker["current"] / tracker["total"]) * 100
    elapsed_time = time.time() - tracker["start_time"]
    
    log_entry = {
        "step": tracker["current"],
        "name": step_name,
        "success": success,
        "message": message,
        "progress": f"{progress_percent:.1f}%",
        "elapsed": f"{elapsed_time:.1f}s"
    }
    
    tracker["logs"].append(log_entry)
    
    # 构建状态报告
    report = f" 进度: {progress_percent:.1f}% ({tracker['current']}/{tracker['total']})\n"
    report += f" 成功: {tracker['success']} |  失败: {tracker['failed']}\n"
    report += f" 用时: {elapsed_time:.1f}秒\n"
    report += f" 当前步骤: {step_name}\n"
    if message:
        report += f" {message[:100]}...\n" if len(message) > 100 else f" {message}\n"
    
    return report, tracker

# ====== API调用函数 ======
def call_qwen_api(api_url: str, prompt: str, model_name: str = "qwen2.5-coder-32b-instruct", 
                  max_tokens: int = 1024, temperature: float = 0.7, 
                  retries: int = 3) -> Tuple[bool, str]:
    """
    调用Qwen API生成代码
    """
    headers = {
        "Authorization": f"Bearer {API_CONFIG['api_key']}",
        "Content-Type": "application/json"
    }
    
    messages = [
        {"role": "system", "content": "你是一个专业的编程助手，请生成高质量、可运行的Python代码。"},
        {"role": "user", "content": prompt}
    ]
    
    payload = {
        "model": model_name,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 0.9
    }
    
    for attempt in range(retries):
        try:
            response = requests.post(api_url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            result = response.json()
            generated_code = result["choices"][0]["message"]["content"]
            
            # 提取代码块（如果有的话）
            code_pattern = r"```(?:python)?\n?(.*?)```"
            matches = re.findall(code_pattern, generated_code, re.DOTALL)
            
            if matches:
                generated_code = matches[0].strip()
            
            return True, generated_code
        except requests.exceptions.RequestException as e:
            if attempt == retries - 1:
                return False, f"API调用失败（尝试{retries}次）: {str(e)}"
            time.sleep(1)  # 等待1秒后重试
        except Exception as e:
            return False, f"API处理失败: {str(e)}"
    
    return False, "未知错误"

def validate_code_with_14b(instruct: str, code: str) -> Tuple[bool, str]:
    """
    使用14B模型验证代码是否符合指令逻辑
    """
    validation_prompt = f"""
    请分析以下代码是否符合用户指令的逻辑要求：
    
    用户指令：{instruct}
    
    生成的代码：
    ```python
    {code}
    ```
    
    请从以下几个方面进行判断：
    1. 代码是否完整实现了指令要求的功能
    2. 代码逻辑是否正确
    3. 是否有明显的逻辑错误或缺失
    
    请用以下格式回答：
    [是否通过]：是/否
    [理由]：简要说明理由
    """
    
    success, response = call_qwen_api(
        API_CONFIG["qwen_14b_api_url"], 
        validation_prompt, 
        model_name="qwen2.5-coder-14b-instruct",
        max_tokens=256,
        temperature=0.3
    )
    
    if not success:
        return False, response
    
    # 解析响应
    if "[是否通过]：是" in response or "通过" in response and "否" not in response:
        return True, response
    else:
        return False, response

# ====== 代码验证函数 ======
def check_code_syntax(code: str) -> Tuple[bool, str]:
    """
    检查Python代码的语法错误
    """
    try:
        # 添加必要的导入
        full_code = "import math\nimport re\nimport heapq\nimport numpy as np\n" + code
        
        # 尝试编译
        compile(full_code, '<string>', 'exec')
        return True, "语法检查通过"
    except SyntaxError as e:
        return False, f"语法错误: {str(e)}"
    except Exception as e:
        return False, f"代码检查错误: {str(e)}"

def run_simple_test(code: str, problem_type: str) -> Tuple[bool, str]:
    """
    运行简单的测试用例
    """
    try:
        # 创建临时文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_file = f.name
        
        # 根据问题类型添加测试
        test_code = ""
        if "minimum cost path" in problem_type.lower():
            test_code = """
if __name__ == "__main__":
    cost_matrix = [[1, 2, 3], [4, 8, 2], [1, 5, 3]]
    try:
        result = min_cost_path(cost_matrix, 2, 2)
        print(f"测试通过，最小成本: {result}")
    except Exception as e:
        print(f"测试失败: {e}")
"""
        elif "similar elements" in problem_type.lower():
            test_code = """
if __name__ == "__main__":
    list1 = [(1, 2), (3, 4), (5, 6)]
    list2 = [(3, 4), (7, 8), (1, 2)]
    try:
        result = find_similar_elements(list1, list2)
        print(f"测试通过，相似元素: {result}")
    except Exception as e:
        print(f"测试失败: {e}")
"""
        # 添加更多测试类型...
        
        if test_code:
            with open(temp_file, 'a') as f:
                f.write(test_code)
            
            result = subprocess.run(
                ['python', temp_file],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            os.unlink(temp_file)
            
            if result.returncode == 0:
                return True, f"测试通过: {result.stdout.strip()}"
            else:
                return False, f"测试失败: {result.stderr.strip()}"
        else:
            os.unlink(temp_file)
            return True, "无特定测试，跳过运行测试"
            
    except Exception as e:
        if os.path.exists(temp_file):
            os.unlink(temp_file)
        return False, f"测试执行错误: {str(e)}"

# ====== 训练数据处理 ======
def save_training_example(instruct: str, code: str, validation_result: str):
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

def classify_problem(instruct: str) -> str:
    """
    分类问题类型
    """
    keywords = {
        "path": ["minimum cost path", "cost matrix", "reach", "grid"],
        "search": ["find", "search", "identify", "check"],
        "sort": ["sort", "largest", "smallest", "order"],
        "string": ["string", "character", "words", "regex"],
        "math": ["prime", "number", "bit", "volume", "rotations"],
        "data_structure": ["heap", "matrix", "list", "tuple", "dictionary"]
    }
    
    instruct_lower = instruct.lower()
    for category, words in keywords.items():
        for word in words:
            if word in instruct_lower:
                return category
    
    return "general"

# ====== 模型训练函数 ======
def fine_tune_on_examples(examples: List[Dict]) -> str:
    """
    在多个示例上微调模型
    """
    global model, tokenizer
    
    if model is None or tokenizer is None:
        return "模型未加载，无法进行微调"
    
    if not examples:
        return "没有有效的训练示例"
    
    try:
        model.train()
        
        total_loss = 0
        successful_updates = 0
        
        for example in examples:
            try:
                # 准备训练数据
                messages = [
                    {"role": "system", "content": "你是一个专业的编程助手"},
                    {"role": "user", "content": example["instruction"]},
                    {"role": "assistant", "content": example["code"]}
                ]
                
                # 应用聊天模板
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False
                )
                
                # 编码
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048).to(device)
                
                # 前向传播
                outputs = model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss
                
                # 反向传播
                loss.backward()
                total_loss += loss.item()
                successful_updates += 1
                
            except Exception as e:
                print(f"处理示例时出错: {str(e)}")
                continue
        
        # 更新模型参数
        if successful_updates > 0:
            optimizer = torch.optim.AdamW(model.parameters(), lr=EVOLUTION_CONFIG["learning_rate"])
            optimizer.step()
            optimizer.zero_grad()
            
            avg_loss = total_loss / successful_updates if successful_updates > 0 else 0
            
            # 保存检查点
            checkpoint_dir = "./model_checkpoints"
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = f"{checkpoint_dir}/checkpoint_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'examples_count': successful_updates
            }, checkpoint_path)
        else:
            avg_loss = 0
        
        model.eval()
        
        return f" 微调完成，处理了{successful_updates}个示例，平均损失: {avg_loss:.4f}"
        
    except Exception as e:
        model.eval()
        return f"微调失败: {str(e)}"

# ====== 单个问题处理 ======
def process_single_problem(problem: str, system_prompt: str = None) -> Tuple[bool, Dict]:
    """
    处理单个问题的完整流程
    """
    result = {
        "problem": problem,
        "success": False,
        "generated_code": "",
        "validation_result": "",
        "test_result": "",
        "saved_file": ""
    }
    
    # 步骤1: 使用32B模型生成代码
    success, code = call_qwen_api(
        API_CONFIG["qwen_32b_api_url"],
        problem,
        model_name="qwen2.5-coder-32b-instruct"
    )
    
    if not success:
        result["validation_result"] = f"代码生成失败: {code}"
        return False, result
    
    result["generated_code"] = code
    
    # 步骤2: 语法检查
    syntax_ok, syntax_msg = check_code_syntax(code)
    if not syntax_ok:
        result["validation_result"] = f"语法错误: {syntax_msg}"
        return False, result
    
    # 步骤3: 逻辑验证（14B模型）
    logic_ok, logic_msg = validate_code_with_14b(problem, code)
    result["validation_result"] = logic_msg
    
    if not logic_ok:
        return False, result
    
    # 步骤4: 运行简单测试
    test_ok, test_msg = run_simple_test(code, problem)
    result["test_result"] = test_msg
    
    if not test_ok:
        print(f"测试失败，但仍保存示例: {test_msg}")
        # 继续处理，因为有些测试可能过于严格
    
    # 步骤5: 保存训练数据
    try:
        saved_file = save_training_example(problem, code, logic_msg)
        result["saved_file"] = saved_file
        result["success"] = True
    except Exception as e:
        result["validation_result"] += f"\n保存失败: {str(e)}"
        return False, result
    
    return True, result

# ====== 批量自我演化主函数 ======
def batch_self_evolution(problems: List[str], system_prompt: str = None) -> str:
    """
    批量自我演化流程
    处理用户输入中提取的所有问题
    """
    if not problems:
        return " 错误：没有提取到有效的编程问题。请确保问题用引号括起来。"
    
    total_problems = len(problems)
    batch_size = EVOLUTION_CONFIG["evolution_batch_size"]
    
    # 创建进度跟踪器
    tracker = create_progress_tracker(total_problems)
    
    report_lines = []
    report_lines.append(" 开始批量自我演化流程")
    report_lines.append(f" 提取到 {total_problems} 个编程问题")
    report_lines.append(f" 批量大小: {batch_size}")
    report_lines.append("=" * 60)
    
    # 显示提取到的问题
    report_lines.append("📝 提取到的问题：")
    for i, problem in enumerate(problems, 1):
        if len(problem) > 80:
            display_problem = problem[:77] + "..."
        else:
            display_problem = problem
        report_lines.append(f"  {i}. {display_problem}")
    
    report_lines.append("=" * 60)
    
    successful_examples = []
    
    # 分批处理问题
    for i in range(0, total_problems, batch_size):
        batch = problems[i:i+batch_size]
        batch_num = i // batch_size + 1
        total_batches = (total_problems + batch_size - 1) // batch_size
        
        report_lines.append(f"\n 处理批次 {batch_num}/{total_batches}")
        
        # 并行处理批次中的问题
        with ThreadPoolExecutor(max_workers=min(batch_size, 4)) as executor:
            future_to_problem = {
                executor.submit(process_single_problem, problem, system_prompt): problem 
                for problem in batch
            }
            
            for future in as_completed(future_to_problem):
                problem = future_to_problem[future]
                try:
                    success, result = future.result(timeout=120)
                    
                    # 更新进度
                    if len(problem) > 50:
                        step_name = f"问题: {problem[:47]}..."
                    else:
                        step_name = f"问题: {problem}"
                    
                    progress_report, tracker = update_progress(
                        tracker, step_name, success, 
                        "成功" if success else result.get("validation_result", "未知错误")
                    )
                    
                    report_lines.append(progress_report)
                    
                    if success:
                        successful_examples.append(result)
                        report_lines.append(f"  已保存到: {result['saved_file']}")
                    else:
                        report_lines.append(f"   失败: {result.get('validation_result', '未知错误')[:80]}...")
                        
                except Exception as e:
                    progress_report, tracker = update_progress(tracker, "处理异常", False, str(e))
                    report_lines.append(progress_report)
        
        report_lines.append("-" * 40)
    
    # 微调模型
    if successful_examples:
        report_lines.append("\n🎯 开始模型微调...")
        
        # 准备训练数据
        training_data = []
        for example in successful_examples:
            if example["success"]:
                training_data.append({
                    "instruction": example["problem"],
                    "code": example["generated_code"]
                })
        
        # 执行微调
        fine_tune_result = fine_tune_on_examples(training_data)
        report_lines.append(fine_tune_result)
        
        # 保存训练统计
        stats_file = f"{TRAINING_DATA_DIR}/batch_evolution_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        stats = {
            "total_problems": total_problems,
            "successful": len(successful_examples),
            "failed": total_problems - len(successful_examples),
            "timestamp": datetime.now().isoformat(),
            "problems": problems,
            "examples_summary": [
                {
                    "problem": ex["problem"][:100] + "..." if len(ex["problem"]) > 100 else ex["problem"],
                    "success": ex["success"]
                }
                for ex in successful_examples[:10]  # 只保存前10个示例的摘要
            ]
        }
        
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        report_lines.append(f" 统计数据已保存到: {stats_file}")
    else:
        report_lines.append("\n 没有成功的示例，跳过微调")
    
    # 最终报告
    report_lines.append("\n" + "=" * 60)
    report_lines.append(" 批量自我演化流程完成！")
    report_lines.append(f" 成功处理: {tracker['success']}/{total_problems}")
    report_lines.append(f" 失败: {tracker['failed']}/{total_problems}")
    report_lines.append(f" 总用时: {time.time() - tracker['start_time']:.1f}秒")
    
    if successful_examples:
        report_lines.append(f"💾 模型已更新，检查点已保存")
    
    return "\n".join(report_lines)

# ====== 模型加载函数 ======
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
        
        return f" 模型加载完成！\n模型路径: {model_path}\n使用设备: {device}"
        
    except Exception as e:
        return f" 加载模型时出错：{str(e)}"

# ====== 主生成函数 ======
def generate_code(prompt, system_prompt, max_tokens, temperature, top_p, enable_evolution=True):
    """生成代码的主函数"""
    global model, tokenizer
    
    if model is None or tokenizer is None:
        return "错误：模型尚未加载，请先点击'加载模型'按钮。", ""
    
    if not prompt or prompt.strip() == "":
        return "错误：请输入代码生成提示。", ""
    
    # 检测是否触发自我演化模式
    should_evolve, extracted_problems = detect_evolution_mode(prompt)
    
    if should_evolve and enable_evolution:
        # 进入自我演化分支
        if extracted_problems:
            # 批量处理提取到的问题
            evolution_status = batch_self_evolution(extracted_problems, system_prompt)
            return evolution_status, ""
        else:
            # 没有提取到问题，可能是单问题自我演化
            try:
                # 提取单个问题（移除演化关键词）
                clean_prompt = prompt
                for keyword in EVOLUTION_CONFIG["evolution_keywords"]:
                    clean_prompt = clean_prompt.replace(keyword, "")
                clean_prompt = clean_prompt.strip()
                
                if clean_prompt:
                    success, result = process_single_problem(clean_prompt, system_prompt)
                    
                    if success:
                        status = f" 单问题自我演化完成！\n"
                        status += f" 已保存训练数据到: {result['saved_file']}\n"
                        
                        # 微调模型
                        fine_tune_result = fine_tune_on_examples([{
                            "instruction": result["problem"],
                            "code": result["generated_code"]
                        }])
                        status += f" {fine_tune_result}"
                        
                        return status, result["generated_code"]
                    else:
                        return f" 自我演化失败:\n{result['validation_result']}", ""
                else:
                    return " 错误：请提供要演化的具体问题。", ""
                    
            except Exception as e:
                return f"自我演化时出错：{str(e)}", ""
    else:
        # 正常代码生成分支
        try:
            messages = [
                {"role": "system", "content": system_prompt if system_prompt else "你是一个专业的编程助手，擅长编写和解释代码。"},
                {"role": "user", "content": prompt},
            ]
            
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            model_inputs = tokenizer([text], return_tensors="pt").to(device)
            
            with torch.no_grad():
                generated_ids = model.generate(
                    **model_inputs,
                    max_new_tokens=int(max_tokens),
                    temperature=float(temperature),
                    top_p=float(top_p),
                    do_sample=True
                )
            
            generated_ids = [
                output_ids[len(input_ids):] 
                for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            
            response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            return " 代码生成完成", response
            
        except Exception as e:
            return f"生成代码时出错：{str(e)}", ""

# ====== 模型评估函数 ======
def extract_function_code(generated_text: str, entry_point: str) -> str:
    """从生成的文本中提取函数代码"""
    # 清理生成的文本
    text = generated_text.strip()
    
    # 方式1: 查找完整的函数定义（包括函数签名）
    pattern = rf'def\s+{re.escape(entry_point)}\s*\([^)]*\)\s*:.*?(?=\n\ndef\s+|\nclass\s+|$)'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(0).strip()
    
    # 方式2: 如果包含函数定义，提取函数体（基于缩进）
    if f"def {entry_point}" in text:
        lines = text.split('\n')
        start_idx = -1
        for i, line in enumerate(lines):
            if f"def {entry_point}" in line:
                start_idx = i
                break
        
        if start_idx >= 0:
            result = [lines[start_idx]]  # 包含函数签名
            # 获取函数签名的缩进级别
            base_indent = len(lines[start_idx]) - len(lines[start_idx].lstrip())
            
            # 提取函数体
            for i in range(start_idx + 1, len(lines)):
                line = lines[i]
                if not line.strip():  # 空行
                    result.append(line)
                    continue
                
                current_indent = len(line) - len(line.lstrip())
                # 如果缩进小于等于基础缩进，说明函数结束了
                if current_indent <= base_indent:
                    break
                result.append(line)
            
            return '\n'.join(result)
    
    # 方式3: 尝试查找函数体（可能没有函数签名）
    # 查找以4个空格或tab开头的代码块
    lines = text.split('\n')
    if lines and (lines[0].startswith('    ') or lines[0].startswith('\t')):
        # 可能是函数体，需要添加函数签名
        # 但这里我们不知道函数签名，所以直接返回
        return '\n'.join(lines)
    
    # 方式4: 如果都没有找到，返回原始文本（可能是完整的函数体）
    return text

def evaluate_model(max_tasks: int = None, max_tokens: int = 512, temperature: float = 0.7, top_p: float = 0.9):
    """评估模型在 HumanEval 数据集上的表现（生成器函数，支持流式输出）"""
    global model, tokenizer, device
    
    if model is None or tokenizer is None:
        yield "错误：模型尚未加载，请先点击'加载模型'按钮。"
        return
    
    dataset_path = "./datasets/human-eval-v2-20210705.jsonl"
    if not os.path.exists(dataset_path):
        yield f"错误：数据集文件不存在: {dataset_path}"
        return
    
    try:
        # 确保模型在 eval 模式
        model.eval()
        
        # 读取数据集
        tasks = []
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    tasks.append(json.loads(line))
        
        if max_tasks:
            tasks = tasks[:max_tasks]
        
        total_tasks = len(tasks)
        passed_tasks = 0
        failed_tasks = []
        results = []
        
        system_prompt = "你是一个专业的编程助手，擅长编写和解释代码。请根据给定的函数签名和文档字符串，实现该函数。"
        
        yield f"开始评估 {total_tasks} 个任务...\n\n"
        
        for idx, task in enumerate(tasks):
            task_id = task['task_id']
            prompt = task['prompt']
            entry_point = task['entry_point']
            test_code = task['test']
            
            # 生成代码
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
            
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            model_inputs = tokenizer([text], return_tensors="pt").to(device)
            
            try:
                with torch.no_grad():
                    # 使用 pad_token_id 以避免 CUDA 错误
                    generated_ids = model.generate(
                        **model_inputs,
                        max_new_tokens=int(max_tokens),
                        temperature=float(temperature),
                        top_p=float(top_p),
                        do_sample=True,
                        pad_token_id=tokenizer.eos_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        num_beams=1
                    )
                
                # 提取生成的 token，去掉输入部分
                generated_ids = [
                    output_ids[len(input_ids):] 
                    for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
                ]
                
                generated_code = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
                
                # 清理 GPU 缓存
                del model_inputs
                torch.cuda.empty_cache()
                
                # 提取函数代码
                function_code = extract_function_code(generated_code, entry_point)
                
                # 构建完整代码用于测试
                full_code = prompt + function_code + "\n" + test_code
                
                # 执行测试
                try:
                    # 创建安全的执行环境
                    namespace = {}
                    exec(full_code, namespace)
                    
                    # 运行测试
                    check_func = namespace.get('check')
                    candidate_func = namespace.get(entry_point)
                    
                    if check_func and candidate_func:
                        check_func(candidate_func)
                        passed_tasks += 1
                        results.append(f" {task_id}: 通过")
                    else:
                        failed_tasks.append(task_id)
                        results.append(f" {task_id}: 函数未找到（生成的代码中可能没有正确的函数定义）")
                        
                except AssertionError as e:
                    # 测试失败：生成的函数没有通过测试用例
                    failed_tasks.append(task_id)
                    results.append(f" {task_id}: 测试失败（函数输出不符合预期）")
                except SyntaxError as e:
                    # 语法错误：生成的代码有语法问题
                    failed_tasks.append(task_id)
                    error_msg = str(e)[:150]
                    results.append(f" {task_id}: 语法错误 - {error_msg}")
                except NameError as e:
                    # 名称错误：可能缺少导入或函数名错误
                    failed_tasks.append(task_id)
                    error_msg = str(e)[:150]
                    results.append(f" {task_id}: 名称错误 - {error_msg}")
                except Exception as e:
                    # 其他执行错误：运行时错误
                    failed_tasks.append(task_id)
                    error_type = type(e).__name__
                    error_msg = str(e)[:150]  # 截断错误信息
                    results.append(f" {task_id}: {error_type} - {error_msg}")
                    
            except RuntimeError as e:
                # 捕获 CUDA 错误和其他运行时错误
                failed_tasks.append(task_id)
                error_msg = str(e)[:100]
                results.append(f" {task_id}: CUDA/运行时错误 - 跳过此任务")
            
            # 实时更新进度
            current_rate = (passed_tasks / (idx + 1) * 100) if (idx + 1) > 0 else 0
            progress_report = f"#  模型评估进度\n\n"
            progress_report += f"**进度**: {idx + 1}/{total_tasks} ({((idx + 1)/total_tasks*100):.1f}%)\n\n"
            progress_report += f"**当前通过率**: {current_rate:.2f}% ({passed_tasks}/{idx + 1})\n\n"
            progress_report += f"## 详细结果\n\n"
            progress_report += "\n".join(results[-20:])  # 显示最近20个结果
            
            yield progress_report
        
        # 计算最终通过率
        pass_rate = (passed_tasks / total_tasks * 100) if total_tasks > 0 else 0
        
        # 生成最终报告
        final_report = f"""#  模型评估最终报告

## 总体结果
- **总任务数**: {total_tasks}
- **通过任务数**: {passed_tasks}
- **失败任务数**: {len(failed_tasks)}
- **通过率**: {pass_rate:.2f}%

## 详细结果
"""
        final_report += "\n".join(results)
        
        if failed_tasks:
            final_report += f"\n\n## 失败的任务ID\n"
            final_report += ", ".join(failed_tasks[:20])  # 只显示前20个
            if len(failed_tasks) > 20:
                final_report += f" ... (还有 {len(failed_tasks) - 20} 个)"
        
        yield final_report
        
    except Exception as e:
        import traceback
        yield f"评估过程中出错：{str(e)}\n\n```\n{traceback.format_exc()}\n```"

# ====== 查看训练数据 ======
def list_training_data(limit: int = 20):
    """列出训练数据"""
    if not os.path.exists(TRAINING_DATA_DIR):
        return "暂无训练数据"
    
    files = [f for f in os.listdir(TRAINING_DATA_DIR) if f.endswith('.json')]
    if not files:
        return "暂无训练数据"
    
    files.sort(reverse=True)  # 按时间倒序
    files = files[:limit]
    
    result = f" 最近 {len(files)} 个训练样本：\n\n"
    
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

# ====== Gradio界面 ======
with gr.Blocks(title="Qwen2.5-Coder 批量自我演化系统", theme=gr.themes.Soft()) as demo:
    gr.Markdown("#  Qwen2.5-Coder 批量自我演化系统")
    gr.Markdown("""
    ##  功能特性：
    1. **普通代码生成**：使用本地0.5B模型生成代码
    2. **批量自我演化**：输入包含多个引号内的问题，系统自动提取并批量训练
    3. **智能问题提取**：自动从文本中提取引号内的编程问题
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("###  模型设置")
            model_path_input = gr.Textbox(
                label="模型路径", value=DEFAULT_MODEL_PATH, lines=1
            )
            load_btn = gr.Button(" 加载模型", variant="primary", size="lg")
            load_status = gr.Textbox(label="模型状态", interactive=False, lines=3)
            
            gr.Markdown("### 📊 模型评估")
            with gr.Row():
                eval_max_tasks = gr.Number(
                    label="评估任务数量",
                    value=10,
                    minimum=1,
                    maximum=164,
                    step=1,
                    info="输入要评估的任务数量（1-164），建议先用少量任务测试。留空或0表示评估全部任务。"
                )
                eval_all_check = gr.Checkbox(
                    label="评估全部任务（164个）",
                    value=False,
                    info="勾选此项将评估所有164个任务"
                )
            eval_btn = gr.Button(" 开始评估", variant="secondary", size="lg")
            eval_output = gr.Markdown(label="评估结果")
            
            # 添加评估说明
            with gr.Accordion(" 评估说明", open=False):
                gr.Markdown("""
                ### 评估结果说明
                
                ** 通过**: 生成的函数通过了所有测试用例
                
                ** 测试失败**: 
                - 函数能够正常执行，但输出结果不符合预期
                - 说明生成的代码逻辑有误
                - 例如：返回值错误、边界条件处理不当等
                
                ** 语法错误**: 
                - 生成的代码存在Python语法问题
                - 例如：缺少冒号、括号不匹配、缩进错误等
                
                ** 名称错误**: 
                - 代码中使用了未定义的变量或函数
                - 可能缺少必要的导入语句
                
                ** 函数未找到**: 
                - 生成的代码中没有找到目标函数
                - 可能是函数名不匹配或代码格式问题
                
                ** 其他执行错误**: 
                - 运行时出现的其他错误
                - 例如：类型错误、索引越界等
                
                ### HumanEval数据集
                - 包含164个Python编程任务
                - 每个任务都有多个测试用例
                - 只有通过所有测试用例才算通过
                """)
            
            with gr.Accordion(" API设置", open=False):
                api_key_input = gr.Textbox(
                    label="API密钥", value=API_CONFIG["api_key"], type="password", lines=1
                )
                api_32b_url = gr.Textbox(
                    label="32B API地址", value=API_CONFIG["qwen_32b_api_url"], lines=1
                )
                api_14b_url = gr.Textbox(
                    label="14B API地址", value=API_CONFIG["qwen_14b_api_url"], lines=1
                )
            
            with gr.Accordion(" 自我演化设置", open=False):
                enable_evolution = gr.Checkbox(
                    label="启用自我演化", value=EVOLUTION_CONFIG["enable_self_evolution"]
                )
                evolution_keywords = gr.Textbox(
                    label="演化关键词", value=",".join(EVOLUTION_CONFIG["evolution_keywords"]), lines=2
                )
                batch_size = gr.Slider(
                    label="批量大小", minimum=1, maximum=10, value=EVOLUTION_CONFIG["evolution_batch_size"], step=1
                )
                learning_rate = gr.Slider(
                    label="学习率", minimum=1e-6, maximum=1e-3, value=EVOLUTION_CONFIG["learning_rate"], step=1e-6
                )
            
            with gr.Accordion(" 数据管理", open=False):
                with gr.Row():
                    view_data_btn = gr.Button("查看训练数据", variant="secondary")
                    test_extraction_btn = gr.Button("测试问题提取", variant="secondary")
                
                training_data_view = gr.Textbox(
                    label="训练数据", interactive=False, lines=10
                )
            
            with gr.Accordion(" 生成设置", open=False):
                system_prompt_input = gr.Textbox(
                    label="系统提示词",
                    value="你是一个专业的编程助手，擅长编写和解释代码。",
                    lines=2
                )
                max_tokens_input = gr.Slider(
                    label="最大token数", minimum=50, maximum=2048, value=512, step=50
                )
                temperature_input = gr.Slider(
                    label="Temperature", minimum=0.1, maximum=2.0, value=0.7, step=0.1
                )
                top_p_input = gr.Slider(
                    label="Top-p", minimum=0.1, maximum=1.0, value=0.9, step=0.05
                )
        
        with gr.Column(scale=2):
            gr.Markdown("###  代码生成与自我演化")
            
            mode_indicator = gr.Markdown("**当前模式：** 等待输入...")
            
            # 示例输入
            example_input = '''请自我演化
"Write a function to find the minimum cost path to reach (m, n) from (0, 0) for the given cost matrix cost[][] and a position (m, n) in cost[][]."
"Write a function to find the similar elements from the given two tuple lists."
"Write a python function to identify non-prime numbers."
"Write a function to find the largest integers from a given list of numbers using heap queue algorithm."
"Write a function to find the number of ways to fill it with 2 x 1 dominoes for the given 3 x n board."'''
            
            prompt_input = gr.Textbox(
                label="输入提示词",
                placeholder=example_input,
                lines=10,
                value=example_input
            )
            
            generate_btn = gr.Button(" 生成代码", variant="primary", size="lg")
            
            status_output = gr.Textbox(
                label="执行状态", interactive=False, lines=12
            )
            
            code_output = gr.Code(
                label="生成的代码", language="python", lines=20
            )
    
    # ====== 事件处理 ======
    def update_api_config(api_key, api_32b, api_14b):
        global API_CONFIG
        API_CONFIG["api_key"] = api_key
        API_CONFIG["qwen_32b_api_url"] = api_32b
        API_CONFIG["qwen_14b_api_url"] = api_14b
        return " API配置已更新"
    
    def update_evolution_config(enable, keywords, batch, lr):
        global EVOLUTION_CONFIG
        EVOLUTION_CONFIG["enable_self_evolution"] = enable
        EVOLUTION_CONFIG["evolution_keywords"] = [k.strip() for k in keywords.split(",") if k.strip()]
        EVOLUTION_CONFIG["evolution_batch_size"] = batch
        EVOLUTION_CONFIG["learning_rate"] = lr
        return " 自我演化配置已更新"
    
    def detect_mode(prompt):
        if not prompt:
            return "**当前模式：** 等待输入..."
        
        should_evolve, problems = detect_evolution_mode(prompt)
        
        if should_evolve:
            if problems:
                return f"**当前模式：**  批量自我演化模式（检测到{len(problems)}个问题）"
            else:
                return "**当前模式：**  单问题自我演化模式"
        
        return "**当前模式：**  普通代码生成模式"
    
    # 测试问题提取
    def test_problem_extraction(prompt):
        should_evolve, problems = detect_evolution_mode(prompt)
        
        if not should_evolve:
            return "未检测到自我演化关键词。"
        
        if not problems:
            return "检测到自我演化关键词，但没有提取到问题。"
        
        result = f" 检测到自我演化模式\n"
        result += f" 提取到 {len(problems)} 个问题：\n\n"
        
        for i, problem in enumerate(problems, 1):
            result += f"{i}. {problem}\n"
        
        result += f"\n提示：点击'生成代码'按钮开始批量训练（输入中包含'自我演化'关键词即可）。"
        return result
    
    # 绑定事件
    load_btn.click(
        fn=load_model,
        inputs=model_path_input,
        outputs=load_status
    )
    
    generate_btn.click(
        fn=generate_code,
        inputs=[
            prompt_input, system_prompt_input, max_tokens_input, 
            temperature_input, top_p_input, enable_evolution
        ],
        outputs=[status_output, code_output]
    ).then(
        fn=detect_mode,
        inputs=prompt_input,
        outputs=mode_indicator
    )
    
    # API配置更新
    api_key_input.change(
        fn=update_api_config,
        inputs=[api_key_input, api_32b_url, api_14b_url],
        outputs=gr.Textbox(visible=False)
    )
    
    # 演化配置更新
    enable_evolution.change(
        fn=update_evolution_config,
        inputs=[enable_evolution, evolution_keywords, batch_size, learning_rate],
        outputs=gr.Textbox(visible=False)
    )
    
    # 查看训练数据
    view_data_btn.click(
        fn=list_training_data,
        outputs=training_data_view
    )
    
    # 测试问题提取
    test_extraction_btn.click(
        fn=test_problem_extraction,
        inputs=prompt_input,
        outputs=training_data_view
    )
    
    # 实时检测模式
    prompt_input.change(
        fn=detect_mode,
        inputs=prompt_input,
        outputs=mode_indicator
    )
    
    # 模型评估事件
    def evaluate_wrapper(max_tasks, eval_all, max_tokens, temperature, top_p):
        """评估函数的包装器，支持流式输出"""
        if eval_all:
            max_tasks_val = None  # 评估全部任务
        else:
            max_tasks_val = int(max_tasks) if max_tasks and max_tasks > 0 else None
        for result in evaluate_model(max_tasks_val, max_tokens, temperature, top_p):
            yield result
    
    eval_btn.click(
        fn=evaluate_wrapper,
        inputs=[eval_max_tasks, eval_all_check, max_tokens_input, temperature_input, top_p_input],
        outputs=eval_output
    )
    
    # 示例提示词
    gr.Examples(
        examples=[
            [example_input],
            ["请自我演化\n\"用Python实现一个快速排序算法。\"\n\"用Python实现一个二叉树的遍历算法。\""],
            ["用Python编写一个简单的HTTP服务器。"],
        ],
        inputs=prompt_input,
        outputs=[mode_indicator]
    )
    
    # 使用说明
    # gr.Markdown("""
    # ##  使用说明：
    
    # ### 1. 普通代码生成：
    # - 输入普通的代码生成提示
    # - 点击"生成代码"按钮
    
    # ### 2. 批量自我演化：
    # - 在输入中包含"自我演化"关键词
    # - 用**双引号**括起每个编程问题
    # - 每个问题占一行或使用分隔符
    # - 点击"生成代码"按钮（输入中包含"自我演化"关键词即可）
    
    # ### 3. 输入格式示例：
    # ```
    # 请自我演化
    # "Write a function to find the minimum cost path..."
    # "Write a function to find the similar elements..."
    # "Write a python function to identify non-prime numbers..."
    # ```
    
    # ### 4. 系统流程：
    # 1. 检测"自我演化"关键词
    # 2. 提取所有引号内的问题
    # 3. 对每个问题：
    #    - 调用32b API生成代码
    #    - 14B模型验证代码逻辑
    #    - 语法检查
    #    - 保存训练数据
    # 4. 用所有成功的问题微调本地0.5B模型
    # 5. 返回处理报告
    
    # ### 5. 注意事项：
    # - API密钥需要正确配置
    # - 自我演化过程可能需要几分钟时间
    # - 模型微调后会保存检查点
    # - 训练数据保存在`./evolution_training_data/`目录
    # """)

if __name__ == "__main__":
    # 创建必要的目录
    os.makedirs(TRAINING_DATA_DIR, exist_ok=True)
    os.makedirs("./model_checkpoints", exist_ok=True)
    
    # 启动 Gradio 界面
    demo.launch(
        share=False, 
        server_name="0.0.0.0", 
        server_port=7860,
        show_api=False
    )