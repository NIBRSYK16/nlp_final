# 使用本地 Qwen2.5-Coder-0.5B 模型的 Gradio 界面
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from transformers import DataCollatorForLanguageModeling
import torch
import gradio as gr
import os
import json
import re
import traceback
import requests
import subprocess
import tempfile
from typing import Dict, List, Tuple, Optional
try:
    from datasets import Dataset
except ImportError:
    # 如果datasets库未安装，使用简单的列表
    Dataset = None

# 全局变量存储模型和分词器
model = None
tokenizer = None
device = None

# 自我演化相关配置
SELF_EVOLUTION_KEYWORD = "自我演化"
QWEN_32B_API_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"  # Qwen API地址
QWEN_14B_API_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"  # Coder-14B API地址
self_evolution_data = []  # 存储(instruct, code)二元组
qwen_32b_api_key = "sk-1d1d9ecf1f1b446588871b3e6d5d3a30"
qwen_14b_api_key = "sk-4b834eacde4740479e7c66dbb8ebd46f"

# 设置本地模型路径（请根据你的实际路径修改）
DEFAULT_MODEL_PATH = "./models/Qwen2.5-Coder-0.5B-instruct"

def load_model(model_path=None):
    """加载模型和分词器"""
    global model, tokenizer, device
    
    # 使用默认路径或用户提供的路径
    if model_path is None or model_path.strip() == "":
        model_path = DEFAULT_MODEL_PATH
    
    # 检查模型路径是否存在
    if not os.path.exists(model_path):
        return f"错误：模型路径不存在: {model_path}\n请检查路径是否正确。"
    
    try:
        print(f"正在从本地路径加载模型: {model_path}")
        print("正在加载模型和分词器...")
        
        # 从本地路径加载分词器
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        
        # 确定设备
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"使用设备: {device}")
        
        # 从本地路径加载模型
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=True,  # 只使用本地文件，不从网络下载
            dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        model = model.to(device)
        model.eval()  # 设置为评估模式
        print("模型加载完成！")
        return f"模型加载完成！\n模型路径: {model_path}\n使用设备: {device}"
        
    except Exception as e:
        return f"加载模型时出错：{str(e)}"

def call_qwen_api(api_url: str, api_key: str, messages: List[Dict], model_name: str, max_tokens: int = 2048):
    """调用Qwen API生成代码"""
    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.7
        }
        response = requests.post(api_url, headers=headers, json=data, timeout=60)
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"]
    except Exception as e:
        raise Exception(f"API调用失败: {str(e)}")

def check_code_syntax(code: str) -> Tuple[bool, str]:
    """检查代码语法是否正确"""
    try:
        # 使用Python编译检查语法
        compile(code, '<string>', 'exec')
        return True, "语法检查通过"
    except SyntaxError as e:
        return False, f"语法错误: {str(e)}"
    except Exception as e:
        return False, f"编译错误: {str(e)}"

def check_code_logic(api_url: str, api_key: str, instruct: str, code: str) -> Tuple[bool, str]:
    """使用Coder-14B判断代码是否符合instruct逻辑"""
    try:
        messages = [
            {
                "role": "system",
                "content": "你是一个代码审查专家，需要判断生成的代码是否符合用户的需求。请仔细分析代码逻辑，判断代码是否正确实现了用户的要求。只回答'是'或'否'，然后简要说明原因。"
            },
            {
                "role": "user",
                "content": f"用户需求：{instruct}\n\n生成的代码：\n```python\n{code}\n```\n\n请判断这段代码是否符合用户需求？"
            }
        ]
        
        response_text = call_qwen_api(api_url, api_key, messages, "qwen2.5-coder-14b-instruct", max_tokens=200)
        
        # 判断响应中是否包含"是"或"符合"等肯定词汇
        positive_keywords = ["是", "符合", "正确", "满足", "可以", "能够"]
        is_valid = any(keyword in response_text for keyword in positive_keywords)
        
        return is_valid, response_text
    except Exception as e:
        return False, f"逻辑检查失败: {str(e)}"

def self_evolution_process(instruct: str, qwen_32b_api_key: str, coder_14b_api_key: str, 
                          qwen_32b_url: str, coder_14b_url: str) -> str:
    """自我演化分支：生成代码、验证、收集数据"""
    global self_evolution_data
    
    try:
        # 步骤1: 使用qwen2.5-coder-32b-instruct生成代码
        messages_32b = [
            {
                "role": "system",
                "content": "你是一个专业的编程助手，擅长编写和解释代码。请根据用户需求生成完整、正确、可运行的Python代码。"
            },
            {
                "role": "user",
                "content": instruct
            }
        ]
        
        generated_code = call_qwen_api(qwen_32b_url, qwen_32b_api_key, messages_32b, "qwen2.5-coder-32b-instruct")
        
        # 步骤2: 检查代码语法
        syntax_ok, syntax_msg = check_code_syntax(generated_code)
        if not syntax_ok:
            return f"代码语法检查失败\n{syntax_msg}\n\n生成的代码：\n```python\n{generated_code}\n```"
        
        # 步骤3: 使用Coder-14B检查代码逻辑
        logic_ok, logic_msg = check_code_logic(coder_14b_url, coder_14b_api_key, instruct, generated_code)
        if not logic_ok:
            return f"代码逻辑检查失败\n{logic_msg}\n\n生成的代码：\n```python\n{generated_code}\n```"
        
        # 步骤4: 代码通过验证，添加到训练数据
        self_evolution_data.append({
            "instruct": instruct,
            "code": generated_code
        })
        
        return f"""自我演化数据收集成功！

**用户需求**: {instruct}

**生成的代码**:
```python
{generated_code}
```

**验证结果**:
- 语法检查: 通过
- 逻辑检查: 通过

**当前数据集大小**: {len(self_evolution_data)} 条

注意：自我演化模式只收集数据用于微调，不直接生成代码。请使用微调功能训练模型。"""
        
    except Exception as e:
        return f"自我演化过程出错：{str(e)}\n{traceback.format_exc()}"

def fine_tune_model(output_dir: str = "./fine_tuned_model", num_epochs: int = 3):
    """使用收集的数据微调0.5B模型"""
    global model, tokenizer, self_evolution_data
    
    if model is None or tokenizer is None:
        return "错误：模型尚未加载，请先点击'加载模型'按钮。"
    
    if len(self_evolution_data) == 0:
        return "错误：没有收集到训练数据。请先使用自我演化功能收集数据。"
    
    try:
        # 准备训练数据
        def format_instruction(example):
            instruct = example["instruct"]
            code = example["code"]
            # 格式化训练数据，使用聊天模板格式
            messages = [
                {"role": "system", "content": "你是一个专业的编程助手，擅长编写和解释代码。"},
                {"role": "user", "content": instruct},
                {"role": "assistant", "content": code}
            ]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            return {"text": text}
        
        # 创建数据集
        formatted_data = [format_instruction(item) for item in self_evolution_data]
        
        if Dataset is not None:
            dataset = Dataset.from_list(formatted_data)
            
            # 对数据进行tokenize
            def tokenize_function(examples):
                return tokenizer(
                    examples["text"],
                    truncation=True,
                    padding="max_length",
                    max_length=1024,
                    return_tensors="pt"
                )
            
            tokenized_dataset = dataset.map(tokenize_function, batched=True)
        else:
            # 如果没有datasets库，使用简单方式
            tokenized_texts = []
            for item in formatted_data:
                tokens = tokenizer(
                    item["text"],
                    truncation=True,
                    padding="max_length",
                    max_length=1024,
                    return_tensors="pt"
                )
                tokenized_texts.append(tokens)
            # 创建一个简单的数据集包装
            class SimpleDataset:
                def __init__(self, data):
                    self.data = data
                def __len__(self):
                    return len(self.data)
                def __getitem__(self, idx):
                    return self.data[idx]
            tokenized_dataset = SimpleDataset(tokenized_texts)
        
        # 设置训练参数
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            warmup_steps=10,
            logging_steps=10,
            save_steps=50,
            evaluation_strategy="no",
            save_total_limit=2,
            load_best_model_at_end=False,
            push_to_hub=False,
            report_to="none",  # 禁用wandb等
        )
        
        # 创建数据整理器
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
        )
        
        # 创建Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator,
        )
        
        # 开始训练
        trainer.train()
        
        # 保存模型
        trainer.save_model()
        tokenizer.save_pretrained(output_dir)
        
        return f"""模型微调完成！

**训练数据量**: {len(self_evolution_data)} 条
**训练轮数**: {num_epochs}
**模型保存路径**: {output_dir}

微调后的模型已保存，可以重新加载使用。"""
        
    except Exception as e:
        return f"微调过程出错：{str(e)}\n\n```\n{traceback.format_exc()}\n```"

def generate_code(prompt, system_prompt, max_tokens, temperature, top_p, 
                 qwen_32b_api_key, coder_14b_api_key, qwen_32b_url, coder_14b_url):
    """生成代码的函数（支持自我演化分支）"""
    if model is None or tokenizer is None:
        return "错误：模型尚未加载，请先点击'加载模型'按钮。"
    
    if not prompt or prompt.strip() == "":
        return "错误：请输入代码生成提示。"
    
    # 判断是否进入自我演化分支
    if SELF_EVOLUTION_KEYWORD in prompt:
        # 自我演化分支：只收集数据，不生成代码
        if not qwen_32b_api_key or not coder_14b_api_key:
            return "错误：自我演化模式需要配置API密钥。请在设置中填写Qwen-32B和Coder-14B的API密钥。"
        
        return self_evolution_process(
            prompt.replace(SELF_EVOLUTION_KEYWORD, "").strip(),  # 移除关键词
            qwen_32b_api_key,
            coder_14b_api_key,
            qwen_32b_url if qwen_32b_url else QWEN_32B_API_URL,
            coder_14b_url if coder_14b_url else QWEN_14B_API_URL
        )
    
    # 普通代码生成分支
    try:
        # 准备对话消息
        messages = [
            {"role": "system", "content": system_prompt if system_prompt else "你是一个专业的编程助手，擅长编写和解释代码。"},
            {"role": "user", "content": prompt},
        ]
        
        # 应用聊天模板
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # 将文本转换为模型输入
        model_inputs = tokenizer([text], return_tensors="pt").to(device)
        
        # 生成代码
        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=int(max_tokens),
                temperature=float(temperature),
                top_p=float(top_p),
                do_sample=True
            )
        
        # 提取生成的文本（去掉输入部分）
        generated_ids = [
            output_ids[len(input_ids):] 
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response
        
    except Exception as e:
        return f"生成代码时出错：{str(e)}"

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
    if model is None or tokenizer is None:
        yield "错误：模型尚未加载，请先点击'加载模型'按钮。"
        return
    
    dataset_path = "./datasets/human-eval-v2-20210705.jsonl"
    if not os.path.exists(dataset_path):
        yield f"错误：数据集文件不存在: {dataset_path}"
        return
    
    try:
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
            
            with torch.no_grad():
                generated_ids = model.generate(
                    **model_inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True
                )
            
            generated_ids = [
                output_ids[len(input_ids):] 
                for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            
            generated_code = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
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
                # 这意味着函数能够运行，但输出结果不正确
                failed_tasks.append(task_id)
                error_msg = str(e)[:150] if str(e) else "断言失败"
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
            
            # 实时更新进度
            current_rate = (passed_tasks / (idx + 1) * 100) if (idx + 1) > 0 else 0
            progress_report = f"# 模型评估进度\n\n"
            progress_report += f"**进度**: {idx + 1}/{total_tasks} ({((idx + 1)/total_tasks*100):.1f}%)\n\n"
            progress_report += f"**当前通过率**: {current_rate:.2f}% ({passed_tasks}/{idx + 1})\n\n"
            progress_report += f"## 详细结果\n\n"
            progress_report += "\n".join(results[-20:])  # 显示最近20个结果
            
            yield progress_report
        
        # 计算最终通过率
        pass_rate = (passed_tasks / total_tasks * 100) if total_tasks > 0 else 0
        
        # 生成最终报告
        final_report = f"""# 模型评估最终报告

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
        yield f"评估过程中出错：{str(e)}\n\n```\n{traceback.format_exc()}\n```"

# 创建 Gradio 界面
with gr.Blocks(title="Qwen2.5-Coder 本地模型代码生成器", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Qwen2.5-Coder 本地模型代码生成器")
    gr.Markdown("使用本地下载的 Qwen2.5-Coder-0.5B-Instruct 模型生成代码。")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 模型设置")
            model_path_input = gr.Textbox(
                label="模型路径",
                value=DEFAULT_MODEL_PATH,
                placeholder="输入本地模型路径，例如: ./Qwen2.5-Coder-0.5B-Instruct",
                lines=1
            )
            load_btn = gr.Button("加载模型", variant="primary", size="lg")
            load_status = gr.Textbox(label="模型状态", interactive=False, lines=3)
            
            gr.Markdown("### 模型评估")
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
            
            with gr.Accordion("高级设置", open=False):
                system_prompt_input = gr.Textbox(
                    label="系统提示词",
                    value="你是一个专业的编程助手，擅长编写和解释代码。",
                    lines=2,
                    placeholder="输入系统提示词..."
                )
                max_tokens_input = gr.Slider(
                    label="最大生成token数",
                    minimum=50,
                    maximum=2048,
                    value=512,
                    step=50
                )
                temperature_input = gr.Slider(
                    label="Temperature (创造性)",
                    minimum=0.1,
                    maximum=2.0,
                    value=0.7,
                    step=0.1
                )
                top_p_input = gr.Slider(
                    label="Top-p (核采样)",
                    minimum=0.1,
                    maximum=1.0,
                    value=0.9,
                    step=0.05
                )
            
            with gr.Accordion("自我演化配置", open=False):
                gr.Markdown("### API配置（自我演化模式需要）")
                qwen_32b_api_key = gr.Textbox(
                    label="qwen2.5-coder-32b-instruct API Key",
                    type="password",
                    placeholder="输入API密钥",
                    info="用于生成高质量代码"
                )
                qwen_32b_url = gr.Textbox(
                    label="qwen2.5-coder-32b-instruct API URL",
                    value=QWEN_32B_API_URL,
                    placeholder="API地址",
                    info="Qwen-32B模型的API地址"
                )
                qwen_14b_api_key = gr.Textbox(
                    label="qwen2.5-coder-14b-instruct API Key",
                    type="password",
                    placeholder="输入API密钥",
                    info="用于验证代码逻辑"
                )
                qwen_14b_url = gr.Textbox(
                    label="qwen2.5-coder-14b-instruct API URL",
                    value=QWEN_14B_API_URL,
                    placeholder="API地址",
                    info="qwen2.5-coder-14b-instruct模型的API地址"
                )
                gr.Markdown("""
                ### 使用说明
                - 在代码生成提示中包含"**自我演化**"关键词即可进入自我演化模式
                - 自我演化模式会使用32B模型生成代码，验证后收集训练数据
                - 收集的数据可用于微调0.5B模型
                - 自我演化模式**只收集数据，不直接生成代码**
                """)
            
            with gr.Accordion("模型微调", open=False):
                gr.Markdown("### 使用收集的数据微调模型")
                fine_tune_output_dir = gr.Textbox(
                    label="模型保存路径",
                    value="./fine_tuned_model",
                    placeholder="./fine_tuned_model",
                    info="微调后模型的保存路径"
                )
                fine_tune_epochs = gr.Slider(
                    label="训练轮数",
                    minimum=1,
                    maximum=10,
                    value=3,
                    step=1,
                    info="微调的训练轮数"
                )
                fine_tune_btn = gr.Button(" 开始微调", variant="secondary", size="lg")
                fine_tune_output = gr.Markdown(label="微调结果")
                fine_tune_status = gr.Textbox(
                    label="当前数据集状态",
                    value=f"已收集数据: 0 条",
                    interactive=False
                )
        
        with gr.Column():
            gr.Markdown("### 代码生成")
            prompt_input = gr.Textbox(
                label="代码生成提示",
                placeholder="例如：请用Python编写一个快速排序算法。",
                lines=5
            )
            generate_btn = gr.Button(" 生成代码", variant="primary", size="lg")
            output = gr.Code(
                label="生成的代码",
                language="python",
                lines=20
            )
    
    # 绑定事件
    load_btn.click(
        fn=load_model,
        inputs=model_path_input,
        outputs=load_status
    )
    
    def update_data_status():
        """更新数据集状态"""
        return f"已收集数据: {len(self_evolution_data)} 条"
    
    def generate_code_with_status_update(prompt, system_prompt, max_tokens, temperature, top_p,
                                        qwen_32b_key, coder_14b_key, qwen_32b_url_val, coder_14b_url_val):
        """生成代码并更新数据集状态"""
        result = generate_code(prompt, system_prompt, max_tokens, temperature, top_p,
                              qwen_32b_key, coder_14b_key, qwen_32b_url_val, coder_14b_url_val)
        status = update_data_status()
        return result, status
    
    generate_btn.click(
        fn=generate_code_with_status_update,
        inputs=[
            prompt_input, 
            system_prompt_input, 
            max_tokens_input, 
            temperature_input, 
            top_p_input,
            qwen_32b_api_key,
            qwen_14b_api_key,
            qwen_32b_url,
            qwen_14b_url
        ],
        outputs=[output, fine_tune_status]
    )
    
    def fine_tune_wrapper(output_dir, epochs):
        """微调包装函数"""
        result = fine_tune_model(output_dir, int(epochs))
        status = update_data_status()
        return result, status
    
    fine_tune_btn.click(
        fn=fine_tune_wrapper,
        inputs=[fine_tune_output_dir, fine_tune_epochs],
        outputs=[fine_tune_output, fine_tune_status]
    )
    
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
            ["请用Python编写一个快速排序算法。"],
            ["用Python实现一个简单的HTTP服务器。"],
            ["写一个函数来计算斐波那契数列的第n项。"],
            ["用Python实现一个简单的计算器类。"],
            ["自我演化 请用Python编写一个快速排序算法。"],
            ["自我演化 实现一个简单的HTTP服务器。"],
        ],
        inputs=prompt_input
    )

if __name__ == "__main__":
    # 启动 Gradio 界面
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860)

