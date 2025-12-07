"""
自我演化核心逻辑模块
"""
import torch
import json
import time
from typing import Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import os

from config import (
    EVOLUTION_CONFIG, 
    API_CONFIG, 
    TRAINING_DATA_DIR, 
    CHECKPOINT_DIR
)
from core.model_manager import get_model
from core.api_client import call_qwen_api, validate_code_with_14b
from core.code_processor import check_code_syntax, run_simple_test
from data.training_data import save_training_example
from utils.text_utils import detect_evolution_mode, extract_problems_from_text
from utils.progress_tracker import create_progress_tracker, update_progress


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
    
    # 步骤1: 使用70B模型生成代码
    success, code = call_qwen_api(
        API_CONFIG["qwen_70b_api_url"],
        problem,
        model_name="Qwen2.5-Coder-70B"
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
    
    # 步骤5: 保存训练数据
    try:
        saved_file = save_training_example(problem, code, logic_msg)
        result["saved_file"] = saved_file
        result["success"] = True
    except Exception as e:
        result["validation_result"] += f"\n保存失败: {str(e)}"
        return False, result
    
    return True, result


def fine_tune_on_examples(examples: List[Dict]) -> str:
    """
    在多个示例上微调模型
    """
    model, tokenizer, device = get_model()
    
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
            os.makedirs(CHECKPOINT_DIR, exist_ok=True)
            checkpoint_path = f"{CHECKPOINT_DIR}/checkpoint_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'examples_count': successful_updates
            }, checkpoint_path)
        else:
            avg_loss = 0
        
        model.eval()
        
        return f"✅ 微调完成，处理了{successful_updates}个示例，平均损失: {avg_loss:.4f}"
        
    except Exception as e:
        model.eval()
        return f"微调失败: {str(e)}"


def batch_self_evolution(problems: List[str], system_prompt: str = None) -> str:
    """
    批量自我演化流程
    """
    if not problems:
        return "❌ 错误：没有提取到有效的编程问题。请确保问题用引号括起来。"
    
    total_problems = len(problems)
    batch_size = EVOLUTION_CONFIG["evolution_batch_size"]
    
    # 创建进度跟踪器
    tracker = create_progress_tracker(total_problems)
    
    report_lines = []
    report_lines.append("🚀 开始批量自我演化流程")
    report_lines.append(f"📋 提取到 {total_problems} 个编程问题")
    report_lines.append(f"📦 批量大小: {batch_size}")
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
        
        report_lines.append(f"\n📁 处理批次 {batch_num}/{total_batches}")
        
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
                        report_lines.append(f"  ✅ 已保存到: {result['saved_file']}")
                    else:
                        report_lines.append(f"  ❌ 失败: {result.get('validation_result', '未知错误')[:80]}...")
                        
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
                for ex in successful_examples[:10]
            ]
        }
        
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        report_lines.append(f"📊 统计数据已保存到: {stats_file}")
    else:
        report_lines.append("\n⚠️ 没有成功的示例，跳过微调")
    
    # 最终报告
    report_lines.append("\n" + "=" * 60)
    report_lines.append("🎉 批量自我演化流程完成！")
    report_lines.append(f"✅ 成功处理: {tracker['success']}/{total_problems}")
    report_lines.append(f"❌ 失败: {tracker['failed']}/{total_problems}")
    report_lines.append(f"⏱️ 总用时: {time.time() - tracker['start_time']:.1f}秒")
    
    if successful_examples:
        report_lines.append(f"💾 模型已更新，检查点已保存")
    
    return "\n".join(report_lines)

def generate_code(prompt, system_prompt, max_tokens, temperature, top_p, enable_evolution=True):
    """生成代码的主函数"""
    model, tokenizer, device = get_model()
    
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
                        status = f"✅ 单问题自我演化完成！\n"
                        status += f"📁 已保存训练数据到: {result['saved_file']}\n"
                        
                        # 微调模型
                        fine_tune_result = fine_tune_on_examples([{
                            "instruction": result["problem"],
                            "code": result["generated_code"]
                        }])
                        status += f"🎯 {fine_tune_result}"
                        
                        return status, result["generated_code"]
                    else:
                        return f"❌ 自我演化失败:\n{result['validation_result']}", ""
                else:
                    return "❌ 错误：请提供要演化的具体问题。", ""
                    
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
            return "✅ 代码生成完成", response
            
        except Exception as e:
            return f"生成代码时出错：{str(e)}", ""