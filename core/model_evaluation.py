"""
模型评估模块 - 处理模型在HumanEval数据集上的评估
"""
import json
import os
import re
import torch
from typing import Tuple, Generator, Dict, List
from config import GENERATION_CONFIG


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
    lines = text.split('\n')
    if lines and (lines[0].startswith('    ') or lines[0].startswith('\t')):
        return '\n'.join(lines)
    
    # 方式4: 如果都没有找到，返回原始文本
    return text


def evaluate_model_on_humaneval(model, tokenizer, device, 
                                max_tasks: int = None, 
                                max_tokens: int = 512, 
                                temperature: float = 0.7, 
                                top_p: float = 0.9) -> Generator[str, None, None]:
    """
    评估模型在 HumanEval 数据集上的表现（生成器函数，支持流式输出）
    
    Args:
        model: 加载的模型
        tokenizer: 分词器
        device: 计算设备
        max_tasks: 最多评估多少个任务（None表示全部）
        max_tokens: 最大生成token数
        temperature: 温度参数
        top_p: nucleus采样参数
    
    Yields:
        str: 评估进度报告（Markdown格式）
    """
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
        
        system_prompt = GENERATION_CONFIG["default_system_prompt"]
        
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
                    results.append(f"✅ {task_id}: 通过")
                else:
                    failed_tasks.append(task_id)
                    results.append(f"❌ {task_id}: 函数未找到（生成的代码中可能没有正确的函数定义）")
                    
            except AssertionError as e:
                failed_tasks.append(task_id)
                error_msg = str(e)[:150] if str(e) else "断言失败"
                results.append(f"❌ {task_id}: 测试失败（函数输出不符合预期）")
            except SyntaxError as e:
                failed_tasks.append(task_id)
                error_msg = str(e)[:150]
                results.append(f"❌ {task_id}: 语法错误 - {error_msg}")
            except NameError as e:
                failed_tasks.append(task_id)
                error_msg = str(e)[:150]
                results.append(f"❌ {task_id}: 名称错误 - {error_msg}")
            except Exception as e:
                failed_tasks.append(task_id)
                error_type = type(e).__name__
                error_msg = str(e)[:150]
                results.append(f"❌ {task_id}: {error_type} - {error_msg}")
            
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
            final_report += ", ".join(failed_tasks[:20])
            if len(failed_tasks) > 20:
                final_report += f" ... (还有 {len(failed_tasks) - 20} 个)"
        
        yield final_report
        
    except Exception as e:
        import traceback
        yield f"评估过程中出错：{str(e)}\n\n```\n{traceback.format_exc()}\n```"


def get_evaluation_help() -> str:
    """获取评估说明文档"""
    return """
# 📖 模型评估说明

## HumanEval 数据集概述
- **总任务数**: 164 个Python编程任务
- **任务类型**: 函数实现任务
- **评估标准**: 函数必须通过所有测试用例才算通过

## 评估结果说明

### ✅ 通过
- 生成的函数通过了所有测试用例
- 说明模型能够正确理解需求并生成有效的代码

### ❌ 测试失败
- 函数能够正常执行，但输出结果不符合预期
- 说明生成的代码逻辑有误
- 可能原因：
  - 返回值错误
  - 边界条件处理不当
  - 算法实现有偏差

### ❌ 语法错误
- 生成的代码存在Python语法问题
- 可能原因：
  - 缺少冒号
  - 括号不匹配
  - 缩进错误
  - 无效的Python语法

### ❌ 名称错误
- 代码中使用了未定义的变量或函数
- 可能原因：
  - 缺少必要的导入语句
  - 引用了不存在的变量
  - 函数名拼写错误

### ❌ 函数未找到
- 生成的代码中没有找到目标函数
- 可能原因：
  - 函数名不匹配
  - 代码格式问题
  - 提取函数代码失败

### ❌ 其他执行错误
- 运行时出现的其他错误
- 可能原因：
  - 类型错误
  - 索引越界
  - 递归深度超限
  - 其他运行时异常

## 性能指标

### 通过率 (Pass Rate)
- 计算方式: (通过任务数 / 总任务数) × 100%
- 用于评估模型的整体性能
- 越高越好，50%以上说明模型具有一定能力

### 评估建议
1. 首先用少量任务（10-20个）进行快速测试
2. 然后逐步增加任务数量进行完整评估
3. 不同的温度/top_p参数可能影响生成质量
4. 建议多次运行取平均值

## 参数说明

### 评估任务数量
- 范围: 1-164
- 默认: 10（建议先用这个测试）
- 建议: 先用少量任务快速测试，满意后再做完整评估

### 最大生成Token数
- 范围: 50-2048
- 默认: 512
- 说明: 越大生成的代码越长，但耗时越多

### Temperature (创造性)
- 范围: 0.1-2.0
- 默认: 0.7
- 说明:
  - 低于1.0: 更稳定、保守的生成
  - 接近0.0: 确定性生成（总是选最可能的token）
  - 高于1.0: 更多样化、富有创意的生成

### Top-p (核采样)
- 范围: 0.1-1.0
- 默认: 0.9
- 说明:
  - 值越小: 采样范围越小，生成越稳定
  - 值为1.0: 考虑所有token，最自由的采样

## 评估时间估计

| 任务数 | 估计时间 |
|------|---------|
| 10   | 2-5分钟  |
| 50   | 10-20分钟 |
| 100  | 20-40分钟 |
| 164  | 30-60分钟 |

注: 实际时间取决于模型大小和硬件配置

## 常见问题

**Q: 为什么评估这么慢?**
A: 需要对每个任务进行代码生成、执行和测试，这都需要时间。建议先用少量任务测试。

**Q: 通过率低是正常的吗?**
A: 对于小模型（0.5B-1.5B），50%以下的通过率是正常的。大模型（32B+）通过率通常在70%以上。

**Q: 能否加速评估过程?**
A: 可以尝试减少max_tokens、降低temperature来加速，但可能影响生成质量。

**Q: 评估结果有缓存吗?**
A: 没有缓存，每次评估都会重新生成和测试代码。
"""
