"""
模型微调管理模块 - 处理模型微调和训练数据管理
"""
import os
import torch
import json
from typing import List, Dict, Tuple
from datetime import datetime
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling

try:
    from datasets import Dataset
except ImportError:
    Dataset = None

from config import TRAINING_DATA_DIR, CHECKPOINT_DIR, EVOLUTION_CONFIG


def fine_tune_model_with_data(model, tokenizer, device, 
                              training_data: List[Dict],
                              output_dir: str = None,
                              num_epochs: int = 3,
                              batch_size: int = 1,
                              learning_rate: float = 5e-5) -> Tuple[bool, str]:
    """
    使用给定的训练数据微调模型
    
    Args:
        model: 要微调的模型
        tokenizer: 分词器
        device: 计算设备
        training_data: 训练数据列表，每个元素为 {"instruct": "...", "code": "..."} 或 {"instruction": "...", "code": "..."}
        output_dir: 模型保存目录
        num_epochs: 训练轮数
        batch_size: 批大小
        learning_rate: 学习率
    
    Returns:
        (success: bool, message: str)
    """
    if model is None or tokenizer is None:
        return False, "错误：模型尚未加载"
    
    if not training_data:
        return False, "错误：没有训练数据"
    
    if output_dir is None:
        output_dir = "./fine_tuned_model"
    
    try:
        # 准备训练数据
        def format_instruction(example):
            # 支持不同的字段名
            instruct = example.get("instruct") or example.get("instruction") or ""
            code = example.get("code", "")
            
            messages = [
                {"role": "system", "content": "你是一个专业的编程助手，擅长编写和解释代码。"},
                {"role": "user", "content": instruct},
                {"role": "assistant", "content": code}
            ]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            return {"text": text}
        
        # 格式化数据
        formatted_data = [format_instruction(item) for item in training_data]
        
        # 创建数据集
        if Dataset is not None:
            dataset = Dataset.from_list(formatted_data)
            
            # Tokenize 数据
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
            # 备选方案：不使用datasets库
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
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=4,
            warmup_steps=10,
            logging_steps=10,
            save_steps=50,
            evaluation_strategy="no",
            save_total_limit=2,
            load_best_model_at_end=False,
            push_to_hub=False,
            report_to="none",
            learning_rate=learning_rate,
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
        
        # 保存微调元数据
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "training_data_size": len(training_data),
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "output_dir": output_dir,
        }
        
        metadata_file = os.path.join(output_dir, "finetune_metadata.json")
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        message = f"""模型微调完成！

**训练数据量**: {len(training_data)} 条
**训练轮数**: {num_epochs}
**批大小**: {batch_size}
**学习率**: {learning_rate}
**模型保存路径**: {output_dir}

微调后的模型已保存，可以重新加载使用。"""
        
        return True, message
        
    except Exception as e:
        import traceback
        return False, f"微调过程出错：{str(e)}\n\n```\n{traceback.format_exc()}\n```"


def get_training_data_from_files(data_dir: str = None) -> List[Dict]:
    """
    从文件中加载训练数据
    
    Args:
        data_dir: 数据目录
    
    Returns:
        训练数据列表
    """
    if data_dir is None:
        data_dir = TRAINING_DATA_DIR
    
    training_data = []
    
    if not os.path.exists(data_dir):
        return training_data
    
    # 遍历目录中的所有JSON文件
    for filename in os.listdir(data_dir):
        if filename.endswith('.json'):
            filepath = os.path.join(data_dir, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                    # 支持不同的数据格式
                    if isinstance(data, list):
                        training_data.extend(data)
                    elif isinstance(data, dict):
                        if "instruct" in data and "code" in data:
                            training_data.append(data)
                        elif "instruction" in data and "code" in data:
                            training_data.append(data)
            except Exception as e:
                print(f"加载文件 {filepath} 时出错: {str(e)}")
    
    return training_data


def get_fine_tune_status() -> str:
    """获取微调状态信息"""
    status = "# 微调状态\n\n"
    
    # 检查训练数据
    training_data = get_training_data_from_files()
    status += f"**收集的训练数据**: {len(training_data)} 条\n\n"
    
    # 检查检查点
    if os.path.exists(CHECKPOINT_DIR):
        checkpoints = [f for f in os.listdir(CHECKPOINT_DIR) if f.endswith('.pt')]
        status += f"**保存的检查点**: {len(checkpoints)} 个\n\n"
    else:
        status += "**保存的检查点**: 0 个\n\n"
    
    # 检查微调模型
    fine_tuned_models = []
    for item in os.listdir('.'):
        if 'fine_tuned' in item and os.path.isdir(item):
            fine_tuned_models.append(item)
    
    if fine_tuned_models:
        status += f"**微调后的模型**: {len(fine_tuned_models)} 个\n\n"
        for model_dir in fine_tuned_models:
            status += f"  - {model_dir}\n"
    else:
        status += "**微调后的模型**: 未发现\n\n"
    
    return status


def get_fine_tune_help() -> str:
    """获取微调功能的帮助文档"""
    return """
# 🔧 模型微调说明

## 什么是模型微调？

模型微调是指在预训练模型的基础上，使用特定的训练数据进行二次训练，以适应特定的任务或域。

## 微调的优势

1. **提高特定任务性能**: 通过在特定数据上训练，提升模型在该任务上的表现
2. **快速适应新需求**: 只需少量数据就能让模型适应新的编码风格或问题类型
3. **保留基础知识**: 保留预训练模型的通用知识，避免从零开始训练
4. **降低计算成本**: 相比完整训练，微调所需的计算资源更少

## 微调流程

### 1. 收集训练数据
- 使用"自我演化"功能收集高质量的代码示例
- 或从 `evolution_training_data/` 目录加载已有的训练数据
- 每条数据包含 `instruction`（问题）和 `code`（代码解决方案）

### 2. 配置微调参数
- **训练轮数** (Epochs): 数据集遍历的次数，通常3-10次
- **批大小** (Batch Size): 每次更新处理的样本数，默认1
- **学习率** (Learning Rate): 参数更新的步长，默认5e-5
- **输出目录**: 微调后模型的保存位置

### 3. 执行微调
- 点击"开始微调"按钮
- 系统将使用收集的数据对模型进行训练
- 微调过程中会保存模型检查点

### 4. 使用微调模型
- 微调后的模型保存在指定目录
- 可以重新加载该模型进行后续代码生成

## 微调参数指南

### 训练轮数 (Epochs)
| 值 | 数据量 | 适用场景 |
|----|------|---------|
| 1  | < 10 | 快速测试 |
| 3  | 10-50 | 标准微调 |
| 5  | 50-100 | 深度微调 |
| 10+ | 100+ | 充分训练 |

**建议**: 
- 数据少（<20条）：2-3轮
- 数据适量（20-100条）：3-5轮
- 数据充足（100+条）：5-10轮

### 学习率 (Learning Rate)
- **较小** (1e-6 ~ 1e-5): 更稳定，学习速度慢
- **中等** (5e-5): 推荐值，平衡稳定性和速度
- **较大** (1e-4+): 学习速度快，但可能不稳定

**建议**: 使用默认值 5e-5，除非有特殊需求

### 批大小 (Batch Size)
- **较小** (1): 更新频繁，但噪声大
- **中等** (2-4): 推荐值，平衡速度和稳定性
- **较大** (8+): 更新频繁度低，但更稳定

**建议**: 对于小模型使用1-2，对于大模型使用2-4

## 微调效果评估

### 如何判断微调是否有效？

1. **使用评估功能**
   - 微调前评估模型性能（基准）
   - 微调后再次评估
   - 比较通过率是否有提升

2. **观察训练损失**
   - 训练过程中损失应该逐渐下降
   - 如果损失没有下降，说明学习率可能太低

3. **定性测试**
   - 输入一些微调数据相关的问题
   - 观察生成的代码质量是否改进

### 微调前后对比示例

**微调前**:
```python
# 生成的代码可能不够符合特定风格或有逻辑错误
def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    # 可能缺少具体实现
```

**微调后**:
```python
# 生成的代码更加稳定和准确
def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort(left) + middle + quick_sort(right)
```

## 常见问题

**Q: 需要多少训练数据才能微调？**
A: 至少 5-10 条高质量数据可以开始，建议 20+ 条获得更好效果。

**Q: 微调会覆盖原始模型吗？**
A: 不会，微调后的模型保存在单独的目录。可以同时保留原始模型和微调模型。

**Q: 如何恢复原始模型？**
A: 原始模型仍在 `./models/Qwen2.5-Coder-0.5B-Instruct` 目录，随时可以重新加载。

**Q: 微调可以进行多次吗？**
A: 可以，每次收集新数据后都可以进行微调，逐步改进模型。

**Q: 微调后模型的大小会增加吗？**
A: 不会，微调只是调整参数，模型大小保持不变。

## 最佳实践

1. **数据质量优先**
   - 确保训练数据正确且高质量
   - 宁可用少量优质数据也不要用大量低质数据

2. **增量微调**
   - 首先用小的数据集测试微调效果
   - 然后逐步增加数据量进行完整微调

3. **定期评估**
   - 每次微调后都运行评估来验证效果
   - 记录不同配置的结果用于对比

4. **保存检查点**
   - 系统会自动保存微调过程中的检查点
   - 如果效果不理想，可以恢复到之前的版本

5. **参数调优**
   - 如果效果不好，尝试调整学习率
   - 通常降低学习率会带来更稳定的训练
"""
