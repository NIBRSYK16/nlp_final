# 使用 Qwen2.5-Coder-3B-Instruct 模型
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# 加载模型和分词器
model_name = "Qwen/Qwen2.5-Coder-0.5B-Instruct"

print("正在加载模型和分词器...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 确定设备
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用设备: {device}")

# 加载模型
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    low_cpu_mem_usage=True
)
model = model.to(device)
model.eval()  # 设置为评估模式

# 准备对话消息
messages = [
    {"role": "system", "content": "你是一个专业的编程助手，擅长编写和解释代码。"},
    {"role": "user", "content": "请用Python编写一个快速排序算法。"},
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
print("正在生成代码...")
with torch.no_grad():
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.9,
        do_sample=True
    )

# 提取生成的文本（去掉输入部分）
generated_ids = [
    output_ids[len(input_ids):] 
    for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print("\n生成的代码：")
print(response)
