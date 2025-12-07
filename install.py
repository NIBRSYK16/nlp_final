from transformers import AutoModel, AutoTokenizer
import os

# 设置镜像
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 下载模型和分词器
model_name = "Qwen/Qwen2.5-0.5B"

print("开始下载模型...")
model = AutoModel.from_pretrained(
    model_name,
    trust_remote_code=True,
    cache_dir="./models"
)

print("开始下载分词器...")
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True,
    cache_dir="./models"
)

print("下载完成！")