# 使用本地 Qwen2.5-Coder-1.5B 模型的优化版 Gradio 界面
# 使用量化等技术加速推理
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import gradio as gr
import os

# 全局变量存储模型和分词器
model = None
tokenizer = None
device = None

# 设置本地模型路径（请根据你的实际路径修改）
DEFAULT_MODEL_PATH = "./Qwen2.5-Coder-1.5B"

def load_model(model_path=None, use_quantization=True, quantization_type="8bit"):
    """加载模型和分词器（优化版）"""
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
        
        # 配置量化（如果使用GPU且启用量化）
        quantization_config = None
        if use_quantization and torch.cuda.is_available():
            if quantization_type == "8bit":
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_threshold=6.0,
                )
                print("使用 8-bit 量化加载模型...")
            elif quantization_type == "4bit":
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                print("使用 4-bit 量化加载模型...")
        
        # 从本地路径加载模型
        load_kwargs = {
            "local_files_only": True,
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }
        
        if quantization_config:
            load_kwargs["quantization_config"] = quantization_config
        else:
            # 如果没有量化，使用半精度（GPU）或全精度（CPU）
            load_kwargs["dtype"] = torch.float16 if torch.cuda.is_available() else torch.float32
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            **load_kwargs
        )
        
        # 如果使用量化，模型已经在GPU上，不需要手动移动
        if not quantization_config:
            model = model.to(device)
        
        model.eval()  # 设置为评估模式
        
        # 尝试使用 torch.compile 加速（PyTorch 2.0+）
        try:
            if hasattr(torch, 'compile') and torch.cuda.is_available():
                print("使用 torch.compile 优化模型...")
                model = torch.compile(model, mode="reduce-overhead")
        except Exception as e:
            print(f"torch.compile 不可用或失败: {e}")
        
        print("模型加载完成！")
        
        # 显示模型信息
        model_size = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**2)
        info = f"✅ 模型加载完成！\n"
        info += f"模型路径: {model_path}\n"
        info += f"使用设备: {device}\n"
        info += f"量化: {'是 (' + quantization_type + ')' if quantization_config else '否'}\n"
        info += f"模型大小: {model_size:.1f} MB"
        
        return info
        
    except Exception as e:
        return f"❌ 加载模型时出错：{str(e)}\n提示：如果使用量化，请确保安装了 bitsandbytes: pip install bitsandbytes"

def generate_code(prompt, system_prompt, max_tokens, temperature, top_p, use_cache=True):
    """生成代码的函数（优化版）"""
    if model is None or tokenizer is None:
        return "错误：模型尚未加载，请先点击'加载模型'按钮。"
    
    if not prompt or prompt.strip() == "":
        return "错误：请输入代码生成提示。"
    
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
        
        # 生成代码（优化参数）
        with torch.no_grad():
            # 使用 torch.inference_mode() 进一步优化（如果可用）
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=int(max_tokens),
                temperature=float(temperature),
                top_p=float(top_p),
                do_sample=True,
                use_cache=use_cache,  # 使用KV缓存加速
                pad_token_id=tokenizer.eos_token_id,  # 避免警告
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

# 创建 Gradio 界面
with gr.Blocks(title="Qwen2.5-Coder 本地模型代码生成器（优化版）", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🚀 Qwen2.5-Coder 本地模型代码生成器（优化版）")
    gr.Markdown("使用量化等技术优化的本地模型，提升推理速度。")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 📁 模型设置")
            model_path_input = gr.Textbox(
                label="模型路径",
                value=DEFAULT_MODEL_PATH,
                placeholder="输入本地模型路径，例如: ./Qwen2.5-Coder-1.5B",
                lines=1
            )
            
            with gr.Row():
                use_quantization_check = gr.Checkbox(
                    label="使用量化加速（需要GPU）",
                    value=True,
                    info="使用8-bit或4-bit量化可以大幅减少显存占用和提升速度"
                )
                quantization_type_dropdown = gr.Dropdown(
                    label="量化类型",
                    choices=["8bit", "4bit"],
                    value="8bit",
                    info="4-bit更快但可能略微降低质量"
                )
            
            load_btn = gr.Button("🔄 加载模型", variant="primary", size="lg")
            load_status = gr.Textbox(label="模型状态", interactive=False, lines=5)
            
            with gr.Accordion("⚙️ 生成参数设置", open=False):
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
                use_cache_check = gr.Checkbox(
                    label="使用KV缓存加速",
                    value=True,
                    info="启用可以加速生成，但会占用更多显存"
                )
        
        with gr.Column():
            gr.Markdown("### 💻 代码生成")
            prompt_input = gr.Textbox(
                label="代码生成提示",
                placeholder="例如：请用Python编写一个快速排序算法。",
                lines=5
            )
            generate_btn = gr.Button("✨ 生成代码", variant="primary", size="lg")
            output = gr.Code(
                label="生成的代码",
                language="python",
                lines=20
            )
    
    # 绑定事件
    def load_model_wrapper(model_path, use_quant, quant_type):
        return load_model(model_path, use_quant, quant_type)
    
    load_btn.click(
        fn=load_model_wrapper,
        inputs=[model_path_input, use_quantization_check, quantization_type_dropdown],
        outputs=load_status
    )
    
    generate_btn.click(
        fn=generate_code,
        inputs=[prompt_input, system_prompt_input, max_tokens_input, temperature_input, top_p_input, use_cache_check],
        outputs=output
    )
    
    # 示例提示词
    gr.Examples(
        examples=[
            ["请用Python编写一个快速排序算法。"],
            ["用Python实现一个简单的HTTP服务器。"],
            ["写一个函数来计算斐波那契数列的第n项。"],
            ["用Python实现一个简单的计算器类。"],
        ],
        inputs=prompt_input
    )
    
    # 添加说明
    gr.Markdown("""
    ### 💡 优化说明
    
    **为什么本地模型比API慢？**
    - API服务器通常使用强大的GPU集群
    - 本地可能使用CPU运行（CPU比GPU慢很多）
    - 没有使用量化等优化技术
    
    **本优化版本包含：**
    - ✅ 8-bit/4-bit量化：减少显存占用，提升速度（需要GPU）
    - ✅ KV缓存：加速生成过程
    - ✅ torch.compile：PyTorch 2.0+编译优化
    - ✅ 半精度推理：减少显存占用
    
    **如果仍然很慢：**
    - 确保使用GPU（量化需要GPU）
    - 安装 bitsandbytes: `pip install bitsandbytes`
    - 考虑使用更小的模型或更专业的推理库（如vLLM）
    """)

if __name__ == "__main__":
    # 启动 Gradio 界面
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860)

