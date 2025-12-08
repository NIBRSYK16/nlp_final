"""
主程序入口
"""
import gradio as gr
from config import (
    DEFAULT_MODEL_PATH, 
    API_CONFIG, 
    EVOLUTION_CONFIG,
    GENERATION_CONFIG
)
from core.model_manager import load_model
from core.evolution_core import generate_code, batch_self_evolution
from data.training_data import list_training_data
from ui.event_handlers import (
    update_api_config, 
    update_evolution_config,
    detect_mode, 
    test_problem_extraction
)
from utils.text_utils import detect_evolution_mode


def create_gradio_interface():
    """创建Gradio界面"""
    with gr.Blocks(title="Qwen2.5-Coder 批量自我演化系统", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🤖 Qwen2.5-Coder 批量自我演化系统")
        gr.Markdown("""
        ## 🚀 功能特性：
        1. **普通代码生成**：使用本地1.5B模型生成代码
        2. **批量自我演化**：输入包含多个引号内的问题，系统自动提取并批量训练
        3. **智能问题提取**：自动从文本中提取引号内的编程问题
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📁 模型设置")
                model_path_input = gr.Textbox(
                    label="模型路径", value=DEFAULT_MODEL_PATH, lines=1
                )
                load_btn = gr.Button("🔄 加载模型", variant="primary", size="lg")
                load_status = gr.Textbox(label="模型状态", interactive=False, lines=3)
                
                with gr.Accordion("🔑 API设置", open=False):
                    api_key_input = gr.Textbox(
                        label="API密钥", value=API_CONFIG["api_key"], type="password", lines=1
                    )
                    api_70b_url = gr.Textbox(
                        label="70B API地址", value=API_CONFIG["qwen_70b_api_url"], lines=1
                    )
                    api_14b_url = gr.Textbox(
                        label="14B API地址", value=API_CONFIG["qwen_14b_api_url"], lines=1
                    )
                
                with gr.Accordion("⚙️ 自我演化设置", open=False):
                    enable_evolution = gr.Checkbox(
                        label="启用自我演化", value=EVOLUTION_CONFIG["enable_self_evolution"]
                    )
                    evolution_keywords = gr.Textbox(
                        label="演化关键词", value=",".join(EVOLUTION_CONFIG["evolution_keywords"]), lines=2
                    )
                    batch_size = gr.Slider(
                        label="批量大小", minimum=1, maximum=10, 
                        value=EVOLUTION_CONFIG["evolution_batch_size"], step=1
                    )
                    learning_rate = gr.Slider(
                        label="学习率", minimum=1e-6, maximum=1e-3, 
                        value=EVOLUTION_CONFIG["learning_rate"], step=1e-6
                    )
                
                with gr.Accordion("📊 数据管理", open=False):
                    with gr.Row():
                        view_data_btn = gr.Button("查看训练数据", variant="secondary")
                        test_extraction_btn = gr.Button("测试问题提取", variant="secondary")
                    
                    training_data_view = gr.Textbox(
                        label="训练数据", interactive=False, lines=10
                    )
                
                with gr.Accordion("⚙️ 生成设置", open=False):
                    system_prompt_input = gr.Textbox(
                        label="系统提示词",
                        value=GENERATION_CONFIG["default_system_prompt"],
                        lines=2
                    )
                    max_tokens_input = gr.Slider(
                        label="最大token数", minimum=50, maximum=2048, 
                        value=GENERATION_CONFIG["default_max_tokens"], step=50
                    )
                    temperature_input = gr.Slider(
                        label="Temperature", minimum=0.1, maximum=2.0, 
                        value=GENERATION_CONFIG["default_temperature"], step=0.1
                    )
                    top_p_input = gr.Slider(
                        label="Top-p", minimum=0.1, maximum=1.0, 
                        value=GENERATION_CONFIG["default_top_p"], step=0.05
                    )
            
            with gr.Column(scale=2):
                gr.Markdown("### 💻 代码生成与自我演化")
                
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
                
                with gr.Row():
                    generate_btn = gr.Button("✨ 生成代码", variant="primary", size="lg")
                    evolve_btn = gr.Button("🚀 执行自我演化", variant="stop", size="lg")
                
                status_output = gr.Textbox(
                    label="执行状态", interactive=False, lines=12
                )
                
                code_output = gr.Code(
                    label="生成的代码", language="python", lines=20
                )
        
        # ====== 绑定事件 ======
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
        
        evolve_btn.click(
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
            inputs=[api_key_input, api_70b_url, api_14b_url],
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
        gr.Markdown("""
        ## 📖 使用说明：
        
        ### 1. 普通代码生成：
        - 输入普通的代码生成提示
        - 点击"生成代码"按钮
        
        ### 2. 批量自我演化：
        - 在输入中包含"自我演化"关键词
        - 用**双引号**括起每个编程问题
        - 每个问题占一行或使用分隔符
        - 点击"执行自我演化"按钮
        
        ### 3. 输入格式示例：
        ```
        请自我演化
        "Write a function to find the minimum cost path..."
        "Write a function to find the similar elements..."
        "Write a python function to identify non-prime numbers..."
        ```
        
        ### 4. 系统流程：
        1. 检测"自我演化"关键词
        2. 提取所有引号内的问题
        3. 对每个问题：
           - 调用70B API生成代码
           - 14B模型验证代码逻辑
           - 语法检查
           - 保存训练数据
        4. 用所有成功的问题微调本地1.5B模型
        5. 返回处理报告
        
        ### 5. 注意事项：
        - API密钥需要正确配置
        - 自我演化过程可能需要几分钟时间
        - 模型微调后会保存检查点
        - 训练数据保存在`./evolution_training_data/`目录
        """)
    
    return demo


if __name__ == "__main__":
    # 启动 Gradio 界面
    demo = create_gradio_interface()
    demo.launch(
        share=False, 
        server_name="0.0.0.0", 
        server_port=7860,
        show_api=False
    )