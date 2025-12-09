"""
主程序入口
"""
import gradio as gr
from config import (
    DEFAULT_MODEL_PATH, 
    API_CONFIG, 
    EVOLUTION_CONFIG,
    GENERATION_CONFIG,
    EVALUATION_CONFIG,
    FINE_TUNE_CONFIG
)
from core.model_manager import load_model
from core.evolution_core import generate_code, batch_self_evolution
from core.model_evaluation import get_evaluation_help
from core.fine_tune_manager import get_fine_tune_help, get_fine_tune_status
from data.training_data import list_training_data
from ui.event_handlers import (
    update_api_config, 
    update_evolution_config,
    detect_mode, 
    test_problem_extraction,
    evaluate_model_wrapper,
    fine_tune_model_wrapper
)
from utils.text_utils import detect_evolution_mode


def create_gradio_interface():
    """创建Gradio界面"""
    with gr.Blocks(title="Qwen2.5-Coder 批量自我演化系统", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# Qwen2.5-Coder 批量自我演化系统")
        gr.Markdown("""
        ## 功能特性：
        1. **普通代码生成**：使用本地0.5B模型生成代码
        2. **批量自我演化**：输入包含多个引号内的问题，系统自动提取并批量训练
        3. **智能问题提取**：自动从文本中提取引号内的编程问题
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 模型设置")
                model_path_input = gr.Textbox(
                    label="模型路径", value=DEFAULT_MODEL_PATH, lines=1
                )
                load_btn = gr.Button("加载模型", variant="primary", size="lg")
                load_status = gr.Textbox(label="模型状态", interactive=False, lines=3)
                
                with gr.Accordion("API设置", open=False):
                    api_key_input = gr.Textbox(
                        label="API密钥", value=API_CONFIG["api_key"], type="password", lines=1
                    )
                    api_32b_url = gr.Textbox(
                        label="32B API地址", value=API_CONFIG["qwen_32b_api_url"], lines=1
                    )
                    api_14b_url = gr.Textbox(
                        label="14B API地址", value=API_CONFIG["qwen_14b_api_url"], lines=1
                    )
                
                with gr.Accordion("自我演化设置", open=False):
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
                
                with gr.Accordion("数据管理", open=False):
                    with gr.Row():
                        view_data_btn = gr.Button("查看训练数据", variant="secondary")
                        test_extraction_btn = gr.Button("测试问题提取", variant="secondary")
                    
                    training_data_view = gr.Textbox(
                        label="训练数据", interactive=False, lines=10
                    )
                
                with gr.Accordion("生成设置", open=False):
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
                
                with gr.Accordion("模型评估", open=False):
                    gr.Markdown("### HumanEval 数据集评估")
                    with gr.Row():
                        eval_max_tasks = gr.Number(
                            label="评估任务数量",
                            value=EVALUATION_CONFIG["default_max_tasks"],
                            minimum=1,
                            maximum=164,
                            step=1,
                            info="输入要评估的任务数量（1-164）"
                        )
                        eval_all_check = gr.Checkbox(
                            label="评估全部任务",
                            value=False,
                            info="勾选此项将评估所有164个任务"
                        )
                    
                    with gr.Row():
                        eval_max_tokens = gr.Slider(
                            label="最大生成token",
                            minimum=50,
                            maximum=2048,
                            value=EVALUATION_CONFIG["default_max_tokens"],
                            step=50
                        )
                        eval_temperature = gr.Slider(
                            label="Temperature",
                            minimum=0.1,
                            maximum=2.0,
                            value=EVALUATION_CONFIG["default_temperature"],
                            step=0.1
                        )
                        eval_top_p = gr.Slider(
                            label="Top-p",
                            minimum=0.1,
                            maximum=1.0,
                            value=EVALUATION_CONFIG["default_top_p"],
                            step=0.05
                        )
                    
                    eval_btn = gr.Button("🚀 开始评估", variant="secondary", size="lg")
                    eval_output = gr.Markdown(label="评估结果")
                    
                    with gr.Accordion("📖 评估说明", open=False):
                        eval_help_text = get_evaluation_help()
                        gr.Markdown(eval_help_text)
                
                with gr.Accordion("模型微调", open=False):
                    gr.Markdown("### 使用收集的数据微调模型")
                    
                    fine_tune_output_dir = gr.Textbox(
                        label="模型保存路径",
                        value=FINE_TUNE_CONFIG["default_output_dir"],
                        placeholder="./fine_tuned_model",
                        info="微调后模型的保存路径"
                    )
                    
                    with gr.Row():
                        fine_tune_epochs = gr.Slider(
                            label="训练轮数",
                            minimum=1,
                            maximum=10,
                            value=FINE_TUNE_CONFIG["default_num_epochs"],
                            step=1,
                            info="微调的训练轮数"
                        )
                        fine_tune_batch_size = gr.Slider(
                            label="批大小",
                            minimum=1,
                            maximum=8,
                            value=FINE_TUNE_CONFIG["default_batch_size"],
                            step=1,
                            info="每批处理的样本数"
                        )
                        fine_tune_lr = gr.Slider(
                            label="学习率",
                            minimum=1e-6,
                            maximum=1e-3,
                            value=FINE_TUNE_CONFIG["default_learning_rate"],
                            step=1e-6,
                            info="模型学习率"
                        )
                    
                    fine_tune_btn = gr.Button(" 开始微调", variant="secondary", size="lg")
                    fine_tune_output = gr.Markdown(label="微调结果")
                    fine_tune_status = gr.Markdown(get_fine_tune_status)
                    
                    with gr.Accordion(" 微调说明", open=False):
                        fine_tune_help_text = get_fine_tune_help()
                        gr.Markdown(fine_tune_help_text)
            
            with gr.Column(scale=2):
                gr.Markdown("### 代码生成与自我演化")
                
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
                    generate_btn = gr.Button("生成代码", variant="primary", size="lg")
                    evolve_btn = gr.Button("执行自我演化", variant="stop", size="lg")
                
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
        
        # 绑定评估事件
        eval_btn.click(
            fn=evaluate_model_wrapper,
            inputs=[eval_max_tasks, eval_all_check, eval_max_tokens, eval_temperature, eval_top_p],
            outputs=eval_output
        )
        
        # 绑定微调事件
        fine_tune_btn.click(
            fn=fine_tune_model_wrapper,
            inputs=[fine_tune_output_dir, fine_tune_epochs, fine_tune_batch_size, fine_tune_lr],
            outputs=fine_tune_output
        ).then(
            fn=lambda: get_fine_tune_status(),
            outputs=fine_tune_status
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
        # ## 功能说明

        # ### 1. 模型加载
        # - 选择或输入本地模型路径
        # - 点击"加载模型"按钮
        # - 模型加载后才能进行其他操作
        
        # ### 2. 普通代码生成
        # - 输入代码生成提示（不包含演化关键词）
        # - 点击"生成代码"按钮
        # - 模型会生成相应的代码
        
        # ### 3. 批量自我演化
        # - 在输入中包含"自我演化"关键词
        # - 用**双引号**括起每个编程问题
        # - 点击"执行自我演化"按钮
        # - 系统会自动提取问题并进行训练
        
        # ### 4. 模型评估
        # - 选择评估任务数量（建议先用10个测试）
        # - 点击"开始评估"按钮
        # - 系统会在HumanEval数据集上评估模型
        # - 支持流式输出，可实时查看进度
        
        # ### 5. 模型微调
        # - 首先使用自我演化功能收集训练数据
        # - 配置微调参数（训练轮数、批大小、学习率）
        # - 点击"开始微调"按钮
        # - 系统会用收集的数据微调模型
        
        # ### 6. 数据管理
        # - 查看训练数据：查看已收集的训练数据
        # - 测试问题提取：测试从输入中提取问题的能力
        
        # ## 工作流程建议
        
        # 1. **首次使用**
        #    - 加载默认模型
        #    - 进行几次普通代码生成测试
        #    - 运行模型评估（少量任务）
        
        # 2. **数据收集**
        #    - 进行自我演化收集高质量训练数据
        #    - 观察生成的代码质量
        #    - 修改提示词以获得更好的结果
        
        # 3. **模型微调**
        #    - 收集20-50条训练数据后进行微调
        #    - 使用默认参数（3轮epoch）
        #    - 微调完成后重新加载微调模型
        
        # 4. **性能对比**
        #    - 微调前后分别进行评估
        #    - 对比通过率是否有提升
        #    - 根据结果调整训练数据或参数
        
        # 5. **持续优化**
        #    - 定期收集新的训练数据
        #    - 进行增量微调（继续训练）
        #    - 评估性能改进情况
        # """)
    
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