"""
核心功能模块
"""
from .model_manager import load_model, get_model
from .api_client import call_qwen_api, validate_code_with_14b
from .code_processor import check_code_syntax, run_simple_test
from .evolution_core import (
    process_single_problem,
    fine_tune_on_examples,
    batch_self_evolution,
    generate_code  # 添加这行
)

__all__ = [
    'load_model',
    'get_model',
    'call_qwen_api',
    'validate_code_with_14b',
    'check_code_syntax',
    'run_simple_test',
    'process_single_problem',
    'fine_tune_on_examples',
    'batch_self_evolution',
    'generate_code'  # 添加这行
]