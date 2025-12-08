"""
界面模块
"""
from .event_handlers import (
    update_api_config,
    update_evolution_config,
    detect_mode,
    test_problem_extraction
)

__all__ = [
    'update_api_config',
    'update_evolution_config',
    'detect_mode',
    'test_problem_extraction'
]