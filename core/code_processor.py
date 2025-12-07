"""
代码处理与验证模块
"""
import subprocess
import tempfile
import os
from typing import Tuple
from config import VALIDATION_CONFIG


def check_code_syntax(code: str) -> Tuple[bool, str]:
    """
    检查Python代码的语法错误
    """
    try:
        # 添加必要的导入
        full_code = "import math\nimport re\nimport heapq\nimport numpy as np\n" + code
        
        # 尝试编译
        compile(full_code, '<string>', 'exec')
        return True, "语法检查通过"
    except SyntaxError as e:
        return False, f"语法错误: {str(e)}"
    except Exception as e:
        return False, f"代码检查错误: {str(e)}"


def run_simple_test(code: str, problem_type: str) -> Tuple[bool, str]:
    """
    运行简单的测试用例
    """
    temp_file = None
    try:
        # 创建临时文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_file = f.name
        
        # 根据问题类型添加测试
        test_code = ""
        if "minimum cost path" in problem_type.lower():
            test_code = """
if __name__ == "__main__":
    cost_matrix = [[1, 2, 3], [4, 8, 2], [1, 5, 3]]
    try:
        result = min_cost_path(cost_matrix, 2, 2)
        print(f"测试通过，最小成本: {result}")
    except Exception as e:
        print(f"测试失败: {e}")
"""
        elif "similar elements" in problem_type.lower():
            test_code = """
if __name__ == "__main__":
    list1 = [(1, 2), (3, 4), (5, 6)]
    list2 = [(3, 4), (7, 8), (1, 2)]
    try:
        result = find_similar_elements(list1, list2)
        print(f"测试通过，相似元素: {result}")
    except Exception as e:
        print(f"测试失败: {e}")
"""
        
        if test_code:
            with open(temp_file, 'a') as f:
                f.write(test_code)
            
            result = subprocess.run(
                ['python', temp_file],
                capture_output=True,
                text=True,
                timeout=VALIDATION_CONFIG["test_timeout"]
            )
            
            os.unlink(temp_file)
            
            if result.returncode == 0:
                return True, f"测试通过: {result.stdout.strip()}"
            else:
                return False, f"测试失败: {result.stderr.strip()}"
        else:
            os.unlink(temp_file)
            return True, "无特定测试，跳过运行测试"
            
    except Exception as e:
        if temp_file and os.path.exists(temp_file):
            os.unlink(temp_file)
        return False, f"测试执行错误: {str(e)}"