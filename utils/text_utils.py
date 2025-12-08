"""
文本处理工具函数
"""
import re
from typing import List, Tuple
from config import EVOLUTION_CONFIG


def extract_problems_from_text(text: str) -> List[str]:
    """
    从文本中提取引号内的编程问题
    """
    # 移除演化关键词
    clean_text = text.lower()
    for keyword in EVOLUTION_CONFIG["evolution_keywords"]:
        clean_text = clean_text.replace(keyword.lower(), "")
    
    # 提取所有引号内的内容
    problems = []
    
    # 提取双引号内容
    double_quote_pattern = r'"([^"]*)"'
    for match in re.findall(double_quote_pattern, text):
        if match.strip() and len(match.strip()) > 10:
            problems.append(match.strip())
    
    # 提取单引号内容
    single_quote_pattern = r"'([^']*)'"
    for match in re.findall(single_quote_pattern, text):
        if match.strip() and len(match.strip()) > 10:
            problems.append(match.strip())
    
    # 如果没有找到引号内容，尝试按行分割
    if not problems:
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if line and len(line) > 20 and not line.startswith('#'):
                clean_line = re.sub(r'^\s*\d+[\.\)]?\s*', '', line)
                clean_line = re.sub(r'^\s*[•\-*]\s*', '', clean_line)
                clean_line = clean_line.strip()
                if clean_line and len(clean_line) > 20:
                    problems.append(clean_line)
    
    # 去重
    unique_problems = []
    seen = set()
    for problem in problems:
        problem_lower = problem.lower()
        if problem_lower not in seen:
            seen.add(problem_lower)
            unique_problems.append(problem)
    
    return unique_problems


def detect_evolution_mode(prompt: str) -> Tuple[bool, List[str]]:
    """
    检测是否进入自我演化模式，并提取问题
    """
    should_evolve = False
    for keyword in EVOLUTION_CONFIG["evolution_keywords"]:
        if keyword.lower() in prompt.lower():
            should_evolve = True
            break
    
    if not should_evolve:
        return False, []
    
    problems = extract_problems_from_text(prompt)
    return True, problems


def classify_problem(instruct: str) -> str:
    """
    分类问题类型
    """
    keywords = {
        "path": ["minimum cost path", "cost matrix", "reach", "grid"],
        "search": ["find", "search", "identify", "check"],
        "sort": ["sort", "largest", "smallest", "order"],
        "string": ["string", "character", "words", "regex"],
        "math": ["prime", "number", "bit", "volume", "rotations"],
        "data_structure": ["heap", "matrix", "list", "tuple", "dictionary"]
    }
    
    instruct_lower = instruct.lower()
    for category, words in keywords.items():
        for word in words:
            if word in instruct_lower:
                return category
    
    return "general"