"""
API客户端模块
"""
import time
import requests
import re
from typing import Tuple
from config import API_CONFIG, VALIDATION_CONFIG


def call_qwen_api(
    api_url: str, 
    prompt: str, 
    model_name: str = "qwen2.5-coder-32b-instruct", 
    max_tokens: int = 1024, 
    temperature: float = 0.7, 
    retries: int = None
) -> Tuple[bool, str]:
    """
    调用Qwen API生成代码
    """
    if retries is None:
        retries = VALIDATION_CONFIG["max_retries"]
    
    headers = {
        "Authorization": f"Bearer {API_CONFIG['api_key']}",
        "Content-Type": "application/json"
    }
    
    messages = [
        {"role": "system", "content": "你是一个专业的编程助手，请生成高质量、可运行的Python代码。"},
        {"role": "user", "content": prompt}
    ]
    
    payload = {
        "model": model_name,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 0.9
    }
    
    for attempt in range(retries):
        try:
            response = requests.post(
                api_url, 
                headers=headers, 
                json=payload, 
                timeout=VALIDATION_CONFIG["timeout_seconds"]
            )
            response.raise_for_status()
            result = response.json()
            generated_code = result["choices"][0]["message"]["content"]
            
            # 提取代码块
            code_pattern = r"```(?:python)?\n?(.*?)```"
            matches = re.findall(code_pattern, generated_code, re.DOTALL)
            
            if matches:
                generated_code = matches[0].strip()
            
            return True, generated_code
        except requests.exceptions.RequestException as e:
            if attempt == retries - 1:
                return False, f"API调用失败（尝试{retries}次）: {str(e)}"
            time.sleep(1)
        except Exception as e:
            return False, f"API处理失败: {str(e)}"
    
    return False, "未知错误"


def validate_code_with_14b(instruct: str, code: str) -> Tuple[bool, str]:
    """
    使用14B模型验证代码是否符合指令逻辑
    """
    validation_prompt = f"""
    请分析以下代码是否符合用户指令的逻辑要求：
    
    用户指令：{instruct}
    
    生成的代码：
    ```python
    {code}
    ```
    
    请从以下几个方面进行判断：
    1. 代码是否完整实现了指令要求的功能
    2. 代码逻辑是否正确
    3. 是否有明显的逻辑错误或缺失
    
    请用以下格式回答：
    [是否通过]：是/否
    [理由]：简要说明理由
    """
    
    success, response = call_qwen_api(
        API_CONFIG["qwen_14b_api_url"], 
        validation_prompt, 
        model_name="qwen2.5-coder-14b-instruct",
        max_tokens=256,
        temperature=0.3
    )
    
    if not success:
        return False, response
    
    # 解析响应
    if "[是否通过]：是" in response or "通过" in response and "否" not in response:
        return True, response
    else:
        return False, response