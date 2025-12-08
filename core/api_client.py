"""
API客户端模块
"""
import time
import re
from typing import Tuple, Optional
from openai import OpenAI
from config import API_CONFIG, VALIDATION_CONFIG

# 初始化 OpenAI 客户端（兼容 DashScope）
_client = None

def get_client() -> OpenAI:
    """获取或创建 OpenAI 客户端实例"""
    global _client
    if _client is None:
        # 根据 region 设置 base_url
        region = API_CONFIG.get("region", "beijing")
        if region == "beijing":
            base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        elif region == "singapore":
            base_url = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
        else:
            base_url = API_CONFIG.get("base_url", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        
        _client = OpenAI(
            api_key=API_CONFIG["api_key"],
            base_url=base_url,
            timeout=VALIDATION_CONFIG["timeout_seconds"]
        )
    return _client


def call_qwen_api(
    prompt: str, 
    model_name: str = "qwen2.5-coder-32b-instruct", 
    max_tokens: int = 1024, 
    temperature: float = 0.7,
    system_prompt: Optional[str] = None,
    retries: int = None
) -> Tuple[bool, str]:
    """
    调用Qwen API生成代码（使用 OpenAI 兼容接口）
    
    Args:
        prompt: 用户提示
        model_name: 模型名称
        max_tokens: 最大token数
        temperature: 温度参数
        system_prompt: 系统提示词，如果为None则使用默认
        retries: 重试次数
        
    Returns:
        (成功标志, 生成的代码或错误信息)
    """
    if retries is None:
        retries = VALIDATION_CONFIG["max_retries"]
    
    # 默认系统提示词
    if system_prompt is None:
        system_prompt = "你是一个专业的编程助手，请生成高质量、可运行的Python代码。"
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]
    
    client = get_client()
    
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=0.9
            )
            
            # 提取生成的代码
            raw_content = response.choices[0].message.content
            
            # 清理代码（去除可能的markdown代码块标记）
            if "```" in raw_content:
                # 提取代码块中的内容
                code_pattern = r"```(?:python)?\n?(.*?)```"
                matches = re.findall(code_pattern, raw_content, re.DOTALL)
                if matches:
                    generated_code = matches[0].strip()
                else:
                    generated_code = raw_content
            else:
                generated_code = raw_content.strip()
            
            return True, generated_code
            
        except Exception as e:
            if attempt == retries - 1:
                return False, f"API调用失败（尝试{retries}次）: {str(e)}"
            time.sleep(1)
    
    return False, "未知错误"


def validate_code_with_14b(instruct: str, code: str, model_name: str = "qwen2.5-coder-14b-instruct") -> Tuple[bool, str]:
    """
    使用指定模型验证代码是否符合指令逻辑
    
    Args:
        instruct: 用户指令
        code: 生成的代码
        model_name: 用于验证的模型名称，默认使用14b模型
        
    Returns:
        (是否通过, 验证响应)
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
    
    system_prompt = "你是一个代码审查专家，需要判断生成的代码是否符合用户的需求。请仔细分析代码逻辑，判断代码是否正确实现了用户的要求。"
    
    success, response = call_qwen_api(
        prompt=validation_prompt,
        model_name=model_name,
        max_tokens=256,
        temperature=0.3,
        system_prompt=system_prompt
    )
    
    if not success:
        return False, response
    
    # 解析响应
    if "[是否通过]：是" in response or ("通过" in response and "否" not in response):
        return True, response
    else:
        return False, response