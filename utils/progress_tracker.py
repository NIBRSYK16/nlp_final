"""
进度跟踪工具
"""
import time
from typing import Dict
from datetime import datetime


def create_progress_tracker(total_steps: int) -> Dict:
    """创建进度跟踪器"""
    return {
        "total": total_steps,
        "current": 0,
        "success": 0,
        "failed": 0,
        "start_time": time.time(),
        "logs": []
    }


def update_progress(tracker: Dict, step_name: str, success: bool = True, message: str = "") -> tuple:
    """更新进度跟踪器"""
    tracker["current"] += 1
    if success:
        tracker["success"] += 1
    else:
        tracker["failed"] += 1
    
    progress_percent = (tracker["current"] / tracker["total"]) * 100
    elapsed_time = time.time() - tracker["start_time"]
    
    log_entry = {
        "step": tracker["current"],
        "name": step_name,
        "success": success,
        "message": message,
        "progress": f"{progress_percent:.1f}%",
        "elapsed": f"{elapsed_time:.1f}s"
    }
    
    tracker["logs"].append(log_entry)
    
    # 构建状态报告
    report = f"进度: {progress_percent:.1f}% ({tracker['current']}/{tracker['total']})\n"
    report += f"成功: {tracker['success']} | 失败: {tracker['failed']}\n"
    report += f"用时: {elapsed_time:.1f}秒\n"
    report += f"当前步骤: {step_name}\n"
    if message:
        if len(message) > 100:
            report += f"{message[:100]}...\n"
        else:
            report += f"{message}\n"
    
    return report, tracker