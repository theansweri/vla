"""
任务相关的数据类型定义
包含所有数据类、枚举和类型定义
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Tuple, Optional


class TaskState(Enum):
    """任务状态枚举"""
    IDLE = "idle"                          # 空闲/巡航
    TASK_DETECTED = "task_detected"        # 检测到任务
    LLM_PROCESSING = "llm_processing"      # 等待LLM处理
    PARSING_TASK = "parsing_task"          # 解析任务
    NAVIGATING = "navigating"              # 导航中
    SEARCHING_DETAIL = "searching_detail"  # 搜索详细位置
    EXECUTING_ACTION = "executing_action"  # 执行动作
    TASK_COMPLETED = "task_completed"      # 任务完成
    WAITING_NEW_TASK = "waiting_new_task"  # 停车后等待新任务
    PARKING_SEARCHING = "parking_searching" # 寻找停车位（不经LLM）


@dataclass
class TaskInfo:
    """解析后的任务信息"""
    ocr_text: str
    time_limit: int  # 秒
    destination: str
    sub_destination: str  # 新增：子目的地（如"一单元"）
    action: str
    time_phrase: str
    location_phrase: str
    action_phrase: str
    # 可选：相对方位（例如 “西边”）
    direction: str = ""


@dataclass
class OCRTextResult:
    """OCR文字检测结果"""
    text: str
    confidence: float
    position: Tuple[float, float]  # 在图像中的位置 (x, y)


@dataclass
class ParkingSearchConstraints:
    """寻找停车位任务的约束条件"""
    time_limit_seconds: int = 120
    speed_limit_kmh: float = 30.0