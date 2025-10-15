"""
VLA (Vision-Language-Action) 任务控制框架 - 四目相机增强版
支持四目相机OCR结果，包含文字位置信息
"""

import time
import re
import math
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt
from matplotlib import font_manager
import os

from metacar import (
    SceneAPI, 
    VehicleControl, 
    VLAExtensionOutput, 
    VLATextOutput,
    SimCarMsg,
    SceneStaticData,
    Vector3,
    GearMode
)
from geometry import Vector2, Vector3, yaw_to_radians

# 导入各个模块
from task_types import TaskState, TaskInfo, ParkingSearchConstraints
from controllers import PIDController
from path_planner import PathPlanner
# from ocr_processor import OCRProcessor
# from ocr_processor_llm import OCRProcessor
from ocr_processor_llm2 import OCRProcessor
from navigation_controller import NavigationController
from parking_finder import ParkingFinder


@dataclass
class OCRDetection:
    """OCR检测结果（简化版）"""
    text: str
    confidence: float
    bbox: List[List[float]]  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    center: Tuple[float, float]  # (x, y) 在图像中的中心坐标


@dataclass
class CameraOCRData:
    """单个相机的OCR数据"""
    camera_id: str  # '0'前目, '1'右目, '2'左目, '3'后目
    camera_name: str  # 相机名称
    detected_texts: List[str]  # 文字列表
    detections: List[OCRDetection]  # 详细检测结果，包含位置
    frame_width: int = 0  # 帧宽度（用于横向区域过滤）


class VLATaskController:
    """VLA任务控制器 - 支持四目相机"""
    
    # 相机名称映射
    CAMERA_NAMES = {
        '0': 'front',
        '1': 'right', 
        '2': 'left',
        '3': 'back'
    }
    
    def __init__(self, scene_static_data: SceneStaticData):
        self.scene_static_data = scene_static_data
        self.buildings = scene_static_data.vla_extension.buildings if scene_static_data.vla_extension else []
        
        # 添加LLM处理状态
        self.waiting_for_llm = False  # 是否正在等待LLM结果
        self.llm_wait_start_time = 0  # 开始等待LLM的时间
        self.max_llm_wait_time = 20.0  # 最大等待时间（秒）

        # 状态管理
        self.current_state = TaskState.IDLE
        self.current_task: Optional[TaskInfo] = None
        self.task_start_time = 0
        self.vla_submitted = False
        self.task_locked = False  # 任务锁定标志
        
        # 目标建筑物
        self.target_building = None
        
        # 初始化各个组件
        self.path_planner = PathPlanner(self.buildings)
        self.ocr_processor = OCRProcessor(self.buildings)
        self.navigation_controller = NavigationController()
        # 停车位相关
        self.parking_finder = ParkingFinder(scene_static_data)
        self.parking_constraints: Optional[ParkingSearchConstraints] = None
        self.parking_search_start_time: float = 0.0
        
        # OCR相关 - 存储各相机的OCR结果
        self.last_ocr_data_by_camera: Dict[str, CameraOCRData] = {}
        
        # 任务来源跟踪
        self.task_source_camera = None  # 任务来自哪个相机
        self.task_source_position = None  # 任务文字在图像中的位置
        
        print(f"VLA任务控制器初始化完成，检测到 {len(self.buildings)} 个建筑物")
        if len(self.buildings) > 0:
            print(f"地图有效范围: X[{self.path_planner.map_min_x:.1f}, {self.path_planner.map_max_x:.1f}], "
                  f"Y[{self.path_planner.map_min_y:.1f}, {self.path_planner.map_max_y:.1f}]")
        for building in self.buildings:
            print(f"  - {building.name} 位置:({building.pos_x:.1f}, {building.pos_y:.1f})")

    def _extract_parking_constraints_from_text(self, text: str) -> ParkingSearchConstraints:
        """从中文文本中提取限时(秒)和限速(km/h)。
        示例："100秒内找到空的停车位停车，并且全程不得超过30km/h" -> 100秒, 30km/h
        未匹配到时使用默认值。
        """
        time_limit = None
        speed_limit = None

        # 提取秒数：优先匹配“(\d+)秒”
        m_time = re.search(r"(\d{1,4})\s*秒", text)
        if m_time:
            try:
                time_limit = int(m_time.group(1))
            except Exception:
                time_limit = None

        # 提取km/h：支持 30km/h、30 km/h、30公里/小时、30千米/小时
        m_speed = re.search(r"(\d{1,3})(?:\s*)?(?:km/?h|公里/?小时|千米/?小时)", text, re.IGNORECASE)
        if not m_speed:
            # 也支持“不得超过30”但未写单位时，保守认为km/h
            m_speed = re.search(r"超过\s*(\d{1,3})", text)
        if m_speed:
            try:
                speed_limit = float(m_speed.group(1))
            except Exception:
                speed_limit = None

        constraints = ParkingSearchConstraints()
        if time_limit is not None:
            constraints.time_limit_seconds = max(5, min(600, time_limit))
        if speed_limit is not None:
            constraints.speed_limit_kmh = max(5.0, min(80.0, speed_limit))
        return constraints
    
    def process_ocr_data(self, ocr_data_dict: Dict) -> None:
        """处理四目相机的OCR数据（支持异步LLM）
        
        Args:
            ocr_data_dict: 字典格式 {
                '0': CameraOCRResult对象或None,
                '1': CameraOCRResult对象或None,
                '2': CameraOCRResult对象或None,
                '3': CameraOCRResult对象或None
            }
        """
        # 转换并存储OCR数据
        self.last_ocr_data_by_camera = {}
        
        for cam_id, ocr_result in ocr_data_dict.items():
            if ocr_result and hasattr(ocr_result, 'detected_text'):
                # 转换检测结果
                detections = []
                if hasattr(ocr_result, 'text_detections'):
                    for det in ocr_result.text_detections:
                        detections.append(OCRDetection(
                            text=det.text,
                            confidence=det.confidence,
                            bbox=det.bbox,
                            center=det.center
                        ))
                
                # 创建相机OCR数据
                frame_width = 0
                if hasattr(ocr_result, 'frame_shape') and ocr_result.frame_shape is not None:
                    try:
                        frame_width = int(ocr_result.frame_shape[1])
                    except Exception:
                        frame_width = 0

                camera_data = CameraOCRData(
                    camera_id=cam_id,
                    camera_name=self.CAMERA_NAMES.get(cam_id, f"相机{cam_id}"),
                    detected_texts=ocr_result.detected_text,
                    detections=detections,
                    frame_width=frame_width
                )
                
                self.last_ocr_data_by_camera[cam_id] = camera_data
        
        # 如果任务已锁定，不再识别新任务，但仍需要更新OCR数据用于搜索子目标
        if self.task_locked:
            # 在SEARCHING_DETAIL状态下，仍然需要更新OCR数据用于搜索子目标和新任务
            if self.current_state != TaskState.SEARCHING_DETAIL:
                return
            else:
                # 在SEARCHING_DETAIL状态下，只更新OCR数据，不进行任务识别
                # OCR数据已经在上面的循环中更新了，直接返回
                return
        
        # 首先检查是否有LLM的异步结果
        if self.waiting_for_llm:
            llm_result = self.ocr_processor.get_llm_result()
            if llm_result:
                print(f"收到LLM异步处理结果: {llm_result.ocr_text}")
                self.current_task = llm_result
                self.task_start_time = time.time()
                self.vla_submitted = False
                self.task_locked = True
                self.current_state = TaskState.TASK_DETECTED
                self.waiting_for_llm = False
                return
            
            # 检查是否超时
            if time.time() - self.llm_wait_start_time > self.max_llm_wait_time:
                print("LLM处理超时，恢复巡航")
                self.waiting_for_llm = False
                self.current_state = TaskState.IDLE

        # 在空闲或等待新任务状态下寻找任务（按相机优先级：前目>后目>右目>左目）
        if self.current_state in (TaskState.IDLE, TaskState.WAITING_NEW_TASK) and not self.waiting_for_llm:
            for cam_id in ['0', '3']:  # 按优先级检查  # , '1', '2'
                if cam_id not in self.last_ocr_data_by_camera:
                    continue
                    
                camera_data = self.last_ocr_data_by_camera[cam_id]
                
                # 遍历该相机的每个检测结果
                for detection in camera_data.detections:
                    # 关键词直达：包含“空”或“停车位”时，直接进入寻找停车位流程（不调用LLM）
                    txt = detection.text.strip()
                    if any(k in txt for k in ["停车位", "空"]):
                        self.parking_constraints = self._extract_parking_constraints_from_text(txt)
                        self.parking_search_start_time = time.time()
                        self.parking_finder.start_search(self.parking_constraints)
                        self.task_locked = True
                        self.task_source_camera = camera_data.camera_name
                        self.task_source_position = detection.center
                        self.current_state = TaskState.PARKING_SEARCHING
                        print(f"触发寻找停车位流程: 文本='{txt}', 限时={self.parking_constraints.time_limit_seconds}s, 限速={self.parking_constraints.speed_limit_kmh}km/h")
                        return
                    print(f"[{camera_data.camera_name}] 尝试解析OCR文字: {detection.text}")
                    task_info = self.ocr_processor.parse_task_instruction(detection.text)
                    
                    if task_info:
                        print(f"快速解析成功: {task_info.ocr_text}")
                        self.current_task = task_info
                        self.task_start_time = time.time()
                        self.vla_submitted = False
                        self.task_locked = True
                        self.task_source_camera = camera_data.camera_name
                        self.task_source_position = detection.center
                        self.current_state = TaskState.TASK_DETECTED
                        return
                    else:
                        # 检查是否已经提交给LLM处理
                        if self.ocr_processor.llm_processing:
                            print(f"OCR文本已提交LLM异步处理，进入等待状态")
                            self.waiting_for_llm = True
                            self.llm_wait_start_time = time.time()
                            self.current_state = TaskState.LLM_PROCESSING
                            self.task_source_camera = camera_data.camera_name
                            self.task_source_position = detection.center
                        return  # 找到任务后立即返回
    
    def check_detail_location_multi_camera(self, target_text: str) -> Tuple[bool, Optional[str], Optional[Tuple[float, float]]]:
        """在多个相机中检查是否找到子目的地
        
        Returns:
            (是否找到, 相机名称, 文字位置)
        """
        # 只允许由左右相机确认子目的地（满足“左/右相机匹配子目标”条件）
        for cam_id, camera_data in self.last_ocr_data_by_camera.items():
            if cam_id not in ['1', '2']:
                continue
            w = camera_data.frame_width if camera_data.frame_width else None
            x_min = ((7.0*w) / 16.0) if w else None
            x_max = ((9.0*w) / 16.0) if w else None
            for detection in camera_data.detections:
                # 若有帧宽度信息，则仅使用横向中间1/3区域（竖条）的检测
                if x_min is not None and x_max is not None:
                    cx = detection.center[0] if detection.center else None
                    if cx is None or not (x_min <= cx <= x_max):
                        continue
                if self.ocr_processor.check_detail_location_found([detection.text], target_text):
                    return True, camera_data.camera_name, detection.center
        
        return False, None, None

    def should_run_ocr(self, vehicle_pos: Vector2) -> bool:
        """根据当前任务状态与距离，决定本帧是否需要运行OCR。
        规则：
        - 空闲(IDLE)：开启OCR以发现任务
        - LLM_PROCESSING、TASK_DETECTED、PARSING_TASK：关闭OCR
        - NAVIGATING：未接近目标建筑前关闭；接近后开启（用于搜索子目标）
        - SEARCHING_DETAIL：开启（搜索子目标或新任务）
        - WAITING_NEW_TASK：开启（停车后等待新任务）
        - TASK_COMPLETED：开启（准备下一任务）
        """
        # 启动早期阶段单独处理，由调用方在frame_count<60时不跑OCR
        if self.current_state == TaskState.IDLE:
            return True
        if self.current_state in (TaskState.LLM_PROCESSING, TaskState.TASK_DETECTED, TaskState.PARSING_TASK):
            return False
        if self.current_state == TaskState.NAVIGATING:
            if not self.target_building:
                return False
            building_pos = Vector2(self.target_building.pos_x, self.target_building.pos_y)
            dist_to_building = math.hypot(building_pos.x - vehicle_pos.x, building_pos.y - vehicle_pos.y)
            # 接近阈值：8米内开始启用OCR搜索子目的地
            # 添加滞后机制，避免在边界附近频繁切换
            if not hasattr(self, '_ocr_enabled_in_navigating'):
                self._ocr_enabled_in_navigating = False
            
            if self._ocr_enabled_in_navigating:
                # 如果已经启用，只有在距离超过10米时才关闭（滞后）
                if dist_to_building > 10.0:
                    self._ocr_enabled_in_navigating = False
                    return False
                return True
            else:
                # 如果未启用，只有在距离小于8米时才启用
                if dist_to_building < 8.0:
                    self._ocr_enabled_in_navigating = True
                    return True
                return False
        if self.current_state == TaskState.SEARCHING_DETAIL:
            return True
        # if self.current_state == TaskState.PARKING_SEARCHING:
        #     return True
        if self.current_state == TaskState.EXECUTING_ACTION:
            return True
        if self.current_state == TaskState.WAITING_NEW_TASK:
            return True
        if self.current_state == TaskState.TASK_COMPLETED:
            return True
        return False
    
    def generate_navigation_trajectory(self, building_name: str, vehicle_pos: Vector2) -> List[Vector3]:
        """生成导航到建筑物的轨迹（包含路径规划避障和等间距采样）"""
        
        # 检查起点是否在地图范围内
        if not self.path_planner.is_point_in_map(vehicle_pos):
            print(f"警告：车辆位置 ({vehicle_pos.x:.1f}, {vehicle_pos.y:.1f}) 超出地图范围！")
            # 将车辆位置限制在地图范围内
            vehicle_pos = Vector2(
                max(self.path_planner.map_min_x, min(self.path_planner.map_max_x, vehicle_pos.x)),
                max(self.path_planner.map_min_y, min(self.path_planner.map_max_y, vehicle_pos.y))
            )
            print(f"调整到边界内: ({vehicle_pos.x:.1f}, {vehicle_pos.y:.1f})")
        
        target_building = None
        for building in self.buildings:
            if building.name == building_name:
                target_building = building
                break
                
        if not target_building:
            return []
        
        # 保存目标建筑物引用
        self.target_building = target_building
        
        building_pos = Vector2(target_building.pos_x, target_building.pos_y)

        # 计算旋转后的四个角点，并据此得到对齐世界坐标轴的AABB
        # 如果是仁安综合医院则设为3.0，否则为5.0
        if target_building.name == "仁安综合医院" or target_building.name == "仁安医院住院部":
            safety_margin = 3.0
        else:
            safety_margin = 5.0  # 安全距离
        half_length = (target_building.length / 2) + safety_margin
        half_width = (target_building.width / 2) + safety_margin

        ori_rad = math.radians(target_building.ori_z)
        cos_ori = math.cos(ori_rad)
        sin_ori = math.sin(ori_rad)

        rotated_corners = []
        for dx, dy in [(half_length, half_width), (-half_length, half_width),
                       (-half_length, -half_width), (half_length, -half_width)]:
            rotated_x = dx * cos_ori - dy * sin_ori
            rotated_y = dx * sin_ori + dy * cos_ori
            rotated_corners.append(building_pos + Vector2(rotated_x, rotated_y))

        # AABB 底边两个角点 (minx, miny) 与 (maxx, miny)
        min_x = min(p.x for p in rotated_corners)
        max_x = max(p.x for p in rotated_corners)
        min_y = min(p.y for p in rotated_corners)

        aabb_left = Vector2(min_x, min_y)
        aabb_right = Vector2(max_x, min_y)

        # 选择 A* 代价更低的角点作为第一目标点
        if self.task_source_camera:
            print(f"  任务来源: {self.task_source_camera}")

        path_to_left = self.path_planner.plan_path(vehicle_pos, aabb_left, target_building)
        path_to_right = self.path_planner.plan_path(vehicle_pos, aabb_right, target_building)

        def path_length(pts):
            if not pts or len(pts) < 2:
                return float('inf')
            total = 0.0
            prev = pts[0]
            for p in pts[1:]:
                total += math.hypot(p.x - prev.x, p.y - prev.y)
                prev = p
            return total

        len_left = path_length(path_to_left)
        len_right = path_length(path_to_right)

        # 如果一条不可达(len为inf或路径为空)，选择可达的；都可达则选更短
        if len_left <= len_right:
            first_corner = aabb_left
            second_corner = aabb_right
            planned_path = path_to_left
            # 如果first_corner是(minx, maxy)，则x再-5米
            # 由于aabb_left是(minx, miny)，但我们要判断maxy的情况
            # 这里实际上aabb_left和aabb_right都是miny，所以无需处理maxy
            # 但为兼容性，假如以后角点有变化，做下判断
            if hasattr(target_building, "pos_y"):
                max_y = max(p.y for p in rotated_corners)
                if abs(first_corner.y - max_y) < 1e-3:
                    first_corner = Vector2(first_corner.x - 5.0, first_corner.y)
        else:
            first_corner = aabb_right
            second_corner = aabb_left
            planned_path = path_to_right
            # 如果first_corner是(maxx, maxy)，则x再+5米
            if hasattr(target_building, "pos_y"):
                max_y = max(p.y for p in rotated_corners)
                max_x = max(p.x for p in rotated_corners)
                if abs(first_corner.y - max_y) < 1e-3 and abs(first_corner.x - max_x) < 1e-3:
                    first_corner = Vector2(first_corner.x + 5.0, first_corner.y)

        # 进入NAVIGATING前注入“远离建筑外边框”的预偏置点：
        # 基于“所有建筑”的旋转外框（含安全边距）计算与自车的全局最近边界点，沿其外法向远离
        def _nearest_point_on_oriented_rect_for_building(world_pt: Vector2, b) -> Tuple[Vector2, float, float, float, float, float]:
            bx = float(getattr(b, "pos_x", 0.0))
            by = float(getattr(b, "pos_y", 0.0))
            ori = math.radians(float(getattr(b, "ori_z", 0.0)))
            dx = world_pt.x - bx
            dy = world_pt.y - by
            neg_ori = -ori
            local_x = dx * math.cos(neg_ori) - dy * math.sin(neg_ori)
            local_y = dx * math.sin(neg_ori) + dy * math.cos(neg_ori)
            half_l = (float(getattr(b, "length", 0.0)) / 2.0) + safety_margin
            half_w = (float(getattr(b, "width", 0.0)) / 2.0) + safety_margin
            clamped_x = max(-half_l, min(half_l, local_x))
            clamped_y = max(-half_w, min(half_w, local_y))
            wx = bx + (clamped_x * math.cos(ori) - clamped_y * math.sin(ori))
            wy = by + (clamped_x * math.sin(ori) + clamped_y * math.cos(ori))
            world_nearest = Vector2(wx, wy)
            dist = math.hypot(world_pt.x - wx, world_pt.y - wy)
            return world_nearest, dist, clamped_x, clamped_y, half_l, half_w, ori

        # 扫描所有建筑，找到全局最近外框点
        nearest_world = None
        nearest_dist = float('inf')
        cx_local = cy_local = 0.0
        half_l = half_w = 0.0
        nearest_ori = 0.0
        nearest_building = None
        for b in getattr(self.path_planner, "buildings", []) or []:
            nw, d, cx, cy, hl, hw, ori_b = _nearest_point_on_oriented_rect_for_building(vehicle_pos, b)
            if d < nearest_dist:
                nearest_world = nw
                nearest_dist = d
                cx_local, cy_local = cx, cy
                half_l, half_w = hl, hw
                nearest_ori = ori_b
                nearest_building = b
        print("nearest_dist:", nearest_dist)
        print("nearest_building:", nearest_building)
        print("nearest_world:", nearest_world)
        # 若离任意建筑外框太近，注入预偏置点
        if nearest_world is not None and nearest_dist < 10.0:
            # 外法向：边界点->车辆 的方向
            dir_x = vehicle_pos.x - nearest_world.x
            dir_y = vehicle_pos.y - nearest_world.y
            norm = math.hypot(dir_x, dir_y)
            if norm > 1e-6:
                dir_x /= norm
                dir_y /= norm
            else:
                # 退化：车辆恰在边界或外框内，基于最近边选择外法向（在最近建筑的局部坐标系中）
                # 与四条边的局部距离（到边界的剩余距离，均为非负）
                d_pos_x = abs(half_l - cx_local)
                d_neg_x = abs(half_l + cx_local)
                d_pos_y = abs(half_w - cy_local)
                d_neg_y = abs(half_w + cy_local)
                # 选择最近的一条边作为外法向方向
                nearest_face_dist = min(d_pos_x, d_neg_x, d_pos_y, d_neg_y)
                if nearest_face_dist == d_pos_x:
                    nx_local, ny_local = 1.0, 0.0
                elif nearest_face_dist == d_neg_x:
                    nx_local, ny_local = -1.0, 0.0
                elif nearest_face_dist == d_pos_y:
                    nx_local, ny_local = 0.0, 1.0
                else:
                    nx_local, ny_local = 0.0, -1.0
                dir_x = nx_local * math.cos(nearest_ori) - ny_local * math.sin(nearest_ori)
                dir_y = nx_local * math.sin(nearest_ori) + ny_local * math.cos(nearest_ori)

            inserted = False
            for away_dist in [6.0, 8.0, 10.0, 12.0, 15.0]:
                pre_point = Vector2(vehicle_pos.x + dir_x * away_dist, vehicle_pos.y + dir_y * away_dist)
                if self.path_planner.is_collision_free(pre_point, target_building):
                    re_path = self.path_planner.plan_path(pre_point, first_corner, target_building)
                    if re_path and len(re_path) >= 2:
                        planned_path = [vehicle_pos, pre_point] + re_path
                        print(f"  已插入基于最近外框法向的预偏置点: ({pre_point.x:.1f}, {pre_point.y:.1f}) （来自 {getattr(nearest_building, 'name', '')}）")
                        inserted = True
                        break

            if not inserted:
                tx, ty = -dir_y, dir_x
                tlen = math.hypot(tx, ty) or 1.0
                tx /= tlen
                ty /= tlen
                for shift, away_dist in [(2.0, 8.0), (3.0, 10.0)]:
                    cand = Vector2(vehicle_pos.x + tx * shift + dir_x * away_dist,
                                   vehicle_pos.y + ty * shift + dir_y * away_dist)
                    if self.path_planner.is_collision_free(cand, target_building):
                        re_path = self.path_planner.plan_path(cand, first_corner, target_building)
                        if re_path and len(re_path) >= 2:
                            planned_path = [vehicle_pos, cand] + re_path
                            print(f"  切向+法向预偏置: ({cand.x:.1f}, {cand.y:.1f}) （来自 {getattr(nearest_building, 'name', '')}）")
                            break

        print(
            f"规划路径: 从 ({vehicle_pos.x:.1f}, {vehicle_pos.y:.1f}) 到 {building_name} 的角点 ({first_corner.x:.1f}, {first_corner.y:.1f})"
        )

        path_2d = []
        path_2d.extend(planned_path)
        print(f"  生成了 {len(planned_path)} 个初始路径点")

        # 记录用于开始细节搜索的第一目标角点
        self.first_search_corner = first_corner
        self._search_triggered = False

        # 再沿底边行驶到另一角点
        path_2d.append(second_corner)
        
        # 重采样路径为等间距点（1米间隔）
        if len(path_2d) > 1:
            resampled_path = self.path_planner.resample_path_uniform(path_2d, self.navigation_controller.path_spacing)
            print(f"  重采样后: {len(resampled_path)} 个等间距路径点（间隔{self.navigation_controller.path_spacing}米）")
            
            # 转换为Vector3格式
            path_3d = [Vector3(p.x, p.y, 0) for p in resampled_path]
        else:
            path_3d = [Vector3(p.x, p.y, 0) for p in path_2d]
        
        return path_3d
    
    def save_navigation_path_image(self, filepath: str = "./navigation_path.png"):
        """保存导航路径图像"""
        navigation_path = self.navigation_controller.navigation_path
        if not navigation_path:
            return
        
        # 设置中文字体（微软雅黑），避免中文显示为方框
        try:
            font_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'font', 'msyh.ttc')
            if os.path.exists(font_path):
                font_manager.fontManager.addfont(font_path)
                plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
                plt.rcParams['axes.unicode_minus'] = False
        except Exception:
            pass

        path_xy = [(p.x, p.y) for p in navigation_path]
        if path_xy:
            xs, ys = zip(*path_xy)
            plt.figure(figsize=(6, 6))
            plt.plot(xs, ys, marker='o', linestyle='-', color='b')
            
            title = "导航路径"
            if self.current_task:
                title = f"导航至 {self.current_task.destination}"
                if self.task_source_camera:
                    title += f"（来源 {self.task_source_camera}）"
            
            plt.title(title)
            plt.xlabel("X 轴")
            plt.ylabel("Y 轴")
            plt.grid(True)
            plt.axis('equal')
            plt.savefig(filepath)
            plt.close()
            print(f"导航路径已保存为 {filepath}")

    def _plan_path_to_parking(self, sim_car_msg: SimCarMsg, park_pose: Tuple[float, float, float]):
        """基于选中的空车位，生成两段目标点路径，并更新导航路径。
        规则：
        - 以车位长边方向（车位yaw±90°的两个方向）外延15m作为第一目标点（挑选无碰撞的一侧）。
        - 第一目标点 -> 车位中心 作为第二段。
        - 使用A*分别从当前车位位置到第一目标点、再到车位点的路径，拼接并重采样。
        - 将路径更新到导航控制器并保存成 navigation_path.png。
        """
        vx = float(sim_car_msg.pose_gnss.pos_x)
        vy = float(sim_car_msg.pose_gnss.pos_y)
        vehicle_pos = Vector2(vx, vy)

        px, py, pyaw_deg = park_pose
        park_center = Vector2(float(px), float(py))
        # 保存选中车位，供减速与刹停判定使用
        self.selected_parking_pose = (float(px), float(py), float(pyaw_deg))

        # 计算长边方向的两个候选第一目标点（沿车位长边方向外延15m）
        # 约定：停车位的 length 为长边；场景yaw为顺时针角度，这里转换为数学正向（逆时针弧度）
        yaw_rad = yaw_to_radians(float(pyaw_deg))
        long_dir = Vector2(math.cos(yaw_rad), math.sin(yaw_rad))
        dir1 = long_dir
        dir2 = Vector2(-long_dir.x, -long_dir.y)
        offset = 30.0
        cand1 = Vector2(park_center.x + dir1.x * offset, park_center.y + dir1.y * offset)
        cand2 = Vector2(park_center.x + dir2.x * offset, park_center.y + dir2.y * offset)

        # 选择无碰撞的一侧作为第一目标点；都可行则选更短总代价
        def cost_for(candidate: Vector2) -> float:
            path_a = self.path_planner.plan_path(vehicle_pos, candidate, None)
            path_b = self.path_planner.plan_path(candidate, park_center, None)
            def path_len(p):
                if not p or len(p) < 2:
                    return float('inf')
                s = 0.0
                for i in range(1, len(p)):
                    s += math.hypot(p[i].x - p[i-1].x, p[i].y - p[i-1].y)
                return s
            la = path_len(path_a)
            lb = path_len(path_b)
            return la + lb

        # 首先过滤掉明显碰撞的第一段直线；若都碰撞，则仍允许A*尝试
        cands = []
        for c in [cand1, cand2]:
            cands.append((c, cost_for(c)))
        cands.sort(key=lambda x: x[1])
        first_goal, best_cost = cands[0]
        print(f"[PARK][PLAN] 车位中心=({park_center.x:.2f},{park_center.y:.2f}) yaw={pyaw_deg:.1f}° 第一目标点=({first_goal.x:.2f},{first_goal.y:.2f}) 预估总代价={best_cost:.1f}")

        # 生成两段路径
        path1 = self.path_planner.plan_path(vehicle_pos, first_goal, None)
        path2 = self.path_planner.plan_path(first_goal, park_center, None)

        # 在车位中心后沿接近方向延申若干米，提升横向跟踪稳定性
        tail_extend = 6.0
        extend_dir = None
        if path2 and len(path2) >= 2:
            # 取从倒数第二个点到车位中心的方向
            p_prev = path2[-2]
            vec_x = park_center.x - p_prev.x
            vec_y = park_center.y - p_prev.y
            norm = math.hypot(vec_x, vec_y) or 1.0
            extend_dir = Vector2(vec_x / norm, vec_y / norm)
        else:
            # 回退：使用车位长边方向
            yaw_rad = yaw_to_radians(float(pyaw_deg))
            extend_dir = Vector2(math.cos(yaw_rad), math.sin(yaw_rad))

        tail_point = Vector2(
            park_center.x + extend_dir.x * tail_extend,
            park_center.y + extend_dir.y * tail_extend,
        )

        # 若尾点可行则规划 park_center -> tail_point 段
        path3 = []
        try:
            if self.path_planner.is_collision_free(tail_point, None):
                path3 = self.path_planner.plan_path(park_center, tail_point, None)
        except Exception:
            path3 = []

        combined = (path1 or []) + (path2[1:] if path2 and path2[:] else [])
        if path3:
            combined += (path3[1:] if path3[:] else [])
        if not combined:
            combined = [vehicle_pos, park_center]

        # 重采样
        resampled = self.path_planner.resample_path_uniform(combined, self.navigation_controller.path_spacing)
        path_3d = [Vector3(p.x, p.y, 0) for p in resampled]

        self.navigation_controller.set_navigation_path(path_3d)
        self.save_navigation_path_image()
    
    def update(self, sim_car_msg: SimCarMsg, ocr_data_dict: Dict, frame_count: int) -> Tuple[VehicleControl, Optional[VLAExtensionOutput]]:
        """主要的更新函数，返回车辆控制指令和VLA输出
        
        Args:
            sim_car_msg: 仿真车辆消息
            ocr_data_dict: 四目相机OCR结果字典
        """
        vehicle_control = VehicleControl()
        vla_output = None
        
        # 获取当前车辆状态
        vehicle_pos = Vector2(sim_car_msg.pose_gnss.pos_x, sim_car_msg.pose_gnss.pos_y)
        vehicle_yaw = sim_car_msg.pose_gnss.ori_z
        vehicle_speed = sim_car_msg.main_vehicle.speed
        
        # 调试打印：状态变化检测
        if not hasattr(self, '_last_debug_state'):
            self._last_debug_state = self.current_state
        if self._last_debug_state != self.current_state:
            print(f"[状态切换] {self._last_debug_state.name} -> {self.current_state.name} (帧{frame_count})")
            self._last_debug_state = self.current_state
        
        # 调试打印：控制指令变化检测（每30帧打印一次）
        if frame_count % 30 == 0:
            print(f"[调试] 状态:{self.current_state.name}, 速度:{vehicle_speed:.2f}, 位置:({vehicle_pos.x:.2f},{vehicle_pos.y:.2f}), 航向:{vehicle_yaw:.1f}°, OCR数据:{bool(ocr_data_dict)}")
        
        
        # 处理OCR数据
        if ocr_data_dict and frame_count > 60:  # 开始阶段画面很不稳定
            if not self.task_locked:
                # 任务未锁定，正常处理OCR数据（可能发现新任务）
                self.process_ocr_data(ocr_data_dict)
            elif self.current_state == TaskState.SEARCHING_DETAIL:
                # 任务已锁定但在搜索子目标状态，仍需要更新OCR数据用于搜索子目标和新任务
                self.process_ocr_data(ocr_data_dict)
        
        # 状态机处理
        if frame_count < 60:
            # 启动阶段，先刹停
            target_speed = 0.0
            throttle, brake = self.navigation_controller.calculate_speed_control(vehicle_speed, target_speed)
            vehicle_control.throttle = throttle
            vehicle_control.brake = brake
            vehicle_control.gear = GearMode.DRIVE

        elif self.current_state == TaskState.IDLE:
            # 巡航状态，寻找任务
            target_speed = 1.0  # 巡航速度
            throttle, brake = self.navigation_controller.calculate_speed_control(vehicle_speed, target_speed)
            vehicle_control.throttle = throttle
            vehicle_control.brake = brake
            vehicle_control.gear = GearMode.DRIVE

        elif self.current_state == TaskState.LLM_PROCESSING:
            # 等待LLM处理时，紧急刹停
            # print(f"等待LLM处理中... (已等待 {time.time() - self.llm_wait_start_time:.1f}秒)")
            vehicle_control.brake = 1.0
            vehicle_control.throttle = 0.0
            vehicle_control.gear = GearMode.DRIVE
            
            # 检查LLM结果
            llm_result = self.ocr_processor.get_llm_result()
            if llm_result:
                print(f"LLM处理完成: {llm_result.destination}")
                self.current_task = llm_result
                self.task_start_time = time.time()
                self.vla_submitted = False
                self.task_locked = True
                self.waiting_for_llm = False
                self.current_state = TaskState.TASK_DETECTED
            elif time.time() - self.llm_wait_start_time > self.max_llm_wait_time:
                print("LLM处理超时，恢复巡航")
                self.waiting_for_llm = False
                self.current_state = TaskState.IDLE
    
        elif self.current_state == TaskState.TASK_DETECTED:
            # 检测到任务，紧急停车
            vehicle_control.brake = 1.0
            vehicle_control.gear = GearMode.DRIVE
            
            if vehicle_speed < 0.1:  # 停车完成
                self.current_state = TaskState.PARSING_TASK
                
        elif self.current_state == TaskState.PARSING_TASK:
            # 解析任务并规划路径
            if self.current_task and not self.vla_submitted:
                # 创建VLA输出
                vla_output = VLAExtensionOutput(
                    text_info=VLATextOutput(
                        ocr_text=self.current_task.ocr_text,
                        time_phrase=self.current_task.time_phrase,
                        location_phrase=self.current_task.location_phrase,
                        action_phrase=self.current_task.action_phrase
                    )
                )
                self.vla_submitted = True
                print(f"提交VLA输出: {self.current_task.ocr_text}")
                
                # 生成导航轨迹（包含路径规划）
                # 如果包含方位信息，则以主要目的地为锚点进行方位后处理
                target_building_name = self.current_task.destination
                if getattr(self.current_task, 'direction', ''):
                    best = self.path_planner.find_building_by_direction(target_building_name, self.current_task.direction)
                    if best is not None:
                        print(f"根据方位解析: 以'{target_building_name}'为锚点，'{self.current_task.direction}'侧 -> '{best.name}'")
                        target_building_name = best.name
                        self.current_task.sub_destination = ""
                        # 清空OCR缓存：子目的地被清空后，应丢弃历史OCR结果避免误触发
                        self.last_ocr_data_by_camera = {}
                        self.task_source_camera = None
                        self.task_source_position = None
                        if hasattr(self, 'ocr_processor') and hasattr(self.ocr_processor, 'reset_cache'):
                            try:
                                self.ocr_processor.reset_cache()
                            except Exception:
                                pass
                    else:
                        print("方位解析未找到合适建筑，fallback 使用原目的地")

                navigation_path = self.generate_navigation_trajectory(
                    target_building_name, vehicle_pos
                )
                self.navigation_controller.set_navigation_path(navigation_path)
                
                # 保存路径图像
                self.save_navigation_path_image()
                
                self.current_state = TaskState.NAVIGATING
                self.navigation_controller.reset_pid()  # 重置PID控制器
                print(f"开始导航到: {target_building_name}")
                if self.current_task.sub_destination:
                    print(f"子目的地: {self.current_task.sub_destination}")
                print("-"*60)
            
            # 保持停车状态
            vehicle_control.brake = 1.0
            vehicle_control.gear = GearMode.DRIVE
            
        elif self.current_state == TaskState.NAVIGATING:
            # 导航到目的地
            # if not self.navigation_controller.is_near_path_end():
            # 使用Stanley算法进行横向控制
            steering, cross_track_error, heading_error = self.navigation_controller.calculate_stanley_control(
                vehicle_pos, vehicle_yaw, vehicle_speed
            )
            
            # 动态调整目标速度（基于偏差 + 距离停车点减速）
            path_progress = self.navigation_controller.get_path_progress()
            target_speed = self.navigation_controller.get_dynamic_target_speed(
                steering, path_progress, cross_track_error, heading_error
            )
            # 追加：距“车位中心/停车点”越近越慢
            try:
                dist_to_goal = None
                # 优先使用选中车位中心
                if hasattr(self, 'selected_parking_pose') and self.selected_parking_pose:
                    px, py, _ = self.selected_parking_pose
                    dist_to_goal = math.hypot(px - vehicle_pos.x, py - vehicle_pos.y)
                # 回退到路径末点
                if dist_to_goal is None and self.navigation_controller.navigation_path:
                    last_pt = self.navigation_controller.navigation_path[-1]
                    dist_to_goal = math.hypot(last_pt.x - vehicle_pos.x, last_pt.y - vehicle_pos.y)

                if dist_to_goal is not None:
                    if dist_to_goal < 15.0:
                        target_speed = min(target_speed, max(1.5, dist_to_goal / 5.0))
                    if dist_to_goal < 3.0:
                        target_speed = min(target_speed, 0.5)
            except Exception:
                pass
            
            # 使用PID进行纵向控制
            throttle, brake = self.navigation_controller.calculate_speed_control(vehicle_speed, target_speed)
            
            vehicle_control.steering = steering
            vehicle_control.throttle = throttle
            vehicle_control.brake = brake
            vehicle_control.gear = GearMode.DRIVE

            # 若“车辆中心接近车位中心”，则刹停并进入停车动作
            try:
                px, py, _ = getattr(self, 'selected_parking_pose', (None, None, None))
                if px is not None and py is not None:
                    d_park = math.hypot(px - vehicle_pos.x, py - vehicle_pos.y)
                    if d_park < 0.5 and vehicle_speed < 0.8:
                        print("接近车位中心，执行停车")
                        self.current_state = TaskState.EXECUTING_ACTION
                        self.navigation_controller.clear_navigation_path()
                        self.navigation_controller.reset_pid()
                        return vehicle_control, vla_output
            except Exception:
                pass
            
            # 通过Stanley算法的追踪点判断是否已超过第一目标角点，若超过则进入搜索子目的地
            if hasattr(self, 'first_search_corner') and self.first_search_corner is not None and not getattr(self, '_search_triggered', False):
                # 获取Stanley当前追踪点
                stanley_idx = self.navigation_controller.current_trajectory_index
                if stanley_idx is not None and stanley_idx < len(self.navigation_controller.navigation_path):
                    stanley_point = self.navigation_controller.navigation_path[stanley_idx]
                    # 判断追踪点是否已超过第一目标角点
                    # 这里假设first_search_corner在路径上，找到其在路径中的索引
                    try:
                        first_idx = next(i for i, pt in enumerate(self.navigation_controller.navigation_path)
                                         if abs(pt.x - self.first_search_corner.x) < 3 and abs(pt.y - self.first_search_corner.y) < 3)
                    except StopIteration:
                        first_idx = None

                    # if first_idx is None:
                    #     print("未找到first_search_corner在路径中的索引（first_idx is None）")
                    #     print(f"first_search_corner: ({self.first_search_corner.x:.2f}, {self.first_search_corner.y:.2f})")
                    #     print("当前路径点如下：")
                    #     for i, pt in enumerate(self.navigation_controller.navigation_path):
                    #         print(f"  idx={i}: ({pt.x:.2f}, {pt.y:.2f})")
                    #     print("Stanley当前追踪点索引：", stanley_idx)
                    #     # 进一步debug：检查first_search_corner与路径点的距离
                    #     min_dist = float('inf')
                    #     min_idx = None
                    #     for i, pt in enumerate(self.navigation_controller.navigation_path):
                    #         dist = math.hypot(pt.x - self.first_search_corner.x, pt.y - self.first_search_corner.y)
                    #         if dist < min_dist:
                    #             min_dist = dist
                    #             min_idx = i
                    #     print(f"first_search_corner与路径最近点的索引: {min_idx}, 距离: {min_dist:.3f}")
                    #     if min_dist > 1.0:
                    #         print("警告：first_search_corner与路径最近点距离较远，可能路径未包含该点或坐标误差较大。")
                    # else:
                    #     print("stanley_idx, first_idx:", stanley_idx, first_idx)

                    if first_idx is not None and stanley_idx >= first_idx:
                        print("Stanley追踪点已超过第一目标角点，开始搜索子目的地")
                        self.current_state = TaskState.SEARCHING_DETAIL
                        self.navigation_controller.reset_pid()
                        self._search_triggered = True
                        # 直接返回以在下一循环执行 SEARCHING_DETAIL 逻辑
                        return vehicle_control, vla_output
            
            # # 只有在接近建筑物时才检查子目的地（避免误识别任务标牌）
            # if self.current_task and self.current_task.sub_destination and self.target_building:
            #     building_pos = Vector2(self.target_building.pos_x, self.target_building.pos_y)
            #     dist_to_building = math.sqrt(
            #         (building_pos.x - vehicle_pos.x)**2 + 
            #         (building_pos.y - vehicle_pos.y)**2
            #     )
                
            #     # 只有距离建筑物10米以内才检查子目的地
            #     if dist_to_building < 10.0:
            #         found, camera_name, position = self.check_detail_location_multi_camera(
            #             self.current_task.sub_destination
            #         )
            #         if found:
            #             print(f"[{camera_name}] 发现子目的地: {self.current_task.sub_destination}")
            #             if position:
            #                 print(f"  位置: ({position[0]:.1f}, {position[1]:.1f})")
            #             self.current_state = TaskState.EXECUTING_ACTION
                    
        elif self.current_state == TaskState.SEARCHING_DETAIL:
            # 在目的地附近低速搜索子目的地
            if self.current_task and self.current_task.sub_destination:
                # 方式1：检查左侧或右侧摄像头OCR识别到的内容和子目标匹配
                found, camera_name, position = self.check_detail_location_multi_camera(
                    self.current_task.sub_destination
                )
                if found:
                    print(f"[{camera_name}] 找到子目的地: {self.current_task.sub_destination}")
                    if position:
                        print(f"  位置: ({position[0]:.1f}, {position[1]:.1f})")
                    self.current_state = TaskState.EXECUTING_ACTION
                else:
                    # 方式2：检查是否有新的任务内容（前或后摄像头OCR识别到了quality_score.is_valid的内容）
                    new_task_detected = False
                    
                    # 检查前目和后目相机的新任务
                    for cam_id in ['0', '3', '1', '2']:  # 前目和后目
                        if cam_id not in self.last_ocr_data_by_camera:
                            continue
                            
                        camera_data = self.last_ocr_data_by_camera[cam_id]
                        
                        # 遍历该相机的每个检测结果
                        for detection in camera_data.detections:
                            print(f"[{camera_data.camera_name}] 检查新任务文本: {detection.text}")
                            
                            # 使用OCR处理器的质量评估
                            quality_score = self.ocr_processor.evaluate_ocr_quality(detection.text)
                            
                            if quality_score.is_valid:
                                print(f"[{camera_data.camera_name}] 检测到有效的新任务文本: {detection.text}")
                                print(f"  质量评分: 总分={quality_score.total_score:.2f}, "
                                      f"有建筑={quality_score.has_building}, "
                                      f"有时间={quality_score.has_time}, "
                                      f"有动作={quality_score.has_action},"
                                      f"有速度吗={quality_score.has_speed}")
                                
                                # 开始执行新的一轮LLM解析新目标的流程
                                task_info = self.ocr_processor.parse_task_instruction(detection.text)
                                
                                if task_info:
                                    print(f"快速解析成功: {task_info.ocr_text}")
                                    # 更新当前任务
                                    self.current_task = task_info
                                    self.task_start_time = time.time()
                                    self.vla_submitted = False
                                    self.task_locked = True
                                    self.task_source_camera = camera_data.camera_name
                                    self.task_source_position = detection.center
                                    self.current_state = TaskState.TASK_DETECTED
                                    new_task_detected = True
                                    break
                                else:
                                    # 检查是否已经提交给LLM处理
                                    if self.ocr_processor.llm_processing:
                                        print(f"OCR文本已提交LLM异步处理，进入等待状态")
                                        self.waiting_for_llm = True
                                        self.llm_wait_start_time = time.time()
                                        self.current_state = TaskState.LLM_PROCESSING
                                        self.task_source_camera = camera_data.camera_name
                                        self.task_source_position = detection.center
                                        new_task_detected = True
                                        break
                        
                        if new_task_detected:
                            break
                    
                    if not new_task_detected:
                        # 继续低速搜索（进一步降速）且保持横向路径跟踪
                        # 横向：Stanley 控制
                        steering, cross_track_error, heading_error = self.navigation_controller.calculate_stanley_control(
                            vehicle_pos, vehicle_yaw, vehicle_speed
                        )
                        vehicle_control.steering = steering

                        # 纵向：低速 PID
                        target_speed = 2.0  # 低速搜索
                        throttle, brake = self.navigation_controller.calculate_speed_control(
                            vehicle_speed, target_speed
                        )
                        vehicle_control.throttle = throttle
                        vehicle_control.brake = brake
                        vehicle_control.gear = GearMode.DRIVE
                        
                        # 打印正在搜索的相机
                        active_cameras = [data.camera_name for data in self.last_ocr_data_by_camera.values() 
                                        if data.detected_texts]
                        if active_cameras:
                            print(f"搜索子目的地 '{self.current_task.sub_destination}' (活跃相机: {', '.join(active_cameras)})")
            else:
                # 没有子目的地：仅允许由左右相机确认检测到文字（仿照check_detail_location_multi_camera的风格）
                detected_any = False
                for cam_id, camera_data in self.last_ocr_data_by_camera.items():
                    if cam_id not in ['1', '2']:
                        continue
                    w = camera_data.frame_width if camera_data.frame_width else None
                    x_min = ((7.0*w) / 16.0) if w else None
                    x_max = ((9.0*w) / 16.0) if w else None
                    for detection in camera_data.detections:
                        # 若有帧宽度信息，则仅使用横向中间1/3区域（竖条）的检测
                        if x_min is not None and x_max is not None:
                            cx = detection.center[0] if detection.center else None
                            if cx is None or not (x_min <= cx <= x_max):
                                continue
                        # 任意文字触发
                        print(f"[{camera_data.camera_name}] 检测到文字: '{detection.text}'，触发任务结束（无子目的地模式）")
                        detected_any = True
                        break
                    if detected_any:
                        break
                if detected_any:
                    self.current_state = TaskState.EXECUTING_ACTION
                else:
                    # 继续低速搜索并保持路径跟踪
                    steering, cross_track_error, heading_error = self.navigation_controller.calculate_stanley_control(
                        vehicle_pos, vehicle_yaw, vehicle_speed
                    )
                    vehicle_control.steering = steering
                    target_speed = 2.0
                    throttle, brake = self.navigation_controller.calculate_speed_control(
                        vehicle_speed, target_speed
                    )
                    vehicle_control.throttle = throttle
                    vehicle_control.brake = brake
                    vehicle_control.gear = GearMode.DRIVE
                
        elif self.current_state == TaskState.PARKING_SEARCHING:
            # 在限定时间和限速内寻找空停车位（不依赖LLM）
            constraints = self.parking_constraints or ParkingSearchConstraints()
            elapsed = time.time() - self.parking_search_start_time
            if frame_count % 30 == 0:
                print(f"[PARK] 状态=PARKING_SEARCHING 已用时:{elapsed:.1f}s / 限时:{getattr(constraints, 'time_limit_seconds', 0)}s, 当前速:{vehicle_speed*3.6:.1f}km/h / 限速:{constraints.speed_limit_kmh:.1f}km/h")

            # 低速巡航搜索，强制限速
            max_ms = constraints.speed_limit_kmh / 3.6
            target_speed = min(3.0, max_ms)
            throttle, brake = self.navigation_controller.calculate_speed_control(vehicle_speed, target_speed)
            vehicle_control.throttle = throttle
            vehicle_control.brake = brake
            vehicle_control.gear = GearMode.DRIVE

            # 由停车查找器推进搜索流程（内部可使用地图/视觉占用等）
            found, park_pose = self.parking_finder.step(sim_car_msg, self.last_ocr_data_by_camera)
            if found:
                print("找到空停车位，开始规划两段路径并前往停车位")
                try:
                    self._plan_path_to_parking(sim_car_msg, park_pose)
                    self.current_state = TaskState.NAVIGATING
                    self.navigation_controller.reset_pid()
                except Exception as e:
                    print(f"规划停车路径时异常: {e}")
                    # 回退到直接停车
                    self.current_state = TaskState.EXECUTING_ACTION
                    self.navigation_controller.clear_navigation_path()
            
        elif self.current_state == TaskState.EXECUTING_ACTION:
            # 执行动作（停车），完成后进入等待新任务状态
            if self.current_task:  #  and self.current_task.action == "停车":
                vehicle_control.brake = 0.1
                vehicle_control.throttle = 0.0
                vehicle_control.gear = GearMode.PARKING
                
                if vehicle_speed < 0.1:
                    print("停车完成，等待新任务")
                    self.current_state = TaskState.WAITING_NEW_TASK
                    # 重置任务锁定，但保留路径为空，准备接收新任务
                    self.current_task = None
                    self.vla_submitted = False
                    self.task_locked = False
                    self.task_source_camera = None
                    self.task_source_position = None
                    self.navigation_controller.clear_navigation_path()
                    self.navigation_controller.reset_pid()
                    # # 清空LLM去重状态，允许相似文本再次提交
                    # try:
                    #     self.ocr_processor.last_valid_ocr_text = None
                    #     self.ocr_processor.last_llm_request_time = 0
                    #     self.ocr_processor.llm_processing = False
                    # except Exception:
                    #     pass
                    # 立即保持停车，不推进到TASK_COMPLETED
                    
        elif self.current_state == TaskState.TASK_COMPLETED:
            # 任务完成，重置状态
            print("任务完成，重置为空闲状态")
            self.current_task = None
            self.vla_submitted = False
            self.task_locked = False  # 解锁任务
            self.task_source_camera = None
            self.task_source_position = None
            self.navigation_controller.clear_navigation_path()  # 清空路径
            self.navigation_controller.reset_pid()  # 重置PID控制器
            self.current_state = TaskState.IDLE
            vehicle_control.gear = GearMode.DRIVE
            # # 清空LLM去重状态，保证下一任务可被正常提交
            # try:
            #     self.ocr_processor.last_valid_ocr_text = None
            #     self.ocr_processor.last_llm_request_time = 0
            #     self.ocr_processor.llm_processing = False
            # except Exception:
            #     pass
            # 重新开始巡航
            target_speed = 0.0
            throttle, brake = self.navigation_controller.calculate_speed_control(vehicle_speed, target_speed)
            vehicle_control.throttle = throttle
            vehicle_control.brake = brake

        elif self.current_state == TaskState.WAITING_NEW_TASK:
            # 保持停车，持续运行OCR寻找新任务
            vehicle_control.brake = 1.0
            vehicle_control.throttle = 0.0
            vehicle_control.gear = GearMode.PARKING
            
            # 在等待新任务时允许OCR运行
            # 复用process_ocr_data逻辑以发现新任务
            if ocr_data_dict:
                # 允许在WAITING_NEW_TASK中识别新任务
                self.task_locked = False
                self.process_ocr_data(ocr_data_dict)
                if self.current_task:
                    print("检测到新任务，准备解析")
                    self.current_state = TaskState.TASK_DETECTED
        
        # # 检查任务超时
        # if (self.current_task and 
        #     time.time() - self.task_start_time > self.current_task.time_limit):
        #     print(f"任务超时 ({self.current_task.time_limit}秒)")
        #     self.current_state = TaskState.TASK_COMPLETED
        
        return vehicle_control, vla_output
    
    def get_navigation_path_for_gui(self) -> List[Tuple[float, float]]:
        """获取当前导航路径，转换为简单的坐标元组（用于GUI显示）"""
        # 将Vector3转换为简单的元组，避免序列化问题
        return [(p.x, p.y) for p in self.navigation_controller.navigation_path]
    
    def get_status_info(self) -> Dict:
        """获取当前状态信息（用于调试）"""
        status = {
            "current_state": self.current_state.name,
            "has_task": self.current_task is not None,
            "vla_submitted": self.vla_submitted,
            "path_progress": self.navigation_controller.get_path_progress(),
            "active_cameras": list(self.last_ocr_data_by_camera.keys()),
            "task_source": self.task_source_camera
        }
        
        # 添加各相机的OCR检测数量
        ocr_counts = {}
        for cam_id, data in self.last_ocr_data_by_camera.items():
            ocr_counts[data.camera_name] = len(data.detected_texts)
        status["ocr_counts_by_camera"] = ocr_counts
        
        if self.current_task:
            elapsed_time = time.time() - self.task_start_time
            status.update({
                "task_destination": self.current_task.destination,
                "task_sub_destination": self.current_task.sub_destination,
                "task_action": self.current_task.action,
                "time_limit": self.current_task.time_limit,
                "elapsed_time": elapsed_time,
                "remaining_time": self.current_task.time_limit - elapsed_time
            })
        
        return status