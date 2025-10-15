"""
导航控制模块
包含Stanley横向控制算法和路径跟踪功能
"""

import math
from typing import List, Tuple
from geometry import Vector2, Vector3, yaw_to_radians
from controllers import PIDController


class NavigationController:
    """导航控制器"""
    
    def __init__(self):
        # 控制参数
        self.stanley_k = 2.5  # Stanley控制增益
        self.path_spacing = 1.0  # 轨迹点间距（米）
        
        # PID速度控制器
        self.speed_pid = PIDController(kp=0.15, ki=0.0, kd=0.0, max_output=1.0, min_output=-1.0)
        
        # 导航状态
        self.navigation_path: List[Vector3] = []
        self.current_trajectory_index = 0
    
    def set_navigation_path(self, path: List[Vector3]):
        """设置导航路径"""
        self.navigation_path = path
        self.current_trajectory_index = 0
    
    def clear_navigation_path(self):
        """清空导航路径"""
        self.navigation_path = []
        self.current_trajectory_index = 0
    
    def find_nearest_path_point(self, vehicle_pos: Vector2, start_idx: int = 0) -> Tuple[int, float]:
        """
        找到路径上最近的点
        返回: (最近点索引, 距离)
        """
        if not self.navigation_path or start_idx >= len(self.navigation_path):
            return -1, float('inf')
        
        min_dist = float('inf')
        min_idx = start_idx
        
        # 搜索范围：从当前索引开始的一定范围内
        search_range = min(20, len(self.navigation_path) - start_idx)
        
        for i in range(start_idx, min(start_idx + search_range, len(self.navigation_path))):
            point = self.navigation_path[i]
            dist = math.sqrt((point.x - vehicle_pos.x)**2 + (point.y - vehicle_pos.y)**2)
            if dist < min_dist:
                min_dist = dist
                min_idx = i
        
        return min_idx, min_dist
    
    def calculate_stanley_control(self, vehicle_pos: Vector2, vehicle_yaw: float, 
                                  vehicle_speed: float) -> Tuple[float, float, float]:
        """
        Stanley横向控制算法
        返回: (转向角度 [-1, 1], 横向偏差, 航向偏差)
        """
        if not self.navigation_path or self.current_trajectory_index >= len(self.navigation_path):
            return 0.0, 0.0, 0.0
        
        # 找到最近的路径点
        nearest_idx, cross_track_error = self.find_nearest_path_point(
            vehicle_pos, self.current_trajectory_index
        )
        
        if nearest_idx < 0:
            return 0.0, 0.0, 0.0
        
        # 更新当前索引
        self.current_trajectory_index = nearest_idx
        
        # 计算前视点（lookahead）
        lookahead_distance = min(12.0, vehicle_speed * 1.5)  # 动态前视距离
        lookahead_distance = max(6.0, lookahead_distance)  # 动态前视距离

        lookahead_idx = self.current_trajectory_index
        accumulated_dist = 0.0
        
        # 找到前视点
        for i in range(self.current_trajectory_index, len(self.navigation_path) - 1):
            point = self.navigation_path[i]
            next_point = self.navigation_path[i + 1]
            segment_dist = math.sqrt(
                (next_point.x - point.x)**2 + (next_point.y - point.y)**2
            )
            accumulated_dist += segment_dist
            if accumulated_dist >= lookahead_distance:
                lookahead_idx = i + 1
                break
        
        # 获取目标点
        if lookahead_idx < len(self.navigation_path):
            target_point = self.navigation_path[lookahead_idx]
        else:
            target_point = self.navigation_path[-1]
        
        # 计算路径切线方向（使用相邻两点）
        if lookahead_idx > 0 and lookahead_idx < len(self.navigation_path):
            prev_point = self.navigation_path[lookahead_idx - 1]
            path_yaw = math.atan2(
                target_point.y - prev_point.y,
                target_point.x - prev_point.x
            )
        else:
            # 直接指向目标点
            path_yaw = math.atan2(
                target_point.y - vehicle_pos.y,
                target_point.x - vehicle_pos.x
            )
        
        # 计算航向误差
        yaw_rad = yaw_to_radians(vehicle_yaw)
        heading_error = path_yaw - yaw_rad
        
        # 归一化到[-π, π]
        while heading_error > math.pi:
            heading_error -= 2 * math.pi
        while heading_error < -math.pi:
            heading_error += 2 * math.pi
        
        # 计算横向误差的符号（车辆在路径的左侧还是右侧）
        nearest_point = self.navigation_path[nearest_idx]
        cross_track_vector = Vector2(
            vehicle_pos.x - nearest_point.x,
            vehicle_pos.y - nearest_point.y
        )
        path_direction = Vector2(math.cos(path_yaw), math.sin(path_yaw))
        # 叉积判断方向
        cross_product = cross_track_vector.x * path_direction.y - cross_track_vector.y * path_direction.x
        signed_cross_track_error = cross_track_error if cross_product > 0 else -cross_track_error
        
        # Stanley控制律
        cross_track_term = math.atan2(self.stanley_k * signed_cross_track_error, max(vehicle_speed, 1.0))
        
        # 总转向角
        steering_angle = heading_error + cross_track_term
        
        # 非线性角度缩放：偏差越大，打角越大
        abs_steering = abs(steering_angle)
        if abs_steering > math.pi/6:  # 30度以上
            scale_factor = 1.8  # 大偏差时放大
        elif abs_steering > math.pi/12:  # 15度以上
            scale_factor = 1.4
        else:
            scale_factor = 1  # 小偏差时缩小
        
        steering_angle *= scale_factor
        
        # 限制在[-1, 1]
        steering = max(-1.0, min(1.0, steering_angle / math.pi))
        steering = -steering  # 方向调整（根据车辆模型）
        
        return steering, signed_cross_track_error, heading_error
    
    def calculate_speed_control(self, current_speed: float, target_speed: float) -> Tuple[float, float]:
        """
        使用PID进行纵向速度控制
        返回: (油门, 刹车)
        """
        speed_error = target_speed - current_speed
        
        # PID控制
        control_output = self.speed_pid.update(speed_error)
        
        if control_output > 0:
            throttle = min(control_output, 1.0)
            brake = 0.0
        else:
            throttle = 0.0
            brake = min(-control_output, 1.0)
        
        return throttle, brake
    
    def get_dynamic_target_speed(self, steering: float, path_progress: float, 
                                cross_track_error: float = 0.0, heading_error: float = 0.0) -> float:
        """
        根据转向角度、路径进度和偏差动态调整目标速度
        steering: 转向角度 [-1, 1]
        path_progress: 路径进度 [0, 1]
        cross_track_error: 横向偏差 (米)
        heading_error: 航向偏差 (弧度)
        """
        base_speed = 6.0  # 基础目标速度
        
        # 根据转向角度调整速度
        if abs(steering) > 0.5:
            base_speed = 3.0  # 转弯时降速
        elif abs(steering) > 0.3:
            base_speed = 4.0
        
        # # 根据偏差进一步降速
        # abs_heading_error = abs(heading_error)
        # if abs_heading_error > math.pi/6:  # 30度以上偏差
        #     base_speed *= 0.7  # 大幅降速
        # elif abs_heading_error > math.pi/12:  # 15度以上偏差
        #     base_speed *= 0.85  # 中等降速
        
        # # 横向偏差过大时降速
        # if abs(cross_track_error) > 2.0:  # 横向偏差超过2米
        #     base_speed *= 0.6
        # elif abs(cross_track_error) > 1.0:  # 横向偏差超过1米
        #     base_speed *= 0.8
        
        # # 接近目标时降速
        # if path_progress > 0.9:  # 路径完成90%以上
        #     base_speed = min(base_speed, 3.0)
        
        return max(1.0, base_speed)  # 最低速度1.0 m/s
    
    def is_near_path_end(self) -> bool:
        """检查是否接近路径终点"""
        if not self.navigation_path:
            return False
        return self.current_trajectory_index >= len(self.navigation_path) - 3
    
    def get_path_progress(self) -> float:
        """获取路径完成进度 [0, 1]"""
        if not self.navigation_path:
            return 0.0
        return self.current_trajectory_index / max(len(self.navigation_path) - 1, 1)
    
    def reset_pid(self):
        """重置PID控制器"""
        self.speed_pid.reset()