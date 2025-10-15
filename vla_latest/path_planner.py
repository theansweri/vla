"""
路径规划模块
包含A*路径规划算法实现
"""

import math
import heapq
from typing import List, Tuple, Optional
from geometry import Vector2


class PathPlanner:
    """A*路径规划器"""
    
    def __init__(self, buildings, grid_size=2.0, safety_margin=8.0):
        """
        初始化路径规划器
        buildings: 建筑物列表
        grid_size: 网格大小（米）
        safety_margin: 与建筑物的安全距离（米）
        """
        self.buildings = buildings
        self.grid_size = grid_size
        self.safety_margin = safety_margin

        # 计算地图边界（基于所有建筑物的范围）
        self.map_min_x = float(260)
        self.map_max_x = float(740)
        self.map_min_y = float(-240)
        self.map_max_y = float(240)

        print(f"地图边界: X[{self.map_min_x:.1f}, {self.map_max_x:.1f}], Y[{self.map_min_y:.1f}, {self.map_max_y:.1f}]")
        
    def find_building_by_name(self, name: str):
        """根据名称精确查找建筑对象"""
        for b in self.buildings:
            if getattr(b, "name", None) == name:
                return b
        return None

    def find_building_by_direction(self, anchor_building_name: str, direction: str, 
                                   max_search_radius: float = 120.0, angle_threshold_deg: float = 60.0):
        """根据锚点建筑与相对方位（东/西/南/北）查找目标建筑。
        规则：
        - 候选建筑需在锚点附近（默认120米内）。
        - 候选建筑相对矢量与方向夹角小于阈值（默认60度）。
        - 在符合条件的建筑中优先距离最近；若无，则选择投影（沿方向分量）最大的。
        返回：建筑对象或None。
        """
        anchor = self.find_building_by_name(anchor_building_name)
        if anchor is None:
            return None
        ax, ay = float(anchor.pos_x), float(anchor.pos_y)

        direction = (direction or "").strip()
        dir_map = {
            "东": (1.0, 0.0),
            "西": (-1.0, 0.0),
            "北": (0.0, 1.0),
            "南": (0.0, -1.0),
        }
        if direction not in dir_map:
            return None
        dx_ref, dy_ref = dir_map[direction]

        def dot(x1, y1, x2, y2):
            return x1 * x2 + y1 * y2
        def norm(x, y):
            return math.sqrt(x * x + y * y)

        angle_threshold = math.radians(angle_threshold_deg)

        best_by_dist = None
        best_dist = float('inf')
        best_by_proj = None
        best_proj = -float('inf')

        for cand in self.buildings:
            if cand is anchor:
                continue
            cx, cy = float(cand.pos_x), float(cand.pos_y)
            vx, vy = (cx - ax), (cy - ay)
            d = norm(vx, vy)
            if d < 1e-6 or d > max_search_radius:
                continue
            # 与参考方向夹角
            cosang = dot(vx, vy, dx_ref, dy_ref) / max(d * 1.0, 1e-6)
            # 只接受投影为正（同向）
            if cosang <= 0:
                continue
            # 角度约束
            if math.acos(max(min(cosang, 1.0), -1.0)) > angle_threshold:
                continue
            # 符合方向，更新候选
            if d < best_dist:
                best_dist = d
                best_by_dist = cand
            # 记录最大投影（沿方向分量）
            proj = dot(vx, vy, dx_ref, dy_ref)
            if proj > best_proj:
                best_proj = proj
                best_by_proj = cand

        return best_by_dist or best_by_proj

    def is_point_in_map(self, point: Vector2) -> bool:
        """检查点是否在地图范围内"""
        return (self.map_min_x <= point.x <= self.map_max_x and 
                self.map_min_y <= point.y <= self.map_max_y)

    def is_point_in_building(self, point: Vector2, building, margin=0) -> bool:
        """检查点是否在建筑物内（考虑旋转和安全边距）"""
        # 建筑物中心
        center = Vector2(building.pos_x, building.pos_y)
        
        # 将点转换到建筑物局部坐标系
        dx = point.x - center.x
        dy = point.y - center.y
        
        # 反向旋转
        ori_rad = -math.radians(building.ori_z)
        local_x = dx * math.cos(ori_rad) - dy * math.sin(ori_rad)
        local_y = dx * math.sin(ori_rad) + dy * math.cos(ori_rad)
        
        # 检查是否在建筑物边界内（加上安全边距）
        half_length = (building.length / 2) + margin
        half_width = (building.width / 2) + margin
        
        return (abs(local_x) <= half_length and abs(local_y) <= half_width)
    
    def is_collision_free(self, point: Vector2, target_building=None) -> bool:
        """检查点是否无碰撞（不在任何建筑物内且在地图范围内）"""
        # 首先检查是否在地图范围内
        if not self.is_point_in_map(point):
            return False
            
        for building in self.buildings:
            # 跳过目标建筑物的检查（允许接近目标）
            if target_building and building.name == target_building.name:
                continue
            if self.is_point_in_building(point, building, self.safety_margin):
                return False
        return True
    
    def is_line_collision_free(self, start: Vector2, end: Vector2, target_building=None, num_checks=10) -> bool:
        """检查两点之间的直线是否无碰撞"""
        for i in range(num_checks + 1):
            t = i / num_checks
            point = Vector2(
                start.x + t * (end.x - start.x),
                start.y + t * (end.y - start.y)
            )
            if not self.is_collision_free(point, target_building):
                return False
        return True
    
    def get_neighbors(self, pos: Vector2, target_building=None) -> List[Vector2]:
        """获取一个位置的可达邻居节点"""
        neighbors = []
        # 8个方向的邻居
        directions = [
            (1, 0), (-1, 0), (0, 1), (0, -1),
            (1, 1), (-1, 1), (1, -1), (-1, -1)
        ]
        
        for dx, dy in directions:
            new_pos = Vector2(
                pos.x + dx * self.grid_size,
                pos.y + dy * self.grid_size
            )
            # 检查新位置是否在地图范围内且无碰撞
            if self.is_point_in_map(new_pos) and self.is_collision_free(new_pos, target_building):
                neighbors.append(new_pos)
        
        return neighbors
    
    def heuristic(self, a: Vector2, b: Vector2) -> float:
        """A*启发式函数（欧几里得距离）"""
        return math.sqrt((a.x - b.x)**2 + (a.y - b.y)**2)
    
    def plan_path(self, start: Vector2, goal: Vector2, target_building=None) -> List[Vector2]:
        """
        使用A*算法规划路径
        start: 起始位置
        goal: 目标位置
        target_building: 目标建筑物（可以接近）
        """
        # 若起点不在可行区域，尝试寻找最近的可行点
        if not self.is_collision_free(start, target_building):
            print("起点处于障碍或越界")
            corrected_start = self._find_nearest_collision_free(start, target_building)
            if corrected_start is not None:
                start = corrected_start
            else:
                print("起点处于障碍或越界且无法修正，返回直线路径")
                return [start, goal]

        # # 若终点不在可行区域，尝试向外寻找最近的可行点
        # if not self.is_collision_free(goal, target_building):
        #     print("目标点处于障碍或越界")
        #     corrected_goal = self._find_nearest_collision_free(goal, target_building)
        #     if corrected_goal is not None:
        #         goal = corrected_goal
        #     else:
        #         print("目标点处于障碍或越界且无法修正，返回直线路径")
        #         return [start, goal]

        # 首先检查是否可以直接到达
        if self.is_line_collision_free(start, goal, target_building):
            return [start, goal]
        
        # 将位置量化到网格
        def quantize(pos: Vector2) -> Tuple[int, int]:
            return (
                round(pos.x / self.grid_size),
                round(pos.y / self.grid_size)
            )
        
        def dequantize(grid_pos: Tuple[int, int]) -> Vector2:
            return Vector2(
                grid_pos[0] * self.grid_size,
                grid_pos[1] * self.grid_size
            )
        
        start_grid = quantize(start)
        goal_grid = quantize(goal)
        
        # A*算法
        open_set = [(0, start_grid)]
        came_from = {}
        g_score = {start_grid: 0}
        f_score = {start_grid: self.heuristic(start, goal)}
        closed_set = set()
        
        max_iterations = 50000  # 防止无限循环
        iteration = 0
        
        while open_set and iteration < max_iterations:
            iteration += 1
            current_f, current_grid = heapq.heappop(open_set)
            
            if current_grid in closed_set:
                continue
                
            closed_set.add(current_grid)
            current = dequantize(current_grid)
            
            # 检查是否到达目标附近
            if self.heuristic(current, goal) < self.grid_size * 4:
                # 重建路径
                path = []
                node = current_grid
                while node in came_from:
                    path.append(dequantize(node))
                    node = came_from[node]
                path.append(start)
                path.reverse()
                path.append(goal)  # 添加精确的目标点
                
                # 路径平滑
                return self.smooth_path(path, target_building)
            
            # 扩展邻居
            for neighbor in self.get_neighbors(current, target_building):
                neighbor_grid = quantize(neighbor)
                
                if neighbor_grid in closed_set:
                    continue
                
                # 计算新的g值
                tentative_g = g_score[current_grid] + self.heuristic(current, neighbor)
                
                if neighbor_grid not in g_score or tentative_g < g_score[neighbor_grid]:
                    came_from[neighbor_grid] = current_grid
                    g_score[neighbor_grid] = tentative_g
                    f = tentative_g + self.heuristic(neighbor, goal)
                    f_score[neighbor_grid] = f
                    heapq.heappush(open_set, (f, neighbor_grid))
        
        # 如果找不到路径，绘制车辆当前位置和所有障碍物，保存调试用图像
        print(f"警告：无法找到避障路径，使用直线路径")
        try:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(8, 8))
            # 绘制地图边界
            ax.set_xlim(self.map_min_x, self.map_max_x)
            ax.set_ylim(self.map_min_y, self.map_max_y)
            # 绘制所有建筑物
            for building in self.buildings:
                cx, cy = building.pos_x, building.pos_y
                l, w = building.length, building.width
                ori = math.radians(building.ori_z)
                # 计算四个角点
                corners = []
                for dx, dy in [(-l/2, -w/2), (-l/2, w/2), (l/2, w/2), (l/2, -w/2)]:
                    x = cx + dx * math.cos(ori) - dy * math.sin(ori)
                    y = cy + dx * math.sin(ori) + dy * math.cos(ori)
                    corners.append([x, y])
                corners.append(corners[0])
                xs, ys = zip(*corners)
                ax.plot(xs, ys, 'k-', linewidth=2)
                ax.fill(xs, ys, color='gray', alpha=0.3)
                ax.text(cx, cy, getattr(building, "name", ""), fontsize=8, ha='center', va='center')
            # 绘制车辆当前位置
            ax.plot(start.x, start.y, 'ro', markersize=10, label='车辆起点')
            ax.text(start.x, start.y, "Start", color='r', fontsize=10)
            # 绘制目标点
            ax.plot(goal.x, goal.y, 'go', markersize=10, label='目标点')
            ax.text(goal.x, goal.y, "Goal", color='g', fontsize=10)
            ax.set_title("A*路径规划失败调试图")
            ax.legend()
            plt.savefig("astar_debug_failed.png", dpi=150)
            plt.close(fig)
            print("已保存A*失败调试图：astar_debug_failed.png")
        except Exception as e:
            print(f"绘制A*失败调试图时出错: {e}")
        return [start, goal]
    
    def smooth_path(self, path: List[Vector2], target_building=None) -> List[Vector2]:
        """平滑路径，去除不必要的中间点"""
        if len(path) <= 2:
            return path
        
        smoothed = [path[0]]
        i = 0
        
        while i < len(path) - 1:
            # 尝试找到最远的可直达点
            j = len(path) - 1
            while j > i + 1:
                if self.is_line_collision_free(path[i], path[j], target_building):
                    break
                j -= 1
            smoothed.append(path[j])
            i = j
        
        return smoothed

    def _find_nearest_collision_free(self, origin: Vector2, target_building=None, max_radius: float = 30.0,
                                     radial_step: float = 1.0, angle_step_deg: float = 15.0) -> Optional[Vector2]:
        """
        在原点周围做同心圆扫描，寻找最近的无碰撞点。
        返回第一个满足 is_collision_free 的位置（优先半径小、角度小）。
        """
        # 如果原点已可行，直接返回
        if self.is_collision_free(origin, target_building):
            return origin

        angle_step = math.radians(angle_step_deg)
        radius = radial_step
        
        while radius <= max_radius:
            angle = 0.0
            while angle < 2 * math.pi:
                candidate = Vector2(
                    origin.x + radius * math.cos(angle),
                    origin.y + radius * math.sin(angle)
                )
                if self.is_collision_free(candidate, target_building):
                    return candidate
                angle += angle_step
            radius += radial_step

        return None
    
    def resample_path_uniform(self, path: List[Vector2], spacing=1.0) -> List[Vector2]:
        """
        将路径重采样为等间距点
        path: 原始路径点列表
        spacing: 点之间的距离（米）
        """
        if len(path) < 2:
            return path
        
        resampled = [path[0]]
        accumulated_dist = 0.0
        
        for i in range(1, len(path)):
            prev_point = path[i-1]
            curr_point = path[i]
            
            # 计算两点之间的距离
            segment_dist = math.sqrt(
                (curr_point.x - prev_point.x)**2 + 
                (curr_point.y - prev_point.y)**2
            )
            
            if segment_dist < 1e-6:  # 跳过重复点
                continue
            
            # 计算这个线段上需要插入多少个点
            num_points = int(segment_dist / spacing)
            
            # 在线段上均匀插入点
            for j in range(1, num_points + 1):
                t = j * spacing / segment_dist
                if t <= 1.0:
                    interp_point = Vector2(
                        prev_point.x + t * (curr_point.x - prev_point.x),
                        prev_point.y + t * (curr_point.y - prev_point.y)
                    )
                    resampled.append(interp_point)
            
            # 如果最后一个点距离上一个采样点超过0.5*spacing，则添加
            if i == len(path) - 1:
                last_resampled = resampled[-1]
                dist_to_end = math.sqrt(
                    (curr_point.x - last_resampled.x)**2 + 
                    (curr_point.y - last_resampled.y)**2
                )
                if dist_to_end > 0.5 * spacing:
                    resampled.append(curr_point)
        
        return resampled