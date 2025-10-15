from typing import Optional, Tuple, Dict, List
import math

from metacar import SceneStaticData, SimCarMsg, ObstacleType
from geometry import Vector2


class ParkingFinder:
    """
    停车位查找器（占位实现）
    - 负责：
      1) 从Scene/地图中提取停车位候选坐标（车位多边形/中心点/朝向）。
      2) 在线检测空车位（基于视觉/占用判断）。
      3) 输出目标停车位位姿，供泊车/停车控制使用。
    """

    def __init__(self, scene_static_data: SceneStaticData):
        self.scene_static_data = scene_static_data
        self._is_searching = False
        self._candidate_slots = []  # [(x, y, yaw), ...]
        self.debug = True
        self._last_debug_info: Dict = {}

    def start_search(self, constraints) -> None:
        self._is_searching = True
        # 步骤1：从地图提取停车位候选（伪代码）
        # self._candidate_slots = self._extract_parking_slots_from_map(self.scene_static_data)
        self._candidate_slots = []
        if self.debug:
            print("[PARK] 启动找车位流程。")

    def reset(self) -> None:
        self._is_searching = False
        self._candidate_slots = []

    def step(self, sim_car_msg: SimCarMsg, ocr_data_by_cam: Dict) -> Tuple[bool, Optional[Tuple[float, float, float]]]:
        if not self._is_searching:
            return False, None

        # 从当前场景障碍物中抽取候选停车位与占用物
        parking_slots = self._extract_parking_slots_from_obstacles(sim_car_msg)
        occupiers = self._extract_occupying_obstacles(sim_car_msg)

        # 根据与自车距离排序候选车位
        car_pos = Vector2(sim_car_msg.pose_gnss.pos_x, sim_car_msg.pose_gnss.pos_y)
        def slot_dist(s):
            sx, sy = s[0], s[1]
            return math.hypot(sx - car_pos.x, sy - car_pos.y)
        parking_slots.sort(key=slot_dist)

        if self.debug:
            print(f"[PARK] 候选车位: {len(parking_slots)} 个，占用体: {len(occupiers)} 个。车辆位置=({car_pos.x:.2f},{car_pos.y:.2f})")
            # 打印所有车位清单
            if parking_slots:
                for i, (sx, sy, syaw, slen, swid) in enumerate(parking_slots):
                    print(f"[PARK][SLOT#{i}] center=({sx:.2f},{sy:.2f}), yaw={syaw:.1f}°, size=({slen:.1f},{swid:.1f})")
            else:
                print("[PARK] 未检测到任何PARKING_SLOT类型障碍物，请检查障碍物类型或地图数据。")
            # 打印所有占用体清单
            if occupiers:
                for i, (ox, oy, oyaw, olen, owid, otype) in enumerate(occupiers):
                    print(f"[PARK][OCC#{i}] type={otype}, center=({ox:.2f},{oy:.2f}), yaw={oyaw:.1f}°, size=({olen:.1f},{owid:.1f})")

        # 逐个检查是否被占用；若一个都没有被识别到，则使用用户提供的假定车位作为fallback
        checked = []
        for slot in parking_slots:
            sx, sy, syaw, slen, swid = slot  # 包含尺寸
            occupied = False
            for ob in occupiers:
                ox, oy, oyaw, olen, owid, _otype = ob
                if self._rects_overlap((sx, sy, syaw, slen, swid), (ox, oy, oyaw, olen, owid)):
                    occupied = True
                    break
            d = math.hypot(sx - car_pos.x, sy - car_pos.y)
            checked.append({"center": (sx, sy), "yaw": syaw, "size": (slen, swid), "dist": d, "occupied": occupied})
            if self.debug:
                occ_str = "占用" if occupied else "空闲"
                print(f"[PARK] 车位@({sx:.2f},{sy:.2f}), yaw={syaw:.1f}°, size=({slen:.1f},{swid:.1f}), 距离={d:.1f}m -> {occ_str}")
            if not occupied:
                # 返回空车位中心与朝向
                self._last_debug_info = {
                    "slots": len(parking_slots),
                    "occupiers": len(occupiers),
                    "picked": {"center": (sx, sy), "yaw": syaw, "size": (slen, swid), "dist": d},
                    "checked": checked[:10],
                }
                return True, (sx, sy, syaw)

        self._last_debug_info = {
            "slots": len(parking_slots),
            "occupiers": len(occupiers),
            "picked": None,
            "checked": checked[:10],
        }
        # Fallback: 使用用户提供的假定车位（用于场景调试）
        # 位置:(491.68,118.45), 航向:270°
        fallback_slot = (491.68, 118.45, 270)
        if self.debug:
            print(f"[PARK] 使用Fallback车位: center=({fallback_slot[0]:.2f},{fallback_slot[1]:.2f}), yaw={fallback_slot[2]:.1f}°")
        return True, fallback_slot

    # 示例：地图中停车位坐标提取（伪代码，仅接口/注释）
    def _extract_parking_slots_from_map(self, scene_static_data: SceneStaticData):
        """
        伪代码：
        - 遍历scene_static_data.vla_extension.parking_areas 或相关结构。
        - 将每个车位的多边形/中心点、朝向等信息转为 (x, y, yaw)。
        - 可筛除与建筑体过近或不可达区域。
        - 返回候选列表。
        """
        slots = []
        # for area in getattr(scene_static_data.vla_extension, 'parking_areas', []) or []:
        #     for slot in area.slots:
        #         slots.append((slot.center_x, slot.center_y, slot.yaw))
        return slots

    def _extract_parking_slots_from_obstacles(self, sim_car_msg: SimCarMsg) -> List[Tuple[float, float, float, float, float]]:
        """
        从障碍物列表抽取停车位：返回 (x, y, yaw, length, width)
        约定：ObstacleType.PARKING_SLOT 的 pos/ori/size 代表车位中心、朝向与尺寸。
        yaw 使用场景角度（度）。
        """
        slots: List[Tuple[float, float, float, float, float]] = []
        for ob in getattr(sim_car_msg, 'obstacles', []) or []:
            if self._is_type(ob, ObstacleType.PARKING_SLOT):
                slots.append((float(ob.pos_x), float(ob.pos_y), float(ob.ori_z), float(ob.length), float(ob.width)))
        return slots

    def _extract_occupying_obstacles(self, sim_car_msg: SimCarMsg) -> List[Tuple[float, float, float, float, float]]:
        """
        提取可能占用车位的障碍物（车辆/静态体等）：返回 (x, y, yaw, length, width)
        """
        candidates: List[Tuple[float, float, float, float, float, str]] = []
        vehicle_like = {
            ObstacleType.CAR,
            ObstacleType.TRUCK,
            ObstacleType.BUS,
            ObstacleType.MOTORCYCLE,
            ObstacleType.BICYCLE,
            ObstacleType.BICYCLE_STATIC,
            ObstacleType.SPECIAL_VEHICLE,
            ObstacleType.PEDESTRIAN,
            ObstacleType.RIDER,
            ObstacleType.DYNAMIC,
            ObstacleType.STATIC,
            ObstacleType.ROAD_OBSTACLE,
        }
        for ob in getattr(sim_car_msg, 'obstacles', []) or []:
            if self._is_one_of_types(ob, vehicle_like):
                otype_str = self._type_str(ob)
                candidates.append((float(ob.pos_x), float(ob.pos_y), float(ob.ori_z), float(ob.length), float(ob.width), otype_str))
        # 也把主车视为占用体，避免选中自身所在位
        mv = getattr(sim_car_msg, 'main_vehicle', None)
        if mv is not None:
            candidates.append((float(sim_car_msg.pose_gnss.pos_x), float(sim_car_msg.pose_gnss.pos_y), float(sim_car_msg.pose_gnss.ori_z), float(mv.length), float(mv.width), 'MAIN_VEHICLE'))
        return candidates

    def _type_value(self, ob) -> Optional[int]:
        t = getattr(ob, 'type', None)
        if t is None:
            return None
        try:
            # Enum case
            return int(getattr(t, 'value', t))
        except Exception:
            try:
                return int(t)
            except Exception:
                return None

    def _is_type(self, ob, enum_member: ObstacleType) -> bool:
        v = self._type_value(ob)
        if v is None:
            return False
        try:
            return v == int(enum_member.value)
        except Exception:
            return False

    def _is_one_of_types(self, ob, enum_set: set) -> bool:
        v = self._type_value(ob)
        if v is None:
            return False
        for em in enum_set:
            try:
                if v == int(em.value):
                    return True
            except Exception:
                continue
        return False

    def _type_str(self, ob) -> str:
        t = getattr(ob, 'type', None)
        if t is None:
            return 'UNKNOWN'
        try:
            # Enum with name
            n = getattr(t, 'name', None)
            if n:
                return str(n)
        except Exception:
            pass
        try:
            return f"{int(t)}"
        except Exception:
            return str(t)

    def _rects_overlap(self, r1, r2) -> bool:
        """
        检测两个旋转矩形是否重叠。
        r = (cx, cy, yaw_deg, length, width)
        使用分离轴定理（SAT）在两个矩形的局部轴上投影判断。
        """
        def corners(cx, cy, yaw_deg, length, width) -> List[Vector2]:
            yaw = math.radians(yaw_deg)
            hx = length / 2.0
            hy = width / 2.0
            local = [
                Vector2(+hx, +hy),
                Vector2(+hx, -hy),
                Vector2(-hx, -hy),
                Vector2(-hx, +hy),
            ]
            world = []
            for p in local:
                pr = p.rotate_rad(-yaw)  # 注意场景yaw为顺时针，geometry里逆时针正，取负号
                world.append(Vector2(cx, cy) + pr)
            return world

        def axes_from_corners(cs: List[Vector2]) -> List[Vector2]:
            # 两条轴：矩形边的法向（标准化）
            axes = []
            for i in range(2):
                edge = Vector2(cs[i+1].x - cs[i].x, cs[i+1].y - cs[i].y)
                # 取法向
                axis = Vector2(-edge.y, edge.x)
                length = math.hypot(axis.x, axis.y) or 1.0
                axes.append(Vector2(axis.x/length, axis.y/length))
            return axes

        def project(cs: List[Vector2], axis: Vector2) -> Tuple[float, float]:
            vals = [c.x*axis.x + c.y*axis.y for c in cs]
            return min(vals), max(vals)

        c1 = corners(*r1)
        c2 = corners(*r2)
        axes = axes_from_corners(c1) + axes_from_corners(c2)
        for axis in axes:
            min1, max1 = project(c1, axis)
            min2, max2 = project(c2, axis)
            if max1 < min2 or max2 < min1:
                return False
        return True



