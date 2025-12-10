"""停车违停检测模块。"""

from dataclasses import dataclass
import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from region_manager import RegionManager, Region


@dataclass
class VehicleOccupancy:
    """车辆在地面上的占地信息。"""

    vehicle_id: str
    footprint: List[Tuple[float, float]]
    yaw_deg: Optional[float] = None
    vehicle_number: Optional[str] = None


@dataclass
class ParkingViolation:
    vehicle_id: str
    violation_types: List[str]
    details: Dict
    vehicle_number: Optional[str] = None


class ParkingViolationChecker:
    """基于区域多边形的违停检测。

    - 禁停区：只要占地与区域交叠比例超过 `cross_line_threshold` 即判定违停。
    - 正常停车区：允许少量越界，超出占地比例大于阈值视为压线；
      若提供区域朝向/车辆朝向，则可检测斜停。
    """

    def __init__(self, region_manager: RegionManager,
                 cross_line_threshold: float = 0.1,
                 slant_angle_threshold: float = 20.0):
        self.region_manager = region_manager
        self.cross_line_threshold = cross_line_threshold
        self.slant_angle_threshold = slant_angle_threshold

    def check(self, vehicle_occupancies: Iterable[VehicleOccupancy]) -> List[ParkingViolation]:
        violations: List[ParkingViolation] = []
        for occupancy in vehicle_occupancies:
            violation_types: List[str] = []
            detail: Dict = {"overlap": [], "angles": []}

            for region in self.region_manager.no_parking_regions:
                ratio = self._intersection_ratio(occupancy.footprint, region.polygon)
                if ratio >= self.cross_line_threshold:
                    violation_types.append("NO_PARKING_ZONE")
                    detail["overlap"].append({"region": region.region_id, "ratio": ratio})

            parking_overlap = 0.0
            parking_region_used: Optional[Region] = None
            for region in self.region_manager.parking_regions:
                ratio = self._intersection_ratio(occupancy.footprint, region.polygon)
                if ratio > parking_overlap:
                    parking_overlap = ratio
                    parking_region_used = region

            if parking_region_used:
                outside_ratio = 1.0 - parking_overlap
                if outside_ratio > self.cross_line_threshold:
                    violation_types.append("CROSSING_LINE")
                    detail["overlap"].append({"region": parking_region_used.region_id, "ratio": parking_overlap})

                if occupancy.yaw_deg is not None:
                    preferred = self._region_preferred_yaw(parking_region_used)
                    if preferred is not None:
                        diff = self._angle_diff(abs(preferred), abs(occupancy.yaw_deg))
                        detail["angles"].append({"region": parking_region_used.region_id, "diff": diff})
                        if diff > self.slant_angle_threshold:
                            violation_types.append("SLANTED_PARKING")
            else:
                # 未落在任何停车区内但有禁停信息时，仍报告最小交叠的禁停区域
                if violation_types:
                    pass

            if violation_types:
                violations.append(
                    ParkingViolation(
                        vehicle_id=occupancy.vehicle_id,
                        vehicle_number=occupancy.vehicle_number,
                        violation_types=sorted(set(violation_types)),
                        details=detail,
                    )
                )

        return violations

    def build_report(self, violations: Sequence[ParkingViolation]) -> Dict:
        """格式化违停车辆编号列表，便于复用发布通道。"""
        violating_numbers = []
        for v in violations:
            number = v.vehicle_number or v.vehicle_id
            if number not in violating_numbers:
                violating_numbers.append(number)
        return {
            "violations": [
                {
                    "vehicle_id": v.vehicle_id,
                    "vehicle_number": v.vehicle_number,
                    "types": v.violation_types,
                    "details": v.details,
                }
                for v in violations
            ],
            "violating_numbers": violating_numbers,
        }

    def _intersection_ratio(self, poly_a: List[Tuple[float, float]], poly_b: List[Tuple[float, float]]) -> float:
        area_a = self._polygon_area(poly_a)
        if area_a <= 0:
            return 0.0
        intersection = self._polygon_intersection_area(poly_a, poly_b)
        return intersection / area_a

    def _region_preferred_yaw(self, region: Region) -> Optional[float]:
        if "preferred_yaw_deg" in region.properties:
            return float(region.properties["preferred_yaw_deg"])
        return self._polygon_orientation(region.polygon)

    def _polygon_area(self, polygon: List[Tuple[float, float]]) -> float:
        area = 0.0
        for i in range(len(polygon)):
            x1, y1 = polygon[i]
            x2, y2 = polygon[(i + 1) % len(polygon)]
            area += x1 * y2 - x2 * y1
        return abs(area) * 0.5

    def _polygon_intersection_area(self, poly1: List[Tuple[float, float]], poly2: List[Tuple[float, float]]) -> float:
        clipped = poly1
        for i in range(len(poly2)):
            clip_start = poly2[i]
            clip_end = poly2[(i + 1) % len(poly2)]
            clipped = self._sutherland_hodgman(clipped, clip_start, clip_end)
            if not clipped:
                return 0.0
        return self._polygon_area(clipped)

    def _sutherland_hodgman(self, subject: List[Tuple[float, float]],
                            edge_start: Tuple[float, float], edge_end: Tuple[float, float]) -> List[Tuple[float, float]]:
        output: List[Tuple[float, float]] = []
        for i in range(len(subject)):
            current = subject[i]
            prev = subject[i - 1]
            curr_in = self._is_inside(current, edge_start, edge_end)
            prev_in = self._is_inside(prev, edge_start, edge_end)
            if curr_in:
                if not prev_in:
                    output.append(self._intersection(prev, current, edge_start, edge_end))
                output.append(current)
            elif prev_in:
                output.append(self._intersection(prev, current, edge_start, edge_end))
        return output

    def _is_inside(self, point: Tuple[float, float], edge_start: Tuple[float, float], edge_end: Tuple[float, float]) -> bool:
        return ((edge_end[0] - edge_start[0]) * (point[1] - edge_start[1]) -
                (edge_end[1] - edge_start[1]) * (point[0] - edge_start[0])) >= 0

    def _intersection(self, p1: Tuple[float, float], p2: Tuple[float, float],
                      edge_start: Tuple[float, float], edge_end: Tuple[float, float]) -> Tuple[float, float]:
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = edge_start
        x4, y4 = edge_end
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-6:
            return p2
        px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
        py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
        return px, py

    def _polygon_orientation(self, polygon: List[Tuple[float, float]]) -> Optional[float]:
        if len(polygon) < 2:
            return None
        dx = polygon[1][0] - polygon[0][0]
        dy = polygon[1][1] - polygon[0][1]
        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            return None
        return math.degrees(math.atan2(dy, dx))

    def _angle_diff(self, a: float, b: float) -> float:
        diff = abs(a - b) % 360.0
        return diff if diff <= 180.0 else 360.0 - diff
