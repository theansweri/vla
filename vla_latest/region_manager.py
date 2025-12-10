"""停车/禁停区域管理模块。"""

from dataclasses import dataclass, field
import json
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple


class RegionType(str, Enum):
    """区域类型枚举。"""

    NO_PARKING = "no_parking"
    PARKING = "parking"


@dataclass
class Region:
    """区域定义。"""

    region_id: str
    polygon: List[Tuple[float, float]]
    region_type: RegionType
    properties: Dict = field(default_factory=dict)


class RegionManager:
    """统一管理禁停区/停车区多边形。"""

    def __init__(self):
        self.regions: List[Region] = []

    def add_region(self, region_type: RegionType, polygon: Sequence[Sequence[float]],
                   region_id: Optional[str] = None, properties: Optional[Dict] = None) -> Region:
        poly = [(float(x), float(y)) for x, y in polygon]
        if len(poly) < 3:
            raise ValueError("polygon must contain at least 3 points")
        region = Region(
            region_id=region_id or f"{region_type.value}_{len(self.regions)}",
            polygon=poly,
            region_type=region_type,
            properties=properties or {},
        )
        self.regions.append(region)
        return region

    def load_from_task_json(self, task_json) -> List[Region]:
        """从任务 JSON 中解析区域多边形。"""
        if isinstance(task_json, str):
            data = json.loads(task_json)
        else:
            data = task_json or {}

        added: List[Region] = []
        no_parking_items = []
        no_parking_items.extend(data.get("no_parking_zones", []) or [])
        no_parking_items.extend(data.get("forbidden_zones", []) or [])
        for item in no_parking_items:
            added.append(self._add_from_dict(item, RegionType.NO_PARKING))

        parking_items = []
        parking_items.extend(data.get("parking_zones", []) or [])
        parking_items.extend(data.get("parking_areas", []) or [])
        for item in parking_items:
            added.append(self._add_from_dict(item, RegionType.PARKING))

        return added

    def _add_from_dict(self, item: Dict, region_type: RegionType) -> Region:
        polygon = item.get("polygon") or item.get("points") or item.get("coords")
        if polygon is None:
            raise ValueError("task json region missing polygon")
        region_id = item.get("id") or item.get("name")
        properties = {k: v for k, v in item.items() if k not in {"polygon", "points", "coords", "id", "name"}}
        return self.add_region(region_type, polygon, region_id, properties)

    def get_regions(self, region_type: Optional[RegionType] = None) -> List[Region]:
        if region_type is None:
            return list(self.regions)
        return [r for r in self.regions if r.region_type == region_type]

    @property
    def parking_regions(self) -> List[Region]:
        return self.get_regions(RegionType.PARKING)

    @property
    def no_parking_regions(self) -> List[Region]:
        return self.get_regions(RegionType.NO_PARKING)
