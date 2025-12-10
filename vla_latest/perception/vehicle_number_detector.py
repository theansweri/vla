"""车辆编号检测与OCR模块。

该模块聚合车辆2D/3D框信息与OCR识别，输出带编号的车辆列表。
"""

from dataclasses import dataclass
import re
from typing import Any, List, Optional, Sequence, Tuple

@dataclass
class TextDetection:
    """OCR 的文本检测输出。"""

    text: str
    confidence: float
    bbox: List[List[float]]
    center: Tuple[float, float]


class OCREngine:
    """最小化的 OCR 引擎接口，需实现 ``detect_text``。"""

    def detect_text(self, image) -> Sequence[TextDetection]:  # pragma: no cover - 接口定义
        raise NotImplementedError


@dataclass
class Vehicle2DBox:
    """像素坐标系下的车辆2D框（左上/右下）。"""

    xmin: float
    ymin: float
    xmax: float
    ymax: float
    vehicle_id: Optional[str] = None

    def as_int_bounds(self, frame_shape: Tuple[int, int, int]) -> Tuple[int, int, int, int]:
        height, width = frame_shape[:2]
        x1 = max(0, int(self.xmin))
        y1 = max(0, int(self.ymin))
        x2 = min(width, int(self.xmax))
        y2 = min(height, int(self.ymax))
        return x1, y1, x2, y2


@dataclass
class Vehicle3DBox:
    """世界坐标系下的车辆3D框描述。"""

    center: Tuple[float, float, float]
    size: Tuple[float, float, float]  # (length, width, height)
    yaw_deg: float


@dataclass
class VehicleNumberDetection:
    """带编号的车辆检测结果。"""

    vehicle_id: str
    vehicle_number: str
    confidence: float
    box2d: Vehicle2DBox
    box3d: Optional[Vehicle3DBox] = None


class VehicleNumberDetector:
    """基于车辆框的车牌/编号OCR检测器。

    使用外部 OCR 引擎在车辆2D框内进行文本检测，并筛选车牌/编号文本。
    """

    def __init__(self, ocr_engine: OCREngine, min_confidence: float = 0.5,
                 plate_pattern: Optional[re.Pattern] = None):
        self.ocr_engine = ocr_engine
        self.min_confidence = min_confidence
        self.plate_pattern = plate_pattern or re.compile(r"[A-Za-z0-9]{4,}")

    def detect(self, frame: Any, vehicle_boxes_2d: Sequence[Vehicle2DBox],
               vehicle_boxes_3d: Optional[Sequence[Optional[Vehicle3DBox]]] = None) -> List[VehicleNumberDetection]:
        """在给定帧中对车辆框执行OCR并输出编号。

        Args:
            frame: 原始BGR或RGB图像。
            vehicle_boxes_2d: 车辆像素框列表。
            vehicle_boxes_3d: 可选的车辆3D框列表（与2D框同索引对齐）。

        Returns:
            带编号文本的检测结果列表。
        """
        if vehicle_boxes_3d is None:
            vehicle_boxes_3d = [None] * len(vehicle_boxes_2d)

        results: List[VehicleNumberDetection] = []
        for idx, (box2d, box3d) in enumerate(zip(vehicle_boxes_2d, vehicle_boxes_3d)):
            x1, y1, x2, y2 = box2d.as_int_bounds(frame.shape)
            if x2 <= x1 or y2 <= y1:
                continue

            crop = frame[y1:y2, x1:x2]
            detections = self.ocr_engine.detect_text(crop) or []
            best = self._select_best_detection(detections)
            if best is None:
                continue

            vehicle_id = box2d.vehicle_id or f"vehicle_{idx}"
            results.append(
                VehicleNumberDetection(
                    vehicle_id=vehicle_id,
                    vehicle_number=best.text,
                    confidence=best.confidence,
                    box2d=box2d,
                    box3d=box3d,
                )
            )

        return results

    def _select_best_detection(self, detections: Sequence[TextDetection]) -> Optional[TextDetection]:
        """从检测列表中过滤并选择最可信的车牌文本。"""
        best: Optional[TextDetection] = None
        for det in detections:
            if det.confidence < self.min_confidence:
                continue
            if not self.plate_pattern.search(det.text):
                continue
            if best is None or det.confidence > best.confidence:
                best = det
        return best
