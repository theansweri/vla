# 车辆编号 OCR 与违停检测接入指南

本文示例展示如何在本地脚本中复用新增的车辆编号检测（OCR）与停车违停检测能力，避免重复实现。

## 依赖与路径
- 本仓库根目录为 `vla_latest/`，示例假设运行脚本位于同仓库根目录或已将仓库根目录加入 `PYTHONPATH`。
- OCR 引擎需要实现最小接口 `OCREngine.detect_text(image) -> Sequence[TextDetection]`，返回的 `TextDetection` 需包含 `text`、`confidence`、`bbox`、`center` 字段。

## 从任务 JSON 初始化区域管理器
```python
from region_manager import RegionManager

# 示例任务 JSON，可替换为接口下发的任务或本地文件解析结果
task_json = {
    "no_parking_zones": [
        {"id": "np1", "polygon": [[0, 0], [10, 0], [10, 5], [0, 5]]}
    ],
    "parking_zones": [
        {"id": "p1", "polygon": [[0, 5], [10, 5], [10, 15], [0, 15]], "preferred_yaw_deg": 90}
    ]
}

region_manager = RegionManager()
region_manager.load_from_task_json(task_json)
```

## 在视觉模块接入车辆编号检测
```python
from perception.vehicle_number_detector import (
    Vehicle2DBox, VehicleNumberDetector, TextDetection, OCREngine,
)

class MyOCREngine(OCREngine):
    def detect_text(self, image):
        # 调用自有 OCR 模型返回 TextDetection 列表
        return [TextDetection(text="粤B12345", confidence=0.92, bbox=[], center=(0, 0))]

ocr_engine = MyOCREngine()
number_detector = VehicleNumberDetector(ocr_engine, min_confidence=0.6)

# 假设已有目标检测输出的车辆框
detections_2d = [Vehicle2DBox(100, 200, 200, 350, vehicle_id="car-1")]
frame = ...  # BGR/RGB 图像

number_results = number_detector.detect(frame, detections_2d)
```

`number_results` 中的每条结果带有 `vehicle_id`、`vehicle_number`、`box2d`、可选的 `box3d`，后续用于违停检测。

## 在规划/评估模块接入违停检测
```python
from planning.parking_violation_checker import ParkingViolationChecker, VehicleOccupancy

checker = ParkingViolationChecker(region_manager,
                                  cross_line_threshold=0.1,
                                  slant_angle_threshold=20.0)

# 将视觉侧输出转成占地多边形与朝向（yaw 可选）
vehicle_occupancies = [
    VehicleOccupancy(
        vehicle_id=nr.vehicle_id,
        vehicle_number=nr.vehicle_number,
        footprint=[(0, 6), (2, 6), (2, 8), (0, 8)],
        yaw_deg=95,
    )
    for nr in number_results
]

violations = checker.check(vehicle_occupancies)
report = checker.build_report(violations)
```

`report` 返回结构:
```json
{
  "violations": [
    {
      "vehicle_id": "car-1",
      "vehicle_number": "粤B12345",
      "types": ["CROSSING_LINE"],
      "details": {"overlap": [...], "angles": [...]}
    }
  ],
  "violating_numbers": ["粤B12345"]
}
```
可以直接复用到现有的结果发布通道。

## 快速检查
- 如果只想验证模块是否正常工作，可运行仓库自带测试：

```bash
python -m unittest discover -s vla_latest/tests
```

- 如需集成到已有脚本，确保在运行前 `PYTHONPATH` 包含仓库根目录，例如：

```bash
export PYTHONPATH=$(pwd)/vla_latest:$PYTHONPATH
python your_script.py
```
```
