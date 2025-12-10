import sys
import pathlib
from typing import List
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from perception.vehicle_number_detector import (
    Vehicle2DBox,
    Vehicle3DBox,
    VehicleNumberDetector,
    TextDetection,
)


class FakeFrame:
    def __init__(self, shape):
        self.shape = shape

    def __getitem__(self, item):
        return self


class FakeOCREngine:
    def __init__(self, detections: List[TextDetection]):
        self.detections = detections
        self.calls = 0

    def detect_text(self, image):
        self.calls += 1
        return self.detections


class VehicleNumberDetectorTests(unittest.TestCase):
    def test_vehicle_number_detector_picks_best_plate(self):
        frame = FakeFrame((20, 20, 3))
        detections = [
            TextDetection(text="noise", confidence=0.9, bbox=[], center=(0, 0)),
            TextDetection(text="AB1234", confidence=0.8, bbox=[], center=(0, 0)),
            TextDetection(text="CD5678", confidence=0.95, bbox=[], center=(0, 0)),
        ]
        detector = VehicleNumberDetector(FakeOCREngine(detections), min_confidence=0.7)

        boxes_2d = [Vehicle2DBox(0, 0, 10, 10, vehicle_id="car-1")]
        boxes_3d = [Vehicle3DBox(center=(0, 0, 0), size=(4, 2, 1.5), yaw_deg=90)]

        results = detector.detect(frame, boxes_2d, boxes_3d)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].vehicle_number, "CD5678")
        self.assertEqual(results[0].box3d, boxes_3d[0])
        self.assertEqual(detector.ocr_engine.calls, 1)

    def test_vehicle_number_detector_filters_low_confidence(self):
        frame = FakeFrame((10, 10, 3))
        detections = [TextDetection(text="XYZ999", confidence=0.2, bbox=[], center=(0, 0))]
        detector = VehicleNumberDetector(FakeOCREngine(detections), min_confidence=0.5)

        boxes_2d = [Vehicle2DBox(0, 0, 9, 9, vehicle_id="car-2")]
        results = detector.detect(frame, boxes_2d)

        self.assertEqual(results, [])
