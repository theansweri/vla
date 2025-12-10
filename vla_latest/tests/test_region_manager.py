import sys
import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from region_manager import RegionManager, RegionType


class RegionManagerTests(unittest.TestCase):
    def test_region_manager_loads_polygons_from_json(self):
        manager = RegionManager()
        payload = {
            "no_parking_zones": [
                {"id": "np-1", "polygon": [[0, 0], [0, 10], [10, 10], [10, 0]]}
            ],
            "parking_zones": [
                {"id": "p-1", "polygon": [[20, 0], [20, 10], [30, 10], [30, 0]], "preferred_yaw_deg": 90}
            ],
        }

        regions = manager.load_from_task_json(payload)

        self.assertEqual(len(regions), 2)
        self.assertEqual(manager.no_parking_regions[0].region_id, "np-1")
        self.assertEqual(manager.parking_regions[0].region_type, RegionType.PARKING)
        self.assertEqual(manager.parking_regions[0].properties["preferred_yaw_deg"], 90)
