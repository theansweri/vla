import sys
import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from planning.parking_violation_checker import ParkingViolationChecker, VehicleOccupancy
from region_manager import RegionManager, RegionType


def square(x, y, size):
    return [
        (x, y),
        (x + size, y),
        (x + size, y + size),
        (x, y + size),
    ]


def setup_checker():
    manager = RegionManager()
    manager.add_region(RegionType.PARKING, square(0, 0, 4), "parking-a", {"preferred_yaw_deg": 0})
    manager.add_region(RegionType.NO_PARKING, square(10, 0, 4), "no-parking-a")
    return ParkingViolationChecker(manager, cross_line_threshold=0.1, slant_angle_threshold=20)


class ParkingViolationCheckerTests(unittest.TestCase):
    def test_normal_parking_no_violation(self):
        checker = setup_checker()
        vehicle = VehicleOccupancy(vehicle_id="car1", footprint=square(0.5, 0.5, 3), yaw_deg=0, vehicle_number="PA123")

        violations = checker.check([vehicle])

        self.assertEqual(violations, [])

    def test_cross_line_violation(self):
        checker = setup_checker()
        vehicle = VehicleOccupancy(vehicle_id="car2", footprint=square(-1, 0.5, 3), yaw_deg=0)

        violations = checker.check([vehicle])

        self.assertEqual(violations[0].vehicle_id, "car2")
        self.assertIn("CROSSING_LINE", violations[0].violation_types)

    def test_no_parking_zone_violation(self):
        checker = setup_checker()
        vehicle = VehicleOccupancy(vehicle_id="car3", footprint=square(10.5, 0.5, 2), yaw_deg=0, vehicle_number="NP999")

        violations = checker.check([vehicle])
        report = checker.build_report(violations)

        self.assertIn("NO_PARKING_ZONE", violations[0].violation_types)
        self.assertEqual(report["violating_numbers"], ["NP999"])

    def test_slanted_parking_violation(self):
        checker = setup_checker()
        vehicle = VehicleOccupancy(vehicle_id="car4", footprint=square(0.5, 0.5, 3), yaw_deg=45)

        violations = checker.check([vehicle])

        self.assertIn("SLANTED_PARKING", violations[0].violation_types)

    def test_small_overlap_not_flagged(self):
        checker = setup_checker()
        vehicle = VehicleOccupancy(vehicle_id="car5", footprint=square(9.8, -0.2, 0.2), yaw_deg=0)

        violations = checker.check([vehicle])

        self.assertEqual(violations, [])
