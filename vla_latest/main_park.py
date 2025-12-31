"""
简化版停车测试程序：
- 启动后直接进入“寻找停车位”流程（绕过OCR/LLM）
- 使用 ParkingFinder（含fallback）确定车位
- 规划两段路径（第一目标点15m外延，第二目标点车位中心），执行前往停车位
- 路径会保存到 navigation_path.png
"""

import os
import sys
LOCAL_METACAR_PATH = os.path.join(os.path.dirname(__file__), "autodrive_api_python-1.0.0")
if os.path.isdir(LOCAL_METACAR_PATH) and LOCAL_METACAR_PATH not in sys.path:
    sys.path.insert(0, LOCAL_METACAR_PATH)

import time
import logging
from typing import Dict

from metacar import SceneAPI, VehicleControl, SimCarMsg, GearMode
from geometry import Vector2
from vla_task_controller import VLATaskController
from task_types import TaskState, ParkingSearchConstraints


logging.basicConfig(
    filename="vla_autodrive.log",
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.DEBUG,
    encoding="utf-8",
)
logger = logging.getLogger(__name__)


def main():
    print("=== 停车测试程序启动 (main_park) ===")

    api = SceneAPI()
    api.connect()
    scene_static = api.get_scene_static_data()

    controller = VLATaskController(scene_static)

    # 在同一个主循环中完成预热与找位，避免提前中断底层套接字
    print("[main_park] 等待场景障碍物加载...")
    warmed = False
    warmup_frames = 0
    max_wait_frames = 90
    started_search = False

    frame_count = 0
    try:
        for sim_car_msg, frames in api.main_loop():
            frame_count += 1

            # 不使用OCR，传空字典
            ocr_results: Dict = {}

            # 预热阶段：观察障碍物是否出现/稳定
            if not warmed:
                warmup_frames += 1
                num_obs = len(getattr(sim_car_msg, 'obstacles', []) or [])
                if warmup_frames % 30 == 0:
                    print(f"[main_park] 预热中: 第{warmup_frames}帧，检测到障碍物{num_obs}个")
                if num_obs > 0 or warmup_frames >= max_wait_frames:
                    warmed = True
                    print("[main_park] 场景预热完成，开始寻找停车位")

            # 预热完成后，只初始化一次找车位流程（绕过OCR/LLM）
            if warmed and not started_search:
                controller.parking_constraints = ParkingSearchConstraints(time_limit_seconds=120, speed_limit_kmh=30.0)
                controller.parking_search_start_time = time.time()
                controller.parking_finder.start_search(controller.parking_constraints)
                controller.task_locked = True
                controller.current_state = TaskState.PARKING_SEARCHING
                # 强制指定目标车位为 [SLOT#3] center=(491.70,118.58), yaw=90.0°
                def _forced_step(sim_car_msg_inner, ocr_dict_inner):
                    return True, (491.70, 118.58, 90.0)
                controller.parking_finder.step = _forced_step  # 覆盖为固定返回
                print("[main_park] 已强制指定车位: center=(491.70,118.58), yaw=90.0°")
                started_search = True
                print("[main_park] 已直接进入寻找停车位流程 (PARKING_SEARCHING)")

            vehicle_control, vla_ext = controller.update(sim_car_msg, ocr_results, frame_count)
            api.set_vehicle_control(vehicle_control, vla_extension=vla_ext)

            # 每秒打印一次车辆位置与状态，并统计障碍物数量
            if frame_count % 30 == 0:
                pos_x, pos_y = sim_car_msg.pose_gnss.pos_x, sim_car_msg.pose_gnss.pos_y
                speed = sim_car_msg.main_vehicle.speed
                yaw = sim_car_msg.pose_gnss.ori_z
                num_obs = len(getattr(sim_car_msg, 'obstacles', []) or [])
                print(f"[main_park] 状态:{controller.current_state.name} 位置:({pos_x:.2f},{pos_y:.2f}) 速度:{speed:.2f} 航向:{yaw:.1f}° 障碍物:{num_obs}")

            # 简单完成条件：进入 WAITING_NEW_TASK 视为已停车
            if controller.current_state == TaskState.WAITING_NEW_TASK:
                print("[main_park] 已完成停车流程，退出。")
                break

    except KeyboardInterrupt:
        print("用户中断程序")
    except Exception as e:
        logger.error(f"main_park 运行异常: {e}")
        print(f"main_park 运行异常: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()


