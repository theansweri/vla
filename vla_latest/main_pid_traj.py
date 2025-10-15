"""
固定目标导航任务入口（用于横向/纵向 PID 与轨迹跟踪调试）

任务配置:
  - 时间: 90 秒
  - 目的地: 锦汇华庭A栋
  - 子目的地: 二单元
  - 动作: 到达

说明:
  - 跳过 OCR/LLM，直接向任务控制器注入固定任务
  - 使用现有 VLATaskController 状态机在 PARSING_TASK 阶段生成路径并开始 NAVIGATING
  - 可在本文件顶部调整横向 Stanley 增益与纵向速度 PID 参数
"""

import time
import logging

from typing import Dict

from metacar import SceneAPI, VehicleControl

from task_types import TaskInfo, TaskState
from vla_task_controller import VLATaskController

# GUI 开关
USE_GUI = True
if USE_GUI:
    # 使用支持导航路径显示的 GUI
    from gui_frame_traj import Dashboard


# ===================== 调参与任务配置 =====================
# 固定任务参数
TIME_LIMIT_SECONDS = 90
DESTINATION = "锦汇华庭A栋"
DESTINATION = "锦汇华庭B栋"
SUB_DESTINATION = "二单元"
ACTION = "到达"

# 控制器参数（用于快速调试）
# 纵向 PID（速度）
SPEED_KP = 0.25
SPEED_KI = 0.00
SPEED_KD = 0.02

# 横向 Stanley 增益
STANLEY_K = 2.5

# 路径重采样间距（米）
PATH_SPACING_M = 1.0


def build_fixed_task() -> TaskInfo:
    ocr_text = f"{TIME_LIMIT_SECONDS}秒内 {DESTINATION} {SUB_DESTINATION} {ACTION}"
    time_phrase = f"{TIME_LIMIT_SECONDS}秒内"
    location_phrase = f"{DESTINATION} {SUB_DESTINATION}"
    action_phrase = ACTION
    return TaskInfo(
        ocr_text=ocr_text,
        time_limit=TIME_LIMIT_SECONDS,
        destination=DESTINATION,
        sub_destination=SUB_DESTINATION,
        action=ACTION,
        time_phrase=time_phrase,
        location_phrase=location_phrase,
        action_phrase=action_phrase,
    )


def main():
    logging.basicConfig(
        filename="vla_autodrive.log",
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.DEBUG,
        encoding="utf-8",
    )
    logger = logging.getLogger(__name__)

    print("=== 固定目标导航任务启动（PID/轨迹调试） ===")
    print(f"任务: {TIME_LIMIT_SECONDS}s 内到达 {DESTINATION} {SUB_DESTINATION}，动作: {ACTION}")

    # 1) 连接仿真并获取场景
    api = SceneAPI()
    api.connect()
    scene_static_data = api.get_scene_static_data()
    if not scene_static_data.vla_extension:
        raise ValueError("当前场景不是VLA场景，无法进行固定导航任务")

    # 2) 创建任务控制器
    task_controller = VLATaskController(scene_static_data)

    # 2.1) 初始化 GUI（显示轨迹）
    dashboard = None
    if USE_GUI:
        dashboard = Dashboard(scene_static_data)

    # 3) 覆盖控制参数，便于 PID/Stanley 调参
    try:
        # 纵向 PID
        speed_pid = task_controller.navigation_controller.speed_pid
        speed_pid.kp = SPEED_KP
        speed_pid.ki = SPEED_KI
        speed_pid.kd = SPEED_KD

        # 横向 Stanley 增益与路径间距
        task_controller.navigation_controller.stanley_k = STANLEY_K
        task_controller.navigation_controller.path_spacing = PATH_SPACING_M
    except Exception as e:
        logger.warning(f"无法设置控制参数: {e}")

    print(
        f"控制参数: Stanley_k={task_controller.navigation_controller.stanley_k}, "
        f"SpeedPID(kp={task_controller.navigation_controller.speed_pid.kp}, "
        f"ki={task_controller.navigation_controller.speed_pid.ki}, "
        f"kd={task_controller.navigation_controller.speed_pid.kd}), "
        f"path_spacing={task_controller.navigation_controller.path_spacing}m"
    )

    # 4) 注入固定任务并切入解析阶段（由控制器生成轨迹并进入导航）
    task_controller.current_task = build_fixed_task()
    task_controller.task_start_time = time.time()
    task_controller.vla_submitted = False
    task_controller.current_state = TaskState.PARSING_TASK

    # 5) 主循环（无 OCR，传空字典）
    start_time = time.time()
    frame_count = 0

    try:
        for sim_car_msg, frames in api.main_loop():
            vehicle_control, vla_extension = task_controller.update(sim_car_msg, {}, frame_count)

            # 将控制指令下发到仿真
            api.set_vehicle_control(vehicle_control, vla_extension=vla_extension)

            # 更新 GUI，包含导航路径
            if USE_GUI and dashboard:
                nav_path = task_controller.get_navigation_path_for_gui()
                dashboard.update(sim_car_msg, nav_path)
                dashboard.update_frame(frames[2].frame)  # 只显示前目


            frame_count += 1

            # 超时自动结束
            if time.time() - start_time >= TIME_LIMIT_SECONDS:
                print("达到固定任务时间上限，准备结束并停车")
                break

    except KeyboardInterrupt:
        print("用户中断，安全停车...")
    except Exception as e:
        print(f"运行异常: {e}")
        logger.error(f"运行异常: {e}")

    # 6) 结束时安全停车
    try:
        stop_cmd = VehicleControl()
        stop_cmd.throttle = 0.0
        stop_cmd.brake = 1.0
        api.set_vehicle_control(stop_cmd, vla_extension=None)
    except Exception:
        pass

    # 7) 关闭 GUI
    try:
        if USE_GUI and dashboard:
            dashboard.quit()
    except Exception:
        pass

    print("固定目标导航任务结束")


if __name__ == "__main__":
    main()


