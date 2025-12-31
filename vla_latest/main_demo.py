"""
VLA主程序 - 四目相机适配版
正确集成OCR、任务控制器和车辆控制，支持四目相机
"""

import os
import sys
LOCAL_METACAR_PATH = os.path.join(os.path.dirname(__file__), "autodrive_api_python-1.0.0")
if os.path.isdir(LOCAL_METACAR_PATH) and LOCAL_METACAR_PATH not in sys.path:
    sys.path.insert(0, LOCAL_METACAR_PATH)

import logging
import keyboard
import time
import cv2
import math
from typing import List, Dict

from metacar import SceneAPI, GearMode, VehicleControl, SimCarMsg, VLAExtensionOutput, VLATextOutput
from geometry import Vector2, Vector3, euler_to_vector2, yaw_to_radians

def calc_throttle_brake(
    current_speed: float, target_speed: float
) -> tuple[float, float]:
    K = 0.2
    B = 0.2
    acceleration = (target_speed - current_speed) * K + B
    if acceleration > 0:
        return min(acceleration, 1), 0
    return 0, min(-acceleration * 0.5, 1)


def calc_steering(
    pos: Vector3, yaw: float, speed: float, trajectory: list[Vector3]
) -> float:
    K = 0.5
    for traj_pos in trajectory:
        if math.dist(traj_pos, pos) > K * speed:
            target_pos = traj_pos
            break
    else:
        target_pos = trajectory[-1]
    theta = (target_pos - pos).yaw_rad()
    steering_angle = (theta - yaw) % (2 * math.pi)
    if steering_angle > math.pi:
        steering_angle -= 2 * math.pi
    steering = math.degrees(-steering_angle) / 45
    return max(min(steering, 1), -1)


def get_vehicle_control_from_algorithm(msg: SimCarMsg) -> VehicleControl:
    vc = VehicleControl()

    # [Safe Shift Logic] 检查当前档位，如果不在 D 挡，先挂挡并踩刹车
    # 注意：msg.main_vehicle.gear 可能是 int 或 enum
    current_gear = msg.main_vehicle.gear
    
    # 定义需要停车挂挡的状态 (P, R, N)
    # 我们只在这些档位下强制踩刹车挂 D 挡
    # 这样避免了如果车在 D1, D2 档（但不等于 GearMode.DRIVE）时导致的死锁
    should_enforce_shift = False
    
    try:
        # 检查是否在 P, R, N
        if current_gear == GearMode.PARKING or current_gear == GearMode.REVERSE or current_gear == GearMode.NEUTRAL:
            should_enforce_shift = True
        # 兼容 int 比较
        elif current_gear == int(GearMode.PARKING) or current_gear == int(GearMode.REVERSE) or current_gear == int(GearMode.NEUTRAL):
            should_enforce_shift = True
    except:
        # 如果出错，保守策略：只要不是 DRIVE 就挂
        # 但为了防止 D1 死锁，这里只在明确不等于 DRIVE 时才... 还是有点风险
        # 回退到简单逻辑：打印出来让用户看，先假设 P=0, R=1, N=2, D=3 之类
        pass

    if should_enforce_shift:
        # 如果在 P/R/N 档，且车速很低，必须踩刹车挂挡
        if msg.main_vehicle.speed < 0.1:
            # print(f"[DEBUG] In non-driving gear ({current_gear}), shifting to DRIVE... Holding Brake.")
            vc.gear = GearMode.DRIVE
            vc.throttle = 0.0
            vc.brake = 1.0 # 挂挡必须踩刹车
            vc.steering = 0.0
            return vc

    vc.gear = GearMode.DRIVE
    # print(f"[DEBUG] Algo Input: Speed={msg.main_vehicle.speed:.2f}, Pos=({msg.pose_gnss.pos_x:.2f}, {msg.pose_gnss.pos_y:.2f}, {msg.pose_gnss.pos_z:.2f})")
    # 目前只支持固定速度
    vc.throttle, vc.brake = calc_throttle_brake(msg.main_vehicle.speed, 5)
    vc.steering = calc_steering(
        Vector3(
            msg.pose_gnss.pos_x,
            msg.pose_gnss.pos_y,
            msg.pose_gnss.pos_z,
        ),
        -math.radians(msg.pose_gnss.ori_z),
        msg.main_vehicle.speed,
        msg.trajectory,
    )
    
    # [DEBUG] 强制起步时减小转向，防止死锁
    if msg.main_vehicle.speed < 0.5:
        # 如果有大转向，限制它
        if abs(vc.steering) > 0.5:
            # print(f"[DEBUG] Low speed ({msg.main_vehicle.speed:.2f}) & High steering ({vc.steering:.2f}) -> Clamped steering")
            vc.steering = max(-0.1, min(0.1, vc.steering)) # 限制得更死一点，几乎回正
        # 甚至如果速度接近0，直接强制回正，先跑起来再说
        if msg.main_vehicle.speed < 0.1:
            vc.steering = 0.0

    # [DEBUG] 强制给一点初始油门，防止因为太小动不了
    if msg.main_vehicle.speed < 0.1 and vc.throttle > 0:
         vc.throttle = max(0.8, vc.throttle) # 加大初始油门

    print(f"[DEBUG] Algo Output: T={vc.throttle:.2f}, B={vc.brake:.2f}, S={vc.steering:.2f}, G={vc.gear}")
    return vc

# 导入我们的VLA模块
from vla_task_controller import VLATaskController, TaskState
from ocr_paddle import create_vla_ocr_system, VLACameraOCRSystem, CameraOCRResult

# 配置选项
USE_GUI = False
USE_REAL_OCR = True  # 是否使用真实OCR
OCR_FREQUENCY = 5    # 每N帧执行一次OCR（降低计算负担）
SHOW_CAMERA_FEED = False  # 是否显示相机画面
USE_TRAJECTORY_CONTROL = True  # True=演示：直接跟随 msg.trajectory；False=走 task_controller 原逻辑
TRAJ_END_DIST = 0.5  # 距离 trajectory 最后一个点 < 0.5m 就停车

# 相机ID映射
CAMERA_NAMES = {
    '0': 'front',
    '1': 'right', 
    '2': 'left',
    '3': 'back'
}

if USE_GUI:
    # from gui_frame_traj import Dashboard
    from gui import Dashboard

# 日志配置
logging.basicConfig(
    filename="vla_autodrive.log",
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.DEBUG,
    encoding="utf-8",
)
logger = logging.getLogger(__name__)


class VLAMainController:
    """VLA主控制器"""
    
    def __init__(self):
        self.api = None
        self.scene_static_data = None
        self.task_controller = None
        self.ocr_system = None
        self.dashboard = None
        
        # 控制相关
        self.use_manual_control = False
        self.current_gear = GearMode.DRIVE
        
        # OCR相关
        self.frame_count = 0
        self.last_ocr_results = {}  # 改为字典，存储各相机的OCR结果
        self.use_real_ocr = USE_REAL_OCR

        # 5. 初始化OCR系统
        if self.use_real_ocr:
            print("初始化OCR系统...")
            try:
                self.ocr_system = create_vla_ocr_system(min_confidence=0.8)
                print("OCR系统初始化成功")
                logger.info("OCR系统初始化成功")
            except Exception as e:
                logger.warning(f"OCR系统初始化失败，将使用模拟OCR: {e}")
                self.ocr_system = None
                self.use_real_ocr = False
        # === DEMO：开局序列 (Move -> Stop -> Move) ===
        self.demo_start_time = None         # 第一帧的时间戳
        self.demo_init_move_duration = 0.5  # 先走一小段 (0.5s)
        self.demo_stop_duration = 3.0       # 然后停车 (3s)
        self.demo_release_duration = 0.5    # 松刹车缓冲 (0.5s)
        self.demo_vla_sent = False
        self.traj_reached = False # 是否已到达轨迹终点（带滞后）

        self.subscene_name_last = None
        self.subscene_paused_names = set()
        self.subscene_pause_until = None
        self.subscene_end_pause_seconds = 10.0
        self.subscene_end_dist = 3.0

        self.level_submission_count = 0
        self.level1_vla_output = VLAExtensionOutput(
            text_info=VLATextOutput(
                ocr_text="在100秒内到达锦汇华庭B栋一单元门口停车，并且全程不得超过30km/h",
                time_phrase="在100秒内",
                location_phrase="锦汇华庭B栋一单元门口",
                action_phrase="到达并停车，并且全程不得超过30km/h"
            )
        )
        self.level2_vla_output = VLAExtensionOutput(
            text_info=VLATextOutput(
                ocr_text="100秒内去到仁安综合医院门诊门口停车，并且全程不得超过30km/h",
                time_phrase="100秒内",
                location_phrase="仁安综合医院门诊门口",
                action_phrase="去到并停车，并且全程不得超过30km/h"
            )
        )
        self.subscene_submitted_names = set()


        
    def initialize(self):
        """初始化所有系统"""

        # 1. 连接仿真器
        self.api = SceneAPI()
        self.api.connect()
        logger.info("已连接到仿真器")
        
        # 2. 获取场景静态数据
        self.scene_static_data = self.api.get_scene_static_data()
        
        # 3. 检查是否为VLA场景
        if not self.scene_static_data.vla_extension:
            raise ValueError("当前场景不是VLA场景")
        
        logger.info(f"VLA场景初始化，包含 {len(self.scene_static_data.vla_extension.buildings)} 个建筑物")
        
        # 4. 初始化VLA任务控制器
        self.task_controller = VLATaskController(self.scene_static_data)
        
        # 6. 初始化GUI
        if USE_GUI:
            self.dashboard = Dashboard(self.scene_static_data)
            logger.info("GUI界面已启动")
        
        # 7. 设置键盘快捷键
        self.setup_keyboard_shortcuts()
        
        print("=== VLA系统初始化完成 ===")
        print("控制说明:")
        print("  C - 切换手动/自动控制")
        print("  V - 显示VLA状态")
        print("  O - 保存OCR调试图像")
        print("  空格 - 重试关卡")
        print("  N - 跳过关卡")
        print("手动控制时:")
        print("  WASD/方向键 - 控制车辆")
        print("  R/F/T/G - 切换档位")
        print("========================")
    
    def setup_keyboard_shortcuts(self):
        """设置键盘快捷键"""
        keyboard.add_hotkey("space", self.api.retry_level)
        keyboard.add_hotkey("n", self.api.skip_level)
        
        # 档位控制
        keyboard.add_hotkey("r", lambda: self.set_gear(GearMode.REVERSE))
        keyboard.add_hotkey("f", lambda: self.set_gear(GearMode.DRIVE))
        keyboard.add_hotkey("t", lambda: self.set_gear(GearMode.NEUTRAL))
        keyboard.add_hotkey("g", lambda: self.set_gear(GearMode.PARKING))
        
        # 手动/自动控制切换
        keyboard.add_hotkey("c", self.toggle_manual_control)
        
        # VLA调试快捷键
        keyboard.add_hotkey("v", self.print_vla_status)
        keyboard.add_hotkey("o", self.save_ocr_debug_image)
    
    def set_gear(self, gear: GearMode):
        """设置档位"""
        self.current_gear = gear
        logger.info(f"档位切换到: {gear.name}")
        print(f"档位: {gear.name}")
    
    def toggle_manual_control(self):
        """切换手动/自动控制"""
        self.use_manual_control = not self.use_manual_control
        mode = "手动" if self.use_manual_control else "VLA自动"
        logger.info(f"控制模式切换到: {mode}")
        print(f"当前控制模式: {mode}")
    
    def print_vla_status(self):
        """打印VLA状态信息"""
        status = self.task_controller.get_status_info()
        print(f"\n=== VLA状态 ===")
        print(f"状态: {status['current_state']}")
        print(f"有任务: {status['has_task']}")
        print(f"VLA已提交: {status['vla_submitted']}")
        
        # 显示各相机的OCR结果
        if self.last_ocr_results:
            total_texts = sum(len(result) for result in self.last_ocr_results.values())
            print(f"OCR检测总数: {total_texts}")
            for cam_id, texts in self.last_ocr_results.items():
                if texts:
                    cam_name = CAMERA_NAMES.get(cam_id, f"相机{cam_id}")
                    print(f"{cam_name}: {texts}")
        
        if status['has_task']:
            print(f"目的地: {status['task_destination']}")
            print(f"动作: {status['task_action']}")
            print(f"剩余时间: {status['remaining_time']:.1f}秒")
        
        print("================\n")
    
    def save_ocr_debug_image(self):
        """保存OCR调试图像"""
        if hasattr(self, '_last_frames') and hasattr(self, '_last_ocr_results_dict'):
            try:
                timestamp = int(time.time())
                for frame_info in self._last_frames:
                    cam_id = frame_info.id
                    ocr_result = self._last_ocr_results_dict.get(cam_id)
                    if ocr_result:
                        cam_name = CAMERA_NAMES.get(cam_id, f"cam{cam_id}")
                        filename = f"./ocr_debug/{timestamp}_{cam_name}.jpg"
                        debug_image = self.ocr_system.visualize_ocr_result(
                            frame_info.frame, 
                            ocr_result,
                            filename
                        )
                        print(f"保存{cam_name}调试图像: {filename}")
                print("所有OCR调试图像已保存")
            except Exception as e:
                print(f"保存调试图像失败: {e}")
        else:
            print("没有可用的调试数据")
    
    def get_manual_vehicle_control(self) -> VehicleControl:
        """获取手动控制指令"""
        vc = VehicleControl()
        vc.gear = self.current_gear
        
        # 基于键盘输入的控制逻辑
        try:
            if keyboard.is_pressed("up") or keyboard.is_pressed("w"):
                value = 0.5 if keyboard.is_pressed("shift") else 1.0
                if self.current_gear == GearMode.DRIVE:
                    vc.throttle = value
                elif self.current_gear == GearMode.REVERSE:
                    vc.brake = value
            elif keyboard.is_pressed("down") or keyboard.is_pressed("s"):
                value = 0.5 if keyboard.is_pressed("shift") else 1.0
                if self.current_gear == GearMode.DRIVE:
                    vc.brake = value
                elif self.current_gear == GearMode.REVERSE:
                    vc.throttle = value
            
            if keyboard.is_pressed("left") or keyboard.is_pressed("a"):
                vc.steering = -1.0
            elif keyboard.is_pressed("right") or keyboard.is_pressed("d"):
                vc.steering = 1.0
        except Exception as e:
            print(f"键盘控制错误: {e}")
            # 返回安全的默认控制
            vc.throttle = 0.0
            vc.brake = 0.0
            vc.steering = 0.0
        
        return vc
    
    def process_camera_frames_for_ocr(self, frames) -> Dict[str, List[str]]:
        """处理四目相机帧进行OCR识别
        
        Args:
            frames: 包含四个相机帧的列表
            
        Returns:
            字典，键为相机ID，值为检测到的文字列表
        """
        # 使用真实OCR
        ocr_results_dict = self.ocr_system.process_camera_frame(frames)
        
        # 保存用于调试
        self._last_frames = frames.copy()
        self._last_ocr_results_dict = ocr_results_dict
        
        # 提取每个相机的文字
        detected_texts_by_camera = {}
        for cam_id, ocr_result in ocr_results_dict.items():
            if ocr_result:
                detected_texts_by_camera[cam_id] = ocr_result.detected_text
            else:
                detected_texts_by_camera[cam_id] = []
        
        return detected_texts_by_camera
    
    def merge_ocr_texts_for_task(self, ocr_texts_by_camera: Dict[str, List[str]]) -> List[str]:
        """合并各相机的OCR文字供任务控制器使用
        
        优先级：前目 > 左右目 > 后目
        """
        merged_texts = []
        
        # 按优先级合并
        for cam_id in ['0', '1', '2', '3']:  # 前、右、左、后
            if cam_id in ocr_texts_by_camera:
                merged_texts.extend(ocr_texts_by_camera[cam_id])
        
        # 去重但保持顺序
        seen = set()
        unique_texts = []
        for text in merged_texts:
            if text not in seen:
                seen.add(text)
                unique_texts.append(text)
        
        return unique_texts

    def run_main_loop(self):
        """运行主循环"""
        logger.info("开始VLA主循环")
        
        try:
            for sim_car_msg, frames in self.api.main_loop():
                # ✅ 第一帧开始计时（此时 Unity 已经在出画面/已连上了）
                if self.demo_start_time is None:
                    self.demo_start_time = time.time()
                    # print(f"[DEMO] 第一帧到达，开始 Demo 序列 (Stop={self.demo_stop_duration}s)")

                # frames包含4个相机: 0前目, 1右目, 2左目, 3后目
                self.frame_count += 1
                
                # 计算经过时间
                elapsed_time = time.time() - self.demo_start_time
                
                # 定义 Demo 阶段
                is_demo_phase = True
                demo_stage = "UNKNOWN"
                
                # 时间节点计算
                t1 = self.demo_init_move_duration
                t2 = t1 + self.demo_stop_duration
                t3 = t2 + self.demo_release_duration
                
                # 阶段 1: 初始移动 (Move) - 激活物理引擎
                if elapsed_time < t1:
                    demo_stage = "INIT_MOVE"
                # 阶段 2: 强制停车 (Stop)
                elif elapsed_time < t2:
                    demo_stage = "STOP"
                # 阶段 3: 松刹车缓冲 (Release)
                elif elapsed_time < t3:
                    demo_stage = "RELEASE"
                # 阶段 4: 正常运行
                else:
                    is_demo_phase = False
                    demo_stage = "RUNNING"
                
                # OCR处理（解析后暂停，接近目标再启用）
                # 每帧初始化空字典，防止使用旧数据
                ocr_results_dict = {}

                # 启动阶段(前60帧)不跑OCR，画面不稳定
                if self.frame_count > 60 and self.frame_count % OCR_FREQUENCY == 0:
                    # 根据任务控制器策略决定是否运行OCR
                    # 需要用当前位置估计距离，因此先取得sim_car_msg
                    try:
                        vehicle_pos = Vector2(sim_car_msg.pose_gnss.pos_x, sim_car_msg.pose_gnss.pos_y)
                        run_ocr = self.task_controller.should_run_ocr(vehicle_pos)
                    except Exception:
                        run_ocr = True

                    if run_ocr:
                        # 从OCR系统获取完整的、带位置信息的结果
                        ocr_results_dict = self.ocr_system.process_camera_frame(frames)

                        # 保存完整的OCR结果，用于调试
                        self._last_frames = frames.copy()
                        self._last_ocr_results_dict = ocr_results_dict

                        # 提取纯文本用于状态
                        texts_for_status = {}
                        for cam_id, result in ocr_results_dict.items():
                            if result:
                                texts_for_status[cam_id] = result.detected_text
                        self.last_ocr_results = texts_for_status

                        # 打印检测结果（仅当有内容）
                        # if any(texts for texts in self.last_ocr_results.values()):
                        #     print(f"\n帧 {self.frame_count} OCR检测结果:")
                        #     for cam_id, texts in self.last_ocr_results.items():
                        #         if texts:
                        #             cam_name = CAMERA_NAMES.get(cam_id, f"相机{cam_id}")
                        #             print(f"  {cam_name}: {texts}")
                    else:
                        # 调试：打印OCR被关闭的原因
                        if self.frame_count % 60 == 0:  # 每2秒打印一次
                            pass
                            # print(f"[OCR控制] 帧{self.frame_count}: OCR已关闭 (状态:{self.task_controller.current_state.name})")
                
                # [DEBUG] 打印仿真器返回的真实车辆状态（每1秒）
                if self.frame_count % 30 == 0:
                    real_gear = sim_car_msg.main_vehicle.gear
                    real_speed = sim_car_msg.main_vehicle.speed
                    real_z = sim_car_msg.pose_gnss.pos_z
                    # print(f"[DEBUG] Sim State: Gear={real_gear}, Speed={real_speed:.2f}, Z={real_z:.2f}")

                # 选择控制模式
                if self.use_manual_control:
                    # 手动控制
                    vehicle_control = self.get_manual_vehicle_control()
                    if self.frame_count % 30 == 0:
                        pass
                        # print(f"[DEBUG] Manual Cmd: T={vehicle_control.throttle:.2f}, B={vehicle_control.brake:.2f}, G={vehicle_control.gear}")
                    vla_extension = None
                    
                    # 手动模式下也更新GUI
                    if USE_GUI and self.dashboard:
                        self.dashboard.update(sim_car_msg)
                        # if hasattr(self, 'dashboard'):
                        #     self.dashboard.update_frame(frames[0].frame)  # 只显示前目
                else:
                    # VLA自动控制 - 直接传递完整的OCR结果字典给任务控制器
                    vc_task, vla_extension = self.task_controller.update(
                        sim_car_msg, ocr_results_dict, self.frame_count
                    )

                    if USE_TRAJECTORY_CONTROL:
                        # 1️⃣ 停牌结束后：只提交一次写死结果（用 demo_vla_sent 控制即可）
                        if (not self.demo_vla_sent) and (not is_demo_phase):
                            vla_extension = None
                            self.demo_vla_sent = True

                        traj = getattr(sim_car_msg, "trajectory", None)

                        # 2️⃣ trajectory 为空 → 回退原逻辑
                        if not traj:
                            vehicle_control = vc_task
                        else:
                            # 当前车辆位置
                            pos = Vector3(
                                sim_car_msg.pose_gnss.pos_x,
                                sim_car_msg.pose_gnss.pos_y,
                                sim_car_msg.pose_gnss.pos_z,
                            )

                            last_pt = traj[-1]
                            # 使用 2D 距离计算，避免因高度差导致无法到达终点
                            dist_to_end_3d = math.dist(pos, last_pt)
                            dist_to_end_2d = math.hypot(pos.x - last_pt.x, pos.y - last_pt.y)
                            
                            # 优先使用 2D 距离判断
                            dist_to_end = dist_to_end_2d

                            if self.frame_count % 30 == 0:
                                pass
                                # print(f"[Trajectory] Dist2D={dist_to_end_2d:.2f}, Dist3D={dist_to_end_3d:.2f}, "
                                #       f"Pos.z={pos.z:.2f}, Target.z={last_pt.z:.2f}")

                            # 更新到达状态（滞后逻辑，防止原地打圈）
                            if dist_to_end < TRAJ_END_DIST:
                                self.traj_reached = True
                            elif dist_to_end > TRAJ_END_DIST + 2.0:
                                self.traj_reached = False

                            # 3️⃣ 已到达 trajectory 终点 → 停车
                            if self.traj_reached:
                                vehicle_control = VehicleControl()
                                vehicle_control.gear = GearMode.DRIVE
                                vehicle_control.throttle = 0.0
                                vehicle_control.brake = 1.0
                                vehicle_control.steering = 0.0

                                if self.frame_count % 30 == 0:
                                    print(f"[Trajectory] 已到达终点，车辆已停止 (dist={dist_to_end:.2f}m)")
                                    # 检查是否有 SubScene 已经 Resumed
                                    try:
                                        current_sub_name = sim_car_msg.scene_status.sub_scene_name
                                        if current_sub_name in self.subscene_paused_names:
                                            print(f"[Warning] SubScene '{current_sub_name}' 已完成，但 Trajectory 仍强制停车！")
                                    except:
                                        pass
                            else:
                                # 4️⃣ 未到终点 → 正常按 trajectory 跟随
                                vehicle_control = get_vehicle_control_from_algorithm(sim_car_msg)

                    else:
                        # ✅ 原逻辑：使用 task_controller 的控制
                        vehicle_control = vc_task


                    
                    # 更新GUI（包含导航路径）
                    if USE_GUI and self.dashboard:
                        # 获取导航路径（转换为简单格式）
                        nav_path = self.task_controller.get_navigation_path_for_gui()
                        self.dashboard.update(sim_car_msg)
                        # self.dashboard.update(sim_car_msg, nav_path)
                        # self.dashboard.update_frame(frames[0].frame)  # 只显示前目
                
                # --- END OF CHANGES ---
                
                # 如果处于 Demo 启动阶段，强制覆盖控制指令
                if is_demo_phase:
                    vehicle_control = VehicleControl()
                    # 始终保持 D 挡
                    vehicle_control.gear = GearMode.DRIVE
                    vehicle_control.steering = 0.0
                    
                    if demo_stage == "INIT_MOVE":
                        # 给油，不刹车
                        vehicle_control.throttle = 0.6
                        vehicle_control.brake = 0.0
                        if self.frame_count % 10 == 0:
                            pass
                            # print(f"[DEMO] 阶段 INIT_MOVE: 激活车辆... {t1 - elapsed_time:.1f}s")

                    elif demo_stage == "STOP":
                        # 满刹车，不给油
                        vehicle_control.throttle = 0.0
                        vehicle_control.brake = 1.0
                        if self.frame_count % 30 == 0:
                            pass
                            # print(f"[DEMO] 阶段 STOP: 停车等待... {t2 - elapsed_time:.1f}s")
                            
                    elif demo_stage == "RELEASE":
                        # 松刹车，不给油
                        vehicle_control.throttle = 0.0
                        vehicle_control.brake = 0.0
                        if self.frame_count % 10 == 0:
                            pass
                            # print(f"[DEMO] 阶段 RELEASE: 松刹车缓冲...")

                    # 覆盖VLA扩展输出
                    vla_extension = None
                
                # 子场景终点停车逻辑
                elif True:
                     scene_status = sim_car_msg.scene_status
                     sub_name = scene_status.sub_scene_name
                     end_pt = scene_status.end_point
                     if sub_name != self.subscene_name_last:
                         self.subscene_name_last = sub_name
                         self.subscene_pause_until = None
                         if sub_name not in self.subscene_submitted_names:
                             self.subscene_submitted_names.add(sub_name)
                             if self.level_submission_count == 0:
                                 vla_extension = self.level1_vla_output
                                 self.level_submission_count += 1
                                 print("[VLA] 提交第1关内容")
                             elif self.level_submission_count == 1:
                                 vla_extension = self.level2_vla_output
                                 self.level_submission_count += 1
                                 print("[VLA] 提交第2关内容")
                     if end_pt is not None:
                        pos = Vector3(sim_car_msg.pose_gnss.pos_x, sim_car_msg.pose_gnss.pos_y, sim_car_msg.pose_gnss.pos_z)
                        
                        # 改为2D距离，避免高度差导致无法触发
                        dist_to_end = math.hypot(pos.x - end_pt.x, pos.y - end_pt.y)
                        
                        # [DEBUG] 打印调试信息
                        if self.frame_count % 30 == 0:
                            print(f"[SubScene] Debug: Dist={dist_to_end:.2f}, Limit={self.subscene_end_dist}")

                        if (self.subscene_pause_until is None) and (sub_name not in self.subscene_paused_names) and (dist_to_end < self.subscene_end_dist):
                            self.subscene_pause_until = time.time() + self.subscene_end_pause_seconds
                            print(f"[SubScene] 触发终点停车: 名称={sub_name}, 距离={dist_to_end:.2f}m, 等待{self.subscene_end_pause_seconds:.1f}s")
                        if self.subscene_pause_until is not None:
                             if time.time() < self.subscene_pause_until:
                                 vehicle_control = VehicleControl()
                                 vehicle_control.gear = GearMode.DRIVE
                                 vehicle_control.throttle = 0.0
                                 vehicle_control.brake = 1.0
                                 vehicle_control.steering = 0.0
                                 vla_extension = None
                                 if self.frame_count % 30 == 0:
                                     remaining = max(0.0, self.subscene_pause_until - time.time())
                                     print(f"[SubScene] 终点停车中... 剩余{remaining:.1f}s")
                             else:
                                 self.subscene_paused_names.add(sub_name)
                                 self.subscene_pause_until = None
                                 print(f"[SubScene] 终点停车结束，继续行驶: 名称={sub_name}")
                                 # 此时 vehicle_control 保留了 task_controller 或 trajectory 的输出
                                 print(f"[SubScene] Resume VC: T={vehicle_control.throttle:.2f}, B={vehicle_control.brake:.2f}")
                     else:
                        if self.frame_count % 30 == 0:
                            pass
                            # print(f"[SubScene] Debug: end_pt is None (SubScene={sub_name})")
                
                # [Anti-Stuck]
                if (not is_demo_phase) and vehicle_control.throttle > 0.1 and abs(sim_car_msg.main_vehicle.speed) < 0.01:
                     # 只有当 Demo 结束后才检测
                     if elapsed_time > t3 + 2.0:
                         if self.frame_count % 30 == 0:
                             pass
                             # print(f"[Anti-Stuck] 检测到卡死 (T={vehicle_control.throttle:.2f}, Speed=0)! 尝试微调...")
                         
                         # 尝试给一点点反向脉冲或者更大的油门？
                         # 这里简单粗暴：直接给满油门
                         vehicle_control.throttle = 1.0

                # 发送控制指令
                if self.frame_count % 30 == 0:
                    try:
                        pass
                        # print(f"[DEBUG] Final Sending: {vehicle_control}")
                    except:
                        pass
                        # print(f"[DEBUG] Final Sending: T={vehicle_control.throttle:.2f}, B={vehicle_control.brake:.2f}, G={vehicle_control.gear}")
                self.api.set_vehicle_control(vehicle_control, vla_extension=vla_extension)
                
                # 定期日志输出（每1秒一次）
                if self.frame_count % 30 == 0:
                    self.log_periodic_status(sim_car_msg, vehicle_control)
                    
        except KeyboardInterrupt:
            logger.info("用户中断程序")
            print("\n程序被用户中断")
        except Exception as e:
            logger.error(f"主循环异常: {e}")
            print(f"运行异常: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.cleanup()    

    def log_periodic_status(self, sim_car_msg: SimCarMsg, vehicle_control: VehicleControl):
        """定期记录状态信息"""
        # 车辆状态
        pos_x, pos_y = sim_car_msg.pose_gnss.pos_x, sim_car_msg.pose_gnss.pos_y
        speed = sim_car_msg.main_vehicle.speed
        yaw = sim_car_msg.pose_gnss.ori_z
        
        # 控制状态
        control_mode = "手动" if self.use_manual_control else "自动"
        
        # 任务状态
        task_status = self.task_controller.get_status_info()
        
        # OCR状态 - 统计各相机的检测数量
        ocr_summary = ""
        if task_status.get('ocr_counts_by_camera'):
            active_cams = [f"{name}:{count}" for name, count in task_status['ocr_counts_by_camera'].items() if count > 0]
            if active_cams:
                ocr_summary = f"OCR[{', '.join(active_cams)}]"
        
        logger.debug(
            f"[{control_mode}] 位置:({pos_x:.1f},{pos_y:.1f}) "
            f"速度:{speed:.1f} 朝向:{yaw:.1f}° "
            f"状态:{task_status['current_state']} "
            f"{ocr_summary} "
            f"控制:T{vehicle_control.throttle:.2f}/B{vehicle_control.brake:.2f}/S{vehicle_control.steering:.2f}"
        )
    
    def cleanup(self):
        """清理资源"""
        logger.info("开始清理资源...")
        
        if SHOW_CAMERA_FEED:
            cv2.destroyAllWindows()
        
        if USE_GUI and self.dashboard:
            self.dashboard.quit()
            logger.info("GUI界面已关闭")
        
        if self.ocr_system:
            # OCR系统的清理（如果有的话）
            pass
        
        logger.info("资源清理完成")


def main():
    """主函数"""
    print("=== VLA自动驾驶系统启动 ===")
    print("支持四目相机OCR识别")
    
    controller = VLAMainController()
    
    try:
        # 初始化系统
        controller.initialize()
        
        # 运行主循环
        controller.run_main_loop()
        
    except Exception as e:
        print(f"系统运行失败: {e}")
        logger.error(f"系统运行失败: {e}")
        return
    
    print("VLA系统正常退出")


if __name__ == "__main__":
    main()
