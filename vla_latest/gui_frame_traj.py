# 在 gui.py 中修改，解决GUI卡住问题

"""
使用 tkinter 和多进程实现的车辆信息显示界面。
之前也尝试过使用多线程，但是 tkinter 的组件只能在创建它的线程中使用，所以只能使用多进程。
"""

from tkinter import *
from tkinter import ttk
from multiprocessing import Process, Pipe, Queue
from multiprocessing.connection import Connection
import time
from enum import Enum, auto
import math
from dataclasses import dataclass
import numpy as np
from scipy.spatial import KDTree
from PIL import Image, ImageTk
import cv2
from typing import List
import io
from geometry import Vector2, Vector3, euler_to_vector2, yaw_to_radians
from metacar import (
    Vector2,
    SceneStaticData,
    SimCarMsg,
    LineType,
)


@dataclass
class LineInfo:
    """绘制用的车道线信息"""

    type: LineType
    points: list[Vector2]


# 在 MsgType 枚举中添加新类型
class MsgType(Enum):
    UPDATE = auto()
    QUIT = auto()

# 在 tk_process_func 的 update 函数后添加 update_path 函数
def update_path(path_points: List[Vector3]):
    """更新导航路径显示"""
    global navigation_path_points
    navigation_path_points = path_points


def tk_process_func(
    conn: Connection, frame_queue: Queue, scene_static_data: SceneStaticData, refresh_interval: int
):
    navigation_path_points = []
    ROADLINE_COLOR_MAP = {
        LineType.MIDDLE_LINE: "dark orange",  # 中线
        LineType.SIDE_LINE: "black",  # 边线
        LineType.SOLID_LINE: "black",  # 实线
        LineType.STOP_LINE: "dark red",  # 停车线
        LineType.ZEBRA_CROSSING: "dark blue",  # 斑马线
        LineType.DASH_LINE: "grey",  # 虚线
    }

    def update(msg: SimCarMsg, nav_path: List[tuple[float, float]] = None):
        string_vars[0].set("{:.3f}".format(msg.main_vehicle.throttle))
        string_vars[1].set("{:.3f}".format(msg.main_vehicle.brake))
        string_vars[2].set("{:.3f}".format(msg.main_vehicle.steering))
        string_vars[3].set(msg.main_vehicle.gear.name)
        string_vars[4].set("{:.3f}".format(msg.main_vehicle.speed))
        string_vars[5].set("{:.2f}".format(msg.pose_gnss.pos_x))
        string_vars[6].set("{:.2f}".format(msg.pose_gnss.pos_y))
        string_vars[7].set("{:.2f}".format(msg.pose_gnss.ori_z))
        # clear canvas
        map_canvas.delete("all")
        canvas_width = map_canvas.winfo_width()
        canvas_height = map_canvas.winfo_height()
        main_vehicle_pos = Vector2(msg.pose_gnss.pos_x, msg.pose_gnss.pos_y)

        SCALE = 10

        def convert_pos(pos: Vector2) -> tuple[float, float]:
            """将场景中的坐标转换为画布上的坐标。"""
            vec = (pos - main_vehicle_pos) * SCALE
            return (canvas_width / 2 + vec.x, canvas_height / 2 - vec.y)

        def draw_rectangle(
            center_pos: Vector2,
            length: float,
            width: float,
            ori_z: float,
            outline_color: str,
        ):
            """绘制带有旋转角度的矩形。"""
            rotate_rad = -math.radians(ori_z)
            a = Vector2(length / 2, width / 2)
            b = Vector2(length / 2, -width / 2)
            c = Vector2(-length / 2, -width / 2)
            d = Vector2(-length / 2, width / 2)
            a, b, c, d = map(
                lambda v: convert_pos(v.rotate_rad(rotate_rad) + center_pos),
                (a, b, c, d),
            )
            # 绘制无填充的多边形
            map_canvas.create_polygon(a, b, c, d, fill="", outline=outline_color)

        # 绘制主车
        draw_rectangle(
            main_vehicle_pos,
            msg.main_vehicle.length,
            msg.main_vehicle.width,
            msg.pose_gnss.ori_z,
            "green",
        )

        # 绘制障碍物
        for obstacle in msg.obstacles:
            draw_rectangle(
                Vector2(obstacle.pos_x, obstacle.pos_y),
                obstacle.length,
                obstacle.width,
                obstacle.ori_z,
                "red",
            )

        # 绘制行驶路线
        trajectory_points = [convert_pos(pt.to_vector2()) for pt in msg.trajectory]
        if len(trajectory_points) > 1:
            map_canvas.create_line(trajectory_points, fill="blue")

        # 在绘制行驶路线之后，绘制导航路径
        trajectory_points = [convert_pos(pt.to_vector2()) for pt in msg.trajectory]
        if len(trajectory_points) > 1:
            map_canvas.create_line(trajectory_points, fill="blue")
        
        # 绘制导航路径（如果有）
        if nav_path and len(nav_path) > 1:
            nav_points = [convert_pos(Vector2(x, y)) for x, y in nav_path]
            # 绘制路径线（用绿色虚线表示规划路径）
            map_canvas.create_line(nav_points, fill="lime green", width=3, dash=(5, 3))
            # 绘制路径点
            for i, point in enumerate(nav_points):
                if i == 0:
                    # 起点用绿色圆圈
                    map_canvas.create_oval(point[0]-5, point[1]-5, point[0]+5, point[1]+5, 
                                        fill="green", outline="dark green")
                elif i == len(nav_points) - 1:
                    # 终点用红色方块
                    map_canvas.create_rectangle(point[0]-5, point[1]-5, point[0]+5, point[1]+5,
                                            fill="red", outline="dark red")
                else:
                    # 中间点用小黄点
                    map_canvas.create_oval(point[0]-3, point[1]-3, point[0]+3, point[1]+3,
                                        fill="yellow", outline="orange")

        # 绘制车道线
        radius = math.hypot(canvas_width, canvas_height) / 2 / SCALE
        min_idx = [-1] * len(line_list)
        max_idx = [-1] * len(line_list)
        point_idxs = kdtree.query_ball_point(
            (main_vehicle_pos.x, main_vehicle_pos.y), radius
        )
        for point_idx in point_idxs:
            rid, pid = point_info[point_idx]
            if min_idx[rid] == -1:
                min_idx[rid] = pid
                max_idx[rid] = pid
            else:
                min_idx[rid] = min(min_idx[rid], pid)
                max_idx[rid] = max(max_idx[rid], pid)
        for line_idx, line in enumerate(line_list):
            cur_min_idx = min_idx[line_idx]
            cur_max_idx = max_idx[line_idx]
            if cur_min_idx == -1:
                continue
            if line.type == LineType.ZEBRA_CROSSING:
                points = [convert_pos(Vector2(pt.x, pt.y)) for pt in line.points]
                map_canvas.create_polygon(
                    points, fill="", outline=ROADLINE_COLOR_MAP[line.type]
                )
                continue
            if cur_min_idx > 0:
                cur_min_idx -= 1
            if cur_max_idx < len(line.points) - 1:
                cur_max_idx += 1
            points = [
                convert_pos(Vector2(pt.x, pt.y))
                for pt in line.points[cur_min_idx : cur_max_idx + 1]
            ]
            map_canvas.create_line(points, fill=ROADLINE_COLOR_MAP[line.type])

        # 判断是否是 VLA 场景
        if scene_static_data.vla_extension:
            # 绘制建筑物
            for building in scene_static_data.vla_extension.buildings:
                # 绘制建筑物边框
                pos = Vector2(building.pos_x, building.pos_y)
                draw_rectangle(
                    pos,
                    building.length,
                    building.width,
                    building.ori_z,
                    "blue",
                )
                # 在建筑物中心显示名称
                center_pos = convert_pos(pos)
                map_canvas.create_text(
                    center_pos[0],
                    center_pos[1],
                    text=building.name,
                    fill="blue",
                )
    
    def update_camera_frame():
        """从队列中获取并更新相机画面"""
        try:
            # 非阻塞地从队列获取最新的帧
            frame_bytes = None
            while not frame_queue.empty():
                frame_bytes = frame_queue.get_nowait()
            
            if frame_bytes is not None:
                # 从字节流恢复图像
                image = Image.open(io.BytesIO(frame_bytes))
                photo = ImageTk.PhotoImage(image)
                
                # 更新Canvas
                camera_canvas.delete("all")
                camera_canvas.create_image(0, 0, anchor=NW, image=photo)
                camera_canvas.image = photo  # 保持引用
        except:
            pass  # 忽略错误，继续运行

    def check_message():
        """检查管道，如果有消息则处理。"""
        start_time = time.time()

        if conn.poll():
            message = conn.recv()
            match message["type"]:
                case MsgType.UPDATE:
                    # 提取导航路径（如果有）
                    nav_path = message.get("nav_path", None)
                    update(message["msg"], nav_path)
                case MsgType.QUIT:
                    root.destroy()
        
        # 更新相机画面
        update_camera_frame()

        # 计算下一次调用 check_message 的时间
        elapsed_time = time.time() - start_time
        next_interval = max(0, refresh_interval - int(elapsed_time * 1000))
        root.after(next_interval, check_message)

    # 先把所有需要绘制的车道线提取出来
    line_list: list[LineInfo] = []
    for road in scene_static_data.roads:
        if road.stop_line:
            line_list.append(LineInfo(LineType.STOP_LINE, road.stop_line))
        for lane in road.lanes:
            line_list.append(
                LineInfo(lane.left_border.type, lane.left_border.path_points)
            )
            line_list.append(
                LineInfo(lane.right_border.type, lane.right_border.path_points)
            )

    # 建 kdtree
    tmp_line_points: list[tuple[float, float]] = []
    point_info: list[tuple[int, int]] = []
    for line_idx, line in enumerate(line_list):
        for point_idx, point in enumerate(line.points):
            tmp_line_points.append((point.x, point.y))
            point_info.append((line_idx, point_idx))
    # 将 tmp_points 转换为 numpy 数组，并 reshape 为 (N, 2) 的形状，避免数组为空时建树失败
    line_points = np.array(tmp_line_points, dtype=float).reshape(-1, 2)
    del tmp_line_points
    kdtree = KDTree(line_points)

    root = Tk()
    root.title("仪表盘")

    mainframe = ttk.Frame(root, padding="4 4 4 4")
    mainframe.grid(column=0, row=0, sticky="NSEW")
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)
    # 设置数值列的最小宽度，使得数值列的宽度不会随着数值的变化而变化
    mainframe.grid_columnconfigure(1, minsize=60)

    # 仪表盘的各个组件
    labels = ["油门", "刹车", "方向盘", "档位", "速度", "x坐标", "y坐标", "航向角"]
    string_vars = [StringVar() for _ in range(len(labels))]

    for row, (label_str, string_var) in enumerate(zip(labels, string_vars)):
        ttk.Label(mainframe, text=label_str).grid(column=0, row=row, sticky=E)
        ttk.Label(mainframe, textvariable=string_var).grid(column=1, row=row, sticky=W)

    # 在这里设置画布的默认大小
    map_canvas = Canvas(
        mainframe,
        width=480,
        height=480,
        borderwidth=2,
        relief="groove",
        background="white",
    )
    map_canvas.grid(column=2, row=0, rowspan=len(labels), sticky="NSEW")
    
    # 新增：相机画面显示（左下角）
    camera_frame = ttk.LabelFrame(mainframe, text="前视相机", padding="2 2 2 2")
    camera_frame.grid(column=0, row=len(labels), columnspan=2, sticky="NSEW", pady=(5, 0))
    
    camera_canvas = Canvas(
        camera_frame,
        width=320,
        height=240,
        borderwidth=1,
        relief="sunken",
        background="black",
    )
    camera_canvas.pack()

    # 窗口大小改变时，将空间分配给画布
    mainframe.grid_columnconfigure(2, weight=1)
    for row in range(len(labels)):
        mainframe.grid_rowconfigure(row, weight=1)

    # 将仪表盘窗口置顶
    root.wm_attributes("-topmost", True)

    root.after_idle(check_message)
    conn.send("ready")
    root.mainloop()


class Dashboard:
    """
    显示车辆信息的仪表盘。
    使用多进程实现，使用 multiprocessing.Pipe 作为进程间通信的方式。
    """

    def __init__(self, scene_static_data: SceneStaticData, refresh_interval: int = 20):
        """
        初始化仪表盘。
        :param scene_static_data: 场景静态数据，用于绘制车道线。
        :param refresh_interval: 刷新间隔，单位毫秒，一定要比 update 方法的调用频率高。
        """
        self._conn, child_conn = Pipe()
        self._frame_queue = Queue(maxsize=2)  # 限制队列大小，避免内存溢出
        self._tk_process = Process(
            target=tk_process_func,
            args=(child_conn, self._frame_queue, scene_static_data, refresh_interval),
        )
        self._tk_process.start()
        # 等待仪表盘初始化完成（包括创建窗口、创建 kdtree 等）
        ready = self._conn.recv()
        if ready != "ready":
            raise RuntimeError("仪表盘初始化失败")
        
        self._frame_counter = 0

    def update(self, msg: SimCarMsg, nav_path: List[tuple[float, float]] = None):
        """更新仪表盘的方法，包含导航路径。"""
        if self._tk_process.is_alive():
            message = {
                "type": MsgType.UPDATE, 
                "msg": msg
            }
            if nav_path:
                message["nav_path"] = nav_path
            self._conn.send(message)

    def update_path(self, path: List[Vector3]):
        """更新导航路径的方法。"""
        if self._tk_process.is_alive():
            self._conn.send({"type": MsgType.UPDATE_PATH, "path": path})
    
    def update_frame(self, frame: np.ndarray):
        """更新相机画面的方法。"""
        if self._tk_process.is_alive():
            # 降低帧率，每10帧更新一次
            self._frame_counter += 1
            if self._frame_counter % 5 == 0:
                try:
                    # 缩放图像
                    height, width = frame.shape[:2]
                    target_width = 320
                    target_height = int(height * target_width / width)
                    frame_resized = cv2.resize(frame, (target_width, target_height))
                    
                    # BGR转RGB
                    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                    
                    # 转为PIL Image并压缩为JPEG字节流
                    image = Image.fromarray(frame_rgb)
                    buffer = io.BytesIO()
                    image.save(buffer, format='JPEG', quality=70)
                    frame_bytes = buffer.getvalue()
                    
                    # 清空队列中的旧帧
                    while not self._frame_queue.empty():
                        try:
                            self._frame_queue.get_nowait()
                        except:
                            break
                    
                    # 非阻塞地放入新帧
                    self._frame_queue.put_nowait(frame_bytes)
                except:
                    pass  # 忽略错误

    def quit(self):
        """退出仪表盘的方法。"""
        if self._tk_process.is_alive():
            self._conn.send({"type": MsgType.QUIT})
        self._tk_process.join()

# 在 main.py 中的调用保持不变：
# if USE_GUI and self.dashboard:
#     self.dashboard.update(sim_car_msg)
#     self.dashboard.update_frame(frame)