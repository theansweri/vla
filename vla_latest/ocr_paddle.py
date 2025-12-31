"""
VLA OCR和相机集成模块 - 基于PytorchPaddleOCR
使用正确的API调用方式获取单目前视相机画面
模型一次加载，支持实时推理
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
import time
from dataclasses import dataclass
import math
from PIL import Image, ImageDraw, ImageFont
import os
from geometry import Vector2, Vector3, euler_to_vector2, yaw_to_radians
import sys
base_dir = os.path.dirname(os.path.abspath(__file__))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

# PytorchPaddleOCR引擎导入
try:
    from pytorchocr.pytorch_paddle import PytorchPaddleOCR
    PYTORCH_PADDLE_OCR_AVAILABLE = True
except ImportError:
    PYTORCH_PADDLE_OCR_AVAILABLE = False


@dataclass
class TextDetection:
    """文字检测结果"""
    text: str
    confidence: float
    bbox: List[List[float]]  # 边界框坐标
    center: Tuple[float, float]  # 文字中心坐标


@dataclass
class CameraOCRResult:
    """相机OCR结果"""
    camera_id: str  # 相机标识
    detected_text: List[str]
    text_detections: List[TextDetection]  # 详细检测结果
    frame_timestamp: float
    frame_shape: Tuple[int, int, int]  # (height, width, channels)


class OCREngine:
    """OCR引擎基类"""
    
    def detect_text(self, image: np.ndarray) -> List[TextDetection]:
        """检测图像中的文字"""
        raise NotImplementedError


class PytorchPaddleOCREngine(OCREngine):
    """PytorchPaddleOCR引擎 - 高性能实时OCR"""
    
    def __init__(self, enable_line_merge: bool = True):
        if not PYTORCH_PADDLE_OCR_AVAILABLE:
            raise ImportError("PytorchPaddleOCR not available. Please check installation.")
        
        print("正在初始化PytorchPaddleOCR模型...")
        self.enable_line_merge = enable_line_merge
        try:
            # 初始化OCR模型，模型会保持在内存中
            self.ocr_model = PytorchPaddleOCR()
            print("PytorchPaddleOCR模型加载成功，准备实时推理...")
            print(f"相邻行合并功能: {'启用' if enable_line_merge else '禁用'}")
        except Exception as e:
            print(f"PytorchPaddleOCR初始化失败: {e}")
            raise RuntimeError(f"无法初始化PytorchPaddleOCR: {e}")
    
    def detect_text(self, image: np.ndarray) -> List[TextDetection]:
        """使用PytorchPaddleOCR检测文字"""
        try:
            # 直接调用模型进行推理
            dt_boxes, rec_res = self.ocr_model(image)
            
            detections = []
            
            # 检查是否有检测结果
            if dt_boxes is not None and rec_res is not None and len(dt_boxes) > 0:
                for box, (text, confidence) in zip(dt_boxes, rec_res):
                    # 转换box格式为列表
                    if hasattr(box, 'tolist'):
                        bbox = box.tolist()
                    else:
                        bbox = box
                    
                    # 确保bbox是正确的格式 [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                    if len(bbox) == 4 and len(bbox[0]) == 2:
                        # 计算中心点
                        bbox_array = np.array(bbox)
                        center_x = np.mean(bbox_array[:, 0])
                        center_y = np.mean(bbox_array[:, 1])
                        
                        # 转换confidence为float
                        if hasattr(confidence, 'item'):
                            conf_value = float(confidence.item())
                        else:
                            conf_value = float(confidence)
                        
                        detections.append(TextDetection(
                            text=str(text),
                            confidence=conf_value,
                            bbox=bbox,
                            center=(center_x, center_y)
                        ))
            
            # 按y坐标排序，保持从上到下的顺序
            detections = sorted(detections, key=lambda d: d.center[1])
            
            # 如果启用行合并，进行相邻行合并
            if self.enable_line_merge and len(detections) > 1:
                detections = self._merge_adjacent_lines(detections)
            
            return detections
            
        except Exception as e:
            print(f"PytorchPaddleOCR推理失败: {e}")
            return []
    
    def _merge_adjacent_lines(self, detections: List[TextDetection]) -> List[TextDetection]:
        """合并相邻行的文字，按从上到下顺序，无空格连接"""
        if len(detections) <= 1:
            return detections
        
        # 估计全局文本倾斜角并在校正坐标系中处理（提高歪斜文本的行判断稳定性）
        skew_angle = self._estimate_global_skew_angle(detections)
        
        def rot_center(det: TextDetection) -> Tuple[float, float]:
            x, y = det.center
            cos_t, sin_t = math.cos(-skew_angle), math.sin(-skew_angle)
            x_p = x * cos_t - y * sin_t
            y_p = x * sin_t + y * cos_t
            return x_p, y_p
        
        # 在旋转后的坐标系中按y'排序（从上到下）
        sorted_detections = sorted(detections, key=lambda d: rot_center(d)[1])
        
        merged = []
        current_group = [sorted_detections[0]]
        
        for i in range(1, len(sorted_detections)):
            prev_det = current_group[-1]  # 当前组的最后一个（最下面的）
            curr_det = sorted_detections[i]
            
            # 计算是否为相邻行（使用旋转坐标系进行判定）
            if self._is_adjacent_line(prev_det, curr_det, skew_angle):
                current_group.append(curr_det)
            else:
                # 处理当前组：如果有多个文字，合并它们
                if len(current_group) > 1:
                    merged_detection = self._merge_line_group(current_group, skew_angle)
                    merged.append(merged_detection)
                else:
                    merged.append(current_group[0])
                
                # 开始新的组
                current_group = [curr_det]
        
        # 处理最后一组
        if len(current_group) > 1:
            merged_detection = self._merge_line_group(current_group, skew_angle)
            merged.append(merged_detection)
        else:
            merged.append(current_group[0])
        
        return merged
    
    def _is_adjacent_line(self, det1: TextDetection, det2: TextDetection, skew_angle: float) -> bool:
        """判断两个检测结果是否为相邻行（在校正后的坐标系中）"""
        cos_t, sin_t = math.cos(-skew_angle), math.sin(-skew_angle)
        
        def rot_point(p: Tuple[float, float]) -> Tuple[float, float]:
            x, y = p
            return (x * cos_t - y * sin_t, x * sin_t + y * cos_t)
        
        # 使用短边长度作为文本高度的估计（比轴对齐高度稳定）
        height1 = self._estimate_box_height(det1.bbox)
        height2 = self._estimate_box_height(det2.bbox)
        avg_height = (height1 + height2) / 2.0
        
        # y'方向的距离
        c1x, c1y = rot_point(det1.center)
        c2x, c2y = rot_point(det2.center)
        y_distance = abs(c2y - c1y)
        
        # 计算在旋转坐标系下的水平投影区间并统计重叠
        def proj_interval_xp(bbox: List[List[float]]) -> Tuple[float, float]:
            xs = []
            for px, py in bbox:
                xp, yp = rot_point((px, py))
                xs.append(xp)
            return (min(xs), max(xs))
        
        l1, r1 = proj_interval_xp(det1.bbox)
        l2, r2 = proj_interval_xp(det2.bbox)
        overlap_left = max(l1, l2)
        overlap_right = min(r1, r2)
        overlap_width = max(0.0, overlap_right - overlap_left)
        width1 = r1 - l1
        width2 = r2 - l2
        min_width = max(min(width1, width2), 1e-6)
        overlap_ratio = overlap_width / min_width
        
        # 相邻行判定条件
        y_condition = y_distance < avg_height * 2.0
        x_gap = min(abs(l1 - r2), abs(l2 - r1))
        horizontal_condition = overlap_ratio > 0.2 or x_gap < avg_height
        
        return y_condition and horizontal_condition
    
    def _merge_line_group(self, detections: List[TextDetection], skew_angle: float) -> TextDetection:
        """合并一组相邻行的检测结果，按从上到下顺序，无空格连接（使用校正坐标排序）"""
        cos_t, sin_t = math.cos(-skew_angle), math.sin(-skew_angle)
        
        def rot_point(p: Tuple[float, float]) -> Tuple[float, float]:
            x, y = p
            return (x * cos_t - y * sin_t, x * sin_t + y * cos_t)
        
        def rot_center(det: TextDetection) -> Tuple[float, float]:
            return rot_point(det.center)
        
        # 确保按y'坐标排序（从上到下）
        sorted_dets = sorted(detections, key=lambda d: rot_center(d)[1])
        
        # 对于每一行，如果有多个文字，按x坐标排序（从左到右）
        lines = []
        current_line = [sorted_dets[0]]
        
        # 将检测结果分组到不同行
        for i in range(1, len(sorted_dets)):
            prev_det = current_line[-1]
            curr_det = sorted_dets[i]
            
            # 如果y'坐标很接近，认为是同一行
            _, py = rot_center(prev_det)
            _, cy = rot_center(curr_det)
            y_diff = abs(cy - py)
            avg_height = (self._estimate_box_height(prev_det.bbox) + self._estimate_box_height(curr_det.bbox)) / 2.0
            
            if y_diff < avg_height * 0.5:  # 同一行
                current_line.append(curr_det)
            else:  # 新的行
                # 对当前行按x坐标排序
                current_line = sorted(current_line, key=lambda d: rot_center(d)[0])
                lines.append(current_line)
                current_line = [curr_det]
        
        # 处理最后一行
        if current_line:
            current_line = sorted(current_line, key=lambda d: rot_center(d)[0])
            lines.append(current_line)
        
        # 合并所有行的文字，按从上到下，从左到右的顺序，无空格连接
        merged_text = ""
        for line in lines:
            line_text = "".join([det.text for det in line])
            merged_text += line_text
        
        # 计算加权平均置信度
        total_area = 0
        weighted_confidence = 0
        
        for det in sorted_dets:
            # 计算检测框面积作为权重
            bbox_array = np.array(det.bbox)
            area = cv2.contourArea(bbox_array.astype(np.float32))
            total_area += area
            weighted_confidence += det.confidence * area
        
        avg_confidence = weighted_confidence / total_area if total_area > 0 else np.mean([d.confidence for d in sorted_dets])
        
        # 合并边界框
        all_points = []
        for det in sorted_dets:
            all_points.extend(det.bbox)
        
        min_x = min(p[0] for p in all_points)
        max_x = max(p[0] for p in all_points)
        min_y = min(p[1] for p in all_points)
        max_y = max(p[1] for p in all_points)
        
        merged_bbox = [[min_x, min_y], [max_x, min_y], [max_x, max_y], [min_x, max_y]]
        
        return TextDetection(
            text=merged_text,
            confidence=avg_confidence,
            bbox=merged_bbox,
            center=((min_x + max_x) / 2, (min_y + max_y) / 2)
        )

    def _estimate_box_height(self, bbox: List[List[float]]) -> float:
        """估计文本框高度（取短边长度，更稳健）"""
        pts = np.array(bbox, dtype=np.float32)
        # 四条边长度
        e = [np.linalg.norm(pts[(i+1) % 4] - pts[i]) for i in range(4)]
        # 邻边两两为宽高，取较小值作为高度
        # 对于一般四边形，近似使用四条边中的两个较小者的平均
        e_sorted = sorted(e)
        height = float(np.mean(e_sorted[:2]))
        return max(height, 1.0)

    def _estimate_global_skew_angle(self, detections: List[TextDetection]) -> float:
        """估计文本的全局倾斜角（弧度，范围约为[-pi/2, pi/2]）"""
        angles: List[float] = []
        for det in detections:
            pts = np.array(det.bbox, dtype=np.float32)
            # 边向量
            v = [pts[(i+1) % 4] - pts[i] for i in range(4)]
            lengths = [np.linalg.norm(vec) for vec in v]
            # 选取较长的边作为文本方向（更接近水平方向）
            long_edge_idx = int(np.argmax(lengths))
            vx, vy = v[long_edge_idx]
            angle = math.atan2(float(vy), float(vx))
            # 归一化到[-pi/2, pi/2]
            while angle <= -math.pi / 2:
                angle += math.pi
            while angle > math.pi / 2:
                angle -= math.pi
            angles.append(angle)
        
        if not angles:
            return 0.0
        # 使用中位数，鲁棒性较好
        angles.sort()
        mid = len(angles) // 2
        if len(angles) % 2 == 1:
            return angles[mid]
        else:
            return 0.5 * (angles[mid - 1] + angles[mid])


class VLACameraOCRSystem:
    """VLA相机OCR系统 - 使用正确的API调用方式"""
    
    def __init__(self, ocr_engine: OCREngine, min_confidence: float = 0.7):
        self.ocr_engine = ocr_engine
        self.min_confidence = min_confidence
        
        # 存储最新的OCR结果
        # 初始化self.latest_ocr_result为四个CameraOCRResult的默认值
        self.latest_ocr_result = {}
        for cam_id in ['0', '1', '2', '3']:
            self.latest_ocr_result[cam_id] = CameraOCRResult(
                camera_id=cam_id,
                detected_text=[],
                text_detections=[],
                frame_timestamp=0.0,
                frame_shape=(0, 0, 0)
            )
        
        # 性能统计
        self.ocr_call_count = 0
        self.total_ocr_time = 0
        
        print(f"VLA OCR系统初始化完成，最小置信度: {min_confidence}")
    
    def process_camera_frame(self, frames, frame_timestamp: float = None):
        """
        处理从api.main_loop()获取的相机帧
        
        Args:
            frame: OpenCV格式的图像帧 (BGR, 640x480)
            frame_timestamp: 帧时间戳
        
        Returns:
            OCR检测结果字典
        """
        if frame_timestamp is None:
            frame_timestamp = time.time()
        
        start_time = time.time()
        all_results = {'0': None, '1': None, '2': None, '3': None}
        for frame_temp in frames:
            frame = frame_temp.frame
            # 执行OCR检测
            detections = self.ocr_engine.detect_text(frame)
            
            # 过滤低置信度结果
            filtered_detections = [d for d in detections if d.confidence >= self.min_confidence]
            
            # 提取文本列表
            detected_texts = [d.text for d in filtered_detections]
            
            # 创建结果对象
            result = CameraOCRResult(
                camera_id=frame_temp.id,
                detected_text=detected_texts,
                text_detections=filtered_detections,
                frame_timestamp=frame_timestamp,
                frame_shape=frame.shape
            )
            
            # 更新缓存
            self.latest_ocr_result[frame_temp.id] = result
            
            all_results[frame_temp.id] = result

        # 更新性能统计
        self.ocr_call_count += 1
        ocr_time = time.time() - start_time
        self.total_ocr_time += ocr_time
        return all_results
    
    def get_latest_ocr_result(self):
        """获取最新的OCR结果"""
        return self.latest_ocr_result
    
    def visualize_ocr_result(self, frame: np.ndarray, ocr_result: CameraOCRResult, 
                           save_path: str = None) -> np.ndarray:
        """
        可视化OCR检测结果
        
        Args:
            frame: 原始图像帧
            ocr_result: OCR检测结果
            save_path: 保存路径（可选）
        
        Returns:
            带有OCR标注的图像
        """
        # 复制图像以避免修改原图（BGR→RGB供PIL渲染中文）
        vis_image_bgr = frame.copy()
        vis_image_rgb = cv2.cvtColor(vis_image_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(vis_image_rgb)
        draw = ImageDraw.Draw(pil_img)
        
        # 加载中文字体
        font_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'font', 'msyh.ttc')
        try:
            font_main = ImageFont.truetype(font_path, 18)
        except Exception:
            font_main = ImageFont.load_default()
        
        # 绘制检测结果
        for detection in ocr_result.text_detections:
            # 绘制边界框
            bbox = np.array(detection.bbox, dtype=int)
            cv2.polylines(vis_image_bgr, [bbox], True, (0, 255, 0), 2)
            
            # 绘制文字和置信度
            center_x, center_y = map(int, detection.center)
            text_with_conf = f"{detection.text} ({detection.confidence:.2f})"
            
            # PIL 计算文本尺寸
            try:
                text_w, text_h = draw.textbbox((0, 0), text_with_conf, font=font_main)[2:4]
            except Exception:
                text_w, text_h = draw.textsize(text_with_conf, font=font_main)
            
            # 绘制文字背景（在PIL图层上）
            bg_left = int(center_x - text_w / 2) - 6
            bg_top = int(center_y - text_h) - 10
            bg_right = int(center_x + text_w / 2) + 6
            bg_bottom = int(center_y) + 6
            draw.rectangle([bg_left, bg_top, bg_right, bg_bottom], fill=(255, 255, 255, 200))
            
            # 绘制文字（黑色描边提高可读性）
            tx = int(center_x - text_w / 2)
            ty = int(center_y - text_h) - 4
            outline_offsets = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
            for dx, dy in outline_offsets:
                draw.text((tx + dx, ty + dy), text_with_conf, font=font_main, fill=(0, 0, 0))
            draw.text((tx, ty), text_with_conf, font=font_main, fill=(255, 0, 0))
        
        # 添加时间戳和统计信息
        info_text = f"OCR检测: {len(ocr_result.text_detections)}个文字"
        try:
            info_w, info_h = draw.textbbox((0, 0), info_text, font=font_main)[2:4]
        except Exception:
            info_w, info_h = draw.textsize(info_text, font=font_main)
        draw.rectangle([8, 8, 8 + info_w + 8, 8 + info_h + 8], fill=(255, 255, 255, 200))
        draw.text((12, 12), info_text, font=font_main, fill=(0, 0, 255))
        
        timestamp_text = f"时间: {time.strftime('%H:%M:%S', time.localtime(ocr_result.frame_timestamp))}"
        try:
            ts_w, ts_h = draw.textbbox((0, 0), timestamp_text, font=font_main)[2:4]
        except Exception:
            ts_w, ts_h = draw.textsize(timestamp_text, font=font_main)
        draw.rectangle([8, 40, 8 + ts_w + 8, 40 + ts_h + 8], fill=(255, 255, 255, 200))
        draw.text((12, 44), timestamp_text, font=font_main, fill=(0, 0, 255))
        
        # 将PIL层绘制的文本合成回BGR图像
        vis_image_rgb = np.array(pil_img)
        vis_image = cv2.cvtColor(vis_image_rgb, cv2.COLOR_RGB2BGR)
        
        # 将边框叠加到最终图像（已在vis_image_bgr上画好边框）
        # 为避免颜色偏差，使用边框图层覆盖到文本图层
        overlay = vis_image_bgr
        alpha = 1.0
        vis_image = cv2.addWeighted(overlay, alpha, vis_image, 1 - alpha, 0)
        
        # 保存图像（如果指定路径）
        if save_path:
            cv2.imwrite(save_path, vis_image)
            print(f"OCR可视化结果已保存到: {save_path}")
        
        return vis_image
    
    def get_performance_stats(self) -> Dict[str, float]:
        """获取性能统计信息"""
        if self.ocr_call_count == 0:
            return {"avg_ocr_time": 0, "total_calls": 0, "total_time": 0}
        
        return {
            "avg_ocr_time": self.total_ocr_time / self.ocr_call_count,
            "total_calls": self.ocr_call_count,
            "total_time": self.total_ocr_time
        }


def create_vla_ocr_system(min_confidence: float = 0.7, enable_line_merge: bool = True, **kwargs) -> VLACameraOCRSystem:
    """
    创建VLA OCR系统的工厂函数
    
    Args:
        min_confidence: 最小置信度 (默认0.7)
        enable_line_merge: 是否启用相邻行合并 (默认True，按从上到下顺序合并)
        **kwargs: 其他参数（保留用于兼容性）
    
    Returns:
        VLACameraOCRSystem实例
    """
    if not PYTORCH_PADDLE_OCR_AVAILABLE:
        raise ImportError(
            "PytorchPaddleOCR not available. Please check installation:\n"
            "Make sure pytorchocr is properly installed and models are available."
        )
    
    try:
        # 创建PytorchPaddleOCR引擎
        ocr_engine = PytorchPaddleOCREngine(enable_line_merge=enable_line_merge)
        print("使用PytorchPaddleOCR引擎")
        
        return VLACameraOCRSystem(ocr_engine, min_confidence)
        
    except Exception as e:
        raise RuntimeError(f"OCR系统初始化失败: {e}")
