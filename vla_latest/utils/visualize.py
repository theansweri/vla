import cv2
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont


def draw_ocr_results(image, ocr_results, show_confidence=True):
    """
    在图像上绘制OCR结果，包括边界框和识别文字
    Args:
        image: 原始图像
        ocr_results: OCR结果，格式为 [[[points], (text, confidence)], ...]
                    其中points为[[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        show_confidence: 是否显示置信度，默认为True
    Returns:
        绘制了检测框和文字的图像
    """
    img = image.copy()

    if not ocr_results or ocr_results[0] is None:
        return img

    # 首先使用OpenCV绘制所有边界框
    for box_text in ocr_results[0]:
        points = np.array(box_text[0], dtype=np.int32)
        cv2.polylines(img, [points], True, (0, 255, 0), 2)

    # 转换为PIL图像以支持中文绘制
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    # 中文字体默认使用微软雅黑
    font_size = 24  # 增大字体大小
    font_path = "font/msyh.ttc"
    font = ImageFont.truetype(font_path, font_size)

    # 使用PIL绘制文本
    for box_text in ocr_results[0]:
        points = np.array(box_text[0], dtype=np.int32)
        text, confidence = box_text[1]

        # 计算文本位置（使用边界框左上角）
        text_x = points[0][0]
        text_y = points[0][1] - 35  # 在框上方显示文本，增加间距
        if text_y < 0:  # 如果文本位置在图像外，则显示在框内
            text_y = points[0][1] + 25

        # 根据show_confidence参数决定是否显示置信度
        text_to_draw = f"{text} ({confidence:.2f})" if show_confidence else text

        # 添加文本背景以提高可读性
        bbox = draw.textbbox((text_x, text_y), text_to_draw, font=font)
        draw.rectangle(
            [bbox[0] - 2, bbox[1] - 2, bbox[2] + 2, bbox[3] + 2], fill=(0, 0, 0)
        )
        draw.text((text_x, text_y), text_to_draw, fill=(0, 255, 0), font=font)

    # 将PIL图像转换回OpenCV格式
    img_result = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    return img_result
