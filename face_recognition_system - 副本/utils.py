from datetime import datetime
import os

import numpy as np


def get_current_time():
    """获取当前时间字符串"""
    return datetime.now().strftime("%H:%M:%S")


def ensure_directory_exists(directory):
    """确保目录存在"""
    os.makedirs(directory, exist_ok=True)
    return directory


def analyze_lbp_features(face_image):
    """
    分析LBP特征（用于调试）
    """
    # 将图像分成8x8的小区域
    height, width = face_image.shape
    cell_height = height // 8
    cell_width = width // 8

    features = []

    for i in range(8):
        for j in range(8):
            # 提取每个小区域
            cell = face_image[i * cell_height:(i + 1) * cell_height,
                   j * cell_width:(j + 1) * cell_width]

            # 计算每个区域的统计信息
            features.append({
                'cell_position': (i, j),
                'mean_intensity': np.mean(cell),
                'std_intensity': np.std(cell),
                'min_intensity': np.min(cell),
                'max_intensity': np.max(cell)
            })

    return features