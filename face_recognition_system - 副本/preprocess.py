import cv2
import os
import numpy as np


def preprocess(user_name, input_dir="data/raw", output_dir="data/processed"):
    """
    预处理人脸图像
    """
    input_user_dir = os.path.join(input_dir, user_name)
    output_user_dir = os.path.join(output_dir, user_name)
    os.makedirs(output_user_dir, exist_ok=True)

    if not os.path.exists(input_user_dir):
        raise Exception(f"用户 {user_name} 的原始数据不存在")

    # 加载人脸检测器
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    processed_count = 0
    for img_name in os.listdir(input_user_dir):
        img_path = os.path.join(input_user_dir, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            continue

        # 检测人脸
        faces = face_cascade.detectMultiScale(img, 1.1, 4)

        for (x, y, w, h) in faces:
            # 裁剪和调整大小
            face = img[y:y + h, x:x + w]
            face = cv2.resize(face, (200, 200))

            # 直方图均衡化
            face = cv2.equalizeHist(face)

            # 保存处理后的图像
            output_path = os.path.join(output_user_dir, img_name)
            cv2.imwrite(output_path, face)
            processed_count += 1

    return processed_count