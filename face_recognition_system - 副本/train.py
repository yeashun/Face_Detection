import os
import cv2
import numpy as np
import pickle


def train(data_dir="data/processed", model_path="lbph_model.yml", labels_path="labels.pkl"):
    """
    训练人脸识别模型
    """
    faces = []
    labels = []
    label_dict = {}
    current_label = 0

    if not os.path.exists(data_dir):
        raise Exception("预处理数据目录不存在")

    # 收集训练数据
    for user_name in os.listdir(data_dir):
        user_dir = os.path.join(data_dir, user_name)
        if not os.path.isdir(user_dir):
            continue

        label_dict[current_label] = user_name

        for img_name in os.listdir(user_dir):
            img_path = os.path.join(user_dir, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if img is not None:
                faces.append(img)
                labels.append(current_label)

        current_label += 1

    if len(faces) == 0:
        raise Exception("没有找到训练数据")

    # 训练模型
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(labels))

    # 保存模型和标签
    recognizer.write(model_path)
    with open(labels_path, "wb") as f:
        pickle.dump(label_dict, f)

    return label_dict