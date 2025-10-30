import cv2
import pickle
import os
import numpy as np


class FaceRecognizer:
    def __init__(self):
        self.cap = None
        self.recognizer = None
        self.names = {}
        self.threshold = 50
        self.detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        # 初始化识别器
        self.setup_recognizer()

    def setup_recognizer(self):
        """设置LBPH识别器参数"""
        self.recognizer = cv2.face.LBPHFaceRecognizer_create(
            radius=2,
            neighbors=16,
            grid_x=8,
            grid_y=8,
        )

    def is_model_trained(self):
        """检查模型是否已训练"""
        return os.path.exists("lbph_model.yml") and os.path.exists("labels.pkl")

    def load_model(self):
        """加载训练好的模型"""
        if not self.is_model_trained():
            raise Exception("模型未训练，请先训练模型")

        self.recognizer.read("lbph_model.yml")

        if os.path.exists("labels.pkl"):
            with open("labels.pkl", "rb") as f:
                self.names = pickle.load(f)
        else:
            raise Exception("标签文件不存在")

    def start_recognition(self, threshold=50, names=None):
        """开始实时识别"""
        if not self.is_model_trained():
            raise Exception("请先训练模型")

        self.threshold = threshold
        self.load_model()

        if names is not None:
            self.names.update(names)

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise Exception("无法打开摄像头")

    def stop_recognition(self):
        """停止识别"""
        if self.cap:
            self.cap.release()
        self.cap = None

    def set_threshold(self, threshold):
        """设置识别阈值"""
        self.threshold = threshold

    def get_frame_with_detection(self):
        """获取带检测结果的帧"""
        if self.cap is None or not self.cap.isOpened():
            return None

        ret, frame = self.cap.read()
        if not ret:
            return None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = self.detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        for (x, y, w, h) in faces:
            try:
                face_roi = gray[y:y + h, x:x + w]
                face_resized = cv2.resize(face_roi, (200, 200), interpolation=cv2.INTER_LINEAR)
                face_processed = cv2.equalizeHist(face_resized)

                id, confidence = self.recognizer.predict(face_processed)
                similarity_score = max(0, min(100, 100 - confidence))

                if confidence < self.threshold and id in self.names:
                    name = self.names[id]
                    color = (0, 255, 0)
                    status = f"{name} ({similarity_score:.1f}%)"
                else:
                    name = "Unknown"
                    color = (0, 0, 255)
                    status = f"{name} ({similarity_score:.1f}%)"

                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, status, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            except Exception as e:
                print(f"预测错误: {e}")
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(frame, "Error", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        return frame

    def predict_face(self, face_image):
        """预测单个人脸"""
        if self.recognizer is None:
            raise Exception("识别器未初始化")

        id, confidence = self.recognizer.predict(face_image)
        similarity = max(0, 100 - confidence)

        return {
            'id': id,
            'confidence': confidence,
            'similarity': similarity,
            'name': self.names.get(id, "Unknown"),
            'is_recognized': confidence < self.threshold and id in self.names
        }

    def recognize_image(self, image_path, threshold=None):
        """识别图片中的人脸"""
        if threshold is not None:
            self.threshold = threshold

        if not self.is_model_trained():
            raise Exception("模型未训练，请先训练模型")

        # 确保模型已加载
        if not hasattr(self.recognizer, 'getThreshold') or self.recognizer.getThreshold() is None:
            self.load_model()

        # 读取图片 - 处理中文路径
        try:
            # 方法1: 使用numpy从文件读取
            with open(image_path, 'rb') as f:
                image_array = np.frombuffer(f.read(), np.uint8)
                image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        except:
            # 方法2: 直接读取
            image = cv2.imread(image_path)

        if image is None:
            raise Exception("无法读取图片文件，请检查文件路径和格式")

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 检测人脸
        faces = self.detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(50, 50)
        )

        results = []

        for i, (x, y, w, h) in enumerate(faces):
            try:
                face_roi = gray[y:y + h, x:x + w]
                if face_roi.size == 0:
                    continue

                face_resized = cv2.resize(face_roi, (200, 200), interpolation=cv2.INTER_LINEAR)
                face_processed = cv2.equalizeHist(face_resized)

                prediction = self.predict_face(face_processed)

                if prediction['is_recognized']:
                    result_text = f"人脸 {i + 1}: {prediction['name']} (相似度: {prediction['similarity']:.1f}%)"
                    color = (0, 255, 0)
                else:
                    result_text = f"人脸 {i + 1}: 未知人脸 (置信度: {prediction['confidence']:.1f})"
                    color = (0, 0, 255)

                results.append(result_text)

                cv2.rectangle(image, (x, y), (x + w, y + h), color, 3)
                cv2.putText(image, f"{prediction['name']} ({prediction['similarity']:.1f}%)",
                            (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            except Exception as e:
                print(f"处理第 {i + 1} 个人脸时出错: {e}")
                continue

        return image, results