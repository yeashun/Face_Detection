import cv2
import os


def capture_faces(user_name, num_samples=150, output_dir="data/raw"):
    """
    采集人脸样本
    """
    # 创建输出目录
    user_dir = os.path.join(output_dir, user_name)
    os.makedirs(user_dir, exist_ok=True)

    # 初始化摄像头
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise Exception("无法打开摄像头")

    # 加载人脸检测器
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    count = 0
    try:
        while count < num_samples:
            ret, frame = cap.read()
            if not ret:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                # 保存人脸区域
                face_img = gray[y:y + h, x:x + w]
                face_path = os.path.join(user_dir, f"{count}.jpg")
                cv2.imwrite(face_path, face_img)
                count += 1

                # 显示框
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"Count: {count}/{num_samples}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("Capturing Faces - Press 'q' to quit", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()