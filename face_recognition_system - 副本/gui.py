import os
import pickle
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QTextEdit, QTabWidget, QListWidget,
    QSlider, QFormLayout, QFrame, QMessageBox, QFileDialog, QProgressBar,
    QGroupBox, QSplitter, QComboBox
)
from PyQt5.QtGui import QImage, QPixmap, QFont, QIcon, QPalette, QColor
from PyQt5.QtCore import QTimer, Qt, QSize

from capture import capture_faces
from preprocess import preprocess
from train import train
from recognize import FaceRecognizer
from utils import get_current_time


class FaceApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("智能人脸识别系统")
        self.setGeometry(100, 50, 1400, 900)
        self.setup_ui()
        self.setup_styles()

        # 模块变量
        self.recognizer = FaceRecognizer()
        self.names = {}
        self.threshold = 50
        self.current_image = None

    def setup_ui(self):
        # 设置应用图标
        self.setWindowIcon(QIcon.fromTheme("camera-web"))

        # 主界面 = TabWidget
        self.tabs = QTabWidget()
        self.tabs.setTabPosition(QTabWidget.North)
        self.tabs.setMovable(False)
        self.setCentralWidget(self.tabs)

        # 添加四个标签页
        self.init_user_tab()
        self.init_train_tab()
        self.init_recognize_tab()
        self.init_image_recognition_tab()  # 新增图片识别页

        # 状态栏
        self.statusBar().showMessage("系统就绪 - 欢迎使用智能人脸识别系统")

    def setup_styles(self):
        # 更现代化的样式
        self.setStyleSheet("""
            QMainWindow {
                background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 1,
                                          stop: 0 #f8f9fa, stop: 1 #e9ecef);
            }
            QTabWidget::pane {
                border: 2px solid #dee2e6;
                background: white;
                border-radius: 12px;
                margin: 5px;
            }
            QTabBar::tab {
                background: #6c757d;
                color: white;
                padding: 12px 24px;
                margin-right: 2px;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                font-weight: bold;
                font-size: 12px;
            }
            QTabBar::tab:selected {
                background: #007bff;
                color: white;
            }
            QTabBar::tab:hover {
                background: #0056b3;
            }
            QPushButton {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                          stop: 0 #007bff, stop: 1 #0056b3);
                color: white;
                border: none;
                padding: 12px 20px;
                border-radius: 8px;
                font-weight: bold;
                font-size: 13px;
                margin: 2px;
            }
            QPushButton:hover {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                          stop: 0 #0056b3, stop: 1 #004085);
            }
            QPushButton:pressed {
                background: #004085;
            }
            QPushButton:disabled {
                background: #6c757d;
                color: #adb5bd;
            }
            QLineEdit, QTextEdit, QListWidget {
                border: 2px solid #ced4da;
                border-radius: 6px;
                padding: 10px;
                background: white;
                font-size: 13px;
                selection-background-color: #007bff;
            }
            QLineEdit:focus, QTextEdit:focus {
                border: 2px solid #007bff;
            }
            QLabel {
                color: #2d3748;
                font-size: 13px;
            }
            QSlider::groove:horizontal {
                border: 1px solid #ced4da;
                height: 10px;
                background: #e9ecef;
                border-radius: 5px;
            }
            QSlider::handle:horizontal {
                background: #007bff;
                border: 2px solid #0056b3;
                width: 22px;
                margin: -6px 0;
                border-radius: 11px;
            }
            QSlider::sub-page:horizontal {
                background: #007bff;
                border-radius: 5px;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #dee2e6;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 15px;
                background: white;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 8px 0 8px;
            }
            QProgressBar {
                border: 2px solid #dee2e6;
                border-radius: 5px;
                text-align: center;
                background: white;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                          stop: 0 #28a745, stop: 1 #20c997);
                border-radius: 3px;
            }
        """)

    # -------------------- 用户管理页 --------------------
    def init_user_tab(self):
        tab = QWidget()
        main_layout = QVBoxLayout()
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(25, 25, 25, 25)

        # 标题
        title = QLabel("👥 用户管理")
        title.setFont(QFont("Microsoft YaHei", 20, QFont.Bold))
        title.setStyleSheet("color: #2d3748; margin-bottom: 25px;")
        main_layout.addWidget(title)

        # 表单区域
        form_group = QGroupBox("用户操作")
        form_layout = QVBoxLayout(form_group)

        # 用户名输入
        name_layout = QHBoxLayout()
        name_label = QLabel("👤 用户名:")
        name_label.setFont(QFont("Microsoft YaHei", 11, QFont.Bold))
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("请输入用户名...")
        self.name_input.setMinimumHeight(45)
        self.name_input.setStyleSheet("font-size: 14px;")
        name_layout.addWidget(name_label)
        name_layout.addWidget(self.name_input)
        form_layout.addLayout(name_layout)

        # 按钮区域
        button_layout = QHBoxLayout()
        btn_capture = QPushButton("📸 采集人脸样本")
        btn_capture.setIconSize(QSize(24, 24))
        btn_capture.setMinimumHeight(50)

        btn_preprocess = QPushButton("⚙️ 预处理人脸数据")
        btn_preprocess.setIconSize(QSize(24, 24))
        btn_preprocess.setMinimumHeight(50)

        button_layout.addWidget(btn_capture)
        button_layout.addWidget(btn_preprocess)
        form_layout.addLayout(button_layout)

        main_layout.addWidget(form_group)

        # 进度条
        self.progress_user = QProgressBar()
        self.progress_user.setVisible(False)
        main_layout.addWidget(self.progress_user)

        # 日志区域
        log_group = QGroupBox("操作日志")
        log_layout = QVBoxLayout(log_group)
        self.log_user = QTextEdit()
        self.log_user.setMaximumHeight(250)
        self.log_user.setStyleSheet("font-family: 'Consolas', 'Courier New'; font-size: 12px;")
        log_layout.addWidget(self.log_user)
        main_layout.addWidget(log_group)

        # 连接信号
        btn_capture.clicked.connect(self.do_capture)
        btn_preprocess.clicked.connect(self.do_preprocess)

        tab.setLayout(main_layout)
        self.tabs.addTab(tab, "👥 用户管理")

    # -------------------- 模型训练页 --------------------
    def init_train_tab(self):
        tab = QWidget()
        main_layout = QVBoxLayout()
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(25, 25, 25, 25)

        # 标题
        title = QLabel("🚀 模型训练")
        title.setFont(QFont("Microsoft YaHei", 18, QFont.Bold))
        title.setStyleSheet("color: #2d3748; margin-bottom: 20px;")
        main_layout.addWidget(title)

        # 训练按钮
        btn_train = QPushButton("🎯 开始训练模型")
        btn_train.setMinimumHeight(55)
        btn_train.setStyleSheet(
            "font-size: 16px; background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1, stop: 0 #28a745, stop: 1 #20c997);")
        main_layout.addWidget(btn_train)

        # 进度条
        self.progress_train = QProgressBar()
        self.progress_train.setVisible(False)
        main_layout.addWidget(self.progress_train)

        # 用户列表区域
        user_group = QGroupBox("训练用户列表")
        user_layout = QVBoxLayout(user_group)
        self.user_list = QListWidget()
        self.user_list.setStyleSheet("font-family: 'Consolas', 'Courier New';")
        user_layout.addWidget(self.user_list)
        main_layout.addWidget(user_group)

        # 统计信息
        stats_layout = QHBoxLayout()
        self.lbl_stats = QLabel("📊 等待训练数据...")
        stats_layout.addWidget(self.lbl_stats)
        stats_layout.addStretch()
        main_layout.addLayout(stats_layout)

        # 日志区域
        log_group = QGroupBox("训练日志")
        log_layout = QVBoxLayout(log_group)
        self.log_train = QTextEdit()
        self.log_train.setMaximumHeight(180)
        self.log_train.setStyleSheet("font-family: 'Consolas', 'Courier New'; font-size: 12px;")
        log_layout.addWidget(self.log_train)
        main_layout.addWidget(log_group)

        btn_train.clicked.connect(self.do_train)

        tab.setLayout(main_layout)
        self.tabs.addTab(tab, "🚀 模型训练")

    # -------------------- 实时识别页 --------------------
    def init_recognize_tab(self):
        tab = QWidget()
        main_layout = QVBoxLayout()
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(25, 25, 25, 25)

        # 标题
        title = QLabel("🔍 实时识别")
        title.setFont(QFont("Microsoft YaHei", 18, QFont.Bold))
        title.setStyleSheet("color: #2d3748; margin-bottom: 20px;")
        main_layout.addWidget(title)

        # 控制区域
        control_group = QGroupBox("识别控制")
        control_layout = QVBoxLayout(control_group)

        # 按钮区域
        btn_layout = QHBoxLayout()
        btn_start = QPushButton("▶️ 开始实时识别")
        btn_start.setMinimumHeight(45)

        btn_stop = QPushButton("⏹️ 停止识别")
        btn_stop.setMinimumHeight(45)
        btn_stop.setStyleSheet(
            "background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1, stop: 0 #dc3545, stop: 1 #c82333);")

        btn_layout.addWidget(btn_start)
        btn_layout.addWidget(btn_stop)
        btn_layout.addStretch()
        control_layout.addLayout(btn_layout)

        # 阈值调节
        threshold_layout = QHBoxLayout()
        threshold_label = QLabel("🎚️ 识别阈值:")
        threshold_label.setFont(QFont("Microsoft YaHei", 11, QFont.Bold))

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(10)
        self.slider.setMaximum(100)
        self.slider.setValue(50)
        self.slider.valueChanged.connect(self.update_threshold)

        self.lbl_threshold = QLabel("65 (值越小越严格)")
        self.lbl_threshold.setMinimumWidth(150)
        self.lbl_threshold.setFont(QFont("Microsoft YaHei", 10))

        threshold_layout.addWidget(threshold_label)
        threshold_layout.addWidget(self.slider)
        threshold_layout.addWidget(self.lbl_threshold)
        control_layout.addLayout(threshold_layout)

        main_layout.addWidget(control_group)

        # 视频显示
        video_group = QGroupBox("实时视频")
        video_layout = QVBoxLayout(video_group)
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(800, 500)
        self.video_label.setText("🎥 视频预览区域\n\n点击\"开始实时识别\"启动摄像头")
        self.video_label.setStyleSheet("""
            color: #6c757d; 
            font-size: 16px; 
            font-family: 'Microsoft YaHei';
            background: #000;
            border-radius: 8px;
        """)
        video_layout.addWidget(self.video_label)
        main_layout.addWidget(video_group)

        # 日志区域
        log_group = QGroupBox("识别日志")
        log_layout = QVBoxLayout(log_group)
        self.log_recog = QTextEdit()
        self.log_recog.setMaximumHeight(120)
        self.log_recog.setStyleSheet("font-family: 'Consolas', 'Courier New'; font-size: 12px;")
        log_layout.addWidget(self.log_recog)
        main_layout.addWidget(log_group)

        btn_start.clicked.connect(self.start_recognition)
        btn_stop.clicked.connect(self.stop_recognition)

        tab.setLayout(main_layout)
        self.tabs.addTab(tab, "🔍 实时识别")

    # -------------------- 图片识别页 --------------------
    def init_image_recognition_tab(self):
        tab = QWidget()
        main_layout = QHBoxLayout()
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(25, 25, 25, 25)

        # 左侧控制面板
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setSpacing(20)

        # 标题
        title = QLabel("🖼️ 图片识别")
        title.setFont(QFont("Microsoft YaHei", 18, QFont.Bold))
        title.setStyleSheet("color: #2d3748;")
        left_layout.addWidget(title)

        # 上传区域
        upload_group = QGroupBox("图片上传")
        upload_layout = QVBoxLayout(upload_group)

        btn_upload = QPushButton("📁 选择图片")
        btn_upload.setMinimumHeight(50)
        btn_upload.clicked.connect(self.upload_image)

        self.lbl_image_path = QLabel("未选择图片")
        self.lbl_image_path.setStyleSheet("color: #6c757d; font-style: italic;")

        upload_layout.addWidget(btn_upload)
        upload_layout.addWidget(self.lbl_image_path)
        left_layout.addWidget(upload_group)

        # 识别控制
        recognize_group = QGroupBox("识别设置")
        recognize_layout = QVBoxLayout(recognize_group)

        threshold_layout = QHBoxLayout()
        threshold_label = QLabel("🎚️ 识别阈值:")
        self.slider_image = QSlider(Qt.Horizontal)
        self.slider_image.setMinimum(10)
        self.slider_image.setMaximum(100)
        self.slider_image.setValue(50)
        self.slider_image.valueChanged.connect(self.update_image_threshold)

        self.lbl_image_threshold = QLabel("50")
        threshold_layout.addWidget(threshold_label)
        threshold_layout.addWidget(self.slider_image)
        threshold_layout.addWidget(self.lbl_image_threshold)
        recognize_layout.addLayout(threshold_layout)

        btn_recognize = QPushButton("🔍 开始识别")
        btn_recognize.setMinimumHeight(45)
        btn_recognize.clicked.connect(self.recognize_image)
        recognize_layout.addWidget(btn_recognize)

        left_layout.addWidget(recognize_group)

        # 识别结果
        result_group = QGroupBox("识别结果")
        result_layout = QVBoxLayout(result_group)

        self.lbl_result = QLabel("等待识别...")
        self.lbl_result.setWordWrap(True)

        result_font = QFont("Microsoft YaHei", 14)  # 14号字体
        self.lbl_result.setFont(result_font)

        self.lbl_result.setStyleSheet("""
            background: #f8f9fa;
            border: 2px solid #dee2e6;
            border-radius: 8px;
            padding: 15px;
            font-family: 'Microsoft YaHei';
            min-height: 100px;
        """)

        result_layout.addWidget(self.lbl_result)
        left_layout.addWidget(result_group)

        left_layout.addStretch()

        # 右侧图片显示
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        image_group = QGroupBox("图片预览")
        image_layout = QVBoxLayout(image_group)

        self.lbl_image_display = QLabel()
        self.lbl_image_display.setAlignment(Qt.AlignCenter)
        self.lbl_image_display.setMinimumSize(600, 400)
        self.lbl_image_display.setStyleSheet("""
            background: #000;
            border: 2px solid #dee2e6;
            border-radius: 8px;
            color: #6c757d;
            font-family: 'Microsoft YaHei';
        """)
        self.lbl_image_display.setText("🖼️ 图片预览区域\n\n请上传图片进行识别")

        image_layout.addWidget(self.lbl_image_display)
        right_layout.addWidget(image_group)

        # 添加到主布局
        main_layout.addWidget(left_panel, 1)
        main_layout.addWidget(right_panel, 2)

        tab.setLayout(main_layout)
        self.tabs.addTab(tab, "🖼️ 图片识别")

    # -------------------- 功能函数 --------------------
    def do_capture(self):
        user = self.name_input.text().strip()
        if not user:
            QMessageBox.warning(self, "⚠️ 警告", "请输入用户名")
            return

        try:
            self.progress_user.setVisible(True)
            self.progress_user.setRange(0, 0)  # 不确定进度
            capture_faces(user)
            self.log_user.append(f"✅ [{get_current_time()}] 采集完成: {user}")
            self.statusBar().showMessage(f"🎉 已采集 {user} 的人脸数据")
        except Exception as e:
            self.log_user.append(f"❌ [{get_current_time()}] 采集失败: {str(e)}")
            QMessageBox.critical(self, "❌ 错误", f"采集失败: {str(e)}")
        finally:
            self.progress_user.setVisible(False)

    def do_preprocess(self):
        user = self.name_input.text().strip()
        if not user:
            QMessageBox.warning(self, "⚠️ 警告", "请输入用户名")
            return

        try:
            self.progress_user.setVisible(True)
            self.progress_user.setRange(0, 0)
            count = preprocess(user)
            self.log_user.append(f"✅ [{get_current_time()}] 预处理完成: {user} ({count}张图像)")
            self.statusBar().showMessage(f"⚙️ 已预处理 {user} 的人脸数据")
        except Exception as e:
            self.log_user.append(f"❌ [{get_current_time()}] 预处理失败: {str(e)}")
            QMessageBox.critical(self, "❌ 错误", f"预处理失败: {str(e)}")
        finally:
            self.progress_user.setVisible(False)

    def do_train(self):
        try:
            self.progress_train.setVisible(True)
            self.progress_train.setRange(0, 0)
            self.names = train()
            self.user_list.clear()
            for k, v in self.names.items():
                self.user_list.addItem(f"👤 ID={k}, 姓名={v}")
            self.log_train.append(f"✅ [{get_current_time()}] 模型训练完成, 共 {len(self.names)} 个用户")
            self.lbl_stats.setText(f"📊 已训练: {len(self.names)} 用户")
            self.statusBar().showMessage("🎯 模型训练完成")
        except Exception as e:
            self.log_train.append(f"❌ [{get_current_time()}] 训练失败: {str(e)}")
            QMessageBox.critical(self, "❌ 错误", f"训练失败: {str(e)}")
        finally:
            self.progress_train.setVisible(False)

    def start_recognition(self):
        try:
            self.recognizer.start_recognition(self.threshold, self.names)
            self.log_recog.append(f"▶️ [{get_current_time()}] 开始实时识别...")
            self.statusBar().showMessage("🔍 实时识别中...")

            # 启动定时器更新画面
            self.timer = QTimer()
            self.timer.timeout.connect(self.update_frame)
            self.timer.start(30)

        except Exception as e:
            self.log_recog.append(f"❌ [{get_current_time()}] 启动识别失败: {str(e)}")
            QMessageBox.critical(self, "❌ 错误", f"启动识别失败: {str(e)}")

    def stop_recognition(self):
        self.recognizer.stop_recognition()
        if hasattr(self, 'timer'):
            self.timer.stop()
        self.video_label.clear()
        self.video_label.setText("🎥 视频预览区域\n\n点击\"开始实时识别\"启动摄像头")
        self.log_recog.append(f"⏹️ [{get_current_time()}] 识别已停止")
        self.statusBar().showMessage("🛑 识别已停止")

    def update_frame(self):
        frame = self.recognizer.get_frame_with_detection()
        if frame is not None:
            # 显示到 QLabel
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            bytes_per_line = ch * w
            qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.video_label.setPixmap(QPixmap.fromImage(qimg).scaled(
                self.video_label.width(),
                self.video_label.height(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            ))

    def update_threshold(self):
        self.threshold = self.slider.value()
        self.lbl_threshold.setText(f"{self.threshold} (值越小越严格)")
        self.recognizer.set_threshold(self.threshold)

    def update_image_threshold(self):
        threshold = self.slider_image.value()
        self.lbl_image_threshold.setText(f"{threshold}")

    def upload_image(self):
        try:
            # 支持更多图片格式
            file_path, _ = QFileDialog.getOpenFileName(
                self, "选择图片", "",
                "图片文件 (*.jpg *.jpeg *.png *.bmp *.tiff *.tif *.webp *.ppm *.pgm);;所有文件 (*.*)"
            )

            if file_path:
                # 处理中文路径问题
                if any(ord(c) > 127 for c in file_path):
                    # 如果是中文路径，尝试使用短路径名（Windows）
                    try:
                        import win32api
                        short_path = win32api.GetShortPathName(file_path)
                        if os.path.exists(short_path):
                            file_path = short_path
                    except:
                        pass

                self.lbl_image_path.setText(f"📄 {os.path.basename(file_path)}")
                self.current_image = file_path

                # 显示图片
                pixmap = QPixmap(file_path)
                if not pixmap.isNull():
                    scaled_pixmap = pixmap.scaled(
                        self.lbl_image_display.width() - 20,
                        self.lbl_image_display.height() - 20,
                        Qt.KeepAspectRatio,
                        Qt.SmoothTransformation
                    )
                    self.lbl_image_display.setPixmap(scaled_pixmap)
                    self.lbl_result.setText("✅ 图片加载成功\n\n点击\"开始识别\"进行分析")
                else:
                    QMessageBox.warning(self, "⚠️ 警告", "无法加载图片文件，可能格式不支持")
                    self.lbl_image_display.setText("🖼️ 图片加载失败\n\n请尝试选择其他格式的图片")

        except Exception as e:
            QMessageBox.critical(self, "❌ 错误", f"上传图片失败: {str(e)}")

    def recognize_image(self):
        if not self.current_image:
            QMessageBox.warning(self, "⚠️ 警告", "请先选择图片")
            return

        # 检查模型是否训练
        if not self.recognizer.is_model_trained():
            QMessageBox.warning(self, "⚠️ 警告", "请先训练模型后再进行识别")
            return

        try:
            # 设置阈值
            threshold = self.slider_image.value()
            self.recognizer.set_threshold(threshold)

            # 确保模型已加载
            try:
                # 识别图片
                result_image, results = self.recognizer.recognize_image(self.current_image, threshold)
            except Exception as e:
                # 如果模型未加载，先加载再识别
                if "not computed" in str(e) or "not initialized" in str(e):
                    self.recognizer.load_model()
                    result_image, results = self.recognizer.recognize_image(self.current_image, threshold)
                else:
                    raise e

            # 显示带标注的图片
            rgb_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qimg = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)

            pixmap = QPixmap.fromImage(qimg)
            scaled_pixmap = pixmap.scaled(
                self.lbl_image_display.width() - 20,
                self.lbl_image_display.height() - 20,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.lbl_image_display.setPixmap(scaled_pixmap)

            # 显示识别结果
            if results:
                result_text = "🎯 识别结果:\n" + "\n".join(results)
                result_text += f"\n\n📊 共检测到 {len(results)} 个人脸"
            else:
                result_text = "❌ 未检测到人脸\n\n💡 提示: 尝试调整图片或确保人脸清晰可见"

            self.lbl_result.setText(result_text)
            self.statusBar().showMessage(f"✅ 图片识别完成 - 检测到 {len(results)} 个人脸")

        except Exception as e:
            error_msg = f"❌ 识别错误: {str(e)}"
            print(f"详细错误: {error_msg}")
            QMessageBox.critical(self, "❌ 错误", error_msg)
            self.lbl_result.setText(error_msg)

    def closeEvent(self, event):
        self.stop_recognition()
        event.accept()