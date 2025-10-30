#主程序入口
import sys
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QFont
from gui import FaceApp

if __name__ == "__main__":
    app = QApplication(sys.argv)

    # 设置应用字体
    font = QFont("Microsoft YaHei", 12)
    app.setFont(font)

    win = FaceApp()
    win.show()
    sys.exit(app.exec_())