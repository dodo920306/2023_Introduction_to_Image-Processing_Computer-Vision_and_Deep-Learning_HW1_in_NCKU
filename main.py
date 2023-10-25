from PyQt5.QtWidgets import QApplication, QWidget, QDesktopWidget, QPushButton, QVBoxLayout, QGroupBox, QHBoxLayout
from Image_Processing import MyWidget as GroupBox1
from Image_Smoothing import MyWidget as GroupBox2
import sys


class MyWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Hw1")
        self.setGeometry(100, 100, 800, 600)
        self.ui()
        screen_geo = QDesktopWidget().screenGeometry()
        widget_geo = self.geometry()
        x = (screen_geo.width() - widget_geo.width()) // 2
        y = (screen_geo.height() - widget_geo.height()) // 2
        self.move(x, y)

    def ui(self):
        mainLayout = QHBoxLayout()
        layout1 = QVBoxLayout()
        layout2 = QVBoxLayout()
        layout3 = QVBoxLayout()

        groupBox1 = GroupBox1()
        groupBox2 = GroupBox2()

        layout2.addWidget(groupBox1)
        layout2.addWidget(groupBox2)

        mainLayout.addLayout(layout1)
        mainLayout.addLayout(layout2)
        mainLayout.addLayout(layout3)

        self.setLayout(mainLayout)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    MainWindow = MyWidget()
    MainWindow.show()
    sys.exit(app.exec_())
