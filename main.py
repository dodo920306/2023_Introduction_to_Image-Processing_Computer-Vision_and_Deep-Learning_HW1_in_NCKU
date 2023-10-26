from PyQt5.QtWidgets import QApplication, QWidget, QDesktopWidget, QPushButton, QVBoxLayout, QGroupBox, QHBoxLayout
from PyQt5.QtCore import QRect
from Image_Processing import MyWidget as GroupBox1
from Image_Smoothing import MyWidget as GroupBox2
from Edge_Detection import MyWidget as GroupBox3
from Transforms import MyWidget as GroupBox4
from VGG19 import MyWidget as GroupBox5
import sys


class MyWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Hw1")
        self.ui()
        screen_geo = QDesktopWidget().screenGeometry()
        widget_geo = self.geometry()
        x = (screen_geo.width() - 2 * widget_geo.width())
        y = (screen_geo.height() - 2 * widget_geo.height())
        self.move(x, y)

    def ui(self):
        mainLayout = QHBoxLayout()
        layout1 = QVBoxLayout()
        layout2 = QVBoxLayout()
        layout3 = QVBoxLayout()

        button1 = QPushButton("Load Image 1")
        button2 = QPushButton("Load Image 2")

        groupBox1 = GroupBox1()
        groupBox2 = GroupBox2()
        groupBox3 = GroupBox3()
        groupBox4 = GroupBox4()
        groupBox5 = GroupBox5()

        layout1.addWidget(button1)
        layout1.addWidget(button2)

        layout2.addWidget(groupBox1)
        layout2.addWidget(groupBox2)
        layout2.addWidget(groupBox3)
        
        layout3.addWidget(groupBox4)
        layout3.addWidget(groupBox5)

        mainLayout.addLayout(layout1)
        mainLayout.addSpacing(20)
        mainLayout.addLayout(layout2)
        mainLayout.addSpacing(20)
        mainLayout.addLayout(layout3)
        mainLayout.setContentsMargins(50, 20, 50, 20)

        self.setLayout(mainLayout)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    MainWindow = MyWidget()
    MainWindow.show()
    sys.exit(app.exec_())
