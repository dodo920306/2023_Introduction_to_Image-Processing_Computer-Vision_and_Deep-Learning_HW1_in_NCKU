from PyQt5.QtWidgets import QApplication, QWidget, QDesktopWidget, QPushButton, QVBoxLayout, QGroupBox, QHBoxLayout, QFileDialog, QLabel
from PyQt5.QtCore import Qt
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
        x = (screen_geo.width() - widget_geo.width()) // 2 - 15
        y = (screen_geo.height() - widget_geo.height()) // 2 - 175
        self.move(x, y)

    def ui(self):
        mainLayout = QHBoxLayout()
        layout1 = QVBoxLayout()
        layout2 = QVBoxLayout()
        layout3 = QVBoxLayout()

        button1 = QPushButton("Load Image 1")
        self.label1 = QLabel()
        button2 = QPushButton("Load Image 2")
        self.label2 = QLabel()

        groupBox1 = GroupBox1()
        groupBox2 = GroupBox2()
        groupBox3 = GroupBox3()
        groupBox4 = GroupBox4()
        groupBox5 = GroupBox5()

        layout1.addWidget(button1)
        layout1.addWidget(self.label1)
        layout1.addWidget(button2)
        layout1.addWidget(self.label2)
        layout1.setAlignment(Qt.AlignVCenter)

        layout2.addWidget(groupBox1)
        layout2.addWidget(groupBox2)
        layout2.addWidget(groupBox3)
        
        layout3.addWidget(groupBox4)
        layout3.addWidget(groupBox5)
        layout3.setAlignment(Qt.AlignTop)

        mainLayout.addLayout(layout1)
        mainLayout.addSpacing(20)
        mainLayout.addLayout(layout2)
        mainLayout.addSpacing(20)
        mainLayout.addLayout(layout3)
        mainLayout.setContentsMargins(50, 20, 50, 20)

        self.setLayout(mainLayout)

        button1.clicked.connect(self.load_image)

    def load_image(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        filename, _ = QFileDialog.getOpenFileName(self, "Select an Image", "", "Images (*.png *.jpg)", options=options)
        self.label1.setText(filename.split('/')[-1])


if __name__ == '__main__':
    app = QApplication(sys.argv)
    MainWindow = MyWidget()
    MainWindow.show()
    sys.exit(app.exec_())
