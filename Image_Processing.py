from PyQt5.QtWidgets import QApplication, QPushButton, QVBoxLayout, QGroupBox
import sys


class MyWidget(QGroupBox):
    def __init__(self):
        super().__init__()
       
        self.ui()

    def ui(self):
        self.setTitle("1. Image Processing")
        layout = QVBoxLayout()

        button1 = QPushButton("1.1 Color Separation")
        button2 = QPushButton("1.2 Color Transformation")
        button3 = QPushButton("1.3 Color Extraction")

        layout.addSpacing(20)
        layout.addWidget(button1)
        layout.addSpacing(30)
        layout.addWidget(button2)
        layout.addSpacing(30)
        layout.addWidget(button3)
        layout.addSpacing(20)

        self.setLayout(layout)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    MainWindow = MyWidget()
    MainWindow.show()
    sys.exit(app.exec_())