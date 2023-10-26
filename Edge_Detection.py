from PyQt5.QtWidgets import QApplication, QPushButton, QVBoxLayout, QGroupBox
import sys


class MyWidget(QGroupBox):
    def __init__(self):
        super().__init__()
       
        self.ui()

    def ui(self):
        self.setTitle("3. Edge Detection")
        layout = QVBoxLayout()

        button1 = QPushButton("3.1 Sobel X")
        button2 = QPushButton("3.2 Sobel Y")
        button3 = QPushButton("3.3 Combination and Threshold")
        button4 = QPushButton("3.4 Gradient Angle")

        layout.addSpacing(20)
        layout.addWidget(button1)
        layout.addSpacing(30)
        layout.addWidget(button2)
        layout.addSpacing(30)
        layout.addWidget(button3)
        layout.addSpacing(30)
        layout.addWidget(button4)
        layout.addSpacing(20)

        self.setLayout(layout)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    MainWindow = MyWidget()
    MainWindow.show()
    sys.exit(app.exec_())
