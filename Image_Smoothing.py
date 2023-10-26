from PyQt5.QtWidgets import QApplication, QPushButton, QVBoxLayout, QGroupBox
import sys


class MyWidget(QGroupBox):
    def __init__(self):
        super().__init__()
       
        self.ui()

    def ui(self):
        self.setTitle("2. Image Smoothing")
        layout = QVBoxLayout()

        button1 = QPushButton("2.1 Gaussian blur")
        button2 = QPushButton("2.2 Bilateral filter")
        button3 = QPushButton("2.3 Median filter")

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
