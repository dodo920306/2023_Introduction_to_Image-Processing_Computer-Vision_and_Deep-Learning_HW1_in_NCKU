from PyQt5.QtWidgets import QApplication, QPushButton, QVBoxLayout, QGroupBox, QLabel
from PyQt5.QtGui import QPixmap
import sys


class MyWidget(QGroupBox):
    def __init__(self):
        super().__init__()
       
        self.ui()

    def ui(self):
        self.setTitle("5. VGG19")
        layout = QVBoxLayout()

        button0 = QPushButton("Load Image")
        button1 = QPushButton("5.1 Show Augmented Images")
        button2 = QPushButton("5.2 Show Model Structure")
        button3 = QPushButton("5.3 Show Acc and Loss ")
        button4 = QPushButton("5.4 Inference")
        button4 = QPushButton("5.4 Inference")
        label = QLabel("Predict=")
        image = QLabel()
        pixmap = QPixmap()
        image.setPixmap(pixmap)

        layout.addWidget(button0)
        layout.addSpacing(10)
        layout.addWidget(button1)
        layout.addSpacing(10)
        layout.addWidget(button2)
        layout.addSpacing(10)
        layout.addWidget(button3)
        layout.addSpacing(10)
        layout.addWidget(button4)
        layout.addWidget(label)
        layout.addWidget(image)
        layout.addSpacing(200)

        self.setLayout(layout)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    MainWindow = MyWidget()
    MainWindow.show()
    sys.exit(app.exec_())
