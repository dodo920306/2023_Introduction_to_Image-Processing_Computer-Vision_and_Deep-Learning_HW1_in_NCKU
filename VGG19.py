from PyQt5.QtWidgets import QApplication, QPushButton, QVBoxLayout, QGroupBox, QLabel, QFileDialog
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
import sys


class MyWidget(QGroupBox):
    def __init__(self):
        super().__init__()
        self.setFixedHeight(545)
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
        self.image = QLabel()
        self.image.setFixedHeight(251)


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
        layout.addWidget(self.image)
        layout.setAlignment(Qt.AlignTop)

        self.setLayout(layout)

        button0.clicked.connect(self.load_image)
        

    def load_image(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        filename, _ = QFileDialog.getOpenFileName(self, "Select an Image", "", "Images (*.png *.jpg)", options=options)
        pixmap = QPixmap(filename)
        self.image.setPixmap(pixmap)
        self.image.setScaledContents(True)
        

if __name__ == '__main__':
    app = QApplication(sys.argv)
    MainWindow = MyWidget()
    MainWindow.show()
    sys.exit(app.exec_())
