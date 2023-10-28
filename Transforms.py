from PyQt5.QtWidgets import QApplication, QHBoxLayout, QPushButton, QVBoxLayout, QGroupBox, QLineEdit, QLabel
from PyQt5.QtGui import QIntValidator, QDoubleValidator
from PyQt5.QtCore import Qt
import sys, cv2, numpy


class MyWidget(QGroupBox):
    def __init__(self):
        super().__init__()
       
        self.ui()

    def ui(self):
        self.setTitle("4. Transforms")
        mainLayout = QVBoxLayout()
        layout = QHBoxLayout()
        layout1 = QVBoxLayout()
        layout2 = QVBoxLayout()
        layout3 = QVBoxLayout()

        label1_1 = QLabel("Rotation:")
        self.input1 = QLineEdit()
        self.input1.setValidator(QDoubleValidator())
        label1_2 = QLabel("deg")

        label2_1 = QLabel("Scaling:")
        self.input2 = QLineEdit()
        self.input2.setValidator(QDoubleValidator())
        label2_2 = QLabel()

        label3_1 = QLabel("Tx:")
        self.input3 = QLineEdit()
        self.input3.setValidator(QIntValidator())
        label3_2 = QLabel("pixel")
        
        label4_1 = QLabel("Ty:")
        self.input4 = QLineEdit()
        self.input4.setValidator(QIntValidator())
        label4_2 = QLabel("pixel")

        button = QPushButton("4. Transforms")

        layout1.addSpacing(3)
        layout1.addWidget(label1_1)
        layout1.addSpacing(9)
        layout1.addWidget(label2_1)
        layout1.addSpacing(9)
        layout1.addWidget(label3_1)
        layout1.addSpacing(9)
        layout1.addWidget(label4_1)
        layout1.setAlignment(Qt.AlignTop)

        layout2.addWidget(self.input1)
        layout2.addWidget(self.input2)
        layout2.addWidget(self.input3)
        layout2.addWidget(self.input4)
        layout2.setAlignment(Qt.AlignTop)

        layout3.addSpacing(3)
        layout3.addWidget(label1_2)
        layout3.addSpacing(9)
        layout3.addWidget(label2_2)
        layout3.addSpacing(9)
        layout3.addWidget(label3_2)
        layout3.addSpacing(9)
        layout3.addWidget(label4_2)
        layout3.setAlignment(Qt.AlignTop)
        
        layout.addLayout(layout1)
        layout.addSpacing(30)
        layout.addLayout(layout2)
        layout.addLayout(layout3)

        mainLayout.addLayout(layout)
        mainLayout.addSpacing(10)
        mainLayout.addWidget(button)
        mainLayout.setAlignment(Qt.AlignTop)

        self.setLayout(mainLayout)

        button.clicked.connect(self.transforms)

    def transforms(self):
        try:
            img = cv2.imread(self.filename1)
            rotation = float(self.input1.text()) if self.input1.text() != '' else 0
            scale = float(self.input2.text()) if self.input2.text() != '' else 1
            tx = int(self.input3.text()) if self.input3.text() != '' else 0
            ty = int(self.input4.text()) if self.input4.text() != '' else 0
            h = img.shape[0]
            w = img.shape[1]
            M = cv2.getRotationMatrix2D((240, 200), rotation, scale)
            img = cv2.warpAffine(img, M, (w, h))
            M = numpy.float32([[1, 0, tx], [0, 1, ty]])
            img = cv2.warpAffine(img, M, (w, h))
            cv2.imshow('transforms', img)
        except AttributeError as e:
            print(e)
            pass

if __name__ == '__main__':
    app = QApplication(sys.argv)
    MainWindow = MyWidget()
    MainWindow.show()
    sys.exit(app.exec_())
