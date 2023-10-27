from PyQt5.QtWidgets import QApplication, QHBoxLayout, QPushButton, QVBoxLayout, QGroupBox, QLineEdit, QLabel
from PyQt5.QtGui import QIntValidator, QDoubleValidator
from PyQt5.QtCore import Qt
import sys


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
        input1 = QLineEdit()
        input1.setValidator(QDoubleValidator())
        label1_2 = QLabel("deg")

        label2_1 = QLabel("Scaling:")
        input2 = QLineEdit()
        input2.setValidator(QDoubleValidator())
        label2_2 = QLabel()

        label3_1 = QLabel("Tx:")
        input3 = QLineEdit()
        input3.setValidator(QIntValidator())
        label3_2 = QLabel("pixel")
        
        label4_1 = QLabel("Ty:")
        input4 = QLineEdit()
        input4.setValidator(QIntValidator())
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


        layout2.addWidget(input1)
        layout2.addWidget(input2)
        layout2.addWidget(input3)
        layout2.addWidget(input4)
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

if __name__ == '__main__':
    app = QApplication(sys.argv)
    MainWindow = MyWidget()
    MainWindow.show()
    sys.exit(app.exec_())
