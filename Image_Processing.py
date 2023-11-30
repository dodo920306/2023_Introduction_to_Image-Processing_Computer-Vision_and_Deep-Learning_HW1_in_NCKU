from PyQt5.QtWidgets import QApplication, QPushButton, QVBoxLayout, QGroupBox, QDesktopWidget
import sys, cv2, numpy


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

        button1.clicked.connect(self.color_separation)
        button2.clicked.connect(self.color_transformation)
        button3.clicked.connect(self.color_extraction)

    def color_separation(self):
        try:
            img = cv2.imread(self.filename1)
            b, g, r = cv2.split(img)
            zeros = numpy.zeros(img.shape[:2], dtype = "uint8")
            cv2.imshow('R', cv2.merge([zeros, zeros, r]))
            cv2.imshow('G', cv2.merge([zeros, g, zeros]))
            cv2.imshow('B', cv2.merge([b, zeros, zeros]))
            # img = cv2.imread(self.filename2)
            # b, g, r = cv2.split(img)
            # cv2.imshow('R2', cv2.merge([zeros, zeros, r]))
            # cv2.imshow('G2', cv2.merge([zeros, g, zeros]))
            # cv2.imshow('B2', cv2.merge([b, zeros, zeros]))
        except AttributeError as e:
            # Image not loaded.
            pass

    def color_transformation(self):
        try:
            img = cv2.imread(self.filename1)
            cv2.imshow('cv2.COLOR_BGR2GRAY for img 1', cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
            b, g, r = cv2.split(img)
            avg = b / 3 + g / 3 + r / 3
            cv2.imshow('Average weighted for img 1', avg)
        except AttributeError:
            # Image not loaded.
            pass

    def color_extraction(self):
        try:
            img = cv2.imread(self.filename1)
            mask = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(mask, (15, 20, 25), (86, 255, 255))
            cv2.imshow('mask for img 1', mask)
            # cv2.bitwise_not(src[, dst[, mask]]) 會在 dst 上的 mask 上白色 (0) 的位置進行 src 的 not 操作
            # Since the src and the mask are the same picture, 
            # this line simply make the white part on the mask black and put it on the img.
            cv2.imshow('img 1 with out yellow and green', cv2.bitwise_not(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), img, mask=mask))

            # img = cv2.imread(self.filename2)
            # mask = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            # mask = cv2.inRange(mask, (15, 20, 25), (86, 255, 255))
            # cv2.imshow('mask for img 2', mask)
            # # cv2.bitwise_not(src[, dst[, mask]]) 會在 dst 上的 mask 上白色 (0) 的位置進行 src 的 not 操作
            # # Since the src and the mask are the same picture, 
            # # this line simply make the white part on the mask black and put it on the img.
            # cv2.imshow('img 2 with out yellow and green', cv2.bitwise_not(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), img, mask=mask))
        except AttributeError:
            # Image not loaded.
            pass


if __name__ == '__main__':
    app = QApplication(sys.argv)
    MainWindow = MyWidget()
    MainWindow.show()
    sys.exit(app.exec_())
