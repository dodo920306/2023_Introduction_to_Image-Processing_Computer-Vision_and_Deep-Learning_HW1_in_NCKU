from PyQt5.QtWidgets import QApplication, QPushButton, QVBoxLayout, QGroupBox
import sys, cv2, numpy

sobel_x = numpy.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
sobel_y = sobel_x.T

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

        button1.clicked.connect(self.sobel_x)
        button2.clicked.connect(self.sobel_y)
        button3.clicked.connect(self.combination_and_threshold)
        button4.clicked.connect(self.gradient_angle)

    def sobel_x(self):
        try:
            img = cv2.imread(self.filename1)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.GaussianBlur(img, (5, 5), 0)
            zero_padding = numpy.zeros((img.shape[0] + 2, img.shape[1] + 2))
            zero_padding[1:-1, 1:-1] = img

            for x in range(img.shape[0]):
                for y in range(img.shape[1]):
                    img[x, y] = (((zero_padding[x : x + 3, y : y + 3] * sobel_x).sum()) ** 2) ** 0.5
            cv2.imshow('sobel_x', cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX))
        except AttributeError as e:
            # Image not loaded.
            pass

    def sobel_y(self):
        try:
            img = cv2.imread(self.filename1)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.GaussianBlur(img, (5, 5), 0)
            zero_padding = numpy.zeros((img.shape[0] + 2, img.shape[1] + 2))
            zero_padding[1:-1, 1:-1] = img

            for x in range(img.shape[0]):
                for y in range(img.shape[1]):
                    img[x, y] = (((zero_padding[x : x + 3, y : y + 3] * sobel_y).sum()) ** 2) ** 0.5
            cv2.imshow('sobel_y', cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX))
        except AttributeError as e:
            # Image not loaded.
            pass

    def combination_and_threshold(self):
        try:
            img = cv2.imread(self.filename1)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.GaussianBlur(img, (5, 5), 0)
            zero_padding = numpy.zeros((img.shape[0] + 2, img.shape[1] + 2))
            zero_padding[1:-1, 1:-1] = img

            for x in range(img.shape[0]):
                for y in range(img.shape[1]):
                    res_x = (zero_padding[x : x + 3, y : y + 3] * sobel_x).sum()
                    res_y = (zero_padding[x : x + 3, y : y + 3] * sobel_y).sum()
                    img[x, y] = (res_x ** 2 + res_y ** 2) ** 0.5

            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
            cv2.imshow('combination', img)
            _, img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)
            cv2.imshow('threshold', img)
        except AttributeError as e:
            # Image not loaded.
            pass

    def gradient_angle(self):
        try:
            img = cv2.imread(self.filename1)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.GaussianBlur(img, (5, 5), 0)
            zero_padding = numpy.zeros((img.shape[0] + 2, img.shape[1] + 2))
            zero_padding[1:-1, 1:-1] = img
            gradient_angle = numpy.zeros_like(img, dtype='uint16')

            for x in range(img.shape[0]):
                for y in range(img.shape[1]):
                    res_x = (zero_padding[x : x + 3, y : y + 3] * sobel_x).sum()
                    res_y = (zero_padding[x : x + 3, y : y + 3] * sobel_y).sum()
                    img[x, y] = (res_x ** 2 + res_y ** 2) ** 0.5
                    gradient_angle[x, y] = (numpy.arctan2(res_y, res_x)) * 180 / numpy.pi + 180
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)

            mask1 = ((gradient_angle >= 120) & (gradient_angle <= 180)).astype(numpy.uint8) * 255
            mask2 = ((gradient_angle >= 210) & (gradient_angle <= 330)).astype(numpy.uint8) * 255

            cv2.imshow('Result 1', cv2.bitwise_and(img, mask1))
            cv2.imshow('Result 2', cv2.bitwise_and(img, mask2))
        except AttributeError as e:
            # Image not loaded.
            pass

if __name__ == '__main__':
    app = QApplication(sys.argv)
    MainWindow = MyWidget()
    MainWindow.show()
    sys.exit(app.exec_())
