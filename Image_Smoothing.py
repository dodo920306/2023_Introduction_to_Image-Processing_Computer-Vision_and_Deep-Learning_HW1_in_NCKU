from PyQt5.QtWidgets import QApplication, QPushButton, QVBoxLayout, QGroupBox
import sys, cv2

def gaussian_blur(m):
    radius = 2 * m + 1
    blurred_image = cv2.GaussianBlur(img, (radius, radius), 0)
    cv2.imshow('gaussian_blur', blurred_image)

def bilateral_filter(m):
    blurred_image = cv2.bilateralFilter(img, 2 * m + 1, 90, 90)
    cv2.imshow('bilateral_filter', blurred_image)

def median_filter(m):
    blurred_image = cv2.medianBlur(img, 2 * m + 1)
    cv2.imshow('median_filter', blurred_image)

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
        button1.clicked.connect(self.gaussian_blur)
        button2.clicked.connect(self.bilateral_filter)
        button3.clicked.connect(self.median_filter)

    def gaussian_blur(self):
        try:
            try:
                # Ensure the window has been closed before.
                cv2.destroyWindow('gaussian_blur')
            except cv2.error as e:
                pass
            global img
            img = cv2.imread(self.filename1)
            cv2.namedWindow('gaussian_blur')
            cv2.createTrackbar('m:', 'gaussian_blur', 1, 5, gaussian_blur)
            gaussian_blur(1)
        except AttributeError as e:
            # Image not loaded.
            pass

    def bilateral_filter(self):
        try:
            try:
                # Ensure the window has been closed before.
                cv2.destroyWindow('bilateral_filter')
            except cv2.error as e:
                pass
            global img
            img = cv2.imread(self.filename1)
            cv2.namedWindow('bilateral_filter')
            cv2.createTrackbar('m:', 'bilateral_filter', 1, 5, bilateral_filter)
            bilateral_filter(1)
        except AttributeError as e:
            # Image not loaded.
            pass

    def median_filter(self):
        try:
            try:
                # Ensure the window has been closed before.
                cv2.destroyWindow('median_filter')
            except cv2.error as e:
                pass
            global img
            img = cv2.imread(self.filename2)
            cv2.namedWindow('median_filter')
            cv2.createTrackbar('m:', 'median_filter', 1, 5, median_filter)
            median_filter(1)
        except AttributeError as e:
            # Image not loaded.
            pass

if __name__ == '__main__':
    app = QApplication(sys.argv)
    MainWindow = MyWidget()
    MainWindow.show()
    sys.exit(app.exec_())
