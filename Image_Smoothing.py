from PyQt5.QtWidgets import QApplication, QDesktopWidget, QPushButton, QVBoxLayout, QGroupBox
import sys


class MyWidget(QGroupBox):
    def __init__(self):
        super().__init__()
       
        self.ui()
        screen_geo = QDesktopWidget().screenGeometry()
        widget_geo = self.geometry()
        x = (screen_geo.width() - widget_geo.width() // 2) // 2
        y = (screen_geo.height() - widget_geo.height() // 2) // 2
        self.move(x, y)

    def ui(self):
        self.setTitle("2. Image Smoothing")
        layout = QVBoxLayout()

        button1 = QPushButton("1.1 Gaussian blur")
        layout.addWidget(button1)

        button2 = QPushButton("2.2 Bilateral filter")
        layout.addWidget(button2)

        button3 = QPushButton("2.3 Median filter")
        layout.addWidget(button3)

        self.setLayout(layout)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    MainWindow = MyWidget()
    MainWindow.show()
    sys.exit(app.exec_())
