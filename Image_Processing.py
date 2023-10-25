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
        self.setTitle("1. Image Processing")
        layout = QVBoxLayout()

        button1 = QPushButton("1.1 Color Separation")
        layout.addWidget(button1)

        button2 = QPushButton("1.2 Color Transformation")
        layout.addWidget(button2)

        button3 = QPushButton("1.3 Color Extraction")
        layout.addWidget(button3)

        self.setLayout(layout)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    MainWindow = MyWidget()
    MainWindow.show()
    sys.exit(app.exec_())
