from PyQt5.QtWidgets import QApplication, QWidget, QDesktopWidget, QPushButton
import sys


class MyWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Hw1")
        screen_geo = QDesktopWidget().screenGeometry()
        widget_geo = self.geometry()
        x = (screen_geo.width() - widget_geo.width() // 2) // 2
        y = (screen_geo.height() - widget_geo.height() // 2) // 2
        self.move(x, y)
        self.ui()

    def ui(self):
        pushButton = QPushButton(self)
        pushButton.setGeometry(100, 70, 113, 32)
        pushButton.setText("Hello World!")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    MainWindow = MyWidget()
    MainWindow.show()
    sys.exit(app.exec_())
