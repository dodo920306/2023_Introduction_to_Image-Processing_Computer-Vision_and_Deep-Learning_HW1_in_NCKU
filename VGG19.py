from PyQt5.QtWidgets import QApplication, QPushButton, QVBoxLayout, QGroupBox, QLabel, QFileDialog
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
from PIL import Image
import sys, torch, os
from torchvision.transforms import v2
from torchvision import models
from torchvision.datasets import CIFAR10
from torchsummary import summary
import matplotlib.pyplot as plt


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
        button3 = QPushButton("5.3 Show Acc and Loss")
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
        button1.clicked.connect(self.show_augmented_img)
        button2.clicked.connect(self.show_model_structure)

    def show_augmented_img(self):
        _, axes = plt.subplots(3, 3, figsize=(5, 5))
        for (filename, ax) in zip(os.listdir('Q5_image/Q5_1/'), axes.ravel()):
            with Image.open('Q5_image/Q5_1/' + filename) as img:
                transforms = v2.Compose([
                    v2.RandomHorizontalFlip(),
                    v2.RandomVerticalFlip(),
                    v2.RandomRotation()
                ])
                ax.imshow(transforms(img))
                ax.set_title(filename.split('.')[0])

        plt.tight_layout()
        plt.show()

    def show_model_structure(self):
        model = models.vgg19_bn(num_classes=10)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        summary(model, (3, 32, 32))

    def show_acc_and_loss(self):
        # transforms = v2.Compose([
        #     v2.RandomHorizontalFlip(p = 0.5),
        #     v2.RandomVerticalFlip(p=0.5),
        #     v2.RandomRotation(30)
        # ])
        
        # trainset = CIFAR10(root='./data', train=True, download=True, transform=transforms)
        # trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=4)

        # testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms)
        # testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=4)

        # model = models.vgg19_bn(num_classes=10)
        # model.train()

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
