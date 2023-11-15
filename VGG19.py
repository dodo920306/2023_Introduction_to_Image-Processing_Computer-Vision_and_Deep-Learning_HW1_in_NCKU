from PyQt5.QtWidgets import QApplication, QPushButton, QVBoxLayout, QGroupBox, QLabel, QFileDialog
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
from PIL import Image
import sys, torch, os, cv2
from torchvision.transforms import v2
from torchvision import models
from torchvision.datasets import CIFAR10
from torchsummary import summary
import matplotlib.pyplot as plt
from collections import OrderedDict


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.vgg19_bn(num_classes=10).to(device)

transforms = v2.Compose([
    v2.RandomHorizontalFlip(),
    v2.RandomVerticalFlip(),
    v2.RandomRotation(30),
])

to_tensor = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True)
])


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
        self.label = QLabel("Predict = ")
        self.image = QLabel()
        self.image.setFixedHeight(128)


        layout.addWidget(button0)
        layout.addSpacing(10)
        layout.addWidget(button1)
        layout.addSpacing(10)
        layout.addWidget(button2)
        layout.addSpacing(10)
        layout.addWidget(button3)
        layout.addSpacing(10)
        layout.addWidget(button4)
        layout.addWidget(self.label)
        layout.addWidget(self.image)
        layout.setAlignment(Qt.AlignTop)

        self.setLayout(layout)

        button0.clicked.connect(self.load_image)
        button1.clicked.connect(self.show_augmented_img)
        button2.clicked.connect(self.show_model_structure)
        button3.clicked.connect(self.show_acc_and_loss)
        button4.clicked.connect(self.inference)

    def show_augmented_img(self):
        _, axes = plt.subplots(3, 3, figsize=(5, 5))
        for (filename, ax) in zip(os.listdir('Q5_image/Q5_1/'), axes.ravel()):
            with Image.open('Q5_image/Q5_1/' + filename) as img:
                ax.imshow(transforms(img))
                ax.set_title(filename.split('.')[0])

        plt.tight_layout()
        plt.show()

    def show_model_structure(self):
        summary(model, (3, 32, 32))

    def show_acc_and_loss(self):
        try:
            img = cv2.imread("training_results.png")
            cv2.imshow("training results", img)
        except cv2.error:
            pass

    def inference(self):
        try:
            classes = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
            model.load_state_dict(torch.load("best_model_weights.pth"))
            model.eval()
            img = Image.open(self.filename)
            transform = v2.Compose([
                transforms,
                to_tensor,
            ])

            with torch.no_grad():
                output = model(transform(img).to(device).unsqueeze(0))

            _, predicted = output.max(1)
            self.label.setText("Predict = " + str(classes[predicted.item()]))
            plt.figure(figsize=(8, 6))
            plt.bar(classes, torch.nn.functional.softmax(output[0], dim=0).cpu())
            plt.xlabel('Class')
            plt.ylabel('Probability')
            plt.title('probability of each class')
            plt.xticks(rotation=45)
            plt.show()
        except AttributeError:
            pass
        except FileNotFoundError:
            pass

    def load_image(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        filename, _ = QFileDialog.getOpenFileName(self, "Select an Image", "", "Images (*.png *.jpg)", options=options)
        if filename != "":
            self.filename = filename
            pixmap = QPixmap(self.filename)
            self.image.setPixmap(pixmap.scaled(128, 128))


if __name__ == '__main__':
    x = 0
    while x == 0:
        x = input("Show widget or training VGG19?\nType 1 for showing widget or 2 for training VGG19: ")
        if x == '1':
            app = QApplication(sys.argv)
            MainWindow = MyWidget()
            MainWindow.show()
            sys.exit(app.exec_())
        elif x == '2':
            # VGG19
            transforms = v2.Compose([
                transforms,
                to_tensor,
            ])

            trainset = CIFAR10(root='./data', train=True, download=True, transform=transforms)
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=4)

            testset = CIFAR10(root='./data', train=False, download=True, transform=transforms)
            testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=4)

            # Cross entropy loss is commonly used as a loss function in classification problems.
            # It measures the difference between the predicted probability distribution (output of the neural network) 
            # and the true probability distribution (the ground truth labels). 
            # The goal during training is to minimize this loss.
            criterion = torch.nn.CrossEntropyLoss()

            # stochastic gradient descent (SGD) optimizer.
            # model.parameters() specifies the parameters (weights and biases) of the neural network that 
            # the optimizer will update during training.
            # lr sets the learning rate, which determines the step size that the optimizer takes during the optimization process.
            # A smaller learning rate can result in slower but more accurate convergence, 
            # while a larger learning rate can speed up the convergence but may lead to overshooting.
            # momentum adds a momentum term to the update, which helps accelerate the optimization in the relevant direction and dampens oscillations.
            # Momentum is a hyperparameter that improves the convergence speed and helps escape local minima.
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

            num_epochs = 80
            best_accuracy = 0.0
            best_model_weights = OrderedDict()
            train_losses, train_accuracies, val_losses, val_accuracies = [], [], [], []

            for epoch in range(num_epochs):
                # train
                model.train()

                running_loss = 0.0
                correct = 0
                total = 0
                for inputs, labels in trainloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    
                    # Before computing the gradients for a new minibatch, 
                    # it is common to zero out the gradients from the previous minibatch.
                    # This is done to prevent the gradients from accumulating across multiple batches.
                    # If you don't zero out the gradients, the new gradients will be added to the existing gradients, 
                    # and it might lead to incorrect updates during optimization.
                    optimizer.zero_grad()

                    # Obtain the model's predictions (outputs) for the input data.
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    # Compute the gradients of the loss with respect to the model parameters, enabling backpropagation.
                    loss.backward()

                    # Update the model parameters using the optimizer and the computed gradients.
                    optimizer.step()

                    # .item() converts the result to a Python number.
                    running_loss += loss.item()
                    # Extracts the class predictions by selecting the index with the maximum value
                    # returns a tuple containing two tensors: 
                    # the maximum value along dimension 1 (the predicted class scores) 
                    # and the index of the maximum value (the predicted class labels).
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()

                # average training loss
                train_loss = running_loss / len(trainloader)
                train_accuracy = 100.0 * correct / total

                # test
                model.eval()
                running_loss = 0.0
                correct = 0
                total = 0

                # PyTorch operations won't build up a computation graph for automatic differentiation.
                # This is done for efficiency during evaluation since you don't need to compute gradients for the parameters.
                with torch.no_grad():
                    for inputs, labels in testloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                        running_loss += loss.item()
                        _, predicted = outputs.max(1)
                        total += labels.size(0)
                        correct += predicted.eq(labels).sum().item()
                val_loss = running_loss / len(testloader)
                val_accuracy = 100.0 * correct / total

                # save
                print(f"Epoch {epoch+1}/{num_epochs}, "
                    f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, "
                    f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")
                
                train_losses.append(train_loss)
                train_accuracies.append(train_accuracy)
                val_losses.append(val_loss)
                val_accuracies.append(val_accuracy)

                if val_accuracy > best_accuracy:
                    best_accuracy = val_accuracy
                    best_model_weights = model.state_dict()
                    
            torch.save(best_model_weights, "best_model_weights.pth")
            
            plt.figure(figsize=(10, 5))
            plt.subplot(2, 1, 1)
            plt.plot(train_losses, label="Train Loss")
            plt.plot(val_losses, label="Val Loss")
            plt.xlabel("epoch")
            plt.ylabel("loss")
            plt.legend()
            plt.title("Loss")

            plt.subplot(2, 1, 2)
            plt.plot(train_accuracies, label="Train Acc")
            plt.plot(val_accuracies, label="Val Acc")
            plt.xlabel("epoch")
            plt.ylabel("accuracy(%)")
            plt.legend()
            plt.title("Accuracy")

            plt.tight_layout()
            plt.savefig("training_results.png")
        else:
            print("Invalid input.")
            x = 0
