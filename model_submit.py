#### This is a test problem  for admission  to summer research camp in University of Dallas Computer Science Departmeent AI lab ###
#### Kyle Lin 03/16/2026 ####

import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import pandas as pd
import os
import numpy as np
from os import walk
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from IPython.display import display, clear_output

### Files are unzipped and saved to my local drive
rootdir = "C:\\UTD\\vehicle_classification\\vehicle_classification_data"

### Upsampling to 224X224 is required for EfficientNet-B0 or RESNET. Otherwise use the original 64X64 resolution #####
# width = 224
# height = 224
width = 64
height = 64

### Import packages and load images

transformer = transforms.Compose(
    [
        transforms.Resize((width, height)),
        transforms.RandomHorizontalFlip(),  ### Argumentation to reduce overfitting ###
        transforms.RandomRotation(15),  ### Argumentation to reduce overfitting ###
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

dataset = datasets.ImageFolder(
    root=rootdir,
    transform=transformer,
)
print(len(dataset))
### 26378 total ####
dataset_size = len(dataset)

#### Split data into 80% training, 20% test #####
train_size = int(0.8 * dataset_size)
test_size = dataset_size - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

### Create data loader ####
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
print(f"Total images: {len(dataset)}")
print(f"Training images: {len(train_dataset)}")
print(f"Testing images: {len(test_dataset)}")

# for images, labels in test_loader:
# print(images.shape)
# print(labels)


### Check label dictionary to make sure it is correct #####
print(dataset.class_to_idx)
### {'Bicycle': 0, 'Bus': 1, 'Car': 2, 'Motorcycle': 3, 'NonVehicles': 4, 'Taxi': 5, 'Truck': 6, 'Van': 7} ###


class SimpleCNN1(nn.Module):
    def __init__(self, num_classes=2):
        super(SimpleCNN1, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(
            64 * 8 * 8, 128
        )  # depends on input size (64x64 -> 8x8 after pooling)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # [batch, 16, 32, 32]
        x = self.pool(F.relu(self.conv2(x)))  # [batch, 32, 16, 16]
        x = self.pool(F.relu(self.conv3(x)))  # [batch, 64, 8, 8]
        x = x.view(-1, 64 * 8 * 8)  # flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # output logits
        return x


class SimpleCNN2(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN2, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),  # 64x64
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32x32
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 16x16
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 8x8
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),  # 1x1
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),  ### reduce overfitting
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class SimpleCNN3(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32x32
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 16x16
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 8x8
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model = SimpleCNN1(num_classes=8).to(device)
model = SimpleCNN2(num_classes=8).to(device)
## model = SimpleCNN3(num_classes=8).to(device)

""" 
#### Try Pretrained Models ######
#### 83.34% accuracy using Efficientnet_B0 model, but it took me 2 hours to run the training loop ###

# model = models.efficientnet_b0(pretrained=True)

### Replace classifier
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 8)

#### Freeze Backbone ######
for param in model.parameters():
    param.requires_grad = False

for param in model.classifier.parameters():
    param.requires_grad = True """

""" #### Try pretrained RESNET18 model, re-train the last layer, it took 5 hours to finish #####
model = models.resnet18(pretrained=True)
model.conv1 = nn.Conv2d(
    3, 64, kernel_size=3, stride=1, padding=1, bias=False
)  ### Do this for small images
model.maxpool = nn.Identity()
### Freeze layers except for final layer
for param in model.parameters():
    param.requires_grad = False

for param in model.fc.parameters():
    param.requires_grad = True

for param in model.layer4.parameters():
    param.requires_grad = True """


### Define Loss + Optimizer
criterion = nn.CrossEntropyLoss()  # good for classification
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


""" 
##### Traning Loop without early stopping ########
num_epochs = 50

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        #### 1. Softmax - > 2. loss = -log(probability_of_correct_class)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}") """


#### Testing / inference, Evaluate Model performance by checking #Correct/#Test
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return total_loss / len(loader), correct / total


#### Add early stopping logic to the training code, stop if loss does not drop after 5 epochs  #########
train_losses = []
train_accuracies = []
val_losses = []
test_accuracies = []
patience = 5
best_val_loss = float("inf")
counter = 0

num_epochs = 50  # large number, then use early stopping logc to stop

for epoch in range(num_epochs):
    model.train()
    running_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = correct / total

    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_acc)

    ### evaluate on validation set
    val_loss, test_acc = evaluate(model, test_loader, criterion, device)
    val_losses.append(val_loss)
    test_accuracies.append(test_acc)

    clear_output(wait=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Loss
    ax1.plot(train_losses, label="Train")
    ax1.plot(val_losses, label="Val")
    ax1.set_title("Loss")
    ax1.legend()

    # Accuracy
    ax2.plot(train_accuracies, label="Train")
    ax2.plot(test_accuracies, label="Test")
    ax2.set_title("Accuracy")
    ax2.legend()

    display(fig)
    plt.close(fig)

    print(f"Epoch {epoch+1}, Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}")

    ###cEarly stopping logic
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        counter = 0
        torch.save(model.state_dict(), "best_model.pth")  # save best model
    else:
        counter += 1
        print(f"No improvement. Counter: {counter}/{patience}")

        if counter >= patience:
            print("Early stopping triggered!")
            break


""" ### Plot curves
plt.figure()
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.title("Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

plt.figure()
plt.plot(train_accuracies, label="Train Acc")
plt.plot(test_accuracies, label="Test Acc")
plt.title("Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show() """


### Load the best model, and re-calculate the accuracy on test dataset and training dataset ###
model.load_state_dict(torch.load("best_model.pth"))

model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Test Accuracy: {100 * correct / total:.2f}%")

model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Train Accuracy: {100 * correct / total:.2f}%")
