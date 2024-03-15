import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class VGG16Model(nn.Module):
    def __init__(self, num_classes=2):
        super(VGG16Model, self).__init__()
        # Load a pre-trained VGG16 model
        self.vgg16 = models.vgg16(pretrained=True)
        # Replace the classifier to fit the number of classes
        self.vgg16.classifier[6] = nn.Linear(4096, num_classes)

    def forward(self, x):
        return self.vgg16(x)

    def train_model(self, train_loader, criterion, optimizer, num_epochs=10):
        self.train()  # Set the model to training mode
        for epoch in range(num_epochs):
            running_loss = 0.0
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}")

    def test_model(self, test_loader):
        self.eval()  # Set the model to evaluation mode
        total = 0
        correct = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = self(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print(f'Accuracy on the test set: {100 * correct / total}%')


# Example usage:
num_classes = 10  # Example: CIFAR-10
vgg16_model = VGG16Model(num_classes=num_classes)

# Assuming CIFAR-10 dataset is being used
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(vgg16_model.parameters(), lr=0.001, momentum=0.9)

# Train and test the model
vgg16_model.train_model(train_loader, criterion, optimizer, num_epochs=10)
vgg16_model.test_model(test_loader)