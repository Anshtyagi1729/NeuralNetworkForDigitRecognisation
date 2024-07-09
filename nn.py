import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# Step 1: Define the Network Class
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

# Step 2: Prepare the MNIST Dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)

testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=64, shuffle=False)

# Step 3: Define Loss Function and Optimizer
criterion = nn.CrossEntropyLoss()
model = Net()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Step 4: Training Loop
epochs = 10
losses = []

for epoch in range(epochs):
    running_loss = 0.0
    for images, labels in trainloader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    epoch_loss = running_loss / len(trainloader)
    losses.append(epoch_loss)
    print(f"Epoch {epoch+1}, Loss: {epoch_loss}")

# Step 5: Evaluation Function
def evaluate_model(model, testloader):
    correct = 0
    total = 0
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        fig.suptitle('Sample Predictions', fontsize=16)
        for i, (images, labels) in enumerate(testloader):
            if i >= 10:
                break
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Plot the image
            image = images[0].cpu().numpy()
            label = labels[0].item()
            prediction = predicted[0].item()
            ax = axes[i // 5, i % 5]
            ax.imshow(np.squeeze(image), cmap='gray')
            ax.set_title(f'Predicted: {prediction}, Actual: {label}')
            ax.axis('off')

    accuracy = 100 * correct / total
    print(f'Accuracy of the network on the {total} test images: {accuracy:.2f}%')
    plt.tight_layout()
    plt.show()

# Step 6: Evaluate the Model
evaluate_model(model, testloader)
