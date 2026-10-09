import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.ToTensor(),
])

train_data = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_data = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

print(f"Train samples: {len(train_data)}")
print(f"Test samples: {len(test_data)}")

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28*28, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

model = MLP()
print(model)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

for epoch in range(3):
    total_loss=0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")


correct = 0
total = 0

test_loader = DataLoader(test_data, batch_size=64)

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted =  torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Test Accuracy: {100 * correct / total:.2f}%")

import matplotlib.pyplot as plt

model.eval()
images, labels = next(iter(test_loader))
outputs = model(images)
_, predicted = torch.max(outputs, 1)

wrong_idx = (predicted != labels).nonzero().flatten()

n = min(5, len(wrong_idx))
fig, axes = plt.subplots(1, n, figsize=(10, 2))
if n == 1:
    axes = [axes]
for i, idx in enumerate(wrong_idx[:5]):
    axes[i].imshow(images[idx][0], cmap="gray")
    axes[i].set_title(f"True: {labels[idx].item()}\nPred: {predicted[idx].item()}")
    axes[i].axis("off")
plt.tight_layout()
plt.savefig("wrong_predictions.png")
print("Saved wrong_predictions.png")