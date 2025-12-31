import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from emnist import extract_training_samples, extract_test_samples

# =========================
# CONFIG
# =========================
BATCH_SIZE = 128
EPOCHS = 10
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATASET = "balanced"

print("Using device:", DEVICE)

# =========================
# LOAD EMNIST
# =========================
X_train, y_train = extract_training_samples(DATASET)
X_test, y_test = extract_test_samples(DATASET)

# Normalize
X_train = X_train.astype(np.float32) / 255.0
X_test  = X_test.astype(np.float32) / 255.0

# PyTorch expects N×C×H×W
X_train = torch.tensor(X_train).unsqueeze(1)
X_test  = torch.tensor(X_test).unsqueeze(1)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test  = torch.tensor(y_test, dtype=torch.long)

train_loader = DataLoader(
    TensorDataset(X_train, y_train),
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_loader = DataLoader(
    TensorDataset(X_test, y_test),
    batch_size=BATCH_SIZE,
    shuffle=False
)

# =========================
# MODEL
# =========================
class EMNIST_CNN(nn.Module):
    def __init__(self, num_classes=47):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

model = EMNIST_CNN().to(DEVICE)

# =========================
# TRAINING
# =========================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for X, y in train_loader:
        X, y = X.to(DEVICE), y.to(DEVICE)

        optimizer.zero_grad()
        out = model(X)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS} - loss: {total_loss:.4f}")

# =========================
# EVALUATION
# =========================
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for X, y in test_loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        out = model(X)
        pred = out.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)

acc = correct / total
print(f"Test accuracy: {acc:.4f}")

# =========================
# SAVE MODEL
# =========================
torch.save(model.state_dict(), "emnist_cnn_balanced.pth")
print("Model saved as emnist_cnn_balanced.pth")
