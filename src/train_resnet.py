import torch
import torch.nn as nn
import torch.optim as optim

from dataset import get_dataloaders
from model_resnet import SatelliteResNet

DATA_DIR = "data/eurosat/EuroSAT"

train_loader, test_loader, classes = get_dataloaders(
    DATA_DIR,
    batch_size=32
)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = SatelliteResNet(
    num_classes=len(classes)
).to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.0001
)

EPOCHS = 5

for epoch in range(EPOCHS):

    model.train()

    running_loss = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    print(
        f"Epoch [{epoch+1}/{EPOCHS}] "
        f"Loss: {running_loss/len(train_loader):.4f}"
    )

torch.save(
    model.state_dict(),
    "models/resnet18_satellite.pth"
)

print("ResNet18 model saved!")