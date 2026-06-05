import torch
from dataset import get_dataloaders
from model import SatelliteCNN

DATA_DIR = "data/eurosat/EuroSAT"

train_loader, test_loader, classes = get_dataloaders(DATA_DIR)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SatelliteCNN(num_classes=len(classes))
model.load_state_dict(torch.load("models/satellite_cnn.pth"))
model.to(device)

model.eval()

correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total

print(f"Accuracy: {accuracy:.2f}%")