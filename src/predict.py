import torch
from PIL import Image
from torchvision import transforms

from src.model import SatelliteCNN

classes = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake"
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SatelliteCNN(num_classes=len(classes))
model.load_state_dict(
    torch.load("models/satellite_cnn.pth", map_location=device)
)
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

def predict_image(image):

    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)

        probabilities = torch.softmax(outputs, dim=1)

        confidence, predicted = torch.max(probabilities, 1)

    return (
        classes[predicted.item()],
        confidence.item() * 100
    )