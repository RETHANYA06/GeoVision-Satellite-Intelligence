import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from torchvision import transforms

from model import SatelliteCNN

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Classes
classes = [
    "AnnualCrop", "Forest", "HerbaceousVegetation",
    "Highway", "Industrial", "Pasture",
    "PermanentCrop", "Residential", "River", "SeaLake"
]

# Load model
model = SatelliteCNN(num_classes=10)
model.load_state_dict(torch.load("models/satellite_cnn.pth", map_location=device))
model.to(device)
model.eval()

# Hooks
features = None
gradients = None

def forward_hook(module, input, output):
    global features
    features = output

def backward_hook(module, grad_in, grad_out):
    global gradients
    gradients = grad_out[0]

# Attach hook to last conv layer
last_conv = model.conv_layers[-1]
last_conv.register_forward_hook(forward_hook)
last_conv.register_full_backward_hook(backward_hook)

# Transform
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

def generate_gradcam(image_path):

    # Load image
    image = Image.open(image_path).convert("RGB")
    original_image = np.array(image)

    img_tensor = transform(image).unsqueeze(0).to(device)
    img_tensor.requires_grad = True

    # Forward pass
    output = model(img_tensor)
    pred_class = output.argmax(dim=1).item()

    # Backward pass
    model.zero_grad()
    output[0, pred_class].backward()

    # Get gradients + features
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

    for i in range(features.shape[1]):
        features[:, i, :, :] *= pooled_gradients[i]

    # Heatmap
    heatmap = torch.mean(features, dim=1).squeeze()
    heatmap = torch.relu(heatmap)

    heatmap = heatmap.cpu().detach().numpy()
    heatmap = heatmap / (np.max(heatmap) + 1e-8)

    # Resize to original image size
    heatmap = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))

    # Convert to color map
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Overlay
    superimposed = cv2.addWeighted(original_image, 0.6, heatmap, 0.4, 0)

    # Show
    plt.figure(figsize=(6, 6))
    plt.imshow(superimposed)
    plt.title(f"Prediction: {classes[pred_class]}")
    plt.axis("off")
    plt.show()

    return classes[pred_class]
if __name__ == "__main__":
    print(
        generate_gradcam(
            "data/eurosat/EuroSAT/Forest/Forest_1.jpg"
        )
    )