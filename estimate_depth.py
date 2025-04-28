from PIL import Image
import torch
import matplotlib.pyplot as plt
import cv2
import numpy as np
from torchvision import transforms

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
midas = torch.hub.load("intel-isl/MiDaS", "DPT_Large")
midas.to(device).eval()

# Correct transform for DPT_Large
transform = transforms.Compose([
    transforms.Resize(384),
    transforms.CenterCrop(384),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Load image
img = Image.open("/home/diligent/Downloads/rgb.png").convert("RGB")

# Apply transform
input_batch = transform(img).unsqueeze(0).to(device)

# Inference
with torch.no_grad():
    prediction = midas(input_batch)
    prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(1),
        size=img.size[::-1],
        mode="bicubic",
        align_corners=False,
    ).squeeze()

depth = prediction.cpu().numpy()
depth_vis = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# Show
plt.imshow(depth_vis, cmap='plasma')
cv2.imwrite("/home/diligent/Downloads/depth.png", depth_vis)
plt.axis('off')
plt.show()
