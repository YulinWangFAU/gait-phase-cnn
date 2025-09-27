#对 CNN 模型的预测结果进行可解释性分析，用 Grad-CAM 生成一张热力图，叠加到输入图像上，显示模型主要关注的区域
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from PIL import Image
from models.cnn_model import SimpleCNN

# Configuration
IMG_PATH = 'data/heatmaps/sample.png'  # Replace with your path
MODEL_PATH = 'models/best_model.pth'
SAVE_PATH = 'gradcam_result.png'
TARGET_LAYER = 'features'  # 'features' is default in SimpleCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# Load image
img = Image.open(IMG_PATH).convert('RGB')
input_tensor = transform(img).unsqueeze(0).to(device)

# Load model
model = SimpleCNN().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# Hook for gradients and activations
grads = None
activations = None

def backward_hook(module, grad_input, grad_output):
    global grads
    grads = grad_output[0]

def forward_hook(module, input, output):
    global activations
    activations = output

# Register hooks
for name, module in model.named_modules():
    if name == TARGET_LAYER:
        module.register_forward_hook(forward_hook)
        module.register_backward_hook(backward_hook)

# Forward pass
output = model(input_tensor)
class_idx = torch.argmax(output, dim=1).item()
score = output[:, class_idx]

# Backward pass
model.zero_grad()
score.backward()

# Compute Grad-CAM
weights = torch.mean(grads, dim=(2, 3), keepdim=True)
cam = torch.sum(weights * activations, dim=1).squeeze()
cam = F.relu(cam)
cam = cam.cpu().detach().numpy()

# Normalize and resize
cam -= cam.min()
cam /= cam.max()
cam = cv2.resize(cam, (128, 128))
heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)

# Overlay on original image
img_np = np.array(img.resize((128, 128)))
overlay = cv2.addWeighted(img_np, 0.5, heatmap, 0.5, 0)

# Save result
cv2.imwrite(SAVE_PATH, overlay[:, :, ::-1])  # RGB to BGR for OpenCV
print(f"Grad-CAM result saved to {SAVE_PATH}")
