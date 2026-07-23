import gradio as gr
import torch
from PIL import Image
import numpy as np
from torchvision import transforms

from model import UNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = UNet()
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

def predict(image):
    # Placeholder
    return image

interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Image(type="pil"),
    title="CloudClear AI",
    description="Upload a cloudy satellite image and generate a cloud-free prediction."
)

interface.launch()
