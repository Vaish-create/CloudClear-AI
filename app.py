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

   
    input_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)
    output = output.squeeze(0).cpu().permute(1,2,0).numpy()

    output = np.clip(output,0,1)

    output = (output * 255).astype(np.uint8)

    output = Image.fromarray(output)
    output = output.resize(original_size)

    return output

interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Image(type="pil"),
    title="CloudClear AI",
    description="Upload a cloudy satellite image and generate a cloud-free prediction."
)

interface.launch()
