import gradio as gr
import torch
from PIL import Image
import numpy as np
import os

from model import UNet

device = torch.device("cpu")
model = UNet()
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.to(device)
model.eval()

def predict(image):
    original_size = image.size
    image_resized = image.convert("RGB").resize((256, 256))
    image_array = np.array(image_resized, dtype=np.float32) / 255.0

    input_tensor = torch.from_numpy(image_array)
    input_tensor = input_tensor.permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
    output = output.squeeze(0).cpu().permute(1, 2, 0).numpy()

    output = np.clip(output, 0, 1)
    output = (output * 255).astype(np.uint8)

    output = Image.fromarray(output)

    output = output.resize(original_size)

    return output

interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload Cloudy Satellite Image"),
    outputs=gr.Image(type="pil", label="Predicted Cloud-Free Image"),
    title="CloudClear AI",
    description="""
CloudClear AI uses a U-Net deep learning model to reconstruct cloud-free satellite images.

Upload a cloudy satellite image and click Submit.
""",
    flagging_mode="never"
)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))

    interface.launch(
        server_name="0.0.0.0",
        server_port=port,
        ssr_mode=False
    )
