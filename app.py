import gradio as gr
import torch
from torchvision import transforms s
from PIL import Image
import numpy as np
from unet_model import UNet
from huggingface_hub import hf_hub_download


weights_path = hf_hub_download(
    repo_id="kinsu2/unet-for-crack-concrete",
    filename="unet_weights_v2.pth"    
)

# Initialize the model
model = UNet()
model.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
model.eval()

# Preprocessing
IMG_HEIGHT, IMG_WIDTH = 128, 128
# FIX 3: Changed 'transform.Compose' to 'transforms.Compose'
# FIX 4: Changed 'transform.resize' to 'transforms.Resize'
transform = transforms.Compose([
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
    transforms.ToTensor()
])

def predict(image):
    orig_w, orig_h = image.size
    img = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        pred = model(img)
        
    
    mask = pred.squeeze().cpu().numpy()
    mask = (mask * 255).astype(np.uint8)
    
    
    mask_img = Image.fromarray(mask).resize((orig_w, orig_h), Image.NEAREST)
    return mask_img

# Gradio interface
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type='pil'),
    outputs=gr.Image(type='pil'),
    title="UNet Crack Segmentation",
    description="Upload a concrete surface image to get predicted crack mask"
)

if __name__ == "__main__":
    demo.launch()
