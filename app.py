import gradio as gr
import torch
from torchvision import transforms 
from PIL import Image
import numpy as np
from unet_model import UNet 



weights_path = "unet_model.pth" 

# Initialize the model and load weights safely
model = UNet()
model.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
model.eval() # Set model to evaluation mode

# Preprocessing pipeline
IMG_HEIGHT, IMG_WIDTH = 128, 128
transform = transforms.Compose([
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
    transforms.ToTensor()
])

def predict(image):
    orig_w, orig_h = image.size
    img = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        pred = model(img)
        # Apply sigmoid activation function to model output within the no_grad context
        pred = torch.sigmoid(pred) 
        
    
    # Process the output tensor into a numpy mask
    mask = pred.squeeze().cpu().numpy()
    mask = (mask * 255).astype(np.uint8)
    
    # Resize mask back to original dimensions using 'L' mode for grayscale
    mask_img = Image.fromarray(mask, mode='L').resize((orig_w, orig_h), Image.NEAREST)
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
