import gradio as gr
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from unet_model import UNet
from huggingface_hub import hf_hub_download

weights_path=hf_hub_download(
    repo_id="kinsu2/unet-for-crack-concrete",
    filename"unet_weights_v2.pth"    
)
#initialize the model
model=UNet()
model.load_state_dict(torch.load(weights_path,map_location="cpu"))
model.eval()
#preprocessing:same as training
IMG_HEIGHT,IMG_WIDTH=128,128
transform=transform.Compose([
    transform.resize((IMG_HEIGHT,IMG_WIDTH)),
    transforms.ToTensor()
])
def predict(image):
    orig_w,orig_h=image.size#originalsize of uploaded image
    img=transform(image).unsqueeze(0)#(1,3,128,128)
    with torch.no_grad():
         pred=model(img)
        
    mask =pred.squeeze(0).squeeze(0).cpu().numpy()
    mask=(mask*255).astype(np.uint8)#grayscale mask
    
    #resize  back to original size
    mask_img=Image.fromarray(mask).resize((orig_w,orig_h),Image.NEAREST)
    return mask_img
#gradio interface
demo=gr.Interface(
    fn=predict,
    inputs=gr.Image(type='pil'),
    outputs=gr.Image(type='pil'),
    title="UNet Crack Segementation",
    description="upload a concrete surface image to get predicte crack mask"
    
)
if __name__=="__main__":
   demo.launch()