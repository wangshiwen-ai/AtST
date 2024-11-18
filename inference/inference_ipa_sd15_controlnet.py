## controlnet openpose
from types import MethodType

import torch
from diffusers import StableDiffusionControlNetPipeline, DDIMScheduler, AutoencoderKL, ControlNetModel
from PIL import Image

from ip_adapter import IPAdapter

base_model_path = "runwayml/stable-diffusion-v1-5"
vae_model_path = "stabilityai/sd-vae-ft-mse"
image_encoder_path = "models/"
ip_ckpt = "models/ip-adapter_sd15.bin"
device = "cuda"

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid

noise_scheduler = DDIMScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    clip_sample=False,
    set_alpha_to_one=False,
    steps_offset=1,
)
vae = AutoencoderKL.from_pretrained(vae_model_path).to(dtype=torch.float16)

# load controlnet
controlnet_model_path = "lllyasviel/control_v11p_sd15_openpose"
controlnet = ControlNetModel.from_pretrained(controlnet_model_path, torch_dtype=torch.float16)
# load SD pipeline
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    base_model_path,
    controlnet=controlnet,
    torch_dtype=torch.float16,
    scheduler=noise_scheduler,
    vae=vae,
    feature_extractor=None,
    safety_checker=None
)

## Openpose
# read image prompt

image = Image.open("assets/images/yishuzi.png")
image.resize((256, 256))

# openpose_image = Image.open("assets/structure_controls/openpose.png")
openpose_image = Image.open("/home/swwang/IP-Adapter/output/Alibaba-PuHuiTi-Bold.png") 
openpose_image.resize((256, 384))

ip_model = IPAdapter(pipe, image_encoder_path, ip_ckpt, device)

images = ip_model.generate(pil_image=image, image=openpose_image, width=512, height=768, num_samples=4, num_inference_steps=50, seed=42)
grid = image_grid(images, 1, 4)

grid.save('demo_control.png')