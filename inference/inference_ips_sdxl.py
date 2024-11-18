from diffusers import StableDiffusionPipeline
from PIL import Image, ImageDraw, ImageFont
import torch
import torch
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipelineLegacy, DDIMScheduler, AutoencoderKL,StableDiffusionXLPipeline, \
    StableDiffusionXLImg2ImgPipeline
from PIL import Image

from ip_adapter import IPAdapter
from ip_adapter import IPAdapterPlusXL, IPAdapterXL
import os
import numpy as np
import cv2

base_model_path = "sdxl_models"
# bash_model_path = '/home/swwang/huggingface/hub/models--runwayml--stable-diffusion-v1-5'
# vae_model_path = "stabilityai/sd-vae-ft-mse"
image_encoder_path = "vit"
ip_ckpt = "pretrained_ip/sdxl_models/ip-adapter-plus_sdxl_vit-h.bin"

from modelscope.utils.constant import Tasks
from modelscope.pipelines import pipeline
import cv2


# exit()

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--ip_ckpt', default="pretrained_ip/sdxl_models/ip-adapter-plus_sdxl_vit-h.bin", type=str)
# parser.add_argument('--ip_ckpt', default='/home/swwang/IP-Adapter/sd-idapter-2/checkpoint-2000/ip_adapter.bin', type=str)
parser.add_argument('--style_path', default="assets/images/chengzi.png", type=str)
parser.add_argument('--img_path', default=["/home/swwang/IP-Adapter/output/Alibaba-PuHuiTi-Bold.png"], type=list)
parser.add_argument('--font_name', default='Alibaba-PuHuiTi-Bold', type=str)
parser.add_argument('--text_prompt', default='', type=str)
parser.add_argument('--char', default='爱')
args = parser.parse_args()

ip_ckpt = args.ip_ckpt
device = "cuda"

def generate_single_char_image(char, font_path):
    image = Image.new('RGB', (512, 512), 'white')
    font = ImageFont.truetype(font_path, 400)  # Adjust the font size as needed
    draw = ImageDraw.Draw(image)
    (left, top, right, bottom) = draw.textbbox((0, 0), char, font=font)
    text_width = max(right-left,5)
    text_height = max(bottom-top,5)
    position = ((512 - text_width) // 2 -left, (512 - text_height) // 2 - top)
    draw.text(position, char, font=font, fill='black')
    # Convert PIL image to NumPy array
    image_np = np.array(image)
    # Identify black regions (assuming black is [0, 0, 0])
    black_mask = np.all(image_np == [0, 0, 0], axis=-1)
    # Generate random noise
    noise = np.random.randint(0, 250, image_np.shape, dtype=np.uint8)
    # Add noise only to black regions
    # image_np[black_mask] = np.clip(image_np[black_mask] + noise[black_mask], 0, 255)
    
    # Convert NumPy array back to PIL image
    # image = Image.fromarray(image_np+noise)
    # image.save('demo.png')
    # image_np = np.array(image)
    # noise = np.random.randint(0, 50, image_np.shape, dtype=np.uint8)
    # image = Image.fromarray(np.clip(noise+image, 0, 255))
    return image
 

from skimage.morphology import skeletonize
from skimage.util import invert
from skimage import morphology


def skeleton_image(image):
    image_np = np.array(image)
    # Convert to grayscale
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    # Threshold the image to binary
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    # Invert the binary image
    binary[binary==255] = 1
    skel, distance = morphology.medial_axis(binary, return_distance=True)
    dist_on_skel = distance * skel
    skeleton = dist_on_skel.astype(np.uint8) * 255
    # binary_inverted = invert(binary)
    # # Perform skeletonization
    # skeleton = skeletonize(binary_inverted)
    
    # Convert skeleton to 8-bit image
    # skeleton = (skeleton * 255).astype(np.uint8)

    skeleton = cv2.cvtColor(skeleton, cv2.COLOR_GRAY2RGB)
    # Convert numpy array back to PIL image
    skeleton_image = Image.fromarray(cv2.bitwise_not(skeleton))
    
    return skeleton_image

def process_image(image):
    # Convert PIL image to numpy array
    image_np = np.array(image)
    # Convert to grayscale
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    # Perform edge detection using Canny
    edges = cv2.Canny(gray, 100, 200)
    # Convert edges to 3-channel image
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    # Convert numpy array back to PIL image
    edge_image = Image.fromarray(cv2.bitwise_not(edges_colored))
    
    return edge_image

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

# vae = AutoencoderKL.from_pretrained(vae_model_path).to(dtype=torch.float16)

# pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
#     base_model_path,
#     torch_dtype=torch.float16,
#     scheduler=noise_scheduler,
#     vae=vae,
#     feature_extractor=None,
#     safety_checker=None
# )

# load SDXL pipeline
pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    'sdxl_models',
    torch_dtype=torch.float16,
    add_watermarker=False,
)

# task = Tasks.image_to_image_generation
# pipe = pipeline(task=task,
#                 model=base_model_path,
#                 use_safetensors=True,
#                 model_revision='v1.0.0',).pipeline

# pipe.to(torch.float16)

image = Image.open(args.style_path)  ## Style
# ip_model = IPAdapter(pipe, image_encoder_path, ip_ckpt, device)
ip_model = IPAdapterXL(pipe, image_encoder_path, ip_ckpt, device)

import os
s_list = [0.75, 0.8, 0.85, 0.9, 1.0]


name = '_'.join([os.path.basename(args.style_path).split('.')[0] , args.ip_ckpt.replace('/', '_')])

os.makedirs(f'exp/{name}', exist_ok=True)

g_image = generate_single_char_image(args.char, f'AnyText/font/{args.font_name}.ttf')
print("Generate char {} in font {} done.".format(args.char, args.font_name))

for s in s_list:
    images = ip_model.generate(pil_image=image, num_samples=4, num_inference_steps=50, seed=42, image=g_image, strength=s, \
                              prompt='''A calligraphic character "{}", with elements about "{}". Details as “{}”. The overall composition is harmonious and aesthetically pleasing.'''.format(args.char, args.char, args.char+args.text_prompt) )
    
    images = [g_image] + images
    grid = image_grid(images, 1, 5)
    output_path = f'exp/{name}/{args.char}_s_{s}_{args.font_name}.png'
    grid.save(output_path)
    print("save to ", output_path)
    # break

