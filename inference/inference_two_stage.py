from diffusers import StableDiffusionPipeline
from PIL import Image, ImageDraw, ImageFont
import torch
import torch
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipelineLegacy, DDIMScheduler, AutoencoderKL,StableDiffusionXLPipeline, \
    StableDiffusionXLImg2ImgPipeline
from PIL import Image

from ip_adapter import IPAdapter
from ip_adapter.ip_adapter import IPAdapterXLDoubleZero, IPAdapterXLDoubleZero640
import os
import numpy as np
import cv2
from utils import generate_chars_image, generate_chars_image_different_size

base_model_path = "sdxl_models"
# bash_model_path = '/home/swwang/huggingface/hub/models--runwayml--stable-diffusion-v1-5'
# vae_model_path = "stabilityai/sd-vae-ft-mse"
image_encoder_path = "vit"

from modelscope.utils.constant import Tasks
from modelscope.pipelines import pipeline
import cv2


# exit()

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--ip_ckpt', default='/home/swwang/IP-Adapter/sdxl-idapter-1s-2ffn-rt/checkpoint-5800/ip_adapter.bin', type=str)
# parser.add_argument('--ip_ckpt', default='/home/swwang/IP-Adapter/sd-idapter-2/checkpoint-2000/ip_adapter.bin', type=str)
parser.add_argument('--style_path', default="assets/images/hua.png", type=str)
parser.add_argument('--bishua_path', default="assets/images/easychar.png", type=str)
parser.add_argument('--img_path', default=["/home/swwang/IP-Adapter/output/Alibaba-PuHuiTi-Bold.png"], type=list)
parser.add_argument('--font_name', default='Alibaba-PuHuiTi-Bold', type=str)
parser.add_argument('--text_prompt', default='', type=str)
parser.add_argument('--char', default='春')
parser.add_argument('--t', default='1', type=str)
args = parser.parse_args()
# args.style_path = args.style_path.split(' ')

ip_ckpt = args.ip_ckpt
device = "cuda"
if args.t == "1":
    from ip_adapter.ip_adapter import IPAdapterXLDoubleZero as IPAdapterXLDoubleZero
    print('Use type 1')
elif args.t == '2':
    from ip_adapter.ip_adapter import IPAdapterXLDoubleZero as IPAdapterXLDoubleZero
    print('Use type 2')
# elif args.t == 'p':
#      from ip_adapter.ip_adapter import IPAdapterXLDoubleZeroPlus as IPAdapterXLDoubleZero
#      print('Use type plus')
# elif args.t == '3':
#      from ip_adapter.ip_adapter import IPAdapterXLDoubleZero3 as IPAdapterXLDoubleZero
#      print('Use type 3')
    
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

def generate_text_image(text, font_path, font_size=100, image_size=(256, 256), output_file='output.png'):
    """
    生成指定字体和文本的图像。

    参数:
    text (str): 要生成的文本（一个或多个字符）。
    font_path (str): 字体文件的路径。
    font_size (int): 字体大小。
    image_size (tuple): 图像大小 (宽, 高)。
    output_file (str): 输出图像文件的路径。
    """
    # 创建空白图像
    image = Image.new('RGB', image_size, (255, 255, 255))
    draw = ImageDraw.Draw(image)

    # 加载指定字体
    font = ImageFont.truetype(font_path, font_size)

    # 计算文本位置，使其居中
    text_width, text_height = draw.textsize(text, font=font)
    position = ((image_size[0] - text_width) // 2, (image_size[1] - text_height) // 2)

    # 绘制文本
    draw.text(position, text, fill=(0, 0, 0), font=font)

    # 保存图像
    image.save(output_file)
    print(f"Image saved to {output_file}")

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

def pad_image_to_512(image):
    target_size = (512, 512)
    width, height = image.size
    
    # 计算需要填充的宽度和高度
    pad_width = (target_size[0] - width) // 2
    pad_height = (target_size[1] - height) // 2
    
    # 使用 ImageOps.expand 方法进行填充，填充颜色为白色
    padded_image = ImageOps.expand(image, (pad_width, pad_height, pad_width, pad_height), fill='white')
    
    # 如果填充后的大小不完全匹配 target_size，进行额外的填充
    padded_image = padded_image.resize(target_size)
    
    return padded_image
# load SDXL pipeline
pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    'sdxl_models',
    torch_dtype=torch.float16,
    add_watermarker=False,
)
from PIL import ImageOps
# image = Image.open(args.style_path)  ## Style
# image = pad_image_to_512(image)
# grey_image = image.convert('L').convert('RGB')
# ip_model = IPAdapter(pipe, image_encoder_path, ip_ckpt, device)
# ip_model = IPAdapterXLDoubleZero(pipe, image_encoder_path, ip_ckpt, device)
# mask = Image.new('L', image.size, 255)  # 创建一个全黑的 mask
# draw = ImageDraw.Draw(mask)
# draw.rectangle([0, 0, image.size[0] // 2, image.size[1]], fill=0)  # 将左半边填充为白色
# 保存 mask 以供检查
# mask.save('left_half_mask.png')
import torchvision.transforms.functional as T
# mask = T.to_tensor(mask.convert('L'))

style_name ="_".join([os.path.basename(s).split('.')[0] for s in args.style_path])if isinstance(args.style_path, list) else os.path.basename(args.style_path).split('.')[0]
name = '_'.join([style_name , args.ip_ckpt.replace('/', '_')])

os.makedirs(f'exp/{name}', exist_ok=True)

if len(list(args.char)) > 1:
    char_image = generate_chars_image_different_size(args.char, f'AnyText/font/{args.font_name}.ttf')
else:
    char_image = generate_single_char_image(args.char, f'AnyText/font/{args.font_name}.ttf', )
print("Generate char {} in font {} done.".format(args.char, args.font_name))

s_list = [0.75, 0.8, 0.85, 0.9, 0.95]


def pad_to_square(image):
    width, height = image.size
    max_side = max(width, height)
    pad_width = (max_side - width) // 2
    pad_height = (max_side - height) // 2
    padded_image = ImageOps.expand(image, (pad_width, pad_height, pad_width, pad_height), fill='white')
    return padded_image

import math
def concatenate_images(image_list):
    widths, heights = 512, 512
    image_num = len(image_list)
    h = math.ceil(math.sqrt(image_num))
    w = math.floor(math.sqrt(image_num))
    if h*w < image_num:
        w += 1
    new_image = Image.new('RGB', (w*512, h*512))
    print(f"Image number {image_num}, concatenated width {w} height {h}")

    x_offset = 0
    y_offset = 0
    for i, img in enumerate(image_list):
        img = img.resize((512, 512))
        new_image.paste(img, (x_offset, y_offset))
        x_offset += 512
        if (i + 1) % w == 0:
            x_offset = 0
            y_offset += 512

    return new_image

os.makedirs(f'exp/{name}/twicediff_easychar', exist_ok=True)
# if args.t == "wcolor":
if True:
    for s in s_list:
        g_image = char_image
        
        if isinstance(args.style_path, list):
            image_list = [Image.open(path) for path in args.style_path]  # 读取每个路径对应的图像对象
            concatenated_image = concatenate_images(image_list)  # 拼接图像
            image = pad_to_square(concatenated_image)  # 填充成方形图像
            image.save('concatenate_image.png')
        else:
            image = Image.open(args.style_path)  ## Style

        if args.t == '2':
            image = Image.open(args.bishua_path)
            
        grey_pil_image = image.convert('L').convert('RGB')
        # image =  Image.open(args.bishua_path).convert('L') ## Style
        # grey_pil_image = image.point(lambda p: p > 200 and 255).convert('RGB')
        # ip_model = IPAdapterXLDoubleZero(pipe, image_encoder_path, '/home/swwang/IP-Adapter/sdxl-idapter-1s-2ffn-nocolor/checkpoint-5800/ip_adapter.bin', device)

        ip_model = IPAdapterXLDoubleZero(pipe, image_encoder_path, '/home/swwang/IP-Adapter/sdxl-idapter-1s-2ffn-easychar/checkpoint-15000/ip_adapter.bin', device)
        
    
        # images = ip_model.generate(pil_image=image.convert("RGB"), grey_pil_image=grey_pil_image, num_samples=4, num_inference_steps=50, seed=42, image=g_image, strength=s, \
        #                           prompt='''A calligraphic Chinese character "{}", with its strokes replaced by images that embody the attributes of "{}". Details as “{}”. The overall composition is harmonious and aesthetically pleasing.'''.format(args.char, args.char, args.char+args.text_prompt) )
        images = ip_model.generate(pil_image=image.convert("RGB"), grey_pil_image=grey_pil_image, num_samples=4, num_inference_steps=50, seed=42, image=g_image, strength=s-0.1, \
                                  prompt='''Generate the structure of 1 Chinese word "{}", simple and white background.'''.format(args.char) )

        s_images = [g_image] + images
        grid = image_grid(s_images, 1, 5)
        output_path = f'exp/{name}/twicediff_easychar/{args.char}_s_{s}_{args.font_name}_1.png'
        grid.save(output_path)

        print("save to ", output_path)
        del ip_model
        
        ip_model = IPAdapterXLDoubleZero(pipe, image_encoder_path, ip_ckpt, device)
     
        if isinstance(args.style_path, list):
            image_list = [Image.open(path) for path in args.style_path]  # 读取每个路径对应的图像对象
            concatenated_image = concatenate_images(image_list)  # 拼接图像
            image = pad_to_square(concatenated_image)  # 填充成方形图像
            
        else:
            image = Image.open(args.style_path)  ## Style
        for i in range(1, 5):
            g_image = s_images[i]
            for sp in s_list:       
                # images = ip_model.generate(pil_image=image, grey_pil_image=image.convert('L').convert("RGB"), num_samples=4, num_inference_steps=50, seed=42, image=g_image, strength=sp, \
                #                   prompt='''A calligraphic Chinese character "{}", with its strokes replaced by images that embody the attributes of "{}". The overall composition is harmonious and aesthetically pleasing.'''.format(args.char, args.char) )
                images = ip_model.generate(pil_image=image, grey_pil_image=image.convert('L').convert("RGB"), num_samples=4, num_inference_steps=50, seed=42, image=g_image, strength=sp, \
                                  prompt='''Color the Chinese word "{}".'''.format(args.char) )
               
                images = [g_image] + images

                grid = image_grid(images, 1, 5)
                output_path = f'exp/{name}/twicediff_easychar/{args.char}_s_{s}_sp_{sp}_{args.font_name}_2_{i}.png'
                grid.save(output_path)
                print("save to ", output_path)



# for s in s_list:
#     images = ip_model.generate(pil_image=image, grey_pil_image=grey_image, no_attn_mask=None, num_samples=4, num_inference_steps=50, seed=42, image=g_image, strength=s, \
#                               prompt="Tag: 1 calligraphy chinese word “{}”. simple and white background. With elements about “{}”. High quality.".format(args.char, args.char), scale=0.8)
    
#     images = [g_image] + images
#     grid = image_grid(images, 1, 5)
#     output_path = f'exp/{name}/{args.char}_s_{s}_{args.font_name}_scale0.8.png'
#     grid.save(output_path)
#     print("save to ", output_path)
#     # break

