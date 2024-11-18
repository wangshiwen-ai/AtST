import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from diffusers import StableDiffusionXLImg2ImgPipeline, DDIMScheduler
from PIL import Image, ImageDraw
import sys
sys.path.append('/root/IP-Adapter2')
from ip_adapter.ip_adapter import IPAdapterXLDoubleZero  # 默认是 IPAdapterXLDoubleZero 类

# 加载参数
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--ip_ckpt', default='/root/autodl-tmp/ip_adapter.bin', type=str)
parser.add_argument('--style_path', default="assets/images/chengzi.png", type=str)
parser.add_argument('--img_path', default=["/home/swwang/IP-Adapter/output/Alibaba-PuHuiTi-Bold.png"], type=list)
parser.add_argument('--font_name', default='Alibaba-PuHuiTi-Bold', type=str)
parser.add_argument('--char', default='橙')
parser.add_argument('--use_pe', type=int, default=0)
parser.add_argument('--t', default='00', type=str)
args = parser.parse_args()

ip_ckpt = args.ip_ckpt
device = "cuda"
print(args.use_pe)
# 选择正确的 IPAdapter 类
if args.t == '00':  ## 这个是没有2ffn的
    from ip_adapter import IPAdapterXLSingle as IPAdapterXLDoubleZero
if args.t == '01':  ## 这个是没有2ffn的
    from ip_adapter import IPAdapterXLSingleDFFN as IPAdapterXLDoubleZero
elif args.t == '10':
    from ip_adapter.ip_adapter import IPAdapterXLDouble as IPAdapterXLDoubleZero
elif args.t == '3':
    from ip_adapter.ip_adapter import IPAdapterXLDoubleZero3 as IPAdapterXLDoubleZero
elif args.t == 'r':
    from ip_adapter.ip_adapter import IPAdapterXLDoubleZeroR as IPAdapterXLDoubleZero

font_name_lst = args.font_name.split(' ')

# 定义生成图片的网格函数
def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols
    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols * w, rows * h))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid

# 加载 SDXL 管道
pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained('/root/autodl-tmp/sdxl-models', torch_dtype=torch.float16, add_watermarker=False)

# 加载输入图片
image = Image.open(args.style_path)  # Style image
grey_image = image.convert('L').convert('RGB')

# 初始化 IP Adapter 模型
ip_model = IPAdapterXLDoubleZero(pipe, "/root/autodl-tmp/vit", ip_ckpt, device, use_pe=args.use_pe)

# 定义生成目录
ckpt_name = (args.ip_ckpt).strip('../autodl-tmp').strip('/root/autodl-tmp').strip('/ip_adapter.bin').replace('/', '_')
style_name = os.path.basename(args.style_path).split('.')[0]
## 方便对比 在一个字+一个元素下设置配置
# 生成字符图片

from utils import generate_single_char_image, generate_chars_image_different_size
g_image_lst = []
for font_name in font_name_lst:
    g_image = generate_single_char_image(args.char, f'AnyText/font/{font_name}.ttf')
    g_image_lst.append(g_image)
    print(f"Generate char {args.char} in font {font_name} done.")

scale=0.8

# 生成不同 strength 的图像并保存
s_list = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95]


for font_name, g_image in zip(font_name_lst, g_image_lst):
    save_dir = f'results/char_{args.char}_{style_name}/{ckpt_name}/{font_name}'
    os.makedirs(save_dir, exist_ok=True)
    for s in s_list:
        images = ip_model.generate(
        pil_image=image,
        grey_pil_image=grey_image,
        no_attn_mask=None,
        num_samples=5,
        num_inference_steps=50,
        seed=42,
        image=g_image,
        strength=s,
        prompt=f"Tag: 1 calligraphy chinese word '{args.char}'. simple and white background. With elements about '{args.char}'. High quality.",
        scale=scale
        )
        # 生成图片网格并保存
        images = [g_image] + [image.resize((512,512))] + images
        grid = image_grid(images, 1, 7)
        output_path = f'{save_dir}/s_{s}_scale_{scale}_nope.png'
        grid.save(output_path)
        print(f"Save to {output_path}")
