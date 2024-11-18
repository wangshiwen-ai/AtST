import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from diffusers import StableDiffusionXLImg2ImgPipeline, DDIMScheduler
from PIL import Image, ImageDraw
import sys
sys.path.append('/root/IP-Adapter2')
from ip_adapter.ip_adapter import IPAdapterXLDoubleZero  # 默认是 IPAdapterXLDoubleZero 类
from utils import *
# 加载参数
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', default='/root/autodl-tmp/', type=str)
parser.add_argument('--ref_char', default="不", type=str)
parser.add_argument('--font_name', default='Alibaba-PuHuiTi-Bold', type=str)
parser.add_argument('--output_dir', default='results', type=str)
parser.add_argument('--char', default='橙')
parser.add_argument('--t', default='11')
args = parser.parse_args()

char_lst = args.char.split(' ')
exp_name = args.exp_name
iter_lst = os.listdir(f"{exp_name}/")
# 过滤出ckpt文件并提取迭代次数
ckpt_iters = [int(f.split('-')[1]) for f in iter_lst if f.startswith('checkpoint')]
# 找到最大的迭代次数
iter = max(ckpt_iters) if ckpt_iters else None

ip_ckpt = f"{exp_name}/checkpoint-{iter}/ip_adapter.bin"
device = "cuda"

font_name = args.font_name.split(' ')

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


# 生成字符图片
from utils import *
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

# 初始化 IP Adapter 模型
ip_model = IPAdapterXLDoubleZero(pipe, "/root/autodl-tmp/vit", ip_ckpt, device, use_pe=True)
import random
# 定义生成目录
ckpt_name = (ip_ckpt).strip('/root/autodl-tmp').strip('/ip_adapter.bin').replace('/', '_')
ref_char = args.ref_char
# chinese_chars = "的一是在不了有和人这中大为上个国我以要他时来用们生到作地于出就分对成会可主发年动同工也能下过子说产种面而方后多定行学法所民得经十三之进着等部度家电力里如水化高自二理起小物现实加量都两体制机当使点从业本去把性好应开它合还因由其些然前外天政四日那社义事平形相全表间样与关各重新线内数正心反你明看原又么利比或但质气第向道命此变条只没结解问意建月公无系军很情者最立代想已通并提直题党程展五果料象员革位入常文总次品式活设及管特件长求老头基资边流路级少图山统接知较将组见计别她手角期根论运农指几九区强放决西被干做必战先回则任取据处队南给色光门即保治北造百规热领七海口东导器压志世金增争济阶油思术极交受联什认六共权收证改清己美再采转更单风切打白教速花带安场身车例真务具万每目至达走积示议声报斗完类八离华名确才科张认马节话米整空元况今集温传土许步群广石记需段研界拉林律叫且究观越织装影算低持音众书布复容儿须际商非验连断深难近矿千周委素技备半办青省列习响约支般史感劳便团往酸历市克何除消构府太准精值号率族维划选标写存候毛亲快效斯院查江型眼王按格养易置派层片始却专状育厂京识适属圆包火住调满县局照参红细引听该铁价严龙飞"
# english_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

# for c in char_lst:
#     print("Ch dict, ", len(list(chinese_chars)))
#     print("En dict, ", len(list(english_chars)))
#     chinese_chars=chinese_chars.replace(c, '')
#     english_chars=english_chars.replace(c, '')
    
# print("New Ch dict, ", len(list(chinese_chars)))
# print("New En dict, ", len(list(english_chars)))

# ref_char = random.choice([random.choice(list(chinese_chars)), random.choice(list(english_chars))])
# ref_char = args.ref_char
print("REFERENCE ", ref_char)
# 加载输入图片
style_name = args.exp_name.split('/')[-1].split('-')[0]
try:
    image = generate_single_char_image(ref_char, f'AnyText/font/{style_name}.ttf', char_size=400) # Style image
    ref_name = ref_char
except:
    image = Image.open(args.ref_char)
    ref_name = os.path.basename(args.ref_char).split('.')[0]
    ## 创建一张全部白色的图
    # image =  Image.new('RGB', (512, 512), 'white')


grey_image = image.convert('L').convert('RGB')

g_image_lst = []
gt_image_lst = []
id_lst = []
for char in char_lst:
    for font in font_name:
        g_image = generate_single_char_image(char, f'AnyText/font/{font}.ttf')
        g_image_lst.append(g_image)
        try:
            gt_image_lst.append(generate_single_char_image(char, f'AnyText/font/{style_name}.ttf'))
        except:
            gt_image_lst.append(g_image)
        print(f"Generate char {char} in font {font} done.")
        id_lst.append((char, font))

scale = 0.8

# 生成不同 strength 的图像并保存
s_list = [0.7, 0.8, 0.85, 0.9, 0.95, 0.97]
# s_list = [0.8, 0.83, 0.85, 0.87]

for (char,font), g_image, gt_image in zip(id_lst, g_image_lst, gt_image_lst):
    save_dir = f'{args.output_dir}/{style_name}/{char}_{ref_char}/{font}/{args.t}'
    all_s_images = []
    grey_image = extract_skeleton(g_image, iter=3)
    
    os.makedirs(f"{save_dir}", exist_ok=True)
    output_path = f"{save_dir}/o_char.png"
    g_image.save(output_path)
    output_path = f"{save_dir}/s_char.png"
    grey_image.save(output_path)
    output_path = f"{save_dir}/prompt_char.png"
    image.save(output_path)
    
    for s in s_list:
        images = ip_model.generate(
        pil_image=image,
        grey_pil_image=grey_image,
        no_attn_mask=None,
        num_samples=6,
        num_inference_steps=50,
        seed=42,
        image=g_image,
        strength=s,
        prompt= "1 word “{}” in specific style: {}".format(char, 'simple and white background.'),
        scale=scale
        )

        # 生成图片网格并保存
        # images = [g_image, grey_image, image.resize((512, 512))] + images
        # grid = image_grid(images, 1, 9)
        # os.makedirs(f"{save_dir}", exist_ok=True)
        # output_path = f"{save_dir}/s_{s}_scale_{scale}.png"
        # grid.save(output_path)
        # print(f"Save to {output_path}")
        os.makedirs(f"{save_dir}/strength_{s}", exist_ok=True)
        for n, i in enumerate(images):
            output_path = f"{save_dir}/strength_{s}/{n}.png"
            i.save(output_path)
            print(f"Save to {output_path}")
