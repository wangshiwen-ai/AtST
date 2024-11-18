import torch
from safetensors.torch import load_file

import argparse
import os

parser = argparse.ArgumentParser(description="Simple example of a training script.")
parser.add_argument(
        "--exp_name",
        type=str,
        default='yeqing-sdxl-idapter-1s-2ffn-rt',
       
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
parser.add_argument(
        "--iter",
        type=int,
        default=4800,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )

args = parser.parse_args()
exp_name = args.exp_name
iter_lst = os.listdir(f"{exp_name}/")
# 过滤出ckpt文件并提取迭代次数
ckpt_iters = [int(f.split('-')[1]) for f in iter_lst if f.startswith('checkpoint')]
# 找到最大的迭代次数
iter = max(ckpt_iters) if ckpt_iters else None

try:
    ckpt = f"{exp_name}/checkpoint-{iter}/pytorch_model.bin"
    sd = torch.load(ckpt, map_location="cpu")
except:
    ckpt = f"{exp_name}/checkpoint-{iter}/pytorch_model.pt"
    sd = torch.load(ckpt, map_location="cpu")
print("load from ", ckpt)

image_proj_sd = {}
ip_sd = {}

for k in sd:
    if k.startswith("unet"):
        pass
    elif k.startswith("image_proj_model"):
        image_proj_sd[k.replace("image_proj_model.", "")] = sd[k]
    elif k.startswith("adapter_modules"):
        ip_sd[k.replace("adapter_modules.", "")] = sd[k]

save_path = f"{exp_name}/checkpoint-{iter}/ip_adapter.bin"
torch.save({"image_proj": image_proj_sd, "ip_adapter": ip_sd}, save_path)
# print(ip_sd.keys())
print("save to ", save_path)