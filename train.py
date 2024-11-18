import os
import random
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import argparse
from pathlib import Path
import json
import itertools
import time

import torch
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from PIL import Image
from transformers import CLIPImageProcessor
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection, CLIPTextModelWithProjection
from ip_adapter.ip_adapter import ImageProjModel2 as ImageProjModel
from utils import extract_color_and_non_color_parts, set_background_white

from ip_adapter.utils import is_torch2_available
if is_torch2_available():
    from ip_adapter.attention_processor import IPAttnProcessor2_0 as IPAttnProcessor, AttnProcessor2_0 as AttnProcessor
else:
    from ip_adapter.attention_processor import AttnProcessor

from ip_adapter.attention_processor import AttnProcessor
# from ip_adapter.attention_processor import IPAttnProcessorDouble as IPAttnProcessor
from ip_adapter.attention_processor import IPAttnProcessorDoubleZero as IPAttnProcessor

from pathlib import Path
def extract_non_digit_characters(input_string):
    return ''.join([char for char in input_string if not char.isdigit()])

import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model=384, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        b, seq_len, d_model = x.size()
        pe = self.pe[:, :seq_len, :]
        x = x + pe.to(x.device)
        return x
 
# Dataset for stage 2/full process. image: color. element: color and grey
class TheirDataset(torch.utils.data.Dataset):

    def __init__(self, json_file, tokenizer, tokenizer_2, aug=False, size=1024, center_crop=True, t_drop_rate=0.05, i_drop_rate=0.05, ti_drop_rate=0.05, image_root_path=""):
        super().__init__()

        self.tokenizer = tokenizer
        self.tokenizer_2 = tokenizer_2
        self.size = size
        self.center_crop = center_crop
        self.i_drop_rate = i_drop_rate
        self.t_drop_rate = t_drop_rate

        self.ti_drop_rate = ti_drop_rate
        self.image_root_path = image_root_path
        self.aug = aug
        self.data = json.load(open(json_file)) # list of dict: [{"image_file": "1.png", "text": "A dog"}]

        self.transform = transforms.Compose([
            transforms.Resize(self.size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

        self.clip_image_processor = CLIPImageProcessor()
        
    def __getitem__(self, idx):
        item = self.data[idx] 
        image_file = item["image_file"]
        name = os.path.basename(image_file).split('.')[0]
        try:
            caption = (Path(self.image_root_path) / Path(image_file)).with_suffix('.caption').read_text().strip()
        except:
            caption = ""
        text = "1 calligraphy Chinese word “{}”: {}".format(extract_non_digit_characters(name), caption)
        # text = 'Tag: 1 chinese word, simple and white background, free structure. Content:' + Path(os.path.join(self.image_root_path, image_file)).with_suffix('.caption').read_text().strip()
        # print(text)
        # read image
        raw_image = Image.open(os.path.join(self.image_root_path, image_file))
        
        ## read ele image
        ele_id = random.choice(range(len(item['char_name']['path'])))
        ele_text = item['char_name']['description'][ele_id]
        ele_path = item['char_name']['path'][ele_id]
        # text = '''A calligraphic character "{}", with elements like "{}". Details as “{}”.'''.format(text, text, ele_text)
        ele_image = Image.open(ele_path).convert('RGB')
        ele_image_grey = ele_image.convert('L')
        ele_image_grey = ele_image_grey.convert('RGB')
        # 随机旋转和缩放增强
        if self.aug:
            rotation_angle = random.uniform(-30, 30)  # 随机旋转角度在-30到30度之间
            scale_factor = random.uniform(0.8, 1.2)  
            ele_image = ele_image.rotate(rotation_angle, resample=Image.BICUBIC)
            width, height = ele_image.size
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            ele_image = ele_image.resize((new_width, new_height), Image.BICUBIC)

        # original size
        original_width, original_height = raw_image.size
        original_size = torch.tensor([original_height, original_width])
        
        image_tensor = self.transform(raw_image.convert("RGB"))
        # random crop
        delta_h = image_tensor.shape[1] - self.size
        delta_w = image_tensor.shape[2] - self.size
        assert not all([delta_h, delta_w])
        
        if self.center_crop:
            top = delta_h // 2
            left = delta_w // 2
        else:
            top = np.random.randint(0, delta_h + 1)
            left = np.random.randint(0, delta_w + 1)
        image = transforms.functional.crop(
            image_tensor, top=top, left=left, height=self.size, width=self.size
        )
        crop_coords_top_left = torch.tensor([top, left]) 

        clip_image = self.clip_image_processor(images=ele_image, return_tensors="pt").pixel_values
        clip_image_grey = self.clip_image_processor(images=ele_image_grey, return_tensors="pt").pixel_values
        
        # drop
        drop_image_embed = 0
        rand_num = random.random()
        if rand_num < self.i_drop_rate:
            drop_image_embed = 1
        elif rand_num < (self.i_drop_rate + self.t_drop_rate):
            text = ""
        elif rand_num < (self.i_drop_rate + self.t_drop_rate + self.ti_drop_rate):
            text = ""
            drop_image_embed = 1

        # get text and tokenize
        text_input_ids = self.tokenizer(
            text,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids
        
        text_input_ids_2 = self.tokenizer_2(
            text,
            max_length=self.tokenizer_2.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids
        
        return {
            "image": image,
            "text_input_ids": text_input_ids,
            "text_input_ids_2": text_input_ids_2,
            "clip_image": [clip_image, clip_image_grey],
            "drop_image_embed": drop_image_embed,
            "original_size": original_size,
            "crop_coords_top_left": crop_coords_top_left,
            "target_size": torch.tensor([self.size, self.size]),
        }
        
    
    def __len__(self):
        return len(self.data)
   
# Dataset for stage 1 structure learning. nocolor, image: grey. element: grey and black
class TheirDataset2(TheirDataset):    
    def __getitem__(self, idx):
        item = self.data[idx] 
        image_file = item["image_file"]
        name = os.path.basename(image_file).split('.')[0]
        if os.path.exists((Path(self.image_root_path) / Path(image_file)).with_suffix('.caption')):
            caption = (Path(self.image_root_path) / Path(image_file)).with_suffix('.caption').read_text().strip()
        else:
            caption = 'simple and white background, ink.'
        text = "Generate the structure of Chinese word “{}” as style: {}".format(extract_non_digit_characters(name), caption)
        # text = 'Tag: 1 chinese word, simple and white background, free structure. Content:' + Path(os.path.join(self.image_root_path, image_file)).with_suffix('.caption').read_text().strip()
        # print(text)
        # read image
        raw_image = Image.open(os.path.join(self.image_root_path, image_file)).convert('L').convert('RGB')
        
        ## read ele image
        ele_id = random.choice(range(len(item['char_name']['path'])))
        ele_text = item['char_name']['description'][ele_id]
        ele_path = item['char_name']['path'][ele_id]
        # text = '''A calligraphic character "{}", with elements like "{}". Details as “{}”.'''.format(text, text, ele_text)
        ele_image = Image.open(ele_path).convert('RGB').convert('L')
        ele_image_grey = ele_image.point(lambda p: p > 200 and 255)
        ele_image_grey = ele_image_grey.convert('RGB')
        ele_image = ele_image.convert('RGB')
        # raw_image.save('debug/raw_image_structure.png')
        # ele_image.save('debug/ele_image_structure.png')
        # ele_image_grey.save('debug/ele_image_grey_structure.png')
        # 随机旋转和缩放增强
        if self.aug:
            rotation_angle = random.uniform(-30, 30)  # 随机旋转角度在-30到30度之间
            scale_factor = random.uniform(0.8, 1.2)  
            ele_image = ele_image.rotate(rotation_angle, resample=Image.BICUBIC)
            width, height = ele_image.size
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            ele_image = ele_image.resize((new_width, new_height), Image.BICUBIC)

        # original size
        original_width, original_height = raw_image.size
        original_size = torch.tensor([original_height, original_width])
        
        image_tensor = self.transform(raw_image.convert("RGB"))
        # random crop
        delta_h = image_tensor.shape[1] - self.size
        delta_w = image_tensor.shape[2] - self.size
        assert not all([delta_h, delta_w])
        
        if self.center_crop:
            top = delta_h // 2
            left = delta_w // 2
        else:
            top = np.random.randint(0, delta_h + 1)
            left = np.random.randint(0, delta_w + 1)
        image = transforms.functional.crop(
            image_tensor, top=top, left=left, height=self.size, width=self.size
        )
        crop_coords_top_left = torch.tensor([top, left]) 

        clip_image = self.clip_image_processor(images=ele_image, return_tensors="pt").pixel_values
        clip_image_grey = self.clip_image_processor(images=ele_image_grey, return_tensors="pt").pixel_values
        
        # drop
        drop_image_embed = 0
        rand_num = random.random()
        if rand_num < self.i_drop_rate:
            drop_image_embed = 1
        elif rand_num < (self.i_drop_rate + self.t_drop_rate):
            text = ""
        elif rand_num < (self.i_drop_rate + self.t_drop_rate + self.ti_drop_rate):
            text = ""
            drop_image_embed = 1

        # get text and tokenize
        text_input_ids = self.tokenizer(
            text,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids
        
        text_input_ids_2 = self.tokenizer_2(
            text,
            max_length=self.tokenizer_2.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids
        
        return {
            "image": image,
            "text_input_ids": text_input_ids,
            "text_input_ids_2": text_input_ids_2,
            "clip_image": [clip_image, clip_image_grey],
            "drop_image_embed": drop_image_embed,
            "original_size": original_size,
            "crop_coords_top_left": crop_coords_top_left,
            "target_size": torch.tensor([self.size, self.size]),
        }

# Dataset for stage 1 structure learning. hold. image: grey element: color and black with image penalty(color)
class TheirDataset3(TheirDataset): 
    def __getitem__(self, idx):
        item = self.data[idx] 
        image_file = item["image_file"]
        name = os.path.basename(image_file).split('.')[0]
        caption = (Path(self.image_root_path) / Path(image_file)).with_suffix('.caption').read_text().strip()
        text = "Color this black and white image of Chinese word “{}” with element: {}".format(extract_non_digit_characters(name), caption)
        raw_image = Image.open(os.path.join(self.image_root_path, image_file)).convert('L').convert('RGB')
        raw_image_rgb = Image.open(os.path.join(self.image_root_path, image_file))

        ## read ele image
        ele_id = random.choice(range(len(item['char_name']['path'])))
        # ele_text = item['char_name']['description'][ele_id]
        ele_path = item['char_name']['path'][ele_id]
        ele_image = Image.open(ele_path)
        ele_image_grey = ele_image.convert('L').convert('RGB')

        # 随机旋转和缩放增强
        if self.aug:
            rotation_angle = random.uniform(-30, 30)  # 随机旋转角度在-30到30度之间
            scale_factor = random.uniform(0.8, 1.2)  
            ele_image = ele_image.rotate(rotation_angle, resample=Image.BICUBIC)
            width, height = ele_image.size
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            ele_image = ele_image.resize((new_width, new_height), Image.BICUBIC)

        # original size
        original_width, original_height = raw_image.size
        original_size = torch.tensor([original_height, original_width])
        
        image_tensor = self.transform(raw_image.convert("RGB"))
        image_tensor_rgb = self.transform(raw_image_rgb.convert("RGB"))
        # random crop
        delta_h = image_tensor.shape[1] - self.size
        delta_w = image_tensor.shape[2] - self.size
        assert not all([delta_h, delta_w])
        
        if self.center_crop:
            top = delta_h // 2
            left = delta_w // 2
        else:
            top = np.random.randint(0, delta_h + 1)
            left = np.random.randint(0, delta_w + 1)
        image = transforms.functional.crop(
            image_tensor, top=top, left=left, height=self.size, width=self.size
        )
        image_rgb = transforms.functional.crop(
            image_tensor_rgb, top=top, left=left, height=self.size, width=self.size
        )
        crop_coords_top_left = torch.tensor([top, left]) 

        clip_image = self.clip_image_processor(images=ele_image, return_tensors="pt").pixel_values
        clip_image_grey = self.clip_image_processor(images=ele_image_grey, return_tensors="pt").pixel_values
        
        # drop
        drop_image_embed = 0
        rand_num = random.random()
        if rand_num < self.i_drop_rate:
            drop_image_embed = 1
        elif rand_num < (self.i_drop_rate + self.t_drop_rate):
            text = ""
        elif rand_num < (self.i_drop_rate + self.t_drop_rate + self.ti_drop_rate):
            text = ""
            drop_image_embed = 1

        # get text and tokenize
        text_input_ids = self.tokenizer(
            text,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids
        
        text_input_ids_2 = self.tokenizer_2(
            text,
            max_length=self.tokenizer_2.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids
        
        return {
            "image": image,
            "rgb_image": image_rgb,
            "text_input_ids": text_input_ids,
            "text_input_ids_2": text_input_ids_2,
            "clip_image": [clip_image, clip_image_grey],
            "drop_image_embed": drop_image_embed,
            "original_size": original_size,
            "crop_coords_top_left": crop_coords_top_left,
            "target_size": torch.tensor([self.size, self.size]),
        }

# Dataset for stage 1 structure learning. image eacychar, no element
class TheirDatasetColor(TheirDataset):     
    def __getitem__(self, idx):
        item = self.data[idx] 
        image_file = item["image_file"]
        name = os.path.basename(image_file).split('.')[0]
        if os.path.exists((Path(self.image_root_path) / Path(image_file)).with_suffix('.caption')):
            caption = (Path(self.image_root_path) / Path(image_file)).with_suffix('.caption').read_text().strip()
        else:
            caption = ''
        text = "Coloring the word “{}” with color: {}".format(extract_non_digit_characters(name), caption)
        # text = 'Tag: 1 chinese word, simple and white background, free structure. Content:' + Path(os.path.join(self.image_root_path, image_file)).with_suffix('.caption').read_text().strip()
        # print(text)
        # read image
        image_path = os.path.join(self.image_root_path, image_file)
        # raw_image = Image.open(image_path)
        raw_image = set_background_white(image_path)
        raw_image.save(image_path)
        # text = '''A calligraphic character "{}", with elements like "{}". Details as “{}”.'''.format(text, text, ele_text)
        ele_image, ele_image_nocolor, ele_image_grey = extract_color_and_non_color_parts(image_path, th=[100, 150])
        # raw_image.save('debug/raw_image_structure.png')
        ele_image.save('processed/yeqing_colored/' + image_file)
        ele_image_nocolor.save('processed/yeqing_black/' + image_file)
        
        # 随机旋转和缩放增强
        if self.aug:
            rotation_angle = random.uniform(-30, 30)  # 随机旋转角度在-30到30度之间
            scale_factor = random.uniform(0.8, 1.2)  
            ele_image = ele_image.rotate(rotation_angle, resample=Image.BICUBIC)
            width, height = ele_image.size
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            ele_image = ele_image.resize((new_width, new_height), Image.BICUBIC)

        # original size
        original_width, original_height = raw_image.size
        original_size = torch.tensor([original_height, original_width])
        
        image_tensor = self.transform(raw_image.convert("RGB"))
        # random crop
        delta_h = image_tensor.shape[1] - self.size
        delta_w = image_tensor.shape[2] - self.size
        assert not all([delta_h, delta_w])
        
        if self.center_crop:
            top = delta_h // 2
            left = delta_w // 2
        else:
            top = np.random.randint(0, delta_h + 1)
            left = np.random.randint(0, delta_w + 1)
        image = transforms.functional.crop(
            image_tensor, top=top, left=left, height=self.size, width=self.size
        )
        crop_coords_top_left = torch.tensor([top, left]) 

        clip_image = self.clip_image_processor(images=ele_image, return_tensors="pt").pixel_values
        clip_image_grey = self.clip_image_processor(images=ele_image_nocolor, return_tensors="pt").pixel_values
        
        # drop
        drop_image_embed = 0
        rand_num = random.random()
        if rand_num < self.i_drop_rate:
            drop_image_embed = 1
        elif rand_num < (self.i_drop_rate + self.t_drop_rate):
            text = ""
        elif rand_num < (self.i_drop_rate + self.t_drop_rate + self.ti_drop_rate):
            text = ""
            drop_image_embed = 1

        # get text and tokenize
        text_input_ids = self.tokenizer(
            text,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids
        
        text_input_ids_2 = self.tokenizer_2(
            text,
            max_length=self.tokenizer_2.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids
        
        return {
            "image": image,
            "text_input_ids": text_input_ids,
            "text_input_ids_2": text_input_ids_2,
            "clip_image": [clip_image, clip_image_grey],
            "drop_image_embed": drop_image_embed,
            "original_size": original_size,
            "crop_coords_top_left": crop_coords_top_left,
            "target_size": torch.tensor([self.size, self.size]),
        }

# Dataset for stage 1 structure learning. image eacychar, no element
class TheirDataset4(TheirDataset):     
    def __getitem__(self, idx):
        item = self.data[idx] 
        image_file = item["image_file"]
        name = os.path.basename(image_file).split('.')[0]
        if os.path.exists((Path(self.image_root_path) / Path(image_file)).with_suffix('.caption')):
            caption = (Path(self.image_root_path) / Path(image_file)).with_suffix('.caption').read_text().strip()
        else:
            caption = 'simple and white background, ink.'
        text = "Generate the structure of Chinese word “{}” as style: {}".format(extract_non_digit_characters(name), caption)
        # text = 'Tag: 1 chinese word, simple and white background, free structure. Content:' + Path(os.path.join(self.image_root_path, image_file)).with_suffix('.caption').read_text().strip()
        # print(text)
        # read image
        raw_image = Image.open(os.path.join(self.image_root_path, image_file)).convert('L').convert('RGB')
        
        # text = '''A calligraphic character "{}", with elements like "{}". Details as “{}”.'''.format(text, text, ele_text)
        ele_image = Image.open(os.path.join(self.image_root_path, image_file))
        ele_image_grey = ele_image.convert('L').convert('RGB')
        # raw_image.save('debug/raw_image_structure.png')
        # ele_image.save('debug/ele_image_structure.png')
        # ele_image_grey.save('debug/ele_image_grey_structure.png')
        # 随机旋转和缩放增强
        if self.aug:
            rotation_angle = random.uniform(-30, 30)  # 随机旋转角度在-30到30度之间
            scale_factor = random.uniform(0.8, 1.2)  
            ele_image = ele_image.rotate(rotation_angle, resample=Image.BICUBIC)
            width, height = ele_image.size
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            ele_image = ele_image.resize((new_width, new_height), Image.BICUBIC)

        # original size
        original_width, original_height = raw_image.size
        original_size = torch.tensor([original_height, original_width])
        
        image_tensor = self.transform(raw_image.convert("RGB"))
        # random crop
        delta_h = image_tensor.shape[1] - self.size
        delta_w = image_tensor.shape[2] - self.size
        assert not all([delta_h, delta_w])
        
        if self.center_crop:
            top = delta_h // 2
            left = delta_w // 2
        else:
            top = np.random.randint(0, delta_h + 1)
            left = np.random.randint(0, delta_w + 1)
        image = transforms.functional.crop(
            image_tensor, top=top, left=left, height=self.size, width=self.size
        )
        crop_coords_top_left = torch.tensor([top, left]) 

        clip_image = self.clip_image_processor(images=ele_image, return_tensors="pt").pixel_values
        clip_image_grey = self.clip_image_processor(images=ele_image_grey, return_tensors="pt").pixel_values
        
        # drop
        drop_image_embed = 0
        rand_num = random.random()
        if rand_num < self.i_drop_rate:
            drop_image_embed = 1
        elif rand_num < (self.i_drop_rate + self.t_drop_rate):
            text = ""
        elif rand_num < (self.i_drop_rate + self.t_drop_rate + self.ti_drop_rate):
            text = ""
            drop_image_embed = 1

        # get text and tokenize
        text_input_ids = self.tokenizer(
            text,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids
        
        text_input_ids_2 = self.tokenizer_2(
            text,
            max_length=self.tokenizer_2.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids
        
        return {
            "image": image,
            "text_input_ids": text_input_ids,
            "text_input_ids_2": text_input_ids_2,
            "clip_image": [clip_image, clip_image_grey],
            "drop_image_embed": drop_image_embed,
            "original_size": original_size,
            "crop_coords_top_left": crop_coords_top_left,
            "target_size": torch.tensor([self.size, self.size]),
        }

class TheirDataset6(TheirDataset):     
    def __getitem__(self, idx):
        item = self.data[idx] 
        image_file = item["image_file"]
        name = os.path.basename(image_file).split('.')[0]
        if os.path.exists((Path(self.image_root_path) / Path(image_file)).with_suffix('.caption')):
            caption = (Path(self.image_root_path) / Path(image_file)).with_suffix('.caption').read_text().strip()
        else:
            caption = 'simple and white background.'
        text = "1 chinese word “{}” in specidic style: {}".format(extract_non_digit_characters(name), caption)
        # text = 'Tag: 1 chinese word, simple and white background, free structure. Content:' + Path(os.path.join(self.image_root_path, image_file)).with_suffix('.caption').read_text().strip()
        # print(text)
        # read image
        raw_image = set_background_white(os.path.join(self.image_root_path, image_file))
        raw_image.save('processed/miai_char_white/' + image_file)
        # text = '''A calligraphic character "{}", with elements like "{}". Details as “{}”.'''.format(text, text, ele_text)
        ele_image = raw_image
        ele_image_grey = ele_image.convert('L').convert('RGB')

 
        # 随机旋转和缩放增强
        if self.aug:
            rotation_angle = random.uniform(-30, 30)  # 随机旋转角度在-30到30度之间
            scale_factor = random.uniform(0.8, 1.2)  
            ele_image = ele_image.rotate(rotation_angle, resample=Image.BICUBIC)
            width, height = ele_image.size
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            ele_image = ele_image.resize((new_width, new_height), Image.BICUBIC)

        # original size
        original_width, original_height = raw_image.size
        original_size = torch.tensor([original_height, original_width])
        
        image_tensor = self.transform(raw_image.convert("RGB"))
        # random crop
        delta_h = image_tensor.shape[1] - self.size
        delta_w = image_tensor.shape[2] - self.size
        assert not all([delta_h, delta_w])
        
        if self.center_crop:
            top = delta_h // 2
            left = delta_w // 2
        else:
            top = np.random.randint(0, delta_h + 1)
            left = np.random.randint(0, delta_w + 1)
        image = transforms.functional.crop(
            image_tensor, top=top, left=left, height=self.size, width=self.size
        )
        crop_coords_top_left = torch.tensor([top, left]) 

        clip_image = self.clip_image_processor(images=ele_image, return_tensors="pt").pixel_values
        clip_image_grey = self.clip_image_processor(images=ele_image_grey, return_tensors="pt").pixel_values
        
        # drop
        drop_image_embed = 0
        rand_num = random.random()
        if rand_num < self.i_drop_rate:
            drop_image_embed = 1
        elif rand_num < (self.i_drop_rate + self.t_drop_rate):
            text = ""
        elif rand_num < (self.i_drop_rate + self.t_drop_rate + self.ti_drop_rate):
            text = ""
            drop_image_embed = 1

        # get text and tokenize
        text_input_ids = self.tokenizer(
            text,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids
        
        text_input_ids_2 = self.tokenizer_2(
            text,
            max_length=self.tokenizer_2.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids
        
        return {
            "image": image,
            "text_input_ids": text_input_ids,
            "text_input_ids_2": text_input_ids_2,
            "clip_image": [clip_image, clip_image_grey],
            "drop_image_embed": drop_image_embed,
            "original_size": original_size,
            "crop_coords_top_left": crop_coords_top_left,
            "target_size": torch.tensor([self.size, self.size]),
        }

from torch.utils.data import Dataset
from collections import defaultdict

class FontWordDataset(TheirDataset):
    def __init__(self, json_file, tokenizer, tokenizer_2, aug=False, size=1024, center_crop=True, t_drop_rate=0.05, i_drop_rate=0.05, ti_drop_rate=0.05, image_root_path=""):
        super().__init__(json_file, tokenizer, tokenizer_2, aug, size, center_crop, t_drop_rate, i_drop_rate, ti_drop_rate, image_root_path)
        self.font_list = os.listdir(image_root_path)
        self.font_dict = {}
        self.word_dict = defaultdict(list)
        self._load_data()
    
    def _load_data(self):
        for font_name in self.font_list:
            font_path = os.path.join(self.image_root_path, font_name)
            if not os.path.isdir(font_path):
                continue
            
            file_names = []
            for file_name in os.listdir(font_path):
                file_path = os.path.join(font_path, file_name)
                if file_name.endswith('.png'):
                    file_names.append(file_path)
                    word = os.path.splitext(file_name)[0]
                    self.word_dict[word].append(file_path)
            
            self.font_dict[font_name] = file_names
    
    def __len__(self):
        return len(self.font_dict)
    
    def __getitem__(self, idx):
        font_list = self.font_dict[self.font_list[idx]]
        ## 随机选择两个字
        tgt_idx, ref_idx = random.sample(font_list, 2)
        tgt_img, ref_img = Image.open(tgt_idx), Image.open(ref_idx)
        tgt_word = os.path.basename(ref_idx).split('.')[0]
        ref_word_list = self.word_dict[tgt_word]
        ref_word = Image.open(random.choice(ref_word_list))
        text = "Generate the Black and White glyph of word {} as the ref style.".format(ref_word)
        tgt_img.save('debug/tgt_image.png')
        ref_img.save('debug/ref_img.png')
        ref_word.save('debug/ref_word.png')
        raw_image = tgt_img
        # text = '''A calligraphic character "{}", with elements like "{}". Details as “{}”.'''.format(text, text, ele_text)
        ele_image = ref_img
        # ele_image_grey = ele_image.convert('L')
        # ele_image_grey = ele_image_grey.convert('RGB')
        ele_image_grey = ref_word
        # 随机旋转和缩放增强
        if self.aug:
            rotation_angle = random.uniform(-30, 30)  # 随机旋转角度在-30到30度之间
            scale_factor = random.uniform(0.8, 1.2)  
            ele_image = ele_image.rotate(rotation_angle, resample=Image.BICUBIC)
            width, height = ele_image.size
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            ele_image = ele_image.resize((new_width, new_height), Image.BICUBIC)

        # original size
        original_width, original_height = raw_image.size
        original_size = torch.tensor([original_height, original_width])
        
        image_tensor = self.transform(raw_image.convert("RGB"))
        # random crop
        delta_h = image_tensor.shape[1] - self.size
        delta_w = image_tensor.shape[2] - self.size
        assert not all([delta_h, delta_w])
        
        if self.center_crop:
            top = delta_h // 2
            left = delta_w // 2
        else:
            top = np.random.randint(0, delta_h + 1)
            left = np.random.randint(0, delta_w + 1)
        image = transforms.functional.crop(
            image_tensor, top=top, left=left, height=self.size, width=self.size
        )
        crop_coords_top_left = torch.tensor([top, left]) 

        clip_image = self.clip_image_processor(images=ele_image, return_tensors="pt").pixel_values
        clip_image_grey = self.clip_image_processor(images=ele_image_grey, return_tensors="pt").pixel_values
        
        # drop
        drop_image_embed = 0
        rand_num = random.random()
        if rand_num < self.i_drop_rate:
            drop_image_embed = 1
        elif rand_num < (self.i_drop_rate + self.t_drop_rate):
            text = ""
        elif rand_num < (self.i_drop_rate + self.t_drop_rate + self.ti_drop_rate):
            text = ""
            drop_image_embed = 1

        # get text and tokenize
        text_input_ids = self.tokenizer(
            text,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids
        
        text_input_ids_2 = self.tokenizer_2(
            text,
            max_length=self.tokenizer_2.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids
        
        return {
            "image": image,
            "text_input_ids": text_input_ids,
            "text_input_ids_2": text_input_ids_2,
            "clip_image": [clip_image, clip_image_grey],
            "drop_image_embed": drop_image_embed,
            "original_size": original_size,
            "crop_coords_top_left": crop_coords_top_left,
            "target_size": torch.tensor([self.size, self.size]),
        }
        
        return 0

def collate_fn(data):
    images = torch.stack([example["image"] for example in data])
    # rgb_images = torch.stack([example["rgb_image"] for example in data])
    text_input_ids = torch.cat([example["text_input_ids"] for example in data], dim=0)
    text_input_ids_2 = torch.cat([example["text_input_ids_2"] for example in data], dim=0)
    clip_images = torch.cat([example["clip_image"][0] for example in data], dim=0)
    clip_images_grey = torch.cat([example["clip_image"][1] for example in data], dim=0)
    drop_image_embeds = [example["drop_image_embed"] for example in data]
    original_size = torch.stack([example["original_size"] for example in data])
    crop_coords_top_left = torch.stack([example["crop_coords_top_left"] for example in data])
    target_size = torch.stack([example["target_size"] for example in data])

    return {
        "images": images,
        # "rgb_image": rgb_images,
        "text_input_ids": text_input_ids,
        "text_input_ids_2": text_input_ids_2,
        "clip_images": [clip_images, clip_images_grey],
        "drop_image_embeds": drop_image_embeds,
        "original_size": original_size,
        "crop_coords_top_left": crop_coords_top_left,
        "target_size": target_size,
    }
    

class IPAdapter(torch.nn.Module):
    """IP-Adapter"""
    def __init__(self, unet, image_proj_model, adapter_modules, ckpt_path=None):
        super().__init__()
        self.unet = unet
        self.image_proj_model = image_proj_model
        self.adapter_modules = adapter_modules

        if ckpt_path is not None:
            self.load_from_checkpoint(ckpt_path)

    def forward(self, noisy_latents, timesteps, encoder_hidden_states, unet_added_cond_kwargs, image_embeds):
        ip_tokens = self.image_proj_model(image_embeds)
        encoder_hidden_states = torch.cat([encoder_hidden_states, ip_tokens], dim=1)
        # Predict the noise residual
        noise_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states, added_cond_kwargs=unet_added_cond_kwargs).sample
        return noise_pred

    def load_from_checkpoint(self, ckpt_path: str):
        # Calculate original checksums
        orig_ip_proj_sum = torch.sum(torch.stack([torch.sum(p) for p in self.image_proj_model.parameters()]))
        orig_adapter_sum = torch.sum(torch.stack([torch.sum(p) for p in self.adapter_modules.parameters()]))

        state_dict = torch.load(ckpt_path, map_location="cpu")

        # Load state dict for image_proj_model and adapter_modules
        self.image_proj_model.load_state_dict(state_dict["image_proj"], strict=True)
        self.adapter_modules.load_state_dict(state_dict["ip_adapter"], strict=True)

        # Calculate new checksums
        new_ip_proj_sum = torch.sum(torch.stack([torch.sum(p) for p in self.image_proj_model.parameters()]))
        new_adapter_sum = torch.sum(torch.stack([torch.sum(p) for p in self.adapter_modules.parameters()]))

        # Verify if the weights have changed
        assert orig_ip_proj_sum != new_ip_proj_sum, "Weights of image_proj_model did not change!"
        assert orig_adapter_sum != new_adapter_sum, "Weights of adapter_modules did not change!"

        print(f"Successfully loaded weights from checkpoint {ckpt_path}")
    
def load_model(path, model):
    ckpt_dirs = os.listdir(path)
    ckpt_dirs = sorted(ckpt_dirs, key=lambda p: int(p.split('-')[-1]))
    ckpt_path = os.path.join(path, ckpt_dirs[-1], 'pytorch_model.pt')
    print("Resume from ", ckpt_path)
    model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
    return int(ckpt_dirs[-1].split('-')[-1])

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="../autodl-tmp/sdxl_models",
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_ip_adapter_path",
        type=str,
        default=None,
        help="Path to pretrained ip adapter model. If not specified weights are initialized randomly.",
    )
    parser.add_argument(
        "--data_json_file",
        type=str,
        default='processed/meta/processed_yeqing_char.json',
        # required=True,
        help="Training data",
    )
    parser.add_argument(
        "--data_root_path",
        type=str,
        default="processed/xhs_char",
        # required=True,
        help="Training data root path",
    )
    parser.add_argument(
        "--image_encoder_path",
        type=str,
        default="vit",
        # required=True,
        help="Path to CLIP image encoder",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="debug",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images"
        ),
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument("--num_train_steps", type=int, default=10000)
    parser.add_argument(
        "--train_batch_size", type=int, default=2, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--noise_offset", type=float, default=None, help="noise offset")
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=2000,
        help=(
            "Save a checkpoint of the training state every X updates"
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='Number of gradient accumulation steps')

    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--img_loss", type=float, default=0.0, help="For distributed training: local_rank")
    parser.add_argument("--denoise_loss", type=float, default=1.0, help="For distributed training: local_rank")
    parser.add_argument("--train_type", type=int, default=0, help="For distributed training: local_rank")
    parser.add_argument("--use_pe", type=int, default=0, help="For distributed training: local_rank")
    parser.add_argument("--resume", type=int, default=0, help="For distributed training: local_rank")

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    # print(env_local_rank)
    # print(torch.cuda.get_device_name())
    # exit()
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args
    
def ddim_reverse_step(xt, noise, alpha_t):
    sqrt_alpha_t = torch.sqrt(alpha_t)
    sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t)
    x0 = (xt - sqrt_one_minus_alpha_t * noise) / sqrt_alpha_t
    return x0

def main():
    args = parse_args()
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        gradient_accumulation_steps=args.gradient_accumulation_steps

    )
    
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
            
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        
    if args.use_pe:
        pe = PositionalEncoding(2048).to(accelerator.device, dtype=weight_dtype)
        print("#### Use position embedding! ####")
    else:
        pe = None
    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    tokenizer_2 = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer_2")
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder_2")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(args.image_encoder_path)
    # freeze parameters of models to save more memory
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    text_encoder_2.requires_grad_(False)
    image_encoder.requires_grad_(False)
    
    #ip-adapter
    num_tokens = 4
    image_proj_model = ImageProjModel(
        cross_attention_dim=unet.config.cross_attention_dim,
        clip_embeddings_dim=image_encoder.config.projection_dim,
        clip_extra_context_tokens=num_tokens,
        pe=pe
    )
    # init adapter modules
    if args.output_dir.endswith('rev'):
        from ip_adapter.attention_processor import IPAttnProcessorDoubleZeroRev as IPAttnProcessor
        print("### Reverse image and textual test ###")
    elif args.output_dir.endswith('single'):
        from ip_adapter.attention_processor import IPAttnProcessorSingle as IPAttnProcessor
        print("### Single info test ###")
    elif args.output_dir.endswith('singledffn'):
        from ip_adapter.attention_processor import IPAttnProcessorSingleDFFN as IPAttnProcessor
        print("### Single and DFFN test ###")
    elif args.output_dir.endswith('1s'):
        from ip_adapter.attention_processor import IPAttnProcessorDouble as IPAttnProcessor
        print("### Double and FFN test ###")
    else:
        from ip_adapter.attention_processor import IPAttnProcessorDoubleZero as IPAttnProcessor
        print("### Right Round ###")

    attn_procs = {}
    unet_sd = unet.state_dict()
    for name in unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]
        if cross_attention_dim is None:
            attn_procs[name] = AttnProcessor()
        else:
            layer_name = name.split(".processor")[0]
            weights = {
                "to_k_ip.weight": unet_sd[layer_name + ".to_k.weight"],
                "to_v_ip.weight": unet_sd[layer_name + ".to_v.weight"],
                # "to_out.0.weight": unet_sd[layer_name + ".to_out.0.weight"],
                # "to_out.0.bias": unet_sd[layer_name + ".to_out.0.bias"],
                # "to_out.1.weights": unet_sd[layer_name + ".to_out.1.weights"],
                # "to_out.1.bias": unet_sd[layer_name + ".to_out.1.weights"],
            }
            attn_procs[name] = IPAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, num_tokens=num_tokens)
            attn_procs[name].load_state_dict(weights, strict=False)
    unet.set_attn_processor(attn_procs)
    adapter_modules = torch.nn.ModuleList(unet.attn_processors.values())
    
    ip_adapter = IPAdapter(unet, image_proj_model, adapter_modules, args.pretrained_ip_adapter_path)
    if args.resume:
        global_step = load_model(args.output_dir, ip_adapter)
    else:
        global_step = 0
    
    #unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device) # use fp32
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    text_encoder_2.to(accelerator.device, dtype=weight_dtype)
    image_encoder.to(accelerator.device, dtype=weight_dtype)
    
    # optimizer
    params_to_opt = itertools.chain(ip_adapter.image_proj_model.parameters(),  ip_adapter.adapter_modules.parameters())
    optimizer = torch.optim.AdamW(params_to_opt, lr=args.learning_rate, weight_decay=args.weight_decay)
    
    # dataloader
    if args.train_type==4:
        print("###train with dataset easychar")
        train_dataset = TheirDataset4(args.data_json_file, tokenizer=tokenizer, tokenizer_2=tokenizer_2, size=args.resolution, image_root_path=args.data_root_path)
    elif args.train_type==0:
        print("###train with dataset 0")
        train_dataset = TheirDataset(args.data_json_file, tokenizer=tokenizer, tokenizer_2=tokenizer_2, size=args.resolution, image_root_path=args.data_root_path)
    elif args.train_type==2:
        train_dataset = TheirDataset2(args.data_json_file, tokenizer=tokenizer, tokenizer_2=tokenizer_2, size=args.resolution, image_root_path=args.data_root_path)
        print("###train with dataset 2")
    elif args.train_type == 3:
        print("###train with dataset 3")
        train_dataset = TheirDataset3(args.data_json_file, tokenizer=tokenizer, tokenizer_2=tokenizer_2, size=args.resolution, image_root_path=args.data_root_path)
    elif args.train_type == 5 :
        print("###train with dataset color")
        train_dataset = TheirDatasetColor(args.data_json_file, tokenizer=tokenizer, tokenizer_2=tokenizer_2, size=args.resolution, image_root_path=args.data_root_path)
    elif args.train_type == 6 :
        print("###train with dataset 6")
        train_dataset = TheirDataset6(args.data_json_file, tokenizer=tokenizer, tokenizer_2=tokenizer_2, size=args.resolution, image_root_path=args.data_root_path)
    elif args.train_type == 7 :
        print("###train with dataset Fontword")
        train_dataset = FontWordDataset(args.data_json_file, tokenizer=tokenizer, tokenizer_2=tokenizer_2, size=args.resolution, image_root_path=args.data_root_path)
    
    print("###train with datasize ", len(train_dataset))
    # train_dataset = YourDataset(args.data_json_file, tokenizer=tokenizer, tokenizer_2=tokenizer_2, size=args.resolution, image_root_path=args.data_root_path)
    # train_dataset = MyDataset(args.data_json_file, tokenizer=tokenizer, tokenizer_2=tokenizer_2, size=args.resolution, image_root_path=args.data_root_path)
    
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )
    
    # Prepare everything with our `accelerator`.
    
    ip_adapter, optimizer, train_dataloader = accelerator.prepare(ip_adapter, optimizer, train_dataloader)
 
    for epoch in range(0, args.num_train_epochs):
        begin = time.perf_counter()
        for step, batch in enumerate(train_dataloader):
            load_data_time = time.perf_counter() - begin
            with accelerator.accumulate(ip_adapter):
                # Convert images to latent space
                with torch.no_grad():
                    # vae of sdxl should use fp32
                    latents = vae.encode(batch["images"].to(accelerator.device, dtype=torch.float32)).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor
                    latents = latents.to(accelerator.device, dtype=weight_dtype)

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                if args.noise_offset:
                    # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                    noise += args.noise_offset * torch.randn((latents.shape[0], latents.shape[1], 1, 1)).to(accelerator.device, dtype=weight_dtype)

                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                image_embeds__ = []
                with torch.no_grad():
                    for i in batch['clip_images']:
                        i.to(accelerator.device, dtype=weight_dtype)
                        image_embeds = image_encoder(i).image_embeds
                    # image_embeds = torch.cat(image_embeds, dim=0)
                        image_embeds_ = []
                        for image_embed, drop_image_embed in zip(image_embeds, batch["drop_image_embeds"]):
                            if drop_image_embed == 1:
                                image_embeds_.append(torch.zeros_like(image_embed))
                            else:
                                image_embeds_.append(image_embed)
                        image_embeds = torch.stack(image_embeds_)  ## B c h w
                        image_embeds__.append(image_embeds)
                image_embeds = image_embeds__

                with torch.no_grad():
                    encoder_output = text_encoder(batch['text_input_ids'].to(accelerator.device), output_hidden_states=True)
                    text_embeds = encoder_output.hidden_states[-2]
                    encoder_output_2 = text_encoder_2(batch['text_input_ids_2'].to(accelerator.device), output_hidden_states=True)
                    pooled_text_embeds = encoder_output_2[0]
                    text_embeds_2 = encoder_output_2.hidden_states[-2]
                    text_embeds = torch.concat([text_embeds, text_embeds_2], dim=-1) # concat
                        
                # add cond
                add_time_ids = [
                    batch["original_size"].to(accelerator.device),
                    batch["crop_coords_top_left"].to(accelerator.device),
                    batch["target_size"].to(accelerator.device),
                ]
                add_time_ids = torch.cat(add_time_ids, dim=1).to(accelerator.device, dtype=weight_dtype)
                unet_added_cond_kwargs = {"text_embeds": pooled_text_embeds, "time_ids": add_time_ids}
                
                noise_pred = ip_adapter(noisy_latents, timesteps, text_embeds, unet_added_cond_kwargs, image_embeds)
            
                if args.denoise_loss > 0.0:
                    loss = args.denoise_loss * F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
                else:
                    loss = 0.0
                ## decode x0
                if args.img_loss > 0.0:
                    sqrt_alpha_prod = noise_scheduler.alphas_cumprod[timesteps].to(device=noise_pred.device, dtype=noise_pred.dtype).flatten()
                    while len(sqrt_alpha_prod.shape) < len(noisy_latents.shape):
                        sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
                    x_pred = ddim_reverse_step(noisy_latents, noise_pred, sqrt_alpha_prod)
                    x_pred = vae.decode(x_pred.to(vae.dtype)).sample.to(noise_pred.dtype)
                    images = batch['rgb_image'].to(x_pred.device, x_pred.dtype)
                    rgb_loss = args.img_loss * F.mse_loss(x_pred.float(), images.float(), reduction="mean")
                    rgb_loss_item = rgb_loss.item()
                else:
                    rgb_loss = 0.0
                    rgb_loss_item = 0.0

                loss += rgb_loss
                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean().item()
                
                # Backpropagate
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
                torch.cuda.empty_cache()
                if accelerator.is_main_process:
                    print("Epoch {}, global step {}, data_time: {:.4f}, time: {:.4f}, step_loss_denoise: {}, rgb: {}".format(
                        epoch, global_step, load_data_time, time.perf_counter() - begin, avg_loss, rgb_loss_item))
            
            global_step += 1
            
            if global_step % args.save_steps == 0:
                save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                # save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                # accelerator.save_state(save_path)
                os.makedirs(save_path, exist_ok=True)
                torch.save(accelerator.get_state_dict(ip_adapter), f"{save_path}/pytorch_model.bin")
    
            
            begin = time.perf_counter()
            
    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
    os.makedirs(save_path, exist_ok=True)
    # accelerator.save_state(save_path)
    torch.save(accelerator.get_state_dict(ip_adapter), f"{save_path}/pytorch_model.pt")

                
if __name__ == "__main__":
    main()    
