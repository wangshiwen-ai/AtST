import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

from PIL import Image, ImageFont, ImageDraw
import numpy as np
import cv2
from skimage import morphology

def generate_single_char_image(char, font_path):
    image = Image.new('RGB', (512, 512), 'white')
    font = ImageFont.truetype(font_path, 400)  # Adjust the font size as needed
    draw = ImageDraw.Draw(image)
    (left, top, right, bottom) = draw.textbbox((0, 0), char, font=font)
    text_width = max(right-left,5)
    text_height = max(bottom-top,5)
    position = ((512 - text_width) // 2 -left, (512 - text_height) // 2 - top)
    draw.text(position, char, font=font, fill='black')
    return image

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

    skeleton = cv2.cvtColor(skeleton, cv2.COLOR_GRAY2RGB)
    # Convert numpy array back to PIL image
    # 骨架太细了，需要膨胀一下
    kernel = np.ones((3, 3), np.uint8)  # 定义膨胀核
    skeleton = cv2.dilate(skeleton, kernel, iterations=1)
    
    skeleton_image = Image.fromarray(skeleton)
    
    return skeleton_image

attn_maps = {}
def hook_fn(name):
    def forward_hook(module, input, output):
        if hasattr(module.processor, "attn_map"):
            attn_maps[name] = module.processor.attn_map
            del module.processor.attn_map

    return forward_hook

def register_cross_attention_hook(unet):
    for name, module in unet.named_modules():
        if name.split('.')[-1].startswith('attn2'):
            module.register_forward_hook(hook_fn(name))

    return unet

def upscale(attn_map, target_size):
    attn_map = torch.mean(attn_map, dim=0)
    attn_map = attn_map.permute(1,0)
    temp_size = None

    for i in range(0,5):
        scale = 2 ** i
        if ( target_size[0] // scale ) * ( target_size[1] // scale) == attn_map.shape[1]*64:
            temp_size = (target_size[0]//(scale*8), target_size[1]//(scale*8))
            break

    assert temp_size is not None, "temp_size cannot is None"

    attn_map = attn_map.view(attn_map.shape[0], *temp_size)

    attn_map = F.interpolate(
        attn_map.unsqueeze(0).to(dtype=torch.float32),
        size=target_size,
        mode='bilinear',
        align_corners=False
    )[0]

    attn_map = torch.softmax(attn_map, dim=0)
    return attn_map

def get_net_attn_map(image_size, batch_size=2, instance_or_negative=False, detach=True):

    idx = 0 if instance_or_negative else 1
    net_attn_maps = []

    for name, attn_map in attn_maps.items():
        attn_map = attn_map.cpu() if detach else attn_map
        attn_map = torch.chunk(attn_map, batch_size)[idx].squeeze()
        attn_map = upscale(attn_map, image_size) 
        net_attn_maps.append(attn_map) 

    net_attn_maps = torch.mean(torch.stack(net_attn_maps,dim=0),dim=0)

    return net_attn_maps

def attnmaps2images(net_attn_maps):

    #total_attn_scores = 0
    images = []

    for attn_map in net_attn_maps:
        attn_map = attn_map.cpu().numpy()
        #total_attn_scores += attn_map.mean().item()

        normalized_attn_map = (attn_map - np.min(attn_map)) / (np.max(attn_map) - np.min(attn_map)) * 255
        normalized_attn_map = normalized_attn_map.astype(np.uint8)
        #print("norm: ", normalized_attn_map.shape)
        image = Image.fromarray(normalized_attn_map)

        #image = fix_save_attn_map(attn_map)
        images.append(image)

    #print(total_attn_scores)
    return images

def is_torch2_available():
    return hasattr(F, "scaled_dot_product_attention")

def get_generator(seed, device):

    if seed is not None:
        if isinstance(seed, list):
            generator = [torch.Generator(device).manual_seed(seed_item) for seed_item in seed]
        else:
            generator = torch.Generator(device).manual_seed(seed)
    else:
        generator = None

    return generator