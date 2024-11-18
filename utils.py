from PIL import Image, ImageFont, ImageDraw
import numpy as np
import cv2
from skimage import morphology

def set_background_white(image_path, th=205):
    ele_image = Image.open(image_path)
    
    # Convert image to NumPy array
    ele_image_np = np.array(ele_image)
    threshold = th

    colored_mask = (ele_image_np[:, :, 0] > threshold) & (ele_image_np[:, :, 1] > threshold) & (ele_image_np[:, :, 2] > threshold)

    ele_image_np[colored_mask] = 255
    return Image.fromarray(ele_image_np)

def extract_color_and_non_color_parts(image_path, th=[100,150]):
    ele_image = Image.open(image_path)
    # Convert image to NumPy array
    ele_image_np = np.array(ele_image)
    
    # Define the threshold for non-colored parts
    threshold = th[1]
    colored_mask = (ele_image_np[:, :, 0] > threshold) | (ele_image_np[:, :, 1] > threshold ) | (ele_image_np[:, :, 2] > threshold)
    threshold = th[0]
    # Create a mask for non-colored parts  黑色部分
    non_colored_mask = (ele_image_np[:, :, 0] < threshold) & (ele_image_np[:, :, 1] < threshold) & (ele_image_np[:, :, 2] < threshold)
    
    # Extract non-colored parts  创建一个全白的图
    black_image_np = np.ones_like(ele_image_np) * 255
    black_image_np[non_colored_mask] = ele_image_np[non_colored_mask]
    
    # Extract colored parts  
    colorful_image_np = np.ones_like(ele_image_np) * 255
    colorful_image_np[colored_mask] = ele_image_np[colored_mask]
    
    # Convert NumPy arrays back to PIL images
    colorful_image = Image.fromarray(colorful_image_np)
    black_image = Image.fromarray(black_image_np)
    
    ele_image_grey = ele_image.convert('L').convert('RGB')
    
    return colorful_image, black_image, ele_image_grey

def generate_single_char_image(char, font_path, char_size=400):
    image = Image.new('RGB', (512, 512), 'white')
    font = ImageFont.truetype(font_path, char_size)  # Adjust the font size as needed
    draw = ImageDraw.Draw(image)
    (left, top, right, bottom) = draw.textbbox((0, 0), char, font=font)
    text_width = max(right-left,5)
    text_height = max(bottom-top,5)
    position = ((512 - text_width) // 2 -left, (512 - text_height) // 2 - top)
    draw.text(position, char, font=font, fill='black')
    return image

def generate_chars_image(char,  font_path, image_size=(512, 512), char_size=None):
    image = Image.new('RGB', image_size, 'white')
    if char_size is None:
        char_size = min(image_size) // len(list(char)) * 0.9
    font = ImageFont.truetype(font_path, char_size)  # Adjust the font size as needed
    draw = ImageDraw.Draw(image)
    (left, top, right, bottom) = draw.textbbox((0, 0), char, font=font)
    text_width = max(right-left,5)
    text_height = max(bottom-top,5)
    position = ((512 - text_width) // 2 -left, (512 - text_height) // 2 - top)
    draw.text(position, char, font=font, fill='black')
    return image

def generate_chars_image_different_size(char,  font_path, image_size=(512, 512), char_size=None):
    image = Image.new('RGB', image_size, 'white')
    draw = ImageDraw.Draw(image)
    if char_size is None:
        # 默认情况下，首字母最大，后续字符逐渐变小
        base_size = min(image_size) // len(char) * 2.5
    else:
        base_size = char_size
    # char_sizes = [base_size - i * (base_size // len(char)) for i in range(len(char))]
    char_sizes =[base_size] + [base_size*0.6 for _ in range(len(char) - 1)]
    print(char_sizes)
    current_x = 50
    for i, c in enumerate(char):
        font = ImageFont.truetype(font_path, char_sizes[i])
        (left, top, right, bottom) = draw.textbbox((0, 0), c, font=font)
        text_width = max(right - left, 5)
        text_height = max(bottom - top, 5)
        position = (current_x, (image_size[1] - text_height) // 2 - top)
        draw.text(position, c, font=font, fill='black')
        current_x += text_width

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
    skeleton = 255 - skeleton
    skeleton_image = Image.fromarray(skeleton)
    
    return skeleton_image
from PIL import Image, ImageFont, ImageDraw
import numpy as np
import cv2
from skimage import morphology
from scipy.ndimage import median_filter
from PIL import Image

def set_background_white(image_path, th=205):
    ele_image = Image.open(image_path)
    
    # Convert image to NumPy array
    ele_image_np = np.array(ele_image)
    threshold = th

    colored_mask = (ele_image_np[:, :, 0] > threshold) & (ele_image_np[:, :, 1] > threshold) & (ele_image_np[:, :, 2] > threshold)

    ele_image_np[colored_mask] = 255
    return Image.fromarray(ele_image_np)
    
def extract_color_and_non_color_parts(image_path, th=150):
    ele_image = Image.open(image_path)
    
    # Convert image to NumPy array
    ele_image_np = np.array(ele_image)
    
    # Define the threshold for non-colored parts
    threshold = th
    colored_mask = (ele_image_np[:, :, 0] > threshold) | (ele_image_np[:, :, 1] > threshold) | (ele_image_np[:, :, 2] > threshold)
    # Create a mask for non-colored parts  黑色部分
    non_colored_mask = (ele_image_np[:, :, 0] < threshold) & (ele_image_np[:, :, 1] < threshold) & (ele_image_np[:, :, 2] < threshold)
    
    # Create a mask for colored parts
    
    # Extract non-colored parts  创建一个全白的图
    black_image_np = np.ones_like(ele_image_np) * 255
    black_image_np[non_colored_mask] = ele_image_np[non_colored_mask]
    
    # Extract colored parts  
    colorful_image_np = np.ones_like(ele_image_np) * 255
    colorful_image_np_2 = (255 - ele_image_np)

    threshold = 50
    colored_mask_2 = (colorful_image_np_2[:, :, 0] < threshold) & (colorful_image_np_2[:, :, 1] < threshold) & (colorful_image_np_2[:, :, 2] < threshold)
    colored_mask = colored_mask & (~colored_mask_2)
    colorful_image_np[colored_mask] = ele_image_np[colored_mask]
    ## 使用中值滤波进行杂色滤除
    # filtered_image_np = median_filter(colorful_image_np, size=3)
    # Convert NumPy arrays back to PIL images
    colorful_image = Image.fromarray(colorful_image_np)
    black_image = Image.fromarray(black_image_np)
    
    ele_image_grey = ele_image.convert('L').convert('RGB')
    
    return colorful_image, black_image, ele_image_grey

def generate_english_words_image(char, font_path, image_size=(512, 512), char_size=None):
    # Ensure the word is "Spring" with the first lette
    # r capitalized
    # print(len(list(char)))
    image = Image.new('RGB', (512, 512), 'white')
    base_size = 512 / len(list(char)) 
    for i in np.arange(1.0, 2.0, 0.25):
        font = ImageFont.truetype(font_path, i*base_size)  # Adjust the font size as needed
        draw = ImageDraw.Draw(image)
        (left, top, right, bottom) = draw.textbbox((0, 0), char, font=font)
        # print((left, top, right, bottom) )
        text_width = max(right-left,5)
        text_height = max(bottom-top,5)
        if text_width < 400:
            continue
        else:
            break
    # print(text_width, text_height)
    position = ((512 - text_width) // 2 -left, (512 - text_height) // 2 - top)
    draw.text(position, char, font=font, fill='black')
    
    return image

from skimage.morphology import skeletonize

def extract_skeleton(image, iter=1):
    image_np = 255 - np.array(image)
    # Convert to grayscale
    gray_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    # Apply binary thresholding
    _, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
    # Apply cross-shaped morphological filter
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
    filtered_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)
    # Skeletonize the image
    skeleton = skeletonize(filtered_image // 255)  # Convert to binary (0, 1) for skeletonize
    # Convert skeleton back to uint8
    skeleton_image = (skeleton * 255).astype(np.uint8)
        # Define the dilation kernel
    dilation_kernel = np.ones((3, 3), np.uint8)
    
    # Dilate the skeleton image
    dilated_skeleton = cv2.dilate(skeleton_image, dilation_kernel, iterations=iter)
    
    return Image.fromarray(255-dilated_skeleton)

if __name__ == "__main__":
    # g_image = generate_svg_img_from_char('花', f'沐瑶软笔手写体')
    # print(g_image.size)
    # g_image = generate_chars_image_different_size('Flower', f'AnyText/font/Alibaba-PuHuiTi-Bold.ttf')
    # # print(1)
    # g_image.save('debug/chuntian.png')
    # print("save to debug/chuntian.png")
    extract_color_and_non_color_parts('/root/IP-Adapter2/processed/yeqing_char/人生海海.png')