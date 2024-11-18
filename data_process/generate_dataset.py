from PIL import Image, ImageDraw, ImageFont
import os
import argparse

parser = argparse.ArgumentParser(description="Simple example of a training script.")
parser.add_argument(
        "--font_name",
        type=str,
        default="yeqing",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
args = parser.parse_args()
# 定义字符集合
chinese_chars = "的一是在不了有和人这中大为上个国我以要他时来用们生到作地于出就分对成会可主发年动同工也能下过子说产种面而方后多定行学法所民得经十三之进着等部度家电力里如水化高自二理起小物现实加量都两体制机当使点从业本去把性好应开它合还因由其些然前外天政四日那社义事平形相全表间样与关各重新线内数正心反你明看原又么利比或但质气第向道命此变条只没结解问意建月公无系军很情者最立代想已通并提直题党程展五果料象员革位入常文总次品式活设及管特件长求老头基资边流路级少图山统接知较将组见计别她手角期根论运农指几九区强放决西被干做必战先回则任取据处队南给色光门即保治北造百规热领七海口东导器压志世金增争济阶油思术极交受联什认六共权收证改清己美再采转更单风切打白教速花带安场身车例真务具万每目至达走积示议声报斗完类八离华名确才科张认马节话米整空元况今集温传土许步群广石记需段研界拉林律叫且究观越织装影算低持音众书布复容儿须际商非验连断深难近矿千周委素技备半办青省列习响约支般史感劳便团往酸历市克何除消构府太准精值号率族维划选标写存候毛亲快效斯院查江型眼王按格养易置派层片始却专状育厂京识适属圆包火住调满县局照参红细引听该铁价严龙飞"
english_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
all_chars = chinese_chars + english_chars

# 字体文件夹路径
font_folder = "/root/IP-Adapter2/AnyText/font"
output_folder = "/root/autodl-tmp/train"
from tqdm import tqdm
# 创建输出文件夹
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

font_list = os.listdir(font_folder)
font_list = [f'{args.font_name}.ttf']
# 遍历字体文件
for font_file in tqdm(font_list):
    if font_file.endswith(".ttf"):
        font_path = os.path.join(font_folder, font_file)
        font_name = os.path.splitext(font_file)[0]
        font_output_folder = os.path.join(output_folder, font_name)
        
        # 创建字体输出文件夹
        if not os.path.exists(font_output_folder):
            os.makedirs(font_output_folder)
        
        # 加载字体
        font = ImageFont.truetype(font_path, 512)
        
        for char in all_chars:
            # 创建白底图像
       
            image = Image.new('RGB', (512, 512), 'white')
            font = ImageFont.truetype(font_path, 400)  # Adjust the font size as needed
            draw = ImageDraw.Draw(image)
            (left, top, right, bottom) = draw.textbbox((0, 0), char, font=font)
            text_width = max(right-left,5)
            text_height = max(bottom-top,5)
            position = ((512 - text_width) // 2 -left, (512 - text_height) // 2 - top)
            draw.text(position, char, font=font, fill='black')
           
            # 保存图像
            char_image_path = os.path.join(font_output_folder, f"{char}.png")
            image.save(char_image_path)
            print("save to ",char_image_path )