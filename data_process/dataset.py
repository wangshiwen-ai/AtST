from torch.utils.data import Dataset
from collections import defaultdict
import os
import glob
import random
from PIL import Image
from paddleocr import PaddleOCR
import shutil
ocr = PaddleOCR(use_angle_cls=True, lang='ch')

def process_char_dict(char_dict, output_dir):
    for font_name, file_list in char_dict.items():
        for file_path in file_list:
            expected_char = os.path.splitext(os.path.basename(file_path))[0]
            try:
                result = ocr.ocr(file_path, cls=True)
            except:
                new_dir = os.path.join(output_dir, f"{font_name}")
                os.makedirs(new_dir, exist_ok=True)
                shutil.copy(file_path, os.path.join(new_dir,f'_{expected_char}.png'))
                continue

            recognized_text = result[0][1][0] if result else ""
            
            
            if recognized_text != expected_char:
                new_dir = os.path.join(output_dir, f"{font_name}")
                os.makedirs(new_dir, exist_ok=True)
                shutil.copy(file_path, os.path.join(new_dir,f'_{expected_char}.png'))

class FontWordDataset(Dataset):
    def __init__(self, root_path='/root/autodl-tmp/dataset/train'):
        self.root_path = root_path
        self.font_list = os.listdir(root_path)
        self.font_dict = {}
        self.word_dict = defaultdict(list)
        self._load_data()
    
    def _load_data(self):
        for font_name in self.font_list:
            font_path = os.path.join(self.root_path, font_name)
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
        output_dir = 'tmp/'
        process_char_dict(self.font_dict, output_dir)
    
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
        txt = "Generate the Black and White glyph of word {} as the ref style.".format(ref_word)
        tgt_img.save('debug/tgt_image.png')
        ref_img.save('debug/ref_img.png')
        ref_word.save('debug/ref_word.png')
        
        return 0

if __name__ == "__main__":
    dataset = FontWordDataset()
    dataset[0]
    print(len(dataset))
