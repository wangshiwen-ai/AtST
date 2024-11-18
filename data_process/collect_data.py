# url = 'https://www.xiaohongshu.com/user/profile/604ce2970000000001008a1a'
## 用于读取图片链接并写入数据
import requests
import os
## 用于浏览器登录
from DrissionPage import ChromiumPage
import time
import json
import re
import argparse

parser = argparse.ArgumentParser(description="Simple example of a training script.")
parser.add_argument(
        "--user_name",
        type=str,
        default="yeqing",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
parser.add_argument(
        "--user_link",
        type=str,
        default="https://www.xiaohongshu.com/user/profile/5b08220ff7e8b904cfdd5ee3",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )

args = parser.parse_args()
user_name = args.user_name
user_link = args.user_link

pattern = re.compile(r'[^\u4e00-\u9fa5a-zA-Z0-9]')
image_path = f'data/{user_name}_data'

os.makedirs(image_path, exist_ok=True)

class XHS:    
    def __init__(self):
        ch = ChromiumPage()
        ch.get("https://www.xiaohongshu.com/explore")
        self.ch = ch

    def save_image_from_post(self, d):
        link_list = []
        self.ch.get(d['link'])
        name = pattern.sub('', d['name'])
        if os.path.exists(f'{image_path}/{name}'):
            print(f'{image_path}/{name} exist, continue')
            return []
        img_list = self.ch.eles('.note-slider-img')# //*[@id="noteContainer"]/div[2]/div/div/div[2]/div/div[9]/div/img
        for i,img in enumerate(img_list):
            link_list.append(img.link)
            request = requests.get(img.link)
            image_data = request.content
            # print(image_data)
            os.makedirs(f'{image_path}/{name}', exist_ok=True)
            with open(image_path+f'/{name}/{i}.png', 'wb') as f:
                f.write(image_data)
                print("Write to ", image_path+f'/{name}/{i}.png')
        return link_list
    
    def spider(self, user_link, t=10):
        self.ch.get(user_link)
        link_list = []
        img_link_list = []
        for _ in range(1, t):    
            time.sleep(2)
            ele = self.ch.eles('.cover ld mask')
            name_ele = self.ch.eles('.title')
            print(len(ele))
            for href,name in zip(ele,name_ele):
                lian = href.link
                na = name.text
                print(na,lian)
                link_list.append({'name':na, 'link': lian})
            self.ch.scroll.to_bottom()

        print("Scroll done\n")

        with open(f'{user_name}.json', 'w', encoding='utf-8') as f:
            json.dump(link_list, f, indent=4, ensure_ascii=False)
            print("write to ", f'{user_name}.json')
        for d in link_list:
            img_link_list.extend(self.save_image_from_post(d))

        print(1)

xhs = XHS()
xhs.spider(user_link)

