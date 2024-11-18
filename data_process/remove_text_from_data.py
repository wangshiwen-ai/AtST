
from paddleocr import PaddleOCR, draw_ocr
from PIL import Image

# Paddleocr目前支持的多语言语种可以通过修改lang参数进行切换
# 例如`ch`, `en`, `fr`, `german`, `korean`, `japan`
ocr = PaddleOCR(use_angle_cls=True, lang="ch")  # need to run only once to download and load model into memory
# img_path = '/home/swwang/IP-Adapter/xhs_data/汉字写意111/0.png'
# result = ocr.ocr(img_path, cls=True)
# for idx in range(len(result)):
#     res = result[idx]
#     for line in res:
#         if line[1][2] > 0.8:
#             print(line)

import os
import glob
# data_dir = '../xhs_data'
data_dir = '/home/swwang/IP-Adapter/exp/hua__data_wsw_IP-Adapter_sd-idapter-2_checkpoint-1500_ip_adapter.bin'
data_dir = '/root/autodl-tmp/dataset/train'
posts = os.listdir(data_dir)
# print(posts)
# img_list = []
# for p in posts:
#     img_dir = os.path.join(data_dir, p)
#     img_list.extend(glob.glob(img_dir+'/*.png'))
import glob
img_list = []
for p in posts:
    img_p = os.path.join(data_dir, p)
    img_list.extend(glob.glob(img_p + '/*.png'))


import multiprocessing
from multiprocessing import Process
from PIL import Image, ImageDraw

def process_image_with_ocr_boxes(image, ocr_boxes, padding=0.):
    # 打开图像
    draw = ImageDraw.Draw(image)
    for ocr_box in ocr_boxes:
        box = ocr_box[0]
        score = ocr_box[1][1]
        if score > 0.8:
            # 加上 padding
            # left top right bottom
            padded_box = [
                box[0][0]*(1-padding), box[0][1]*(1-padding), box[2][0]*(1+padding), box[2][1]*(1+padding)
            ]
            # 绘制白色填充框
            draw.rectangle(padded_box, fill='white')

    # 保存处理后的图像
    return image
# result_dir = '../processed'
result_dir = 'ocr_fontimage'
faults_list = []
os.makedirs(result_dir, exist_ok=True)

def ocr_image(img_path):
    result = ocr.ocr(img_path, cls=True)
    image = Image.open(img_path).convert('RGB')
    try:
        for idx in range(len(result)):
            res = result[idx]
            for line in res:
                print(line)
        result = result[0]
        
        boxes = [line[0] for line in result]
        txts = [line[1][0] for line in result]
        scores = [line[1][1] for line in result]
        im_show = draw_ocr(image, boxes, txts, scores, font_path='/home/swwang/IP-Adapter/PaddleOCR/doc/fonts/simfang.ttf')
        img_path = img_path.strip('../')
        save_dir = os.path.dirname(img_path)
        os.makedirs(f'{result_dir}/{save_dir}', exist_ok=True)
        im_show = Image.fromarray(im_show)
    except:
        im_show = image
        print("No result")
    im_show.save(f'{result_dir}/{img_path}')
    print("Save to ", f'{result_dir}/{img_path}')
    # im_show = Image.fromarray(im_show)
    # im_show.save('result.jpg')
import shutil
def process_img(img_path):
    # print("Processsing ", img_path)
    result = ocr.ocr(img_path, cls=True)
    result = result[0]
    # try:
    #     im_show = process_image_with_ocr_boxes(image, result)
    # except:
    #     print("Some fault in imags", img_path)
    #     faults_list.append(img_path)
    #     im_show = image
    new_path = os.path.join('tmp', img_path.strip(data_dir))
    os.makedirs(os.path.dirname(new_path), exist_ok=True)
    char = os.path.basename(img_path).split('.')[0]
    if result is not None:
        if result[0][1][0] != char:
            print(img_path, result[0][1][0])
            shutil.copy(img_path, new_path)
    else:
        print(img_path)
        shutil.copy(img_path, new_path)
    # img_path = img_path.strip('../')
    # save_dir = os.path.dirname(img_path)
    # os.makedirs(f'{result_dir}/{save_dir}', exist_ok=True)
    # im_show.save(f'{result_dir}/{img_path}')
    # print("Save to ", f'{result_dir}/{img_path}')

for img in img_list:
    # process_img(img)
    process_img(img)

# with open('faults.txt', 'w') as f:
#     f.write()
# def init_cuda():
#     # 初始化 CUDA 驱动程序
#     import paddle
#     paddle.device.set_device('gpu')

# with multiprocessing.Pool(processes=4,initializer=init_cuda) as pool:  # 4 是进程数，可以根据需要调整
#     # 使用 map 方法并行处理任务
#     results = pool.map(process_img, img_list)

# def print_fun(i):
#     print(i)

# if __name__ == '__main__':
#     p_list = []
#     for i in img_list:
#         p=Process(target=print_fun,args=(i,))
#         p.start()
#         p_list.append(p)

#     print(p_list)
#     for p in p_list:
#         p.join()

# 显示结果
# from PIL import Image
# result = result[0]
# image = Image.open(img_path).convert('RGB')
# boxes = [line[0] for line in result]
# txts = [line[1][0] for line in result]
# scores = [line[1][1] for line in result]
# im_show = draw_ocr(image, boxes, txts, scores, font_path='/home/swwang/IP-Adapter/PaddleOCR/doc/fonts/simfang.ttf')
# im_show = Image.fromarray(im_show)
# im_show.save('result.jpg')
