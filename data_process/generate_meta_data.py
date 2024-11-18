data_dir = 'processed/yeqing_char'
# element_dir = 'processed/xhs_char_element'
data_meta_dir = 'processed/meta'

import glob
import os
import shutil
import re

os.makedirs(data_meta_dir, exist_ok=True)
paths = os.listdir(data_dir)
paths = [{"image_file": p, "text": re.sub(r'\d+', '', p.split('.')[0]), "char_name": p.split('.')[0]} for p in paths]

# if element_dir:
#     element_paths = os.listdir(element_dir)
# else:
#     element_paths = []

# print(len(paths))
# print(len(element_paths))
# import re

# paths = [{"image_file": p, "text": re.sub(r'\d+', '', p.split('.')[0]), "char_name": p.split('.')[0]} for p in paths]
# char_to_ele = {}
# for ele in element_paths:
#     # print(ele)
#     char_name = ele.split('.')[0].split('_')[0]
#     description = ele.split('.')[0].split('_')[-1]
    
#     if description:
#         description = description.strip('e') + '，水墨风'
#     else:
#         description = description + '水墨风'

#     if char_to_ele and char_name in char_to_ele:
#         char_to_ele[char_name]['description'].append(description)
#         char_to_ele[char_name]['path'].append(f'{element_dir}/{ele}')
#     else:
#         char_to_ele[char_name] = {}
#         char_to_ele[char_name]['description'] = [description]
#         char_to_ele[char_name]['path'] = [f'{element_dir}/{ele}']

# for p in paths:
#     if p['char_name'] in char_to_ele:
#         p['char_name'] = char_to_ele[p['char_name']]
#     else:
#         f = p['image_file']
#         p['char_name'] = {'description': ["水墨画"], 'path': [f'{data_dir}/{f}']}

import json

output_file_path = os.path.join(data_meta_dir, data_dir.replace('/', '_') + '.json')
print("#### SIZE ", len(paths), " ####")

with open(output_file_path, 'w', encoding='utf-8') as f:
    json.dump(paths, f, ensure_ascii=False, indent=4)
    print("Save to ", output_file_path)

# with open(os.path.join(data_meta_dir, data_dir.replace('/', '_')+'.json'), 'w', encoding='utf-8') as f:
#     json.dump(paths,f, ensure_ascii=False, indent=4)
#     print("Save to ", os.path.join(data_meta_dir, data_dir.replace('/', '_')+'.json'))

    