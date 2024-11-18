
model_type='vit_h'
checkpoint = 'models\sam_vit_h_4b8939.pth'
img_path = 'output\Alibaba-PuHuiTi-Bold.png'
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import cv2
sam = sam_model_registry[model_type](checkpoint=checkpoint)
mask_generator = SamAutomaticMaskGenerator(sam)
img = cv2.imread(img_path)
masks = mask_generator.generate(img)
