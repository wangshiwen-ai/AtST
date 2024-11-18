export CUDA_VISIBLE_DEVICES=0
# accelerate launch --num_processes 2 --multi_gpu --mixed_precision "fp16" \

python \
  train.py \
  --pretrained_model_name_or_path="/root/autodl-tmp/sdxl-models" \
  --image_encoder_path="/root/autodl-tmp/vit" \
  --data_json_file='processed/meta/processed_yeqing_char.json' \
  --data_root_path="processed/yeqing_char" \
  --mixed_precision="fp16" \
  --resolution=512 \
  --train_batch_size=2 \
  --gradient_accumulation_steps=4 \
  --dataloader_num_workers=4 \
  --learning_rate=1e-04 \
  --weight_decay=0.01 \
  --output_dir="/root/autodl-tmp/yeqing-sdxl-idapter-1s-2ffn-new-00-single" \
  --save_steps=3000 \
  --img_loss=0.0 \
  --denoise_loss=1.0 \
  --train_type=5

python \
  train.py \
  --pretrained_model_name_or_path="/root/autodl-tmp/sdxl-models" \
  --image_encoder_path="/root/autodl-tmp/vit" \
  --data_json_file='processed/meta/processed_yeqing_char.json' \
  --data_root_path="processed/yeqing_char" \
  --mixed_precision="fp16" \
  --resolution=512 \
  --train_batch_size=2 \
  --gradient_accumulation_steps=4 \
  --dataloader_num_workers=4 \
  --learning_rate=1e-04 \
  --weight_decay=0.01 \
  --output_dir="/root/autodl-tmp/yeqing-sdxl-idapter-1s-2ffn-new-10-1s" \
  --save_steps=3000 \
  --img_loss=0.0 \
  --denoise_loss=1.0 \
  --train_type=5
  
## stage1-1
# python \
#   train.py \
#   --pretrained_model_name_or_path="/root/autodl-tmp/sdxl_models" \
#   --image_encoder_path="/root/autodl-tmp/vit" \
#   --data_json_file='processed/meta/processed_yeqing_char.json' \
#   --data_root_path="processed/yeqing_char" \
#   --mixed_precision="fp16" \
#   --resolution=512 \
#   --train_batch_size=2 \
#   --gradient_accumulation_steps=4 \
#   --dataloader_num_workers=4 \
#   --learning_rate=1e-04 \
#   --weight_decay=0.01 \
#   --output_dir="/root/autodl-tmp/sdxl-idapter-1s-2ffn-nocolor" \
#   --save_steps=3000 \
#   --img_loss=0.0 \
#   --denoise_loss=1.0 \
#   --train_type=2

## stage1-2
# python \
#   train.py \
#   --pretrained_model_name_or_path="/root/autodl-tmp/sdxl_models" \
#   --image_encoder_path="/root/autodl-tmp/vit" \
#   --data_json_file='processed/meta/processed_yeqing_pure_char.json' \
#   --data_root_path="processed/yeqing_pure_char" \
#   --mixed_precision="fp16" \
#   --resolution=512 \
#   --train_batch_size=2 \
#   --gradient_accumulation_steps=4 \
#   --dataloader_num_workers=4 \
#   --learning_rate=1e-04 \
#   --weight_decay=0.01 \
#   --output_dir="/root/autodl-tmp/sdxl-idapter-1s-2ffn-easychar" \
#   --save_steps=3000 \
#   --img_loss=0.0 \
#   --denoise_loss=1.0 \
#   --train_type=4

## rev
# python \
#   train.py \
#   --pretrained_model_name_or_path="/root/autodl-tmp/sdxl_models" \
#   --image_encoder_path="/root/autodl-tmp/vit" \
#   --data_json_file='processed/meta/processed_yeqing_char.json' \
#   --data_root_path="processed/yeqing_char" \
#   --mixed_precision="fp16" \
#   --resolution=512 \
#   --train_batch_size=2 \
#   --gradient_accumulation_steps=4 \
#   --dataloader_num_workers=4 \
#   --learning_rate=1e-04 \
#   --weight_decay=0.01 \
#   --output_dir="/root/autodl-tmp/sdxl-idapter-1s-2ffn-rev" \
#   --save_steps=3000 \
#   --img_loss=0.0 \
#   --denoise_loss=1.0 \
#   --train_type=0 

## 3
# python \
#   train.py \
#   --pretrained_model_name_or_path="/root/autodl-tmp/sdxl_models" \
#   --image_encoder_path="/root/autodl-tmp/vit" \
#   --data_json_file='processed/meta/processed_yeqing_char.json' \
#   --data_root_path="processed/yeqing_char" \
#   --mixed_precision="fp16" \
#   --resolution=512 \
#   --train_batch_size=2 \
#   --gradient_accumulation_steps=4 \
#   --dataloader_num_workers=4 \
#   --learning_rate=1e-04 \
#   --weight_decay=0.01 \
#   --output_dir="/root/autodl-tmp/sdxl-idapter-1s-3ffn" \
#   --save_steps=3000 \
#   --img_loss=0.0 \
#   --denoise_loss=1.0 \
#   --train_type=0 

## 00
# python \
#   train.py \
#   --pretrained_model_name_or_path="/root/autodl-tmp/sdxl_models" \
#   --image_encoder_path="/root/autodl-tmp/vit" \
#   --data_json_file='/home/swwang/IP-Adapter2/processed/meta/processed_yeqing_char.json' \
#   --data_root_path="/home/swwang/IP-Adapter2/processed/yeqing_char" \
#   --mixed_precision="fp16" \
#   --resolution=512 \
#   --train_batch_size=2 \
#   --gradient_accumulation_steps=4 \
#   --dataloader_num_workers=4 \
#   --learning_rate=1e-04 \
#   --weight_decay=0.01 \
#   --output_dir="/root/autodl-tmp/sdxl-idapter-1s-single" \
#   --save_steps=3000 \
#   --img_loss=0.0 \
#   --denoise_loss=1.0 \
#   --train_type=0 

## 01
# python \
#   train.py \
#   --pretrained_model_name_or_path="/root/autodl-tmp/sdxl_models" \
#   --image_encoder_path="/root/autodl-tmp/vit" \
#   --data_json_file='processed/meta/processed_yeqing_char.json' \
#   --data_root_path="processed/yeqing_char" \
#   --mixed_precision="fp16" \
#   --resolution=512 \
#   --train_batch_size=2 \
#   --gradient_accumulation_steps=4 \
#   --dataloader_num_workers=4 \
#   --learning_rate=1e-04 \
#   --weight_decay=0.01 \
#   --output_dir="/root/autodl-tmp/sdxl-idapter-1s-singledffn" \
#   --save_steps=3000 \
#   --img_loss=0.0 \
#   --denoise_loss=1.0 \
#   --train_type=0 

## 10
# python \
#   train.py \
#   --pretrained_model_name_or_path="/root/autodl-tmp/sdxl_models" \
#   --image_encoder_path="/root/autodl-tmp/vit" \
#   --data_json_file='processed/meta/processed_yeqing_char.json' \
#   --data_root_path="processed/yeqing_char" \
#   --mixed_precision="fp16" \
#   --resolution=512 \
#   --train_batch_size=2 \
#   --gradient_accumulation_steps=4 \
#   --dataloader_num_workers=4 \
#   --learning_rate=1e-04 \
#   --weight_decay=0.01 \
#   --output_dir="/root/autodl-tmp/sdxl-idapter-1s" \
#   --save_steps=3000 \
#   --img_loss=0.0 \
#   --denoise_loss=1.0 \
#   --train_type=0 

## 11
