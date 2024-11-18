export CUDA_VISIBLE_DEVICES=0
# accelerate launch --num_processes 2 --multi_gpu --mixed_precision "fp16" \

## stage1-1
# python \
#   train.py \
#   --pretrained_model_name_or_path="/home/swwang/IP-Adapter/sdxl_models" \
#   --image_encoder_path="/home/swwang/IP-Adapter/vit" \
#   --data_json_file='processed/meta/processed_xhs_char.json' \
#   --data_root_path="processed/xhs_char" \
#   --mixed_precision="fp16" \
#   --resolution=512 \
#   --train_batch_size=2 \
#   --gradient_accumulation_steps=4 \
#   --dataloader_num_workers=4 \
#   --learning_rate=1e-04 \
#   --weight_decay=0.01 \
#   --output_dir="/home/swwang/IP-Adapter/sdxl-idapter-1s-2ffn-nocolor" \
#   --save_steps=3000 \
#   --img_loss=0.0 \
#   --denoise_loss=1.0 \
#   --train_type=2

## stage1-2
# python \
#   train.py \
#   --pretrained_model_name_or_path="/home/swwang/IP-Adapter/sdxl_models" \
#   --image_encoder_path="/home/swwang/IP-Adapter/vit" \
#   --data_json_file='processed/meta/processed_xhs_pure_char.json' \
#   --data_root_path="processed/xhs_pure_char" \
#   --mixed_precision="fp16" \
#   --resolution=512 \
#   --train_batch_size=2 \
#   --gradient_accumulation_steps=4 \
#   --dataloader_num_workers=4 \
#   --learning_rate=1e-04 \
#   --weight_decay=0.01 \
#   --output_dir="/home/swwang/IP-Adapter/sdxl-idapter-1s-2ffn-easychar" \
#   --save_steps=3000 \
#   --img_loss=0.0 \
#   --denoise_loss=1.0 \
#   --train_type=4

## rev
# python \
#   train.py \
#   --pretrained_model_name_or_path="/home/swwang/IP-Adapter/sdxl_models" \
#   --image_encoder_path="/home/swwang/IP-Adapter/vit" \
#   --data_json_file='processed/meta/processed_xhs_char.json' \
#   --data_root_path="processed/xhs_char" \
#   --mixed_precision="fp16" \
#   --resolution=512 \
#   --train_batch_size=2 \
#   --gradient_accumulation_steps=4 \
#   --dataloader_num_workers=4 \
#   --learning_rate=1e-04 \
#   --weight_decay=0.01 \
#   --output_dir="/home/swwang/IP-Adapter/sdxl-idapter-1s-2ffn-rev" \
#   --save_steps=3000 \
#   --img_loss=0.0 \
#   --denoise_loss=1.0 \
#   --train_type=0 

## 3
# python \
#   train.py \
#   --pretrained_model_name_or_path="/home/swwang/IP-Adapter/sdxl_models" \
#   --image_encoder_path="/home/swwang/IP-Adapter/vit" \
#   --data_json_file='processed/meta/processed_xhs_char.json' \
#   --data_root_path="processed/xhs_char" \
#   --mixed_precision="fp16" \
#   --resolution=512 \
#   --train_batch_size=2 \
#   --gradient_accumulation_steps=4 \
#   --dataloader_num_workers=4 \
#   --learning_rate=1e-04 \
#   --weight_decay=0.01 \
#   --output_dir="/home/swwang/IP-Adapter/sdxl-idapter-1s-3ffn" \
#   --save_steps=3000 \
#   --img_loss=0.0 \
#   --denoise_loss=1.0 \
#   --train_type=0 

## 00
# python \
#   train.py \
#   --pretrained_model_name_or_path="/home/swwang/IP-Adapter/sdxl_models" \
#   --image_encoder_path="/home/swwang/IP-Adapter/vit" \
#   --data_json_file='/home/swwang/IP-Adapter2/processed/meta/processed_xhs_char.json' \
#   --data_root_path="/home/swwang/IP-Adapter2/processed/xhs_char" \
#   --mixed_precision="fp16" \
#   --resolution=512 \
#   --train_batch_size=2 \
#   --gradient_accumulation_steps=4 \
#   --dataloader_num_workers=4 \
#   --learning_rate=1e-04 \
#   --weight_decay=0.01 \
#   --output_dir="/home/swwang/IP-Adapter/sdxl-idapter-1s-single" \
#   --save_steps=3000 \
#   --img_loss=0.0 \
#   --denoise_loss=1.0 \
#   --train_type=0 

## 01
# python \
#   train.py \
#   --pretrained_model_name_or_path="/home/swwang/IP-Adapter/sdxl_models" \
#   --image_encoder_path="/home/swwang/IP-Adapter/vit" \
#   --data_json_file='processed/meta/processed_xhs_char.json' \
#   --data_root_path="processed/xhs_char" \
#   --mixed_precision="fp16" \
#   --resolution=512 \
#   --train_batch_size=2 \
#   --gradient_accumulation_steps=4 \
#   --dataloader_num_workers=4 \
#   --learning_rate=1e-04 \
#   --weight_decay=0.01 \
#   --output_dir="/home/swwang/IP-Adapter/sdxl-idapter-1s-singledffn" \
#   --save_steps=3000 \
#   --img_loss=0.0 \
#   --denoise_loss=1.0 \
#   --train_type=0 

## 10
# python \
#   train.py \
#   --pretrained_model_name_or_path="/home/swwang/IP-Adapter/sdxl_models" \
#   --image_encoder_path="/home/swwang/IP-Adapter/vit" \
#   --data_json_file='processed/meta/processed_xhs_char.json' \
#   --data_root_path="processed/xhs_char" \
#   --mixed_precision="fp16" \
#   --resolution=512 \
#   --train_batch_size=2 \
#   --gradient_accumulation_steps=4 \
#   --dataloader_num_workers=4 \
#   --learning_rate=1e-04 \
#   --weight_decay=0.01 \
#   --output_dir="/home/swwang/IP-Adapter/sdxl-idapter-1s-pe" \
#   --save_steps=3000 \
#   --img_loss=0.0 \
#   --denoise_loss=1.0 \
#   --train_type=0 \
#   --use_pe true

## 11
python \
  train.py \
  --pretrained_model_name_or_path="/root/autodl-tmp/sdxl-models" \
  --image_encoder_path="/root/autodl-tmp/vit" \
  --data_json_file='processed/meta/processed_xhs_char.json' \
  --data_root_path="processed/xhs_char" \
  --mixed_precision="fp16" \
  --resolution=512 \
  --train_batch_size=2 \
  --gradient_accumulation_steps=4 \
  --dataloader_num_workers=4 \
  --learning_rate=1e-04 \
  --weight_decay=0.01 \
  --output_dir="/root/autodl-tmp/sdxl-idapter-1s-2ffn-rt-nore" \
  --save_steps=3000 \
  --img_loss=0.0 \
  --denoise_loss=1.0 \
  --train_type=0 
