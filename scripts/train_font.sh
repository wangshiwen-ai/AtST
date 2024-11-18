## 11
CUDA_VISIBLE_DEVICES=0
python \
  train.py \
  --pretrained_model_name_or_path="/root/autodl-tmp/sdxl-models" \
  --image_encoder_path="/root/autodl-tmp/vit" \
  --data_json_file='processed/meta/processed_xhs_char.json' \
  --data_root_path="/root/autodl-tmp/train" \
  --mixed_precision="fp16" \
  --resolution=512 \
  --train_batch_size=2 \
  --gradient_accumulation_steps=2 \
  --dataloader_num_workers=4 \
  --learning_rate=1e-04 \
  --weight_decay=0.01 \
  --output_dir="/root/autodl-tmp/sdxl-idapter-font-single" \
  --save_steps=3000 \
  --img_loss=0.0 \
  --denoise_loss=1.0 \
  --train_type=7 \
  --use_pe=0 \
  --resume=1
