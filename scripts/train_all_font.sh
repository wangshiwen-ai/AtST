#!/bin/bash

# Alibaba-PuHuiTi-Regular 无衬线
# GenRyuMinTW-Bold  衬线
# 851手写杂字体 不规则
# 锐字真言体 形状
# 王漢宗顏楷體繁 垂感
# 王漢宗新粗標魏碑空心 空心
# 王漢宗空疊圓繁 空心2
# 手书体 手写体

fonts=(
#   "Alibaba-PuHuiTi-Regular"  # 无衬线
#   "GenRyuMinTW-Bold"         # 衬线
#   "851手写杂字体"          # 不规则
#   "锐字真言体"           # 形状
#   "王漢宗顏楷體繁"       # 垂感
#   "王漢宗新粗標魏碑空心"   # 空心
#   "王漢宗空疊圓繁"         # 空心2
#   "手书体"                 # 手写体
#   "miai_char_white"
#    "千图小兔体"
#    "xhs_pure_char"
   "王漢宗波卡體空陰"
#    "麦克笔手绘体"
#    "装甲明朝体"
#    "钟齐志莽行书"
    # "贤二体"
    # "演示佛系体"
)

for font in "${fonts[@]}"; do
    echo =============================
    echo ${font}
    # python data_process/generate_dataset.py \
    #     --font_name ${font}
    # # echo ====== Generate ${font} Done ======
    python train_all_font.py \
    --pretrained_model_name_or_path="/root/autodl-tmp/sdxl-models" \
    --image_encoder_path="/root/autodl-tmp/vit" \
    --data_root_path="/root/autodl-tmp/train/${font}" \
    --mixed_precision="fp16" \
    --resolution=512 \
    --train_batch_size=2 \
    --gradient_accumulation_steps=4 \
    --dataloader_num_workers=4 \
    --learning_rate=1e-04 \
    --weight_decay=0.01 \
    --output_dir="/root/autodl-tmp/all_fonts/${font}-rt" \
    --save_steps=6000 \
    --img_loss=0.0 \
    --denoise_loss=1.0 \
    --train_type=0
    # echo =============================
    # echo Training ${font} Done Start converting.
    # python convert.py \
    #     --exp_name "/root/autodl-tmp/all_fonts/${font}-rt"
    # echo =============================
    # echo Convert ${font} Done Start inferencing.


    # python inference/test_fonts.py \
    #     --exp_name "/root/autodl-tmp/all_fonts/${font}" \
    #     --char "A B C D E F G"
    # echo %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # break
done