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
  
  # "miai_char_white"
  # "xhs_pure_char"
#   "锐字真言体"           # 形状
#   "王漢宗顏楷體繁"       # 垂感
#   "王漢宗新粗標魏碑空心"   # 空心
#   "王漢宗空疊圓繁"         # 空心2
    # "千图小兔体"                 # 手写体
    # "麦克笔手绘体"
    # "王漢宗波卡體空陰"  # √
    # "851手写杂字体"          # √
    # "装甲明朝体"  # √
    # "贤二体" # √
    "演示佛系体"  # √
    # "钟齐志莽行书"

)

for font in "${fonts[@]}"; do
    echo =============================
    echo ${font}
    # python convert.py \
    #     --exp_name "/root/autodl-tmp/all_fonts/${font}"
    # echo =============================
    # echo Convert ${font} Done Start inferencing.
    #  GenRyuMinTW-Light
    # --ref_char /root/autodl-tmp/train/miai_char_white/不.png \ 

    python inference/test_fonts_collect.py \
        --exp_name "/root/autodl-tmp/all_fonts/${font}" \
        --font_name "GenRyuMinTW-Light 钟齐志莽行书 851手写杂字体 贤二体" \
        --ref_char "X Y M 小 日 子" \
        --char "V P R 计 算 机" \
        --output_dir results_collect \
        --t "10"

      # python inference/test_fonts.py \
      #   --exp_name "/root/autodl-tmp/all_fonts/${font}-single" \
      #   --font_name "GenRyuMinTW-Light" \
      #   --ref_char H \
      #   --char "E F G" \
      #   --output_dir results_ablation \
      #   --t "00"

      # python inference/test_fonts.py \
      #   --exp_name "/root/autodl-tmp/all_fonts/${font}-singledffn" \
      #   --font_name "GenRyuMinTW-Light" \
      #   --ref_char H \
      #   --char "A B C D E F G" \
      #   --output_dir results_ablation \
      #   --t "01"

      # python inference/test_fonts.py \
      #   --exp_name "/root/autodl-tmp/all_fonts/${font}-1s" \
      #   --font_name "GenRyuMinTW-Light" \
      #   --ref_char H \
      #   --char "A B C D E F G" \
      #   --output_dir results_ablation \
      #   --t "10"

    echo %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # break
done