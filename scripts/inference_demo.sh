export CUDA_VISIBLE_DEVICES=0

# python inference/inference.py \
#     --char 春 \
#     --style_path assets/images/genryu.png \
#     --font_name "Alibaba-PuHuiTi-Regular GenRyuMinTW-Bold 851手写杂字体 锐字真言体 王漢宗顏楷體繁 王漢宗新粗標魏碑空心 王漢宗空疊圓繁 手书体" \
#     --ip_ckpt  /root/autodl-tmp/sdxl-idapter-font-single/checkpoint-5200/ip_adapter.bin \
#     --use_pe=0 \
#     --t '00'


python inference/inference.py \
    --char 春 \
    --style_path assets/images/hua.png \
    --font_name "Alibaba-PuHuiTi-Regular GenRyuMinTW-Bold 851手写杂字体 锐字真言体 王漢宗顏楷體繁 王漢宗新粗標魏碑空心 王漢宗空疊圓繁 手书体" \
    --ip_ckpt  /root/autodl-tmp/sdxl-idapter-1s-2ffn-rt/checkpoint-5800/ip_adapter.bin \
    --use_pe=0 \
    --t '11'

# python inference/inference.py \
#     --char 秋 \
#     --style_path assets/images/qiu.jpg \
#     --font_name "Alibaba-PuHuiTi-Regular GenRyuMinTW-Bold 851手写杂字体 锐字真言体 王漢宗顏楷體繁 王漢宗新粗標魏碑空心 王漢宗空疊圓繁 手书体" \
#     --ip_ckpt  /root/autodl-tmp/sdxl-idapter-1s-2ffn-rt/checkpoint-5800/ip_adapter.bin \
#     --use_pe=0 \
#     --t '11'

# python inference/inference.py \
#     --char 夏 \
#     --style_path assets/images/xia1.jpg \
#     --font_name "Alibaba-PuHuiTi-Regular GenRyuMinTW-Bold 851手写杂字体 锐字真言体 王漢宗顏楷體繁 王漢宗新粗標魏碑空心 王漢宗空疊圓繁 手书体" \
#     --ip_ckpt  /root/autodl-tmp/sdxl-idapter-1s-2ffn-rt/checkpoint-5800/ip_adapter.bin \
#     --use_pe=0 \
#     --t '11'

# python inference/inference.py \
#     --char 夏 \
#     --style_path assets/images/bingjiling.png \
#     --font_name "Alibaba-PuHuiTi-Regular GenRyuMinTW-Bold 851手写杂字体 锐字真言体 王漢宗顏楷體繁 王漢宗新粗標魏碑空心 王漢宗空疊圓繁 手书体" \
#     --ip_ckpt  /root/autodl-tmp/sdxl-idapter-1s-2ffn-rt/checkpoint-5800/ip_adapter.bin \
#     --use_pe=0 \
#     --t '11'

# python inference/inference.py \
#     --char 冬 \
#     --style_path assets/images/dong.jpg \
#     --font_name "Alibaba-PuHuiTi-Regular GenRyuMinTW-Bold 851手写杂字体 锐字真言体 王漢宗顏楷體繁 王漢宗新粗標魏碑空心 王漢宗空疊圓繁 手书体" \
#     --ip_ckpt  /root/autodl-tmp/sdxl-idapter-1s-2ffn-rt/checkpoint-5800/ip_adapter.bin \
#     --use_pe=0 \
#     --t '11'


# export CUDA_VISIBLE_DEVICES=1

# python inference/inference.py \
#     --char 春 \
#     --style_path assets/images/hua.png \
#     --font_name "Alibaba-PuHuiTi-Regular GenRyuMinTW-Bold 851手写杂字体 锐字真言体 王漢宗顏楷體繁 王漢宗新粗標魏碑空心 王漢宗空疊圓繁 手书体" \
#     --ip_ckpt /root/autodl-tmp/sdxl-idapter-1s-2ffn-rt-pe/checkpoint-5800/ip_adapter.bin \
#     --use_pe=1 \
#     --t '11'

# python inference/inference.py \
#     --char 秋 \
#     --style_path assets/images/qiu.jpg \
#     --font_name "Alibaba-PuHuiTi-Regular GenRyuMinTW-Bold 851手写杂字体 锐字真言体 王漢宗顏楷體繁 王漢宗新粗標魏碑空心 王漢宗空疊圓繁 手书体" \
#     --ip_ckpt /root/autodl-tmp/sdxl-idapter-1s-2ffn-rt-pe/checkpoint-5800/ip_adapter.bin \
#     --use_pe=1 \
#     --t '11'

# python inference/inference.py \
#     --char 夏 \
#     --style_path assets/images/xia1.jpg \
#     --font_name "Alibaba-PuHuiTi-Regular GenRyuMinTW-Bold 851手写杂字体 锐字真言体 王漢宗顏楷體繁 王漢宗新粗標魏碑空心 王漢宗空疊圓繁 手书体" \
#     --ip_ckpt /root/autodl-tmp/sdxl-idapter-1s-2ffn-rt-pe/checkpoint-5800/ip_adapter.bin \
#     --use_pe=1 \
#     --t '11'

# python inference/inference.py \
#     --char 夏 \
#     --style_path assets/images/xia.jpg \
#     --font_name "Alibaba-PuHuiTi-Regular GenRyuMinTW-Bold 851手写杂字体 锐字真言体 王漢宗顏楷體繁 王漢宗新粗標魏碑空心 王漢宗空疊圓繁 手书体" \
#     --ip_ckpt /root/autodl-tmp/sdxl-idapter-1s-2ffn-rt-pe/checkpoint-5800/ip_adapter.bin \
#     --use_pe=1 \
#     --t '11'

# python inference/inference.py \
#     --char 冬 \
#     --style_path assets/images/dong.jpg \
#     --font_name "Alibaba-PuHuiTi-Regular GenRyuMinTW-Bold 851手写杂字体 锐字真言体 王漢宗顏楷體繁 王漢宗新粗標魏碑空心 王漢宗空疊圓繁 手书体" \
#     --ip_ckpt /root/autodl-tmp/sdxl-idapter-1s-2ffn-rt-pe/checkpoint-5800/ip_adapter.bin \
#     --use_pe=1 \
#     --t '11'
# # Alibaba-PuHuiTi-Regular 无衬线
# # GenRyuMinTW-Bold  衬线
# # 851手写杂字体 不规则
# # 锐字真言体 形状
# # 王漢宗顏楷體繁 垂感
# # 王漢宗新粗標魏碑空心 空心
# # 王漢宗空疊圓繁 空心2
# # 手书体 手写体

