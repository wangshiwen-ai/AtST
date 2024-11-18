export CUDA_VISIBLE_DEVICES=0

# python inference/inference.py \
#     --char 春 \
#     --style_path assets/images/hua.png \
#     --font_name Alibaba-PuHuiTi-Bold  \
#     --ip_ckpt /root/autodl-tmp/miai-sdxl-idapter-1s-2ffn-rt/checkpoint-6600/ip_adapter.bin \
#     --t '11'

# python inference/inference.py \
#     --char 春 \
#     --style_path assets/images/miai.png \
#     --font_name Alibaba-PuHuiTi-Bold  \
#     --ip_ckpt /root/autodl-tmp/miai-sdxl-idapter-1s-2ffn-rt/checkpoint-6600/ip_adapter.bin \
#     --t '11'

# python inference/inference.py \
#     --char Lake \
#     --style_path assets/images/hua.png \
#     --font_name Alibaba-PuHuiTi-Bold  \
#     --ip_ckpt /root/autodl-tmp/yeqing-sdxl-idapter-1s-2ffn-rt/checkpoint-9500/ip_adapter.bin \
#     --t '11'

python inference/inference.py \
    --char 湖 \
    --style_path assets/images/hu.png \
    --font_name Alibaba-PuHuiTi-Bold  \
    --ip_ckpt /root/autodl-tmp/yeqing-sdxl-idapter-1s-2ffn-rt/checkpoint-9500/ip_adapter.bin \
    --t '11'

# python inference/inference.py \
#     --char 春 \
#     --style_path /home/swwang/IP-Adapter/assets/images/hua.png \
#     --font_name Alibaba-PuHuiTi-Bold  \
#     --ip_ckpt /home/swwang/IP-Adapter/sdxl-idapter-1s-single/checkpoint-5800/ip_adapter.bin \
#     --t '00'

# python inference/inference.py \
#     --char 春 \
#     --style_path /home/swwang/IP-Adapter/assets/images/hua.png \
#     --font_name Alibaba-PuHuiTi-Bold  \
#     --ip_ckpt /home/swwang/IP-Adapter/sdxl-idapter-1s/checkpoint-5800/ip_adapter.bin \
#     --t '10'

# python inference/inference.py \
#     --char 春 \
#     --style_path /home/swwang/IP-Adapter/assets/images/hua.png \
#     --font_name Alibaba-PuHuiTi-Bold  \
#     --ip_ckpt /home/swwang/IP-Adapter/sdxl-idapter-1s-singledffn/checkpoint-5800/ip_adapter.bin \
#     --t '01'

# python inference/inference.py \
#     --char 春 \
#     --style_path /home/swwang/IP-Adapter/assets/images/hua.png \
#     --font_name Alibaba-PuHuiTi-Bold  \
#     --ip_ckpt /home/swwang/IP-Adapter/sdxl-idapter-1s-3ffn/checkpoint-5800/ip_adapter.bin \
#     --t '3'

# python inference/inference.py \
#     --char 春 \
#     --style_path /home/swwang/IP-Adapter/assets/images/hua.png \
#     --font_name Alibaba-PuHuiTi-Bold  \
#     --ip_ckpt /home/swwang/IP-Adapter/sdxl-idapter-1s-2ffn-rev/checkpoint-5800/ip_adapter.bin \
#     --t 'r'


