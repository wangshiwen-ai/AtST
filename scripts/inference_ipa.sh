export CUDA_VISIBLE_DEVICES=0
# python inference_from_char.py \
#     --char 花 \
#     --style_path assets/images/yishuzi.png \
#     --font_name Alibaba-PuHuiTi-Bold \
#     --ip_ckpt models/ip-adapter_sd15.bin

# python inference_from_char.py \
#     --char 花 \
#     --style_path assets/images/yishuzi.png \
#     --font_name Alibaba-PuHuiTi-Bold \
#     --ip_ckpt /home/swwang/IP-Adapter/sd-idapter-2/checkpoint-1000/ip_adapter.bin

# python inference_from_char.py \
#     --char 花 \
#     --style_path assets/images/yishuzi.png \
#     --font_name 钟齐志莽行书 \
#     --ip_ckpt models/ip-adapter_sd15.bin

# python inference_from_char.py \
#     --char 花 \
#     --style_path assets/images/yishuzi.png \
#     --font_name 钟齐志莽行书 \
#     --ip_ckpt /home/swwang/IP-Adapter/sd-idapter-2/checkpoint-1000/ip_adapter.bin

# python inference_from_char.py \
#     --char 花 \
#     --style_path assets/images/hua-bi.png \
#     --font_name 钟齐志莽行书 \
#     --ip_ckpt models/ip-adapter_sd15.bin

python inference_from_char.py \
    --char 花 \
    --style_path assets/images/hua.png \
    --font_name 钟齐志莽行书 \
    --text_prompt 水墨风 \
    --ip_ckpt /home/swwang/IP-Adapter/sdxl-idapter-1/checkpoint-1000/ip_adapter.bin

