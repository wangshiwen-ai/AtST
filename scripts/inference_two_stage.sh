export CUDA_VISIBLE_DEVICES=1
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

python inference/inference_two_stage.py \
    --char 春 \
    --style_path assets/images/hua.png \
    --bishua_path assets/images/easychar.png \
    --font_name Alibaba-PuHuiTi-Bold  \
    --text_prompt 水墨风 \
    --ip_ckpt /home/swwang/IP-Adapter/sdxl-idapter-1s-2ffn-rt/checkpoint-5800/ip_adapter.bin \
    --t '1'

