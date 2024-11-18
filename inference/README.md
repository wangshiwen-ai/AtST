# Inference directory

- `inference.py` is the base config of our model, with **Primary and Auxiliary Mutual Attention Adapters with Dual-Feed-Forward Networks**
- `inference_two_stage.py` is the plus inference process of pur model, with
    - first stage: use the grey image is element prompt or the specific stoke style image to generate the char image with desired structure or texture.
    - second stage: use the colorful element image to fusion with the image generated in the first stage.
- `inference_ipa*.py` is the baseline of ip-adapter, which is designed to compare with our work.

Default config is one stage model.




