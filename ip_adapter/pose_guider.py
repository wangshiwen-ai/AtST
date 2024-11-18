from typing import Tuple

import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from diffusers.models.modeling_utils import ModelMixin
from einops import rearrange


class InflatedConv3d(nn.Conv2d):
    def forward(self, x):
        video_length = x.shape[2]

        x = rearrange(x, "b c f h w -> (b f) c h w")
        x = super().forward(x)
        x = rearrange(x, "(b f) c h w -> b c f h w", f=video_length)

        return x

def zero_module(module):
    # Zero out the parameters of a module and return it.
    for p in module.parameters():
        p.detach().zero_()
    return module


class PoseGuider(ModelMixin):
    def __init__(
        self,
        conditioning_embedding_channels: int,
        conditioning_channels: int = 3,
        block_out_channels: Tuple[int] = (16, 32, 64, 128),
    ):
        super().__init__()

        # self.conv_in_sample = InflatedConv3d(
        #     conditioning_channels, block_out_channels[0], kernel_size=3, padding=1
        # )

        self.conv_in = InflatedConv3d(
            conditioning_channels, block_out_channels[0], kernel_size=3, padding=1
        )

        self.blocks = nn.ModuleList([])

        for i in range(len(block_out_channels) - 1):
            channel_in = block_out_channels[i]
            channel_out = block_out_channels[i + 1]
            self.blocks.append(
                InflatedConv3d(channel_in, channel_in, kernel_size=3, padding=1)
            )
            self.blocks.append(
                InflatedConv3d(
                    channel_in, channel_out, kernel_size=3, padding=1, stride=2
                )
            )

        self.conv_out = zero_module(
            InflatedConv3d(
                block_out_channels[-1],
                conditioning_embedding_channels,
                kernel_size=3,
                padding=1,
            )
        )

    def forward(self, conditioning, sample=None):
        # sample = self.conv_in_sample(sample)
        embedding = self.conv_in(conditioning)
        embedding = F.silu(embedding)

        for block in self.blocks:
            embedding = block(embedding)
            embedding = F.silu(embedding)

        embedding = self.conv_out(embedding)

        return embedding
import torch.nn as nn

import math

class PositionalEncoding(nn.Module):

    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000, device='cuda'):
        super(PositionalEncoding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).to(device)
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = x + self.pe[:, :x.size(1)].detach()
        return self.dropout(x).transpose(1, 2)

if __name__=="__main__":
    import torch
    import sys
    sys.path.append('.')
    from utils import skeleton_image
    from PIL import Image
    from torchvision import transforms
    size = 512
    transform = transforms.Compose([
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ])

    cond_transform = transforms.Compose([
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
            ])

    controlnet_openpose_path = 'models/pose_guider.pth'
    pose_guider = PoseGuider(
            conditioning_embedding_channels=320, block_out_channels=(16, 32, 96, 256)
        ).to(device="cuda")
    state_dict = torch.load(controlnet_openpose_path)
    pose_guider.load_state_dict(state_dict)
    img = '/home/swwang/IP-Adapter/processed/xhs_char/兴.png'
    img_pil = Image.open(img)
    skeleton = skeleton_image(img_pil)
    skeleton = cond_transform(skeleton).unsqueeze(0).to(device='cuda').unsqueeze(2) ## b, 3, 1, 256, 256
    embeddings = pose_guider(skeleton).squeeze(2)  ## b, 320, 1, 32, 32  b, 320, 32, 32
    print(embeddings.shape)
    # pos_emb = PositionalEncoding(320, 0.1)
    # skeleton = pos_emb(embeddings.view(1, 320, -1)).view(1, 320, 32, 32)
    # print(1)