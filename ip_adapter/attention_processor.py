# modified from https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch
import os
import matplotlib.pyplot as plt
from einops import rearrange
import torchvision.transforms.functional as T
from PIL import Image
import numpy as np
from torchvision.utils import save_image
RES=16

def retain_top_half(attn_map):
    # 将 attn_map 展平为一维
    flat_attn_map = attn_map.view(-1)
    # 计算中位数
    threshold = torch.median(flat_attn_map)
    # 创建掩码，只保留大于或等于中位数的值
    mask = attn_map >= threshold
    # 应用掩码，将小于中位数的值设置为零
    filtered_attn_map = attn_map * mask.to(attn_map.dtype)
    return filtered_attn_map

def visualize_attn_map(attn_map, res=RES):
    # visualize the attn map
    # best visualization in 16 * 16 resolution
    global avg_store, attn_store
    b, l, _, _ = attn_store[0].shape
    avg = torch.zeros(b, l, res, res, device=attn_store[0].device)
    for attn in attn_store:
        if attn.shape[-1] == res:
            avg += attn.squeeze(0)
    avg /= len(attn_store)
    avg_store.append(avg.unsqueeze(0))

## custom 
class AttnProcessor(nn.Module):
    r"""
    Default processor for performing attention-related computations.
    """

    def __init__(
        self,
        hidden_size=None,
        cross_attention_dim=None,
    ):
        super().__init__()

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        # no_attn_mask=None,
        *args,
        **kwargs,
    ):
        residual = hidden_states
        # encoder_attention_mask = kwargs['no_attn_mask']
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)
        
        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states

## original
class IPAttnProcessor(nn.Module):
    r"""
    Attention processor for IP-Adapater.
    Args:
        hidden_size (`int`):
            The hidden size of the attention layer.
        cross_attention_dim (`int`):
            The number of channels in the `encoder_hidden_states`.
        scale (`float`, defaults to 1.0):
            the weight scale of image prompt.
        num_tokens (`int`, defaults to 4 when do ip_adapter_plus it should be 16):
            The context length of the image features.
    """

    def __init__(self, hidden_size, cross_attention_dim=None, scale=1.0, num_tokens=4):
        super().__init__()

        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.scale = scale
        self.num_tokens = num_tokens

        self.to_k_ip = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
        self.to_v_ip = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        # no_attn_mask=None,
        **cross_attention_kwargs,
        # *args,
        # **kwargs,
    ):
        no_attn_mask = cross_attention_kwargs.pop('no_attn_mask', None)
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        else:
            # get encoder_hidden_states, ip_hidden_states
            end_pos = encoder_hidden_states.shape[1] - self.num_tokens
            encoder_hidden_states, ip_hidden_states = (
                encoder_hidden_states[:, :end_pos, :],
                encoder_hidden_states[:, end_pos:, :],
            )
            if attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # for ip-adapter
        ip_key = self.to_k_ip(ip_hidden_states)
        ip_value = self.to_v_ip(ip_hidden_states)

        ip_key = attn.head_to_batch_dim(ip_key)
        ip_value = attn.head_to_batch_dim(ip_value)

        ip_attention_probs = attn.get_attention_scores(query, ip_key, None)
        self.attn_map = ip_attention_probs
        ip_hidden_states = torch.bmm(ip_attention_probs, ip_value)
        ip_hidden_states = attn.batch_to_head_dim(ip_hidden_states)

        hidden_states = hidden_states + self.scale * ip_hidden_states

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states

## no pamma no 2ffn
class IPAttnProcessorSingle(IPAttnProcessor):
    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        # no_attn_mask=None,
        **cross_attention_kwargs,
        # *args,
        # **kwargs,
    ):
        no_attn_mask = cross_attention_kwargs.pop('no_attn_mask', None)
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        else:
            # get encoder_hidden_states, ip_hidden_states
            end_pos = encoder_hidden_states.shape[1] - self.num_tokens *2
            encoder_hidden_states, ip_hidden_states = (
                encoder_hidden_states[:, :end_pos, :],
                encoder_hidden_states[:, end_pos:end_pos+self.num_tokens, :],
            )
            if attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # for ip-adapter
        ip_key = self.to_k_ip(ip_hidden_states)
        ip_value = self.to_v_ip(ip_hidden_states)

        ip_key = attn.head_to_batch_dim(ip_key)
        ip_value = attn.head_to_batch_dim(ip_value)

        ip_attention_probs = attn.get_attention_scores(query, ip_key, None)
        self.attn_map = ip_attention_probs
        ip_hidden_states = torch.bmm(ip_attention_probs, ip_value)
        ip_hidden_states = attn.batch_to_head_dim(ip_hidden_states)

        hidden_states = hidden_states + self.scale * ip_hidden_states

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states

## no pamma has 2ffn
class IPAttnProcessorSingleDFFN(IPAttnProcessor):
    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        # no_attn_mask=None,
        **cross_attention_kwargs,
        # *args,
        # **kwargs,
    ):
        no_attn_mask = cross_attention_kwargs.pop('no_attn_mask', None)
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        else:
            # get encoder_hidden_states, ip_hidden_states
            end_pos = encoder_hidden_states.shape[1] - self.num_tokens *2
            encoder_hidden_states, ip_hidden_states = (
                encoder_hidden_states[:, :end_pos, :],
                encoder_hidden_states[:, end_pos:end_pos+self.num_tokens, :],
            )
            if attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

         # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)
        
        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual
            
        residual = hidden_states
        # for ip-adapter
        ip_key = self.to_k_ip(ip_hidden_states)
        ip_value = self.to_v_ip(ip_hidden_states)

        ip_key = attn.head_to_batch_dim(ip_key)
        ip_value = attn.head_to_batch_dim(ip_value)

        ip_attention_probs = attn.get_attention_scores(query, ip_key, None)
        self.attn_map = ip_attention_probs
        ip_hidden_states = torch.bmm(ip_attention_probs, ip_value)
        ip_hidden_states = attn.batch_to_head_dim(ip_hidden_states)

        hidden_states = hidden_states + self.scale * ip_hidden_states

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states

## pamma no 2ffn
class IPAttnProcessorDouble(nn.Module):
    r"""
    Attention processor for IP-Adapater.
    Args:
        hidden_size (`int`):
            The hidden size of the attention layer.
        cross_attention_dim (`int`):
            The number of channels in the `encoder_hidden_states`.
        scale (`float`, defaults to 1.0):
            the weight scale of image prompt.
        num_tokens (`int`, defaults to 4 when do ip_adapter_plus it should be 16):
            The context length of the image features.
    """

    def __init__(self, hidden_size, cross_attention_dim=None, scale=1.0, num_tokens=4):
        super().__init__()

        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.scale = scale
        self.num_tokens = num_tokens
        # self.feature_attention = 
        self.to_k_ip = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
        self.to_v_ip = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)

        self.to_ks_ip = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
        self.attn_scores = []
        self.simliarity = []

        self.attn_maps_text = []
        self.attn_maps_ip = []
        self.attn_maps_ips = []

    def save_attention_map(self, attn_map, filename):
        plt.figure(figsize=(10, 10))
        plt.imshow(attn_map.numpy(), cmap='viridis')
        plt.colorbar()
        plt.savefig(filename)
        plt.close()

    def visualize_attention_maps(self, save_dir, output_base):
        # Calculate mean attention map for each type and save it
        def calculate_and_save_mean_map(attn_maps, filename_suffix):
            mean_map = torch.mean(torch.mean(torch.stack(attn_maps), dim=0), dim=1).unsqueeze(1).float()
            # print(mean_map.shape)
            filename = os.path.join(f"{output_base}_{filename_suffix}_res{mean_map.shape[-1]}.png")
            save_image(mean_map, filename, normalize=True)

        # Save each type of attention map
        calculate_and_save_mean_map(self.attn_maps_text, "text_attn_map")
        calculate_and_save_mean_map(self.attn_maps_ip, "ip_attn_map")
        calculate_and_save_mean_map(self.attn_maps_ips, "ips_attn_map")

        # self.to_vs_ip = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
    
    def save_attention_map(self, attn_map, filename):
        plt.figure(figsize=(10, 10))
        plt.imshow(attn_map.cpu().detach().numpy(), cmap='viridis')
        plt.colorbar()
        plt.savefig(filename)
        plt.close()
    
    def get_attention_scores(self, attn, q, k, no_attn_mask, is_cross=True, attention_mask=None, degrate=0.01,):
        h = attn.heads
        batch_size = q.shape[0] // h  ## (b h) n d
        sim = torch.einsum("b i d, b j d -> b i j", q, k) * attn.scale
        if attention_mask is not None:
            attention_mask = attention_mask.reshape(batch_size, -1)
            max_neg_value = -torch.finfo(sim.dtype).max
            attention_mask = attention_mask[:, None, :].repeat(h, 1, 1)
            sim.masked_fill_(~attention_mask, max_neg_value)

        # self.simliarity.append(((sim.shape[1])**0.5, torch.sum(sim).item())) ## (b h) n m
        # if sim.shape[1] ** 0.5 == RES:
        #     print(self.simliarity[-1])
        attn = sim.softmax(dim=-1)
        
        if is_cross:
            attn = rearrange(attn, 'b (h w) t -> b t h w', h=int((attn.shape[1])**0.5))
            if degrate != 1 and no_attn_mask is not None:
                cur_mask = T.resize(no_attn_mask, attn.shape[2:]).bool().squeeze()
                attn[:, :, cur_mask] *= degrate
                # attn[:, :, ~cur_mask] *= 1.
            # if attn.shape[2] == RES:
            #     visualize_attn_map(self.visualize_dir)
            #     self.attn_scores.append(attn)  # ([16, 77, RES, RES])
            attn = rearrange(attn, 'b t h w -> b (h w) t')

        return attn
    
    def head_to_batch_dim(self, tensor: torch.Tensor, head_size=8) -> torch.Tensor:
    
        
        batch_size, seq_len, dim = tensor.shape
        tensor = tensor.reshape(batch_size, seq_len , head_size, dim // head_size)
        tensor = tensor.permute(0, 2, 1, 3)
        tensor = tensor.reshape(batch_size * head_size, seq_len, dim // head_size)

        return tensor
    
    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        # no_attn_mask=None,
        **condition_kwargs,
    ):
        # def cosine_similarity(tensor1, tensor2):
        #     tensor1_flat = tensor1.view(tensor1.size(0), -1)
        #     tensor2_flat = tensor2.view(tensor2.size(0), -1)
        #     similarity = F.cosine_similarity(tensor1_flat, tensor2_flat, dim=1)
        #     return similarity
        no_attn_mask = condition_kwargs.get('no_attn_mask')
        residual = hidden_states
        # if no_attn_mask is None:
        #     print("No attn mask")
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        else:
            # get encoder_hidden_states, ip_hidden_states
            end_pos = encoder_hidden_states.shape[1] - self.num_tokens - self.num_tokens
            encoder_hidden_states, ip_hidden_states, ips_hidden_states = (
                encoder_hidden_states[:, :end_pos, :],
                encoder_hidden_states[:, end_pos:encoder_hidden_states.shape[1] - self.num_tokens, :],
                encoder_hidden_states[:, encoder_hidden_states.shape[1] - self.num_tokens:, :],
            )
            if attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        # encoder_hidden_states += torch.bmm(F.softmax(torch.bmm(encoder_hidden_states, attn.to_k_ip(ip_hidden_states))), ip_hidden_states)
        
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        
        query = attn.head_to_batch_dim(query)
        # t_query = self.head_to_batch_dim(encoder_hidden_states, attn.heads)
        t_query = encoder_hidden_states
        # i_key = self.head_to_batch_dim(ip_hidden_states,attn.heads)
        i_key = ip_hidden_states

        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)
        # attention_probs = attn.get_attention_score(query, key, no_attn_mask)
        attention_probs = attn.get_attention_scores(query, key, attention_mask)  ## b 1024 77

        self.attn_map_t = attention_probs
        # self.save_attention_map(self.attn_map_t[0], 'attn_map_t.png')
        # h = attn.heads
        # ti_attn_map = torch.bmm(t_query, i_key.transpose(1, 2)) * attn.scale  ## b 77 4
        # ti_attn_map = torch.sum(ti_attn_map.softmax(dim=1), dim=-1).unsqueeze(1).repeat_interleave(attn.heads, 0)
        # attention_probs *= ti_attn_map

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # for ip-adapter
        ip_key = self.to_k_ip(ip_hidden_states)
        ip_value = self.to_v_ip(ip_hidden_states)

        ip_key = attn.head_to_batch_dim(ip_key)
        ip_value = attn.head_to_batch_dim(ip_value)

        # ip_attention_probs = attn.get_attention_scores(query, ip_key, attention_mask)
        ip_attention_probs = self.get_attention_scores(attn, query, ip_key, no_attn_mask)

        self.attn_map = ip_attention_probs
         # for ips-adapter
        # self.save_attention_map(self.attn_map[0], 'attn_map.png')

        ips_key = self.to_ks_ip(ips_hidden_states)
        # ips_value = self.to_vs_ip(ips_hidden_states)

        ips_key = attn.head_to_batch_dim(ips_key)
        # ips_value = attn.head_to_batch_dim(ips_value)

        # ips_attention_probs = attn.get_attention_scores(query, ips_key, attention_mask)
        ips_attention_probs = self.get_attention_scores(attn, query, ips_key, no_attn_mask)
        
        self.attn_map_s = ips_attention_probs

        # self.save_attention_map(self.attn_map_s[0], 'attn_map.png')
        # ips_hidden_states = torch.bmm(ips_attention_probs, ip_value)
        # ips_hidden_states = attn.batch_to_head_dim(ips_hidden_states)
        ip_attention_probs = (ips_attention_probs + ip_attention_probs + torch.abs(ips_attention_probs - ip_attention_probs)) / 2
        
        ## 计算attn_maps & attn_map 的相似度
        ## 只保留attn map最大的一半
        # ip_attention_probs = retain_top_half(ip_attention_probs)

        ip_hidden_states = torch.bmm(ip_attention_probs, ip_value)
        ip_hidden_states = attn.batch_to_head_dim(ip_hidden_states)

        hidden_states = hidden_states + self.scale * ip_hidden_states
       
        # hidden_states = hidden_states + self.scale * ips_hidden_states

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)
        # residual = attn.to_out[1](residual)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = 0.5 * hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        # visualize_attn_map()

        return hidden_states

## backup
class IPAttnProcessorDoubleZeroBackup(IPAttnProcessorDouble):
    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        # no_attn_mask=None,
        **condition_kwargs,
    ):
        # def cosine_similarity(tensor1, tensor2):
        #     tensor1_flat = tensor1.view(tensor1.size(0), -1)
        #     tensor2_flat = tensor2.view(tensor2.size(0), -1)
        #     similarity = F.cosine_similarity(tensor1_flat, tensor2_flat, dim=1)
        #     return similarity
        no_attn_mask = condition_kwargs.get('no_attn_mask')
        residual = hidden_states
        # if no_attn_mask is None:
        #     print("No attn mask")
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        else:
            # get encoder_hidden_states, ip_hidden_states
            end_pos = encoder_hidden_states.shape[1] - self.num_tokens - self.num_tokens
            encoder_hidden_states, ip_hidden_states, ips_hidden_states = (
                encoder_hidden_states[:, :end_pos, :],
                encoder_hidden_states[:, end_pos:encoder_hidden_states.shape[1] - self.num_tokens, :],
                encoder_hidden_states[:, encoder_hidden_states.shape[1] - self.num_tokens:, :],
            )
            if attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        # encoder_hidden_states += torch.bmm(F.softmax(torch.bmm(encoder_hidden_states, attn.to_k_ip(ip_hidden_states))), ip_hidden_states)
        
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        
        query = attn.head_to_batch_dim(query)
        # t_query = self.head_to_batch_dim(encoder_hidden_states, attn.heads)
        t_query = encoder_hidden_states
        # i_key = self.head_to_batch_dim(ip_hidden_states,attn.heads)
        i_key = ip_hidden_states

        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        # attention_probs = attn.get_attention_score(query, key, no_attn_mask)
        attention_probs = self.get_attention_scores(attn, query, key, attention_mask)  ## b 1024 77

        self.attn_map_t = attention_probs
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)
        # residual = attn.to_out[1](residual)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = 0.5 * hidden_states + residual

        residual = hidden_states
        # print('Right 1 rnd')
        # for ip-adapter
        ip_key = self.to_k_ip(ip_hidden_states)
        ip_value = self.to_v_ip(ip_hidden_states)

        ip_key = attn.head_to_batch_dim(ip_key)
        ip_value = attn.head_to_batch_dim(ip_value)

        # ip_attention_probs = attn.get_attention_scores(query, ip_key, attention_mask)
        ip_attention_probs = self.get_attention_scores(attn, query, ip_key, no_attn_mask)

        self.attn_map = ip_attention_probs
     
        ips_key = self.to_ks_ip(ips_hidden_states)

        ips_key = attn.head_to_batch_dim(ips_key)

        ## BEGIN
        # query = attn.to_q(hidden_states)
        # query = attn.head_to_batch_dim(query)
        ## END

        ips_attention_probs = self.get_attention_scores(attn, query, ips_key, no_attn_mask)
        
        self.attn_map_s = ips_attention_probs

        # self.save_attention_map(self.attn_map_s[0], 'attn_map.png')
        # ips_hidden_states = torch.bmm(ips_attention_probs, ip_value)
        # ips_hidden_states = attn.batch_to_head_dim(ips_hidden_states)
        ip_attention_probs = (ips_attention_probs + ip_attention_probs + torch.abs(ips_attention_probs - ip_attention_probs)) / 2
        
        ## 计算attn_maps & attn_map 的相似度
        ## 只保留attn map最大的一半
        # ip_attention_probs = retain_top_half(ip_attention_probs)

        ip_hidden_states = torch.bmm(ip_attention_probs, ip_value)
        ip_hidden_states = attn.batch_to_head_dim(ip_hidden_states)
        ## BEGIN
        # hidden_states = self.scale * ip_hidden_states
        ## END
        hidden_states = hidden_states + self.scale * ip_hidden_states

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)
        # residual = attn.to_out[1](residual)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = 1/2 * hidden_states + residual

        # print('Right 2 rnd')
        # for ip-adapter
        hidden_states = hidden_states / attn.rescale_output_factor

        # visualize_attn_map()

        return hidden_states

## 2ffn + pamma
class IPAttnProcessorDoubleZero(IPAttnProcessorDouble):

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        # no_attn_mask=None,
        **condition_kwargs,
    ):
        no_attn_mask = condition_kwargs.get('no_attn_mask')
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        else:
            end_pos = encoder_hidden_states.shape[1] - self.num_tokens - self.num_tokens
            encoder_hidden_states, ip_hidden_states, ips_hidden_states = (
                encoder_hidden_states[:, :end_pos, :],
                encoder_hidden_states[:, end_pos:encoder_hidden_states.shape[1] - self.num_tokens, :],
                encoder_hidden_states[:, encoder_hidden_states.shape[1] - self.num_tokens:, :],
            )
            if attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = self.get_attention_scores(attn, query, key, attention_mask)  # text attention
        
        # self.attn_maps_text.append(attention_probs.detach().cpu().view(batch_size, -1, int((attention_probs.shape[1])**0.5), int((attention_probs.shape[1])**0.5)))

        hidden_states = torch.bmm(attention_probs, value)
        
        #### FIX the BUG
        # linear proj
        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)
        # residual = attn.to_out[1](residual)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = 0.5 * hidden_states + residual

        residual = hidden_states
        #### BUG end 

        # For ip-adapter
        ip_key = self.to_k_ip(ip_hidden_states)
        ip_value = self.to_v_ip(ip_hidden_states)

        ip_key = attn.head_to_batch_dim(ip_key)
        ip_value = attn.head_to_batch_dim(ip_value)

        ip_attention_probs = self.get_attention_scores(attn, query, ip_key, no_attn_mask)  # ip attention
        # self.attn_maps_ip.append(ip_attention_probs.detach().cpu().view(batch_size, -1, int((ip_attention_probs.shape[1])**0.5), int((ip_attention_probs.shape[1])**0.5)))

        # For ips-adapter
        ips_key = self.to_ks_ip(ips_hidden_states)
        ips_key = attn.head_to_batch_dim(ips_key)

        ips_attention_probs = self.get_attention_scores(attn, query, ips_key, no_attn_mask)  # ips attention
        # self.attn_maps_ips.append(ips_attention_probs.detach().cpu().view(batch_size, -1, int((ips_attention_probs.shape[1])**0.5), int((ips_attention_probs.shape[1])**0.5)))
        # Combine and update hidden states
        ip_attention_probs = (ips_attention_probs + ip_attention_probs + torch.abs(ips_attention_probs - ip_attention_probs)) / 2
        ip_hidden_states = torch.bmm(ip_attention_probs, ip_value)
        ip_hidden_states = attn.batch_to_head_dim(ip_hidden_states)
        # hidden_states = hidden_states + self.scale * ip_hidden_states
        hidden_states = self.scale * ip_hidden_states

        # Linear proj
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = 0.5 * hidden_states + residual

        return hidden_states

## 2ffnrev + pamma
class IPAttnProcessorDoubleZeroRev(IPAttnProcessorDoubleZero):
    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        # no_attn_mask=None,
        **condition_kwargs,
    ):
       
        no_attn_mask = condition_kwargs.get('no_attn_mask')
        residual = hidden_states
        # if no_attn_mask is None:
        #     print("No attn mask")
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        else:
            # get encoder_hidden_states, ip_hidden_states
            end_pos = encoder_hidden_states.shape[1] - self.num_tokens - self.num_tokens
            encoder_hidden_states, ip_hidden_states, ips_hidden_states = (
                encoder_hidden_states[:, :end_pos, :],
                encoder_hidden_states[:, end_pos:encoder_hidden_states.shape[1] - self.num_tokens, :],
                encoder_hidden_states[:, encoder_hidden_states.shape[1] - self.num_tokens:, :],
            )
            if attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        # encoder_hidden_states += torch.bmm(F.softmax(torch.bmm(encoder_hidden_states, attn.to_k_ip(ip_hidden_states))), ip_hidden_states)
        ip_key = self.to_k_ip(ip_hidden_states)
        ip_value = self.to_v_ip(ip_hidden_states)

        ip_key = attn.head_to_batch_dim(ip_key)
        ip_value = attn.head_to_batch_dim(ip_value)

        # ip_attention_probs = attn.get_attention_scores(query, ip_key, attention_mask)
        ip_attention_probs = self.get_attention_scores(attn, query, ip_key, no_attn_mask)

        self.attn_map = ip_attention_probs
     
        ips_key = self.to_ks_ip(ips_hidden_states)
        # ips_value = self.to_vs_ip(ips_hidden_states)

        ips_key = attn.head_to_batch_dim(ips_key)
        # ips_attention_probs = attn.get_attention_scores(query, ips_key, attention_mask)
        ips_attention_probs = self.get_attention_scores(attn, query, ips_key, no_attn_mask)
        
        self.attn_map_s = ips_attention_probs

        ip_attention_probs = (ips_attention_probs + ip_attention_probs + torch.abs(ips_attention_probs - ip_attention_probs)) / 2
   
        ip_hidden_states = torch.bmm(ip_attention_probs, ip_value)
        ip_hidden_states = attn.batch_to_head_dim(ip_hidden_states)

        hidden_states += self.scale * ip_hidden_states
        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)
        # residual = attn.to_out[1](residual)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
        # if attn.residual_connection:
        #     hidden_states = 0.5 * hidden_states + residual
        residual = hidden_states

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        # attention_probs = attn.get_attention_score(query, key, no_attn_mask)
        attention_probs = attn.get_attention_scores(query, key, attention_mask)  ## b 1024 77

        self.attn_map_t = attention_probs
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)
        # residual = attn.to_out[1](residual)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = 1/2 * hidden_states + residual

        # print('Right 2 rnd')
        # for ip-adapter
        hidden_states = hidden_states / attn.rescale_output_factor

        # visualize_attn_map()

        return hidden_states


## 3ffn + pamma
class IPAttnProcessorDoubleZero3(IPAttnProcessorDoubleZero):
    def __init__(self, hidden_size, cross_attention_dim=None, scale=1.0, num_tokens=4):
        super().__init__(hidden_size, cross_attention_dim=None, scale=1.0, num_tokens=4)

        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.scale = scale
        self.num_tokens = num_tokens
        # self.feature_attention = 
        self.to_k_ip = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
        self.to_v_ip = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)

        self.to_ks_ip = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
        self.to_vs_ip = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
    
    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        # no_attn_mask=None,
        **condition_kwargs,
    ):
        # def cosine_similarity(tensor1, tensor2):
        #     tensor1_flat = tensor1.view(tensor1.size(0), -1)
        #     tensor2_flat = tensor2.view(tensor2.size(0), -1)
        #     similarity = F.cosine_similarity(tensor1_flat, tensor2_flat, dim=1)
        #     return similarity
        no_attn_mask = condition_kwargs.get('no_attn_mask')
        residual = hidden_states
        # if no_attn_mask is None:
        #     print("No attn mask")
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        else:
            # get encoder_hidden_states, ip_hidden_states
            end_pos = encoder_hidden_states.shape[1] - self.num_tokens - self.num_tokens
            encoder_hidden_states, ip_hidden_states, ips_hidden_states = (
                encoder_hidden_states[:, :end_pos, :],
                encoder_hidden_states[:, end_pos:encoder_hidden_states.shape[1] - self.num_tokens, :],
                encoder_hidden_states[:, encoder_hidden_states.shape[1] - self.num_tokens:, :],
            )
            if attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        # encoder_hidden_states += torch.bmm(F.softmax(torch.bmm(encoder_hidden_states, attn.to_k_ip(ip_hidden_states))), ip_hidden_states)
        
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        
        query = attn.head_to_batch_dim(query)

        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        # attention_probs = attn.get_attention_score(query, key, no_attn_mask)
        attention_probs = attn.get_attention_scores(query, key, attention_mask)  ## b 1024 77

        self.attn_map_t = attention_probs
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)
        # residual = attn.to_out[1](residual)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = 0.5 * hidden_states + residual

        residual = hidden_states
        # print('Right 1 rnd')
        # for ip-adapter
        ip_key = self.to_k_ip(ip_hidden_states)
        ip_value = self.to_v_ip(ip_hidden_states)

        ip_key = attn.head_to_batch_dim(ip_key)
        ip_value = attn.head_to_batch_dim(ip_value)

        ip_attention_probs = self.get_attention_scores(attn, query, ip_key, no_attn_mask)

        self.attn_map = ip_attention_probs
     
        ips_key = self.to_ks_ip(ips_hidden_states)
        ips_key = attn.head_to_batch_dim(ips_key)
        ips_value = self.to_vs_ip(ips_hidden_states)
        ips_value = attn.head_to_batch_dim(ips_value)

        ips_attention_probs = self.get_attention_scores(attn, query, ips_key, no_attn_mask)
        
        self.attn_map_s = ips_attention_probs
        
        ip_hidden_states = torch.bmm(ip_attention_probs, ip_value)
        ip_hidden_states = attn.batch_to_head_dim(ip_hidden_states)
        ## BEGIN
        # hidden_states = self.scale * ip_hidden_states
        ## END
        hidden_states = hidden_states + self.scale * ip_hidden_states

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)
        # residual = attn.to_out[1](residual)
        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = 1/3 * hidden_states + residual
            
        ips_hidden_states = torch.bmm(ips_attention_probs, ips_value)
        ips_hidden_states = attn.batch_to_head_dim(ips_hidden_states)
        hidden_states = hidden_states + self.scale * ips_hidden_states

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)
        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = 1/3 * hidden_states + residual
        
        

        # print('Right 2 rnd')
        # for ip-adapter
        hidden_states = hidden_states / attn.rescale_output_factor

        # visualize_attn_map()

        return hidden_states

class AttnProcessor2_0(torch.nn.Module):
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(
        self,
        hidden_size=None,
        cross_attention_dim=None,
    ):
        super().__init__()
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        *args,
        **kwargs,
    ):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states

class IPAttnProcessor2_0(torch.nn.Module):
    r"""
    Attention processor for IP-Adapater for PyTorch 2.0.
    Args:
        hidden_size (`int`):
            The hidden size of the attention layer.
        cross_attention_dim (`int`):
            The number of channels in the `encoder_hidden_states`.
        scale (`float`, defaults to 1.0):
            the weight scale of image prompt.
        num_tokens (`int`, defaults to 4 when do ip_adapter_plus it should be 16):
            The context length of the image features.
    """

    def __init__(self, hidden_size, cross_attention_dim=None, scale=1.0, num_tokens=4):
        super().__init__()

        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.scale = scale
        self.num_tokens = num_tokens

        self.to_k_ip = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
        self.to_v_ip = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        *args,
        **kwargs,
    ):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        else:
            # get encoder_hidden_states, ip_hidden_states
            end_pos = encoder_hidden_states.shape[1] - self.num_tokens
            encoder_hidden_states, ip_hidden_states = (
                encoder_hidden_states[:, :end_pos, :],
                encoder_hidden_states[:, end_pos:, :],
            )
            if attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # for ip-adapter
        ip_key = self.to_k_ip(ip_hidden_states)
        ip_value = self.to_v_ip(ip_hidden_states)

        ip_key = ip_key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        ip_value = ip_value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        ip_hidden_states = F.scaled_dot_product_attention(
            query, ip_key, ip_value, attn_mask=None, dropout_p=0.0, is_causal=False
        )
        with torch.no_grad():
            self.attn_map = query @ ip_key.transpose(-2, -1).softmax(dim=-1)
            #print(self.attn_map.shape)

        ip_hidden_states = ip_hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        ip_hidden_states = ip_hidden_states.to(query.dtype)

        hidden_states = hidden_states + self.scale * ip_hidden_states

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states

## for controlnet
class CNAttnProcessor:
    r"""
    Default processor for performing attention-related computations.
    """

    def __init__(self, num_tokens=4):
        self.num_tokens = num_tokens

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None, *args, **kwargs,):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        else:
            end_pos = encoder_hidden_states.shape[1] - self.num_tokens
            encoder_hidden_states = encoder_hidden_states[:, :end_pos]  # only use text
            if attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states

class CNAttnProcessor2_0:
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(self, num_tokens=4):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")
        self.num_tokens = num_tokens

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        *args,
        **kwargs,
    ):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        else:
            end_pos = encoder_hidden_states.shape[1] - self.num_tokens
            encoder_hidden_states = encoder_hidden_states[:, :end_pos]  # only use text
            if attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states
