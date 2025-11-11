import math
import torch
import torch.fft as fft
from torch import nn
import torch.nn.functional as F
from inspect import isfunction
import cv2
import torchvision.transforms as T
import numpy as np
def exists(x):
    return x is not None




class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class NormDownsample(nn.Module):
    def __init__(self,in_ch,out_ch,scale=0.5,use_norm=False):
        super(NormDownsample, self).__init__()
        self.use_norm=use_norm
        if self.use_norm:
            self.norm=LayerNorm(out_ch)
        self.prelu = nn.PReLU()
        self.down = nn.Sequential(
            nn.Conv2d(in_ch, out_ch,kernel_size=3,stride=1, padding=1, bias=False),
            nn.UpsamplingBilinear2d(scale_factor=scale))
    def forward(self, x):
        x = self.down(x)
        x = self.prelu(x)
        if self.use_norm:
            x = self.norm(x)
            return x
        else:
            return x

class NormUpsample(nn.Module):
    def __init__(self, in_ch,out_ch,scale=2,use_norm=False):
        super(NormUpsample, self).__init__()
        self.use_norm=use_norm
        if self.use_norm:
            self.norm=LayerNorm(out_ch)
        self.prelu = nn.PReLU()
        self.up_scale = nn.Sequential(
            nn.Conv2d(in_ch,out_ch,kernel_size=3,stride=1, padding=1, bias=False),
            nn.UpsamplingBilinear2d(scale_factor=scale))
        self.up = nn.Conv2d(out_ch*2,out_ch,kernel_size=1,stride=1, padding=0, bias=False)
            
    def forward(self, x,y):
        x = self.up_scale(x)
        x = torch.cat([x, y],dim=1)
        x = self.up(x)
        x = self.prelu(x)
        if self.use_norm:
            return self.norm(x)
        else:
            return x


# Cross Attention Block
class CAB(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(CAB, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.kv = nn.Conv2d(dim, dim*2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim*2, dim*2, kernel_size=3, stride=1, padding=1, groups=dim*2, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, y):
        b, c, h, w = x.shape

        q = self.q_dwconv(self.q(x))
        kv = self.kv_dwconv(self.kv(y))
        k, v = kv.chunk(2, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = nn.functional.softmax(attn,dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out
    

# Intensity Enhancement Layer
class IEL(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=2.66, bias=False):
        super(IEL, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)
        
        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)
        self.dwconv1 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, groups=hidden_features, bias=bias)
        self.dwconv2 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, groups=hidden_features, bias=bias)
       
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

        self.Tanh = nn.Tanh()
    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x1 = self.Tanh(self.dwconv1(x1)) + x1
        x2 = self.Tanh(self.dwconv2(x2)) + x2
        x = x1 * x2
        x = self.project_out(x)
        return x
  
  
# Lightweight Cross Attention
class HV_LCA(nn.Module):
    def __init__(self, dim,num_heads, bias=False):
        super(HV_LCA, self).__init__()
        self.gdfn = IEL(dim) # IEL and CDL have same structure
        self.norm = LayerNorm(dim)
        self.ffn = CAB(dim, num_heads, bias)
        
    def forward(self, x, y):
        x = x + self.ffn(self.norm(x),self.norm(y))
        x = self.gdfn(self.norm(x))
        return x
    
class I_LCA(nn.Module):
    def __init__(self, dim,num_heads, bias=False):
        super(I_LCA, self).__init__()
        self.norm = LayerNorm(dim)
        self.gdfn = IEL(dim)
        self.ffn = CAB(dim, num_heads, bias=bias)
        
    def forward(self, x, y):
        x = x + self.ffn(self.norm(x),self.norm(y))
        x = x + self.gdfn(self.norm(x)) 
        return x










# PositionalEncoding Source： https://github.com/lmnt-com/wavegrad/blob/master/src/wavegrad/model.py
class PositionalEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, noise_level):
        count = self.dim // 2
        step = torch.arange(count, dtype=noise_level.dtype,
                            device=noise_level.device) / count
        encoding = noise_level.unsqueeze(
            1) * torch.exp(-math.log(1e4) * step.unsqueeze(0))
        encoding = torch.cat(
            [torch.sin(encoding), torch.cos(encoding)], dim=-1)
        return encoding

class FeatureWiseAffine(nn.Module):
    def __init__(self, in_channels, out_channels, use_affine_level=False):
        super(FeatureWiseAffine, self).__init__()
        self.use_affine_level = use_affine_level
        self.noise_func = nn.Sequential(
            nn.Linear(in_channels, out_channels*(1+self.use_affine_level))
        )

    def forward(self, x, noise_embed):
        batch = x.shape[0]
        if self.use_affine_level:
            gamma, beta = self.noise_func(noise_embed).view(
                batch, -1, 1, 1).chunk(2, dim=1)
            x = (1 + gamma) * x + beta
        else:
            x = x + self.noise_func(noise_embed).view(batch, -1, 1, 1)
        return x

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

# model
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class Upsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Conv2d(dim, dim, 3, padding=1)

    def forward(self, x):
        return self.conv(self.up(x))


class Downsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


# building block modules
class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=32, dropout=0):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(groups, dim),
            Swish(),
            nn.Dropout(dropout) if dropout != 0 else nn.Identity(),
            nn.Conv2d(dim, dim_out, 3, padding=1)
        )

    def forward(self, x):
        return self.block(x)


class AttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.norm = nn.GroupNorm(32, channels)
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        self.proj_out = nn.Conv2d(channels, channels, 1)
        
    def forward(self, x):
        b, c, h, w = x.shape
        x_norm = self.norm(x)
        q = self.q(x_norm).view(b, c, -1)
        k = self.k(x_norm).view(b, c, -1)
        v = self.v(x_norm).view(b, c, -1)
        
        attn = torch.softmax(torch.bmm(q.transpose(1, 2), k) / (c ** 0.5), dim=-1)
        out = torch.bmm(v, attn.transpose(1, 2)).view(b, c, h, w)
        return x + self.proj_out(out)


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, time_emb_dim=None, dropout=0, norm_groups=32):
        super().__init__()
        self.mlp = nn.Sequential(
            Swish(),
            nn.Linear(time_emb_dim, dim_out)
        ) if exists(time_emb_dim) else None
        self.noise_func = FeatureWiseAffine(
            time_emb_dim, dim_out, use_affine_level=False)

        self.block1 = Block(dim, dim_out, groups=norm_groups)
        self.block2 = Block(dim_out, dim_out, groups=norm_groups, dropout=dropout)
        self.res_conv = nn.Conv2d(
            dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb):
        h = self.block1(x)
        h = self.noise_func(h, time_emb)
        h = self.block2(h)
        return h + self.res_conv(x)
        
        
from einops import rearrange

class SelfAttention(nn.Module):
    def __init__(self, in_channel, n_head=1, norm_groups=32):
        super().__init__()

        self.n_head = n_head

        self.norm = nn.GroupNorm(norm_groups, in_channel)
        self.qkv = nn.Conv2d(in_channel, in_channel * 3, 1, bias=False)
        self.out = nn.Conv2d(in_channel, in_channel, 1)


        self.dim1=in_channel
        self.heads=4

        self.i_lca=I_LCA(dim=self.dim1,num_heads=self.heads)

    def forward(self, input):
        # print("input.shape:",input.shape)

        input= self.i_lca(input,input)


        batch, channel, height, width = input.shape
        n_head = self.n_head
        head_dim = channel // n_head
        
        norm = self.norm(input)
        qkv = self.qkv(norm).view(batch, n_head, head_dim * 3, height, width)
        query, key, value = qkv.chunk(3, dim=2)  # bhdyx

        attn = torch.einsum(
            "bnchw, bncyx -> bnhwyx", query, key
        ).contiguous() / math.sqrt(channel)
        attn = attn.view(batch, n_head, height, width, -1)
        attn = torch.softmax(attn, -1)
        attn = attn.view(batch, n_head, height, width, height, width)

        out = torch.einsum("bnhwyx, bncyx -> bnchw", attn, value).contiguous()
        out = self.out(out.view(batch, channel, height, width))

        return out + input







class ResnetBlocWithAttn(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, norm_groups=32, dropout=0, with_attn=False):
        super().__init__()
        self.with_attn = with_attn
        self.res_block = ResnetBlock(
            dim, dim_out, time_emb_dim, norm_groups=norm_groups, dropout=dropout)
        if with_attn:
            self.attn = SelfAttention(dim_out, norm_groups=norm_groups)

    def forward(self, x, time_emb):
        x = self.res_block(x, time_emb)
        if(self.with_attn):
            x = self.attn(x)
        return x


class UNet(nn.Module):
    def __init__(
        self,
        in_channel=6,
        out_channel=3,
        inner_channel=32,
        norm_groups=32,
        channel_mults=(1, 1, 2, 2, 4),
        attn_res=(8),
        res_blocks=3,
        dropout=0,
        with_noise_level_emb=True,
        image_size=128
    ):
        super().__init__()
        if with_noise_level_emb:
            noise_level_channel = inner_channel
            self.noise_level_mlp = nn.Sequential(
                PositionalEncoding(inner_channel),
                nn.Linear(inner_channel, inner_channel * 4),
                Swish(),
                nn.Linear(inner_channel * 4, inner_channel)
            )
        else:
            noise_level_channel = None
            self.noise_level_mlp = None


        num_mults = len(channel_mults)
        pre_channel = inner_channel
        feat_channels = [pre_channel]
        now_res = image_size
        downs = [nn.Conv2d(in_channel, inner_channel,
                           kernel_size=3, padding=1)]
        for ind in range(num_mults):
            is_last = (ind == num_mults - 1)
            use_attn = (now_res in attn_res)
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, res_blocks):
                downs.append(ResnetBlocWithAttn(
                    pre_channel, channel_mult, time_emb_dim=noise_level_channel, norm_groups=norm_groups, dropout=dropout, with_attn=use_attn))
                feat_channels.append(channel_mult)
                pre_channel = channel_mult
            if not is_last:
                downs.append(Downsample(pre_channel))
                feat_channels.append(pre_channel)
                now_res = now_res//2
        self.downs = nn.ModuleList(downs)

        self.mid = nn.ModuleList([
            ResnetBlocWithAttn(pre_channel, pre_channel, time_emb_dim=noise_level_channel, norm_groups=norm_groups,
                               dropout=dropout, with_attn=True),
            ResnetBlocWithAttn(pre_channel, pre_channel, time_emb_dim=noise_level_channel, norm_groups=norm_groups, 
                                dropout=dropout, with_attn=False)
        ])

        ups = []
        for ind in reversed(range(num_mults)):
            is_last = (ind < 1)
            use_attn = (now_res in attn_res)
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, res_blocks+1):
                ups.append(ResnetBlocWithAttn(
                    pre_channel+feat_channels.pop(), channel_mult, time_emb_dim=noise_level_channel, dropout=dropout, norm_groups=norm_groups, with_attn=use_attn))
                pre_channel = channel_mult
            if not is_last:
                ups.append(Upsample(pre_channel))
                now_res = now_res*2

        self.ups = nn.ModuleList(ups)

        self.final_conv = Block(pre_channel, default(out_channel, in_channel), groups=norm_groups)

        self.var_conv = nn.Sequential(*[
            nn.Conv2d(pre_channel, pre_channel, 3, padding=(3//2), bias=True), 
            nn.ELU(), 
            nn.Conv2d(pre_channel, pre_channel, 3, padding=(3//2), bias=True),
            nn.ELU(), 
            nn.Conv2d(pre_channel, 3, 3, padding=(3//2), bias=True),
            nn.ELU()
        ])
        # self.swish = Swish()
    def default_conv(in_channels, out_channels, kernel_size, bias=True):
        return nn.Conv2d(
            in_channels, out_channels, kernel_size,
            padding=(kernel_size//2), bias=bias)

    def forward(self, x, noise):
        noise_level = self.noise_level_mlp(noise) if exists(self.noise_level_mlp) else None
        feats = []
        for layer in self.downs:
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(x, noise_level)
            else:
                x = layer(x)
            feats.append(x)

        for layer in self.mid:
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(x, noise_level)
            else:
                x = layer(x)

        for layer in self.ups:
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(torch.cat((x, feats.pop()), dim=1), noise_level)
            else:
                x = layer(x)
        return self.final_conv(x), self.var_conv(x)



# FreeU
def Fourier_filter(x, threshold, scale):
    # FFT
    x_freq = fft.fftn(x, dim=(-2, -1))
    x_freq = fft.fftshift(x_freq, dim=(-2, -1))
    
    B, C, H, W = x_freq.shape
    mask = torch.ones((B, C, H, W)).cuda() 

    crow, ccol = H // 2, W //2
    mask[..., crow - threshold:crow + threshold, ccol - threshold:ccol + threshold] = scale
    x_freq = x_freq * mask

    # IFFT
    x_freq = fft.ifftshift(x_freq, dim=(-2, -1))
    x_filtered = fft.ifftn(x_freq, dim=(-2, -1)).real
    
    return x_filtered

def SNR_filter(x_in, threshold, scale):

    blur_transform = T.GaussianBlur(kernel_size=15, sigma=3)
    blur_x_in = blur_transform(x_in)
    gray_blur_x_in = blur_x_in[:, 0:1, :, :] * 0.299 + blur_x_in[:, 1:2, :, :] * 0.587 + blur_x_in[:, 2:3, :, :] * 0.114
    gray_x_in = x_in[:, 0:1, :, :] * 0.299 + x_in[:, 1:2, :, :] * 0.587 + x_in[:, 2:3, :, :] * 0.114
    noise = torch.abs(gray_blur_x_in - gray_x_in)
    mask = torch.div(gray_x_in, noise + 0.0001)

    batch_size = mask.shape[0]
    height = mask.shape[2]
    width = mask.shape[3]
    mask_max = torch.max(mask.view(batch_size, -1), dim=1)[0]
    mask_max = mask_max.view(batch_size, 1, 1, 1)
    mask_max = mask_max.repeat(1, 1, height, width)
    mask = mask * 1.0 / (mask_max+0.0001)
    mask = torch.clamp(mask, min=0, max=1.0)
    mask = mask.float()

    return mask

    
class Free_UNet(UNet):
    """
    :param b1: backbone factor of the first stage block of decoder.
    :param b2: backbone factor of the second stage block of decoder.
    :param s1: skip factor of the first stage block of decoder.
    :param s2: skip factor of the second stage block of decoder.
    """

    def __init__(
        self,
        b1=1.3,
        b2=1.4,
        s1=0.9,
        s2=0.2,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.b1 = b1 
        self.b2 = b2
        self.s1 = s1
        self.s2 = s2

    def forward(self, h, noise):
        # what we need is only x and noise
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param context: conditioning plugged in via crossattn
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        hs = []
        noise_level = self.noise_level_mlp(noise) if exists(self.noise_level_mlp) else None



        for layer in self.downs:
            if isinstance(layer, ResnetBlocWithAttn):
                h = layer(h, noise_level)
            else:
                h = layer(h)
            hs.append(h)
        
        for layer in self.mid:
            if isinstance(layer, ResnetBlocWithAttn):
                h = layer(h, noise_level)
            else:
                h = layer(h)

        for layer in self.ups:
            # --------------- FreeU code -----------------------
            # Only operate on the first two stages
            if h.shape[1] == 256:
                hs_ = hs.pop()
                hidden_mean = h.mean(1).unsqueeze(1)
                B = hidden_mean.shape[0]
                hidden_max, _ = torch.max(hidden_mean.view(B, -1), dim=-1, keepdim=True) 
                hidden_min, _ = torch.min(hidden_mean.view(B, -1), dim=-1, keepdim=True)
                hidden_mean = (hidden_mean - hidden_min.unsqueeze(2).unsqueeze(3)) / (hidden_max - hidden_min).unsqueeze(2).unsqueeze(3)

                h[:,:128] = h[:,:128] * ((self.b1 - 1 ) * hidden_mean + 1)
                hs_ = Fourier_filter(hs_, threshold=1, scale=self.s1)
                hs.append(hs_)
            if h.shape[1] == 128:
                hs_ = hs.pop()
                hidden_mean = h.mean(1).unsqueeze(1)
                B = hidden_mean.shape[0]
                hidden_max, _ = torch.max(hidden_mean.view(B, -1), dim=-1, keepdim=True) 
                hidden_min, _ = torch.min(hidden_mean.view(B, -1), dim=-1, keepdim=True)
                hidden_mean = (hidden_mean - hidden_min.unsqueeze(2).unsqueeze(3)) / (hidden_max - hidden_min).unsqueeze(2).unsqueeze(3)

                h[:,:64] = h[:,:64] * ((self.b2 - 1 ) * hidden_mean + 1)
                hs_ = Fourier_filter(hs_, threshold=1, scale=self.s2)
                hs.append(hs_)
            # ---------------------------------------------------------
            # exit(print(h.shape, hs_.shape)) # torch.Size([1, 256, 26, 38]) torch.Size([1, 256, 26, 38])
            # print(h.shape, hs_.shape)
            # h = torch.cat((h, hs_), dim=1)

            if isinstance(layer, ResnetBlocWithAttn):
                h = layer(torch.cat((h, hs.pop()), dim=1), noise_level)
            else:
                h = layer(h)
        return self.final_conv(h), self.var_conv(h)

class LAUNet(UNet):
    """
    :param b1: backbone factor of the first stage block of decoder.
    :param b2: backbone factor of the second stage block of decoder.
    :param s1: skip factor of the first stage block of decoder.
    :param s2: skip factor of the second stage block of decoder.
    """

    def __init__(
        self,
        b1=1.3,
        b2=1.4,
        s1=0.9,
        s2=0.2,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.b1 = b1 
        self.b2 = b2
        self.s1 = s1
        self.s2 = s2

    def forward(self, h, noise):
        # what we need is only x and noise
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param context: conditioning plugged in via crossattn
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        hs = []
        noise_level = self.noise_level_mlp(noise) if exists(self.noise_level_mlp) else None



        for layer in self.downs:
            if isinstance(layer, ResnetBlocWithAttn):
                h = layer(h, noise_level)
            else:
                h = layer(h)
            hs.append(h)
        
        for layer in self.mid:
            if isinstance(layer, ResnetBlocWithAttn):
                h = layer(h, noise_level)
            else:
                h = layer(h)

        for layer in self.ups:
            # --------------- FreeU code -----------------------
            # Only operate on the first two stages
            if h.shape[1] == 256:
                hs_ = hs.pop()
                hidden_mean = h.mean(1).unsqueeze(1)
                B = hidden_mean.shape[0]
                hidden_max, _ = torch.max(hidden_mean.view(B, -1), dim=-1, keepdim=True) 
                hidden_min, _ = torch.min(hidden_mean.view(B, -1), dim=-1, keepdim=True)
                hidden_mean = (hidden_mean - hidden_min.unsqueeze(2).unsqueeze(3)) / (hidden_max - hidden_min).unsqueeze(2).unsqueeze(3)

                h[:,:128] = h[:,:128] * ((self.b1 - 1 ) * hidden_mean + 1)
                hs_ = Fourier_filter(hs_, threshold=1, scale=self.s1)
                hs.append(hs_)
            if h.shape[1] == 128:
                hs_ = hs.pop()
                hidden_mean = h.mean(1).unsqueeze(1)
                B = hidden_mean.shape[0]
                hidden_max, _ = torch.max(hidden_mean.view(B, -1), dim=-1, keepdim=True) 
                hidden_min, _ = torch.min(hidden_mean.view(B, -1), dim=-1, keepdim=True)
                hidden_mean = (hidden_mean - hidden_min.unsqueeze(2).unsqueeze(3)) / (hidden_max - hidden_min).unsqueeze(2).unsqueeze(3)

                h[:,:64] = h[:,:64] * ((self.b2 - 1 ) * hidden_mean + 1)
                hs_ = Fourier_filter(hs_, threshold=1, scale=self.s2)
                hs.append(hs_)
            # ---------------------------------------------------------
            # exit(print(h.shape, hs_.shape)) # torch.Size([1, 256, 26, 38]) torch.Size([1, 256, 26, 38])
            # print(h.shape, hs_.shape)
            # h = torch.cat((h, hs_), dim=1)

            if isinstance(layer, ResnetBlocWithAttn):
                h = layer(torch.cat((h, hs.pop()), dim=1), noise_level)
            else:
                h = layer(h)
        return self.final_conv(h), self.var_conv(h)

class LAUNet(UNet):
    """
    :param b1: backbone factor of the first stage block of decoder.
    :param b2: backbone factor of the second stage block of decoder.
    :param s1: skip factor of the first stage block of decoder.
    :param s2: skip factor of the second stage block of decoder.
    """

    def __init__(
        self,
        b1=1.3,
        b2=1.4,
        s1=0.9,
        s2=0.2,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.b1 = b1 
        self.b2 = b2
        self.s1 = s1
        self.s2 = s2

    def forward(self, h, noise):
        # what we need is only x and noise
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param context: conditioning plugged in via crossattn
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        hs = []
        noise_level = self.noise_level_mlp(noise) if exists(self.noise_level_mlp) else None



        for layer in self.downs:
            # print(h.shape)
            if isinstance(layer, ResnetBlocWithAttn):
                h = layer(h, noise_level)
            else:
                h = layer(h)
            hs.append(h)
        # print("\n")
        for layer in self.mid:
            # print(h.shape)
            if isinstance(layer, ResnetBlocWithAttn):
                h = layer(h, noise_level)
            else:
                h = layer(h)
        # print("\n")
        for layer in self.ups:
            # print(h.shape)
            # --------------- FreeU code -----------------------
            # Only operate on the first two stages
            if h.shape[1] == 256:
                hs_ = hs.pop()
                hidden_mean = h.mean(1).unsqueeze(1)
                B = hidden_mean.shape[0]
                hidden_max, _ = torch.max(hidden_mean.view(B, -1), dim=-1, keepdim=True) 
                hidden_min, _ = torch.min(hidden_mean.view(B, -1), dim=-1, keepdim=True)
                hidden_mean = (hidden_mean - hidden_min.unsqueeze(2).unsqueeze(3)) / (hidden_max - hidden_min).unsqueeze(2).unsqueeze(3)

                h[:,:64] = h[:,:64] * ((self.b1 - 1 ) * hidden_mean + 1)
                hs_ = Fourier_filter(hs_, threshold=1, scale=self.s1)
                hs.append(hs_)
            # if h.shape[1] == 128 and h.shape[2] == 104:
            #     hs_ = hs.pop()
            #     hidden_mean = h.mean(1).unsqueeze(1)
            #     B = hidden_mean.shape[0]
            #     hidden_max, _ = torch.max(hidden_mean.view(B, -1), dim=-1, keepdim=True) 
            #     hidden_min, _ = torch.min(hidden_mean.view(B, -1), dim=-1, keepdim=True)
            #     hidden_mean = (hidden_mean - hidden_min.unsqueeze(2).unsqueeze(3)) / (hidden_max - hidden_min).unsqueeze(2).unsqueeze(3)

            #     h[:,:64] = h[:,:64] * ((self.b2 - 1 ) * hidden_mean + 1)
            #     hs_ = Fourier_filter(hs_, threshold=1, scale=self.s2)
            #     hs.append(hs_)
            # ---------------------------------------------------------
            # exit(print(h.shape, hs_.shape)) # torch.Size([1, 256, 26, 38]) torch.Size([1, 256, 26, 38])
            # print(h.shape, hs_.shape)
            # h = torch.cat((h, hs_), dim=1)
            if isinstance(layer, ResnetBlocWithAttn):
                h = layer(torch.cat((h, hs.pop()), dim=1), noise_level)
            else:
                h = layer(h)
        
        # exit(-1)
        return self.final_conv(h), self.var_conv(h)