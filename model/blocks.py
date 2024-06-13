""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from torch.nn.init import _calculate_fan_in_and_fan_out
import numpy as np
import string
# from functools import partial

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
    def forward(self, x):
        return nn.Sigmoid()(self.conv(x))
    
def _calculate_correct_fan(tensor, mode):
    """
    copied and modified from https://github.com/pytorch/pytorch/blob/master/torch/nn/init.py#L337
    """
    mode = mode.lower()
    valid_modes = ['fan_in', 'fan_out', 'fan_avg']
    if mode not in valid_modes:
        raise ValueError("Mode {} not supported, please use one of {}".format(mode, valid_modes))

    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    return fan_in if mode == 'fan_in' else fan_out

def kaiming_uniform_(tensor, gain=1., mode='fan_in'):
    r"""Fills the input `Tensor` with values according to the method
    described in `Delving deep into rectifiers: Surpassing human-level
    performance on ImageNet classification` - He, K. et al. (2015), using a
    uniform distribution. The resulting tensor will have values sampled from
    :math:`\mathcal{U}(-\text{bound}, \text{bound})` where
    .. math::
        \text{bound} = \text{gain} \times \sqrt{\frac{3}{\text{fan\_mode}}}
    Also known as He initialization.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        gain: multiplier to the dispersion
        mode: either ``'fan_in'`` (default) or ``'fan_out'``. Choosing ``'fan_in'``
            preserves the magnitude of the variance of the weights in the
            forward pass. Choosing ``'fan_out'`` preserves the magnitudes in the
            backwards pass.
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.kaiming_uniform_(w, mode='fan_in')
    """
    fan = _calculate_correct_fan(tensor, mode)
    var = gain / max(1., fan)
    bound = math.sqrt(3.0 * var)  # Calculate uniform bounds from standard deviation
    with torch.no_grad():
        return tensor.uniform_(-bound, bound)

def variance_scaling(scale, mode, distribution,
                     in_axis=1, out_axis=0,
                     dtype=torch.float32,
                     device='cpu'):
    """Ported from JAX. """

    def _compute_fans(shape, in_axis=1, out_axis=0):
        receptive_field_size = np.prod(shape) / shape[in_axis] / shape[out_axis]
        fan_in = shape[in_axis] * receptive_field_size
        fan_out = shape[out_axis] * receptive_field_size
        return fan_in, fan_out

    def init(shape, dtype=dtype, device=device):
        fan_in, fan_out = _compute_fans(shape, in_axis, out_axis)
        if mode == "fan_in":
            denominator = fan_in
        elif mode == "fan_out":
            denominator = fan_out
        elif mode == "fan_avg":
            denominator = (fan_in + fan_out) / 2
        else:
            raise ValueError(
                "invalid mode for variance scaling initializer: {}".format(mode))
        variance = scale / denominator
        if distribution == "normal":
            return torch.randn(*shape, dtype=dtype, device=device) * np.sqrt(variance)
        elif distribution == "uniform":
            return (torch.rand(*shape, dtype=dtype, device=device) * 2. - 1.) * np.sqrt(3 * variance)
        else:
            raise ValueError("invalid distribution for variance scaling initializer")

    return init

def variance_scaling_init_(tensor, scale):
    return kaiming_uniform_(tensor, gain=1e-10 if scale == 0 else scale, mode='fan_avg')

def dense(in_channels, out_channels, init_scale=1.):
    lin = nn.Linear(in_channels, out_channels)
    variance_scaling_init_(lin.weight, scale=init_scale)
    nn.init.zeros_(lin.bias)
    return lin

def default_init(scale=1.):
    """The same initialization used in DDPM."""
    scale = 1e-10 if scale == 0 else scale
    return variance_scaling(scale, 'fan_avg', 'uniform')

def ddpm_conv3x3(in_planes, out_planes, stride=1, bias=True, dilation=1, init_scale=1., padding=1, groups=1):
    """3x3 convolution with DDPM initialization."""
    conv = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=padding,
                     dilation=dilation, bias=bias, groups=groups)
    conv.weight.data = default_init(init_scale)(conv.weight.data.shape)
    nn.init.zeros_(conv.bias)
    return conv

def ddpm_conv1x1(in_planes, out_planes, stride=1, bias=True, init_scale=1., padding=0, groups=1):
    """1x1 convolution with DDPM initialization."""
    conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=padding, bias=bias, groups=1)
    conv.weight.data = default_init(init_scale)(conv.weight.data.shape)
    nn.init.zeros_(conv.bias)
    return conv

conv1x1 = ddpm_conv1x1
conv3x3 = ddpm_conv3x3
default_init = default_init
# dense = dense

class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input / torch.sqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)

class AdaptiveGroupNorm(nn.Module):
    def __init__(self, num_groups, in_channel, style_dim):
        super().__init__()

        self.norm = nn.GroupNorm(
            num_groups, in_channel, affine=False, eps=1e-6)
        self.style = dense(style_dim, in_channel * 2)

        self.style.bias.data[:in_channel] = 1
        self.style.bias.data[in_channel:] = 0

    def forward(self, input, style):
        style = self.style(style).unsqueeze(2).unsqueeze(3)
        gamma, beta = style.chunk(2, 1)
        out = self.norm(input)
        out = gamma * out + beta

        return out

class Combine(nn.Module):
    """Combine information from skip connections."""

    def __init__(self, dim1, dim2, method='cat'):
        super().__init__()
        self.Conv_0 = conv1x1(dim1, dim2)
        self.method = method

    def forward(self, x, y):
        h = self.Conv_0(x)
        if self.method == 'cat':
            return torch.cat([h, y], dim=1)
        elif self.method == 'sum':
            return h + y
        else:
            raise ValueError(f'Method {self.method} not recognized.')

def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)

class AttentionBlock(nn.Module): ### self attention ### 
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)

        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)

        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)

        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)

        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x):
        h_ = x

        h_ = self.norm(h_)
        
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape

        q = q.reshape(b, c, h*w)
        
        q = q.permute(0, 2, 1)   # b,hw,c
        
        k = k.reshape(b, c, h*w)  # b,c,hw
        
        w_ = torch.bmm(q, k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]

        w_ = w_ * (int(c)**(-0.5))

        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b, c, h*w)
        w_ = w_.permute(0, 2, 1)   # b,hw,hw (first hw of k, second of q)

        # b, c, hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]

        h_ = torch.bmm(v, w_)
        h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)

        return x + h_

class MultiHead_CrossAttention(nn.Module):
    def __init__(self, num_heads = 8, hidden_size=32*24 , attn_dropout=0.):
        super(MultiHead_CrossAttention, self).__init__()

        self.num_attention_heads = num_heads
        self.attention_head_size = int(hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.out = nn.Linear(hidden_size, hidden_size)

        self.attn_dropout = nn.Dropout(attn_dropout)
        self.proj_dropout = nn.Dropout(attn_dropout)

        self.softmax = nn.Softmax(dim=-1)

    def transpose_for_scores(self, x):
        
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, query, key, value):
        
        mixed_query_layer = self.query(query)
        mixed_key_layer = self.key(key)
        mixed_value_layer = self.value(value)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        # weights = attention_probs if self.vis else None

        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output

# class Edge_Enhancement_Layer(nn.Module):
#     def __init__(self, act, in_ch, out_ch=None):
#         super().__init__()
#         out_ch = out_ch if out_ch else in_ch
#         self.Conv_x = conv3x3(in_ch, out_ch)
#         self.Conv_y = conv3x3(in_ch, out_ch)
        
#         self.sobel_x = torch.nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
#         self.sobel_y = torch.nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        
#         # Define Sobel operators
#         self.sobel_x.weight = nn.Parameter(torch.tensor([[[[-1, 0, 1], 
#                                                            [-2, 0, 2], 
#                                                            [-1, 0, 1]]]*in_ch]*out_ch).cuda().float(), requires_grad=False)
        
#         self.sobel_y.weight = nn.Parameter(torch.tensor([[[[1, 2, 1], 
#                                                            [0, 0, 0], 
#                                                            [-1, -2, -1]]]*in_ch]*out_ch).cuda().float(), requires_grad=False)
        
#         self.act = act 

#     def forward(self, x):
#         # Calculate gradients
#         grad_x = self.sobel_x(x)
#         grad_y = self.sobel_y(x)

#         grad_x = self.act(self.Conv_x(grad_x))
#         grad_y = self.act(self.Conv_y(grad_y))

#         x = torch.sqrt(grad_x**2 + grad_y**2)

#         return x 

from DWT_IDWT.DWT_IDWT_layer import DWT_2D, IDWT_2D

class WaveletDownsample(nn.Module):
    def __init__(self, in_ch=None, out_ch=None):
        super().__init__()
        out_ch = out_ch if out_ch else in_ch
        self.weight = nn.Parameter(torch.zeros(out_ch, in_ch * 4, 3, 3))
        self.weight.data = default_init()(self.weight.data.shape)
        self.bias = nn.Parameter(torch.zeros(out_ch))

        self.dwt = DWT_2D("haar")

    def forward(self, x):
        xLL, xLH, xHL, xHH = self.dwt(x)

        x = torch.cat((xLL, xLH, xHL, xHH), dim=1) / 2.

        x = F.conv2d(x, self.weight, stride=1, padding=1)
        x = x + self.bias.reshape(1, -1, 1, 1)

        return x

class WaveletResnetBlock_Adagn(nn.Module): 
    ### equal to freq-aware (up)downsampling block ### in wavelet diffusion 
    
    def __init__(self, act, in_ch, out_ch=None, 
        clip_dim=None, zemb_dim=None, 
        up=False, down=False, dropout=0., 
        skip_rescale=True, init_scale=0., hi_in_ch=None, cross_attn=False):
            
        super().__init__()
        out_ch = out_ch if out_ch else in_ch

        self.GroupNorm_0 = AdaptiveGroupNorm(
            min(in_ch // 4, 32), in_ch, zemb_dim)

        self.up = up
        self.down = down
        self.cross_attn = cross_attn
        
        self.Conv_0 = conv3x3(in_ch, out_ch)

        if self.cross_attn:
            ### addition clip infor ### 
            self.Dense_cross_attn = nn.Linear(clip_dim, clip_dim)
            self.Dense_cross_attn.weight.data = default_init()(self.Dense_cross_attn.weight.shape)
            nn.init.zeros_(self.Dense_cross_attn.bias)

            self.GroupNorm_cross_attn_0 = AdaptiveGroupNorm(
            min(out_ch // 4, 32), out_ch, zemb_dim)
            self.Conv_in_cross_attn = conv3x3(out_ch, clip_dim)

            self.Multi_head_cross_attn = MultiHead_CrossAttention(hidden_size=clip_dim, num_heads=4)
            
            self.GroupNorm_cross_attn_1 = AdaptiveGroupNorm(
            min(clip_dim // 4, 32), clip_dim, zemb_dim)
            self.Conv_out_cross_attn = conv3x3(clip_dim, out_ch)

        if clip_dim is not None:
            self.Dense_0 = nn.Linear(clip_dim, out_ch)
            self.Dense_0.weight.data = default_init()(self.Dense_0.weight.shape)
            nn.init.zeros_(self.Dense_0.bias)

        self.GroupNorm_1 = AdaptiveGroupNorm(
            min(out_ch // 4, 32), out_ch, zemb_dim)
        
        self.Dropout_0 = nn.Dropout(dropout)
        self.Conv_1 = conv3x3(out_ch, out_ch, init_scale=init_scale)

        if in_ch != out_ch or up or down:
            self.Conv_2 = conv1x1(in_ch, out_ch)
        
        self.skip_rescale = skip_rescale
        self.act = act
        self.in_ch = in_ch
        self.out_ch = out_ch

        if self.up:
            self.convH_0 = conv3x3(hi_in_ch * 3, out_ch * 3, groups=3)
        
        self.dwt = DWT_2D("haar")
        self.iwt = IDWT_2D("haar")
        # self.edge = Edge_Enhancement_Layer(act=act, in_ch=out_ch)
    
    def forward(self, x, clip_embeddings=None, zemb=None, skipH=None):

        h = self.act(self.GroupNorm_0(x, zemb))
        h = self.Conv_0(h)

        if self.cross_attn:
            
            addition_clip = self.Dense_cross_attn(self.act(clip_embeddings))

            h = self.act(self.GroupNorm_cross_attn_0(h, zemb))
            h = self.Conv_in_cross_attn(h)
            
            h_size = h.size()
            h_flt = torch.flatten(h,2).permute(0,2,1) ### B x Clip_dim x Resolution => B x Resolution x Clip_dim ### h_cross_attn = h_cross_attn.permute(0,2,1)
            h_cross_attn = self.Multi_head_cross_attn(query = h_flt, key = addition_clip.unsqueeze(1), value = addition_clip.unsqueeze(1))
            h_cross_attn = h_cross_attn.permute(0,2,1) ### Re permute back  
            h = h_cross_attn.contiguous().view(h_size)
            
            h = self.act(self.GroupNorm_cross_attn_1(h, zemb))
            h = self.Conv_out_cross_attn(h)

        if self.in_ch != self.out_ch or self.up or self.down:
            x = self.Conv_2(x)

        hH = None
        if self.up:
            D = h.size(1)
            skipH = self.convH_0(torch.cat(skipH, dim=1) / 2.) * 2.
            h = self.iwt(2. * h, skipH[:, :D],
                         skipH[:, D: 2 * D], skipH[:, 2 * D:])
            x = self.iwt(2. * x, skipH[:, :D],
                         skipH[:, D: 2 * D], skipH[:, 2 * D:])

        elif self.down:
            h, hLH, hHL, hHH = self.dwt(h)
            x, xLH, xHL, xHH = self.dwt(x)
            hH, _ = (hLH, hHL, hHH), (xLH, xHL, xHH)

            h, x = h / 2., x / 2.  # shift range of ll

        if clip_embeddings is not None:
            h += self.Dense_0(self.act(clip_embeddings))[:, :, None, None]

        h = self.act(self.GroupNorm_1(h, zemb))
        h = self.Dropout_0(h)
        h = self.Conv_1(h)
        
        # h = self.edge(h)

        if not self.skip_rescale:
            out = x + h
        else:
            out = (x + h) / np.sqrt(2.)

        if not self.down:
            return out
        return out, hH

class WaveletResnetBlock_GN(nn.Module): 
    ### equal to freq-aware (up)downsampling block ### in wavelet diffusion 
    
    def __init__(self, act, in_ch, out_ch=None, 
        clip_dim=None, 
        up=False, down=False, dropout=0.1, 
        skip_rescale=True, init_scale=0., hi_in_ch=None):
            
        super().__init__()
        out_ch = out_ch if out_ch else in_ch

        self.GroupNorm_0 = nn.GroupNorm(
            min(in_ch // 4, 32), in_ch)

        self.up = up
        self.down = down
 
        self.Conv_0 = conv3x3(in_ch, out_ch)

        if clip_dim is not None:
            self.Dense_0 = nn.Linear(clip_dim, out_ch)
            self.Dense_0.weight.data = default_init()(self.Dense_0.weight.shape)
            nn.init.zeros_(self.Dense_0.bias)

        self.GroupNorm_1 = nn.GroupNorm(
            min(out_ch // 4, 32), out_ch)
        
        self.Dropout_0 = nn.Dropout(dropout)
        self.Conv_1 = conv3x3(out_ch, out_ch, init_scale=init_scale)

        if in_ch != out_ch or up or down:
            self.Conv_2 = conv1x1(in_ch, out_ch)
        
        self.skip_rescale = skip_rescale
        self.act = act
        self.in_ch = in_ch
        self.out_ch = out_ch

        if self.up:
            self.convH_0 = conv3x3(hi_in_ch * 3, out_ch * 3, groups=3)
        
        self.dwt = DWT_2D("haar")
        self.iwt = IDWT_2D("haar")
    
    def forward(self, x, clip_embeddings=None, skipH=None):

        h = self.act(self.GroupNorm_0(x))
        h = self.Conv_0(h)

        if self.in_ch != self.out_ch or self.up or self.down:
            x = self.Conv_2(x)

        hH = None
        if self.up:
            D = h.size(1)
            skipH = self.convH_0(torch.cat(skipH, dim=1) / 2.) * 2.
            h = self.iwt(2. * h, skipH[:, :D],
                         skipH[:, D: 2 * D], skipH[:, 2 * D:])
            x = self.iwt(2. * x, skipH[:, :D],
                         skipH[:, D: 2 * D], skipH[:, 2 * D:])

        elif self.down:
            h, hLH, hHL, hHH = self.dwt(h)
            x, xLH, xHL, xHH = self.dwt(x)
            hH, _ = (hLH, hHL, hHH), (xLH, xHL, xHH)

            h, x = h / 2., x / 2.  # shift range of ll

        if clip_embeddings is not None:
            h += self.Dense_0(self.act(clip_embeddings))[:, :, None, None]

        h = self.act(self.GroupNorm_1(h))
        h = self.Dropout_0(h)
        h = self.Conv_1(h)

        if not self.skip_rescale:
            out = x + h
        else:
            out = (x + h) / np.sqrt(2.)

        if not self.down:
            return out
        return out, hH