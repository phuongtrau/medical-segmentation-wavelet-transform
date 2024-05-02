from .blocks import *
import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm 
import functools

from DWT_IDWT.DWT_IDWT_layer import DWT_2D, IDWT_2D
from einops import rearrange

""" Full assembly of the parts to form the complete network """


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

from .blocks import MultiHead_CrossAttention, AttentionBlock, \
                        WaveletResnetBlock_Adagn, WaveletDownsample, Combine, \
                        conv3x3, default_init, PixelNorm, dense

class Wavelet_Segmentation(nn.Module): ### combine with implicit warping module ### 
    def __init__(self, config):
        super(Wavelet_Segmentation, self).__init__()
        self.config = config
        self.act = act = nn.SiLU()
        self.z_emb_dim = z_emb_dim = config.wavelet_model.z_emb_dim

        self.patch_size = config.wavelet_model.patch_size
        assert config.wavelet_model.current_resolution % self.patch_size == 0

        self.nf = nf = config.wavelet_model.num_channels_dae
        ch_mult = config.wavelet_model.ch_mult
        self.num_res_blocks = num_res_blocks = config.wavelet_model.num_res_blocks
        self.attn_resolutions = attn_resolutions = config.wavelet_model.attn_resolutions

        dropout = config.wavelet_model.dropout
        # resamp_with_conv = config.wavelet_model.resamp_with_conv
        self.num_resolutions = num_resolutions = len(ch_mult)
        
        self.all_resolutions = all_resolutions = [
            (config.wavelet_model.current_resolution // self.patch_size) // (2 ** i) for i in range(num_resolutions)]

        self.use_clip = use_clip = config.wavelet_model.use_clip
        self.use_cross_attn = config.wavelet_model.cross_attn
        # fir = config.wavelet_model.fir
        # fir_kernel = config.wavelet_model.fir_kernel
        init_scale = 0.
        self.skip_rescale = skip_rescale = config.wavelet_model.skip_rescale

        # combine_method = config.wavelet_model.progressive_combine.lower()
        # combiner = functools.partial(Combine, method=combine_method)

        ### start construct the main Unet ### 
        mains = [] ### main Unet 
        
        ### this one replace the time embedding by clip emb of clothe - only for main Unet ### 
        if use_clip: 
            mains.append(nn.Linear(config.clip.clip_chn, nf * 4)) ### the clip emb dimension is 512 
            mains[-1].weight.data = default_init()(mains[-1].weight.shape)
            nn.init.zeros_(mains[-1].bias)
            mains.append(nn.Linear(nf * 4, nf * 4))
            mains[-1].weight.data = default_init()(mains[-1].weight.shape)
            nn.init.zeros_(mains[-1].bias)
        
        AttnBlock = functools.partial(AttentionBlock)

        # Cross_AttnBlock = functools.partial(MultiHead_CrossAttention,
        #                               num_heads = 4,
        #                               attn_dropout = 0.2)

        ResnetBlock = functools.partial(WaveletResnetBlock_Adagn,
                                        act=act,
                                        dropout=dropout,
                                        init_scale=init_scale,
                                        skip_rescale=skip_rescale,
                                        clip_dim = nf*4,
                                        zemb_dim=z_emb_dim)

        WaveDownBlock = functools.partial(WaveletDownsample)

        #### Downsampling block

        channels = config.wavelet_model.num_channels * self.patch_size**2
        
        input_pyramid_ch = channels

        mains.append(conv3x3(channels, nf))

        hs_c = [nf]
        hs_c2 = [nf]

        in_ch = nf
        for i_level in range(num_resolutions):
            
            for i_block in range(num_res_blocks):
                out_ch = nf * ch_mult[i_level]

                if all_resolutions[i_level] in attn_resolutions:
                    mains.append(ResnetBlock(in_ch=in_ch, out_ch=out_ch, cross_attn=self.use_cross_attn))
                    in_ch = out_ch
                    mains.append(AttnBlock(in_channels=in_ch))
                else:
                    mains.append(ResnetBlock(in_ch=in_ch, out_ch=out_ch))
                    in_ch = out_ch
                hs_c.append(in_ch)
            
            if i_level != num_resolutions - 1:
                
                hs_c2.append(in_ch)
                
                mains.append(ResnetBlock(down=True, in_ch=in_ch))
                
                mains.append(WaveDownBlock(
                    in_ch=input_pyramid_ch, out_ch=in_ch))

                input_pyramid_ch = in_ch

                hs_c.append(in_ch)

        ##### middle #
        in_ch = hs_c[-1]

        mains.append(ResnetBlock(in_ch=in_ch, cross_attn=self.use_cross_attn))
        mains.append(AttnBlock(in_channels=in_ch))
        mains.append(ResnetBlock(in_ch=in_ch, cross_attn= self.use_cross_attn))


        # pyramid_ch = 0

        #### Upsampling block 
        for i_level in reversed(range(num_resolutions)):
            for i_block in range(num_res_blocks + 1):
                out_ch = nf * ch_mult[i_level]
                
                if all_resolutions[i_level] in attn_resolutions:
                    mains.append(ResnetBlock(in_ch=in_ch + hs_c.pop(),
                                           out_ch=out_ch, cross_attn= self.use_cross_attn))
                    in_ch = out_ch
                    mains.append(AttnBlock(in_channels=in_ch))
                else:
                    mains.append(ResnetBlock(in_ch=in_ch + hs_c.pop(),
                                           out_ch=out_ch, cross_attn= self.use_cross_attn))
                    in_ch = out_ch
            
            if i_level != 0:
                
                mains.append(ResnetBlock(
                    in_ch=in_ch, up=True, hi_in_ch=hs_c2.pop()))

        channels_out = config.wavelet_model.num_out_channels
        mains.append(nn.GroupNorm(num_groups=min(in_ch // 4, 32),
                                    num_channels=in_ch, eps=1e-6))
        mains.append(conv3x3(in_ch, channels_out, init_scale=init_scale))

        self.mains_modules = nn.ModuleList(mains)

        mapping_layers = [PixelNorm(),
                          dense(config.wavelet_model.nz, z_emb_dim),
                          self.act, ]
        
        for _ in range(config.wavelet_model.n_mlp):
            mapping_layers.append(dense(z_emb_dim, z_emb_dim))
            mapping_layers.append(self.act)

        self.z_transform = nn.Sequential(*mapping_layers)

        self.dwt = DWT_2D("haar")
        self.iwt = IDWT_2D("haar")

    def forward(self, x, z, clip_feature):

        # patchify
        x = rearrange(x, "n c (h p1) (w p2) -> n (p1 p2 c) h w",
                      p1=self.patch_size, p2=self.patch_size)
        
        zemb = self.z_transform(z)

        mains = self.mains_modules ### main Unet ###
        
        m_idx = 0

        if self.use_clip :
            clip_emb = mains[m_idx](clip_feature.type(torch.cuda.FloatTensor))
            m_idx += 1
            clip_emb = mains[m_idx](self.act(clip_emb))
            m_idx += 1
        else:
            clip_emb = None

        if not self.config.wavelet_model.centered:
            # If input data is in [0, 1]
            x = 2 * x - 1.

        # Downsampling block
        input_pyramid = x
        
        hs = [mains[m_idx](x)]
        skipHs = []
        m_idx += 1

        for i_level in range(self.num_resolutions):
            # Residual blocks for this resolution
            for i_block in range(self.num_res_blocks):
                h = mains[m_idx](x = hs[-1], clip_embeddings = clip_emb, zemb = zemb)
                m_idx += 1

                if h.shape[-2] in self.attn_resolutions:
                    h = mains[m_idx](h)
                    m_idx += 1

                hs.append(h)
            
            if i_level != self.num_resolutions - 1:

                h, skipH = mains[m_idx](x = h, clip_embeddings = clip_emb, zemb = zemb)
                skipHs.append(skipH)

                m_idx += 1

                input_pyramid = mains[m_idx](input_pyramid)
                m_idx += 1

                if self.skip_rescale:
                        input_pyramid = (input_pyramid + h) / np.sqrt(2.)

                else:
                    input_pyramid = input_pyramid + h

                h = input_pyramid
                hs.append(h)

        h = hs[-1]

        #### middle ### First Freq Bottelneck Block ###
        h, hlh, hhl, hhh = self.dwt(h)
        h = mains[m_idx](x = h / 2., clip_embeddings = clip_emb, zemb = zemb)
        h = self.iwt(h * 2., hlh, hhl, hhh)

        m_idx += 1

        # attn block
        h = mains[m_idx](h)
        m_idx += 1

        #### middle ### Second Freq Bottelneck Block ###
        h, hlh, hhl, hhh = self.dwt(h)
        h = mains[m_idx](x = h/2., clip_embeddings = clip_emb, zemb = zemb)
        h = self.iwt(h * 2., hlh, hhl, hhh)

        m_idx += 1


        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):

                h = mains[m_idx](x = torch.cat([h, hs.pop()], dim=1), clip_embeddings = clip_emb, zemb = zemb)
                
                m_idx += 1
            
                if h.shape[-2] in self.attn_resolutions:
                    h = mains[m_idx](h)
                    m_idx += 1

            if i_level != 0:
                h = mains[m_idx](x = h, clip_embeddings = clip_emb, zemb = zemb, skipH=skipHs.pop())
                m_idx += 1

        assert not hs

        h = self.act(mains[m_idx](h))
        m_idx += 1
        h = mains[m_idx](h)
        m_idx += 1
        assert m_idx == len(mains)

        # unpatchify
        h = rearrange(h, "n (c p1 p2) h w -> n c (h p1) (w p2)",
                      p1=self.patch_size, p2=self.patch_size)

        return h

class Wavelet_Segmentation_GN(nn.Module): ### combine with implicit warping module ### 
    def __init__(self, config):
        super(Wavelet_Segmentation_GN, self).__init__()
        self.config = config
        self.act = act = nn.ReLU()
        self.z_emb_dim = z_emb_dim = config.wavelet_model.z_emb_dim

        self.patch_size = config.wavelet_model.patch_size
        assert config.wavelet_model.current_resolution % self.patch_size == 0

        self.nf = nf = config.wavelet_model.num_channels_dae
        ch_mult = config.wavelet_model.ch_mult
        self.num_res_blocks = num_res_blocks = config.wavelet_model.num_res_blocks
        self.attn_resolutions = attn_resolutions = config.wavelet_model.attn_resolutions

        dropout = config.wavelet_model.dropout
        # resamp_with_conv = config.wavelet_model.resamp_with_conv
        self.num_resolutions = num_resolutions = len(ch_mult)
        
        self.all_resolutions = all_resolutions = [
            (config.wavelet_model.current_resolution // self.patch_size) // (2 ** i) for i in range(num_resolutions)]

        self.use_clip = use_clip = config.wavelet_model.use_clip
        # fir = config.wavelet_model.fir
        # fir_kernel = config.wavelet_model.fir_kernel
        init_scale = 0.
        self.skip_rescale = skip_rescale = config.wavelet_model.skip_rescale

        combine_method = config.wavelet_model.progressive_combine.lower()
        combiner = functools.partial(Combine, method=combine_method)

        ### start construct the main and branch Unet ### 
        mains = [] ### main Unet 
        
        ### this one replace the time embedding by clip emb of clothe - only for main Unet ### 
        if use_clip: 
            mains.append(nn.Linear(config.clip.clip_chn, nf * 4)) ### the clip emb dimension is 512 
            mains[-1].weight.data = default_init()(mains[-1].weight.shape)
            nn.init.zeros_(mains[-1].bias)
            mains.append(nn.Linear(nf * 4, nf * 4))
            mains[-1].weight.data = default_init()(mains[-1].weight.shape)
            nn.init.zeros_(mains[-1].bias)
        
        AttnBlock = functools.partial(AttentionBlock)

        ResnetBlock = functools.partial(WaveletResnetBlock_Adagn,
                                        act=act,
                                        dropout=dropout,
                                        init_scale=init_scale,
                                        skip_rescale=skip_rescale,
                                        clip_dim = nf*4)

        WaveDownBlock = functools.partial(WaveletDownsample)

        #### Downsampling block

        channels = config.wavelet_model.num_channels * self.patch_size**2
        
        input_pyramid_ch = channels

        mains.append(conv3x3(channels, nf))

        hs_c = [nf]
        hs_c2 = [nf]

        in_ch = nf
        for i_level in range(num_resolutions):
            
            for i_block in range(num_res_blocks):
                out_ch = nf * ch_mult[i_level]
            
                if all_resolutions[i_level] in attn_resolutions:
                    mains.append(ResnetBlock(in_ch=in_ch, out_ch=out_ch, cross_attn = True))
                    in_ch = out_ch
                    mains.append(AttnBlock(in_channels=in_ch))
                else:
                    mains.append(ResnetBlock(in_ch=in_ch, out_ch=out_ch))
                    in_ch = out_ch
                hs_c.append(in_ch)
            
            if i_level != num_resolutions - 1:
                
                hs_c2.append(in_ch)
                
                mains.append(ResnetBlock(down=True, in_ch=in_ch))
                
                mains.append(WaveDownBlock(
                    in_ch=input_pyramid_ch, out_ch=in_ch))

                input_pyramid_ch = in_ch

                hs_c.append(in_ch)

        ##### middle #
        in_ch = hs_c[-1]

        mains.append(ResnetBlock(in_ch=in_ch,cross_attn = True))
        mains.append(AttnBlock(in_channels=in_ch))
        mains.append(ResnetBlock(in_ch=in_ch,cross_attn = True))


        pyramid_ch = 0

        #### Upsampling block 
        for i_level in reversed(range(num_resolutions)):
            for i_block in range(num_res_blocks + 1):
                out_ch = nf * ch_mult[i_level]
                
                if all_resolutions[i_level] in attn_resolutions:
                    mains.append(ResnetBlock(in_ch=in_ch + hs_c.pop(),
                                           out_ch=out_ch, cross_attn=True))
                    in_ch = out_ch
                    mains.append(AttnBlock(in_channels=in_ch))
                else:
                    mains.append(ResnetBlock(in_ch=in_ch + hs_c.pop(),
                                           out_ch=out_ch))
                    in_ch = out_ch

            if i_level != 0:
                
                mains.append(ResnetBlock(
                    in_ch=in_ch, up=True, hi_in_ch=hs_c2.pop()))

        channels_out = config.wavelet_model.num_out_channels
        mains.append(nn.GroupNorm(num_groups=min(in_ch // 4, 32),
                                    num_channels=in_ch, eps=1e-6))
        mains.append(conv3x3(in_ch, channels_out, init_scale=init_scale))

        self.mains_modules = nn.ModuleList(mains)

        mapping_layers = [PixelNorm(),
                          dense(config.wavelet_model.nz, z_emb_dim),
                          self.act, ]
        
        for _ in range(config.wavelet_model.n_mlp):
            mapping_layers.append(dense(z_emb_dim, z_emb_dim))
            mapping_layers.append(self.act)

        self.z_transform = nn.Sequential(*mapping_layers)

        self.dwt = DWT_2D("haar")
        self.iwt = IDWT_2D("haar")

    def forward(self, x, clip_feature):

        # patchify
        x = rearrange(x, "n c (h p1) (w p2) -> n (p1 p2 c) h w",
                      p1=self.patch_size, p2=self.patch_size)
        
        # zemb = self.z_transform(z)

        mains = self.mains_modules ### main Unet ###
        
        m_idx = 0

        if self.use_clip :
            clip_emb = mains[m_idx](clip_feature.type(torch.cuda.FloatTensor))
            m_idx += 1
            clip_emb = mains[m_idx](self.act(clip_emb))
            m_idx += 1
        else:
            clip_emb = None

        if not self.config.wavelet_model.centered:
            # If input data is in [0, 1]
            x = 2 * x - 1.

        # Downsampling block
        input_pyramid = x
        
        hs = [mains[m_idx](x)]
        skipHs = []
        m_idx += 1

        for i_level in range(self.num_resolutions):
            # Residual blocks for this resolution
            for i_block in range(self.num_res_blocks):
                h = mains[m_idx](x = hs[-1], clip_embeddings = clip_emb)
                m_idx += 1

                if h.shape[-2] in self.attn_resolutions:
                    h = mains[m_idx](h)
                    m_idx += 1

                hs.append(h)
            
            if i_level != self.num_resolutions - 1:

                h, skipH = mains[m_idx](x = h, clip_embeddings = clip_emb)
                skipHs.append(skipH)

                m_idx += 1

                input_pyramid = mains[m_idx](input_pyramid)
                m_idx += 1

                if self.skip_rescale:
                        input_pyramid = (input_pyramid + h) / np.sqrt(2.)

                else:
                    input_pyramid = input_pyramid + h

                h = input_pyramid
                hs.append(h)

        h = hs[-1]

        #### middle ### First Freq Bottelneck Block ###
        h, hlh, hhl, hhh = self.dwt(h)
        h = mains[m_idx](x = h / 2., clip_embeddings = clip_emb)
        h = self.iwt(h * 2., hlh, hhl, hhh)

        m_idx += 1

        # attn block
        h = mains[m_idx](h)
        m_idx += 1

        #### middle ### Second Freq Bottelneck Block ###
        h, hlh, hhl, hhh = self.dwt(h)
        h = mains[m_idx](x = h/2., clip_embeddings = clip_emb)
        h = self.iwt(h * 2., hlh, hhl, hhh)

        m_idx += 1


        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):

                h = mains[m_idx](x = torch.cat([h, hs.pop()], dim=1), clip_embeddings = clip_emb)
                
                m_idx += 1
            
                if h.shape[-2] in self.attn_resolutions:
                    h = mains[m_idx](h)
                    m_idx += 1

            if i_level != 0:
                h = mains[m_idx](x = h, clip_embeddings = clip_emb, skipH=skipHs.pop())
                m_idx += 1

        assert not hs

        h = self.act(mains[m_idx](h))
        m_idx += 1
        h = mains[m_idx](h)
        m_idx += 1
        assert m_idx == len(mains)

        # unpatchify
        h = rearrange(h, "n (c p1 p2) h w -> n c (h p1) (w p2)",
                      p1=self.patch_size, p2=self.patch_size)

        return h
    
### Discriminator model ### 
class SpectralDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(SpectralDiscriminator, self).__init__()
        # no need to use bias as BatchNorm2d has affine parameters
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        kw = 4
        padw = 1
        sequence = [spectral_norm(nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw)),
                    nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                spectral_norm(nn.Conv2d(ndf * nf_mult_prev, ndf *
                              nf_mult, kernel_size=kw, stride=2, padding=padw)),
                norm_layer(ndf * nf_mult,affine=True, track_running_stats=True),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            spectral_norm(nn.Conv2d(ndf * nf_mult_prev, ndf *
                          nf_mult, kernel_size=kw, stride=1, padding=padw)),
            norm_layer(ndf * nf_mult,affine=True, track_running_stats=True),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [spectral_norm(nn.Conv2d(ndf * nf_mult,
                                   1, kernel_size=kw, stride=1, padding=padw))]
        if use_sigmoid:
            sequence += [nn.Sigmoid()]
        self.model = nn.Sequential(*sequence)
    
    def forward(self, input):
        """Standard forward."""
        return self.model(input)