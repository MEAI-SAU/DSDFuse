import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import functools
from model.diffusers.module_util import (
    NonLinearity,
    Upsample, Downsample,
    default_conv,
    ResBlock,
    LinearAttention,
    PreNorm, Residual, Identity)


def build_batch_norm_layer(num_features):
    return nn.BatchNorm2d(num_features)


class Encode(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, ch=64, ch_mult=[1, 2, 4, 4], embed_dim=4):
        super().__init__()
        self.depth = len(ch_mult)

        block_class = functools.partial(ResBlock, conv=default_conv, act=NonLinearity())

        self.init_conv1 = default_conv(in_ch, ch, 3)
        self.init_conv2 = default_conv(in_ch, ch, 3)

        self.encoder1 = nn.ModuleList([])
        self.encoder2 = nn.ModuleList([])

        ch_mult = [1] + ch_mult
        for i in range(self.depth):
            dim_in = ch * ch_mult[i]
            dim_out = ch * ch_mult[i + 1]
            self.encoder1.append(nn.ModuleList([
                block_class(dim_in=dim_in, dim_out=dim_in),
                block_class(dim_in=dim_in, dim_out=dim_in),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))) if i == (self.depth - 1) else Identity(),
                Downsample(dim_in, dim_out) if i != (self.depth - 1) else default_conv(dim_in, dim_out)
            ]))
            self.encoder2.append(nn.ModuleList([
                block_class(dim_in=dim_in, dim_out=dim_in),
                block_class(dim_in=dim_in, dim_out=dim_in),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))) if i == (self.depth - 1) else Identity(),
                Downsample(dim_in, dim_out) if i != (self.depth - 1) else default_conv(dim_in, dim_out)
            ]))

        mid_dim = ch * ch_mult[-1]

        self.latent_conv1 = block_class(dim_in=mid_dim, dim_out=embed_dim)
        self.latent_conv2 = block_class(dim_in=mid_dim, dim_out=embed_dim)

        self.conv_fuse = nn.ModuleList()
        self.num_channels = [
            ch * ch_mult[0], ch * ch_mult[0], ch * ch_mult[0],
            ch * ch_mult[1], ch * ch_mult[1],
            ch * ch_mult[2], ch * ch_mult[2],
            ch * ch_mult[3], ch * ch_mult[3]
        ]
        for i in range(len(self.num_channels)):
            self.conv_fuse.append(
                nn.Sequential(
                    nn.Conv2d(self.num_channels[i] * 2, self.num_channels[i], kernel_size=3, padding=1, stride=1, dilation=1, groups=1),
                    build_batch_norm_layer(self.num_channels[i]),
                    nn.LeakyReLU(0.2, True),
                )
            )

        self.last_codec_state = None

    def check_image_size(self, x, h, w):
        s = int(math.pow(2, self.depth))
        mod_pad_h = (s - h % s) % s
        mod_pad_w = (s - w % s) % s
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def forward(self, x1, x2):
        self.H, self.W = x1.shape[2:]
        x1 = self.check_image_size(x1, self.H, self.W)
        x2 = self.check_image_size(x2, self.H, self.W)

        x1 = self.init_conv1(x1)
        x2 = self.init_conv2(x2)
        h1 = [x1]
        h2 = [x2]
        fused_skips = []

        for b1, b2, attn, downsample in self.encoder1:
            x1 = b1(x1)
            h1.append(x1)
            x1 = b2(x1)
            x1 = attn(x1)
            h1.append(x1)
            x1 = downsample(x1)

        x1 = self.latent_conv1(x1)

        for b1, b2, attn, downsample in self.encoder2:
            x2 = b1(x2)
            h2.append(x2)
            x2 = b2(x2)
            x2 = attn(x2)
            h2.append(x2)
            x2 = downsample(x2)

        x2 = self.latent_conv2(x2)

        for i in range(len(h1)):
            pair = torch.cat((h1[i], h2[i]), dim=1)
            fused_skips.append(self.conv_fuse[i](pair))

        codec_state = {
            'structure_skips': h1,
            'function_skips': h2,
            'fused_skips': fused_skips,
        }
        self.last_codec_state = codec_state
        return x1, x2, codec_state


class Decode(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, ch=64, ch_mult=[1, 2, 4, 4], embed_dim=4):
        super().__init__()
        self.depth = len(ch_mult)

        block_class = functools.partial(ResBlock, conv=default_conv, act=NonLinearity())

        self.decoder = nn.ModuleList([])
        ch_mult = [1] + ch_mult
        for i in range(self.depth):
            dim_in = ch * ch_mult[i]
            dim_out = ch * ch_mult[i + 1]
            self.decoder.insert(0, nn.ModuleList([
                block_class(dim_in=dim_out + dim_in, dim_out=dim_out),
                block_class(dim_in=dim_out + dim_in, dim_out=dim_out),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))) if i == (self.depth - 1) else Identity(),
                Upsample(dim_out, dim_in) if i != 0 else default_conv(dim_out, dim_in)
            ]))

        mid_dim = ch * ch_mult[-1]
        self.post_latent_conv = block_class(dim_in=embed_dim, dim_out=mid_dim)
        self.final_conv = nn.Conv2d(ch, out_ch, 3, 1, 1)
        self.last_codec_state = None

    def check_image_size(self, x, h, w):
        s = int(math.pow(2, self.depth))
        mod_pad_h = (s - h % s) % s
        mod_pad_w = (s - w % s) % s
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def _resolve_skips(self, h):
        if isinstance(h, dict):
            self.last_codec_state = h
            if 'fused_skips' in h and h['fused_skips'] is not None:
                return h['fused_skips']
            if 'structure_skips' in h and 'function_skips' in h:
                h = (h['structure_skips'], h['function_skips'])

        if isinstance(h, (tuple, list)) and len(h) == 2 and all(isinstance(item, (list, tuple)) for item in h):
            structure_skips, function_skips = h
            fused = []
            for s_feat, f_feat in zip(structure_skips, function_skips):
                if s_feat.shape[-2:] != f_feat.shape[-2:]:
                    f_feat = F.interpolate(f_feat, size=s_feat.shape[-2:], mode='bilinear', align_corners=False)
                fused.append(torch.cat((s_feat, f_feat), dim=1))
            return fused

        return h

    def forward(self, x, h):
        h = self._resolve_skips(h)
        x = self.post_latent_conv(x)
        for i, (b1, b2, attn, upsample) in enumerate(self.decoder):
            x = torch.cat([x, h[-(i * 2 + 1)]], dim=1)
            x = b1(x)

            x = torch.cat([x, h[-(i * 2 + 2)]], dim=1)
            x = b2(x)
            x = attn(x)

            x = upsample(x)

        x = self.final_conv(x + h[0])
        x = torch.tanh(x)
        return x


if __name__ == '__main__':
    test_sample1 = torch.randn(1, 1, 256, 256).cuda(1)
    test_sample2 = torch.randn(1, 1, 256, 256).cuda(1)

    LDM_Enc = Encode(in_ch=1,
                     out_ch=1,
                     ch=8,
                     ch_mult=[4, 4, 4, 8],
                     embed_dim=8).cuda(1).eval()
    LDM_Dec = Decode(in_ch=1,
                     out_ch=1,
                     ch=8,
                     ch_mult=[4, 4, 4, 8],
                     embed_dim=8).cuda(1).eval()

    x1, x2, h = LDM_Enc(test_sample1, test_sample2)
    x = LDM_Dec(x2, h)

    print(x.shape)
    total = sum([param.nelement() for param in LDM_Enc.parameters()])
    print("Number of parameters: %.2fM" % (total / 1e6))
