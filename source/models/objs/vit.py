import torch

import torch.nn as nn
import torch.nn.functional as F
import math
import einops
from einops.layers.torch import Rearrange, Reduce
from source.layers.common_layers import PatchEmbedding, RGBNormalize, LayerNormForImage
from source.layers.common_fns import positionalencoding2d
from source.layers.common_layers import Attention
from source.layers.common_layers import Reshape


class TransformerBlock(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        mlp_dim,
        dropout=0.0,
        hw=None,
        gta=False,
    ):
        super().__init__()
        self.layernorm1 = LayerNormForImage(embed_dim)
        self.attn = Attention(
            embed_dim,
            num_heads,
            weight="conv",
            gta=gta,
            hw=hw,
        )
        self.layernorm2 = LayerNormForImage(embed_dim)
        self.mlp = nn.Sequential(
            nn.Conv2d(embed_dim, mlp_dim, 1, 1, 0),
            nn.GELU(),
            nn.Conv2d(mlp_dim, embed_dim, 1, 1, 0),
            nn.Dropout(dropout),
        )

    def forward(self, src, T):
        xs = []
        # Repeat attention T times
        for _ in range(T):
            src2 = self.layernorm1(src)
            src2 = self.attn(src2)
            src = src + src2
            xs.append(src)

        src2 = self.layernorm2(src)
        src2 = self.mlp(src2)
        src = src + src2
        return src, xs


class ViT(nn.Module):
    # ViT with iterlative self-attention
    def __init__(
        self,
        imsize=128,
        psize=8,
        ch=128,
        blocks=1,
        heads=4,
        mlp_dim=256,
        T=8,
        maxpool=False,
        gta=False,
        autorescale=False,
    ):
        super().__init__()
        self.T = T
        self.psize = psize
        self.autorescale = autorescale
        self.patchfy = nn.Sequential(
            RGBNormalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            nn.Conv2d(3, ch, kernel_size=psize, stride=psize, padding=0),
        )
        if not gta:
            self.pos_embed = nn.Parameter(
                positionalencoding2d(ch, imsize // psize, imsize // psize)
                .reshape(-1, imsize // psize, imsize // psize)
            )

        self.transformer_encoder = nn.ModuleList(
            [
                TransformerBlock(
                    ch,
                    heads,
                    mlp_dim,
                    0.0,
                    hw=[imsize // psize, imsize // psize],
                    gta=gta,
                )
                for _ in range(blocks)
            ]
        )

        self.out = torch.nn.Sequential(
            LayerNormForImage(ch),
            (
                nn.AdaptiveMaxPool2d((1, 1))
                if not maxpool
                else nn.AdaptiveMaxPool2d((1, 1))
            ),
            Reshape(-1, ch),
            nn.Linear(ch, 4 * ch),
            nn.ReLU(),
            nn.Linear(4 * ch, ch),
        )

    def feature(self, x):
        if self.autorescale and (
            x.shape[2] != self.imsize or x.shape[3] != self.imsize
        ):
            x = F.interpolate(
                x,
                (self.imsize, self.imsize),
                mode="bilinear",
            )
        x = self.patchfy(x)
        if hasattr(self, "pos_embed"):
            x = x + self.pos_embed[None]
        xs = [x]
        for block in self.transformer_encoder:
            x, _xs = block(x, self.T)
            xs.append(_xs)
        return x, xs

    def forward(self, x, return_xs=False):
        x, xs = self.feature(x)
        x = self.out(x)
        if return_xs:
            return x, xs
        else:
            return x
