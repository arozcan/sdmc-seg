# ldm/models/transunet.py
# TransUNet (ViT + UNet decoder) - adapted for 2D medical segmentation
# Minimal, self-contained version inspired by official TransUNet repo.

from __future__ import annotations

import math
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
# Config helpers
# -------------------------

@dataclass
class TransformerCfg:
    mlp_dim: int = 3072
    num_heads: int = 12
    num_layers: int = 12
    attention_dropout_rate: float = 0.0
    dropout_rate: float = 0.1


@dataclass
class PatchesCfg:
    # either size=(16,16) or grid=(16,16) for hybrid (ResNet + ViT)
    size: Tuple[int, int] = (16, 16)
    grid: Optional[Tuple[int, int]] = None


@dataclass
class ResNetCfg:
    num_layers: Tuple[int, int, int] = (3, 4, 9)  # like R50-ViT-B/16 in TransUNet repo
    width_factor: int = 1


@dataclass
class TransUNetConfig:
    patches: PatchesCfg = field(default_factory=PatchesCfg)
    hidden_size: int = 768
    transformer: TransformerCfg = field(default_factory=TransformerCfg)

    # decoder
    decoder_channels: Tuple[int, int, int, int] = (256, 128, 64, 16)
    n_skip: int = 3
    skip_channels: List[int] = field(default_factory=lambda: [512, 256, 64, 16])

    # bookkeeping
    classifier: str = "seg"
    n_classes: int = 3  # will be overwritten by wrapper
    resnet: Optional[ResNetCfg] = None


def get_vit_b16_config(n_classes: int = 3) -> TransUNetConfig:
    cfg = TransUNetConfig()
    cfg.patches = PatchesCfg(size=(16, 16), grid=None)
    cfg.hidden_size = 768
    cfg.transformer = TransformerCfg(mlp_dim=3072, num_heads=12, num_layers=12, attention_dropout_rate=0.0, dropout_rate=0.1)
    cfg.decoder_channels = (256, 128, 64, 16)
    cfg.n_skip = 0
    cfg.skip_channels = [0, 0, 0, 0]
    cfg.classifier = "seg"
    cfg.n_classes = n_classes
    cfg.resnet = None
    return cfg


def get_r50_vit_b16_config(n_classes: int = 3) -> TransUNetConfig:
    cfg = get_vit_b16_config(n_classes=n_classes)
    cfg.patches = PatchesCfg(size=(16, 16), grid=(16, 16))  # enables hybrid ResNetV2
    cfg.resnet = ResNetCfg(num_layers=(3, 4, 9), width_factor=1)
    cfg.n_skip = 3
    cfg.skip_channels = [1024, 512, 256, 64]
    return cfg


# -------------------------
# TransUNet building blocks
# -------------------------

def swish(x):
    return x * torch.sigmoid(x)


class Attention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, attn_drop: float, proj_drop: float, vis: bool = False):
        super().__init__()
        self.vis = vis
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        self.all_head_size = self.num_heads * self.head_dim

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.out = nn.Linear(hidden_size, hidden_size)
        self.attn_dropout = nn.Dropout(attn_drop)
        self.proj_dropout = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

    def transpose_for_scores(self, x):
        # (B, N, hidden) -> (B, heads, N, head_dim)
        B, N, _ = x.shape
        x = x.view(B, N, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        q = self.transpose_for_scores(self.query(hidden_states))
        k = self.transpose_for_scores(self.key(hidden_states))
        v = self.transpose_for_scores(self.value(hidden_states))

        attn = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        attn = self.softmax(attn)
        weights = attn if self.vis else None
        attn = self.attn_dropout(attn)

        context = torch.matmul(attn, v)  # (B, heads, N, head_dim)
        context = context.permute(0, 2, 1, 3).contiguous()
        B, N, _, _ = context.shape
        context = context.view(B, N, self.all_head_size)

        out = self.out(context)
        out = self.proj_dropout(out)
        return out, weights


class Mlp(nn.Module):
    def __init__(self, hidden_size: int, mlp_dim: int, drop: float):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, hidden_size)
        self.act = torch.nn.functional.gelu
        self.dropout = nn.Dropout(drop)

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_dim: int, attn_drop: float, drop: float, vis: bool = False):
        super().__init__()
        self.attention_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.ffn_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads, attn_drop=attn_drop, proj_drop=attn_drop, vis=vis)
        self.ffn = Mlp(hidden_size, mlp_dim, drop=drop)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, w = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, w


# -------------------------
# Hybrid ResNetV2 (skip features)
# (kept close to repo behavior but simplified)
# -------------------------

class StdConv2d(nn.Conv2d):
    def forward(self, x):
        w = self.weight
        v, m = torch.var_mean(w, dim=[1, 2, 3], keepdim=True, unbiased=False)
        w = (w - m) / torch.sqrt(v + 1e-5)
        return F.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)


def conv3x3(cin, cout, stride=1, groups=1, bias=False):
    return StdConv2d(cin, cout, kernel_size=3, stride=stride, padding=1, bias=bias, groups=groups)


def conv1x1(cin, cout, stride=1, bias=False):
    return StdConv2d(cin, cout, kernel_size=1, stride=stride, padding=0, bias=bias)


class PreActBottleneck(nn.Module):
    def __init__(self, cin, cout=None, cmid=None, stride=1):
        super().__init__()
        cout = cout or cin
        cmid = cmid or cout // 4

        self.gn1 = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv1 = conv1x1(cin, cmid, bias=False)
        self.gn2 = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv2 = conv3x3(cmid, cmid, stride, bias=False)
        self.gn3 = nn.GroupNorm(32, cout, eps=1e-6)
        self.conv3 = conv1x1(cmid, cout, bias=False)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = None
        if (stride != 1) or (cin != cout):
            self.downsample = conv1x1(cin, cout, stride, bias=False)
            self.gn_proj = nn.GroupNorm(cout, cout)

    def forward(self, x):
        residual = x
        if self.downsample is not None:
            residual = self.downsample(x)
            residual = self.gn_proj(residual)

        y = self.relu(self.gn1(self.conv1(x)))
        y = self.relu(self.gn2(self.conv2(y)))
        y = self.gn3(self.conv3(y))
        y = self.relu(residual + y)
        return y


class ResNetV2(nn.Module):
    def __init__(self, block_units=(3, 4, 9), width_factor=1):
        super().__init__()
        width = int(64 * width_factor)
        self.width = width

        self.root = nn.Sequential(
            StdConv2d(3, width, kernel_size=7, stride=2, bias=False, padding=3),
            nn.GroupNorm(32, width, eps=1e-6),
            nn.ReLU(inplace=True),
        )

        # stage widths: width*4, width*8, width*16
        self.block1 = nn.Sequential(*([PreActBottleneck(width, width * 4, width)] +
                                      [PreActBottleneck(width * 4, width * 4, width) for _ in range(block_units[0] - 1)]))

        self.block2 = nn.Sequential(*([PreActBottleneck(width * 4, width * 8, width * 2, stride=2)] +
                                      [PreActBottleneck(width * 8, width * 8, width * 2) for _ in range(block_units[1] - 1)]))

        self.block3 = nn.Sequential(*([PreActBottleneck(width * 8, width * 16, width * 4, stride=2)] +
                                      [PreActBottleneck(width * 16, width * 16, width * 4) for _ in range(block_units[2] - 1)]))

    def forward(self, x):
        # Return x (deep feature) and list of skip features (high->low resolution order for decoder)
        feats = []
        x = self.root(x)                  # 1/2
        feats.append(x)
        x = F.max_pool2d(x, 3, stride=2)  # 1/4

        x = self.block1(x)                # 1/4
        feats.append(x)
        x = self.block2(x)                # 1/8
        feats.append(x)
        x = self.block3(x)                # 1/16
        feats.append(x)

        # TransUNet decoder expects features reversed (deep->shallow)
        return x, feats[::-1]


# -------------------------
# Embeddings + Encoder
# -------------------------

class Embeddings(nn.Module):
    def __init__(self, cfg: TransUNetConfig, img_size: Tuple[int, int], in_channels: int = 3):
        super().__init__()
        self.cfg = cfg
        self.img_size = img_size

        self.hybrid = cfg.patches.grid is not None
        self.hybrid_model = None

        if self.hybrid:
            assert cfg.resnet is not None, "Hybrid mode requires cfg.resnet"
            self.hybrid_model = ResNetV2(block_units=cfg.resnet.num_layers, width_factor=cfg.resnet.width_factor)
            in_channels = self.hybrid_model.width * 16  # matches repo assumption

            # patch size derived from grid (repo logic)
            grid_h, grid_w = cfg.patches.grid
            patch_h = img_size[0] // 16 // grid_h
            patch_w = img_size[1] // 16 // grid_w
            self.patch_size = (patch_h * 16, patch_w * 16)
        else:
            self.patch_size = cfg.patches.size

        n_patches = (img_size[0] // self.patch_size[0]) * (img_size[1] // self.patch_size[1])

        self.patch_embeddings = nn.Conv2d(in_channels, cfg.hidden_size, kernel_size=self.patch_size, stride=self.patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, cfg.hidden_size))
        self.dropout = nn.Dropout(cfg.transformer.dropout_rate)

    def forward(self, x):
        features = None
        if self.hybrid:
            x, features = self.hybrid_model(x)  # x: deep conv feature

        x = self.patch_embeddings(x)       # (B, hidden, H', W')
        x = x.flatten(2).transpose(-1, -2) # (B, N, hidden)

        x = x + self.position_embeddings
        x = self.dropout(x)
        return x, features


class Encoder(nn.Module):
    def __init__(self, cfg: TransUNetConfig, vis: bool = False):
        super().__init__()
        self.vis = vis
        self.layers = nn.ModuleList([
            Block(
                hidden_size=cfg.hidden_size,
                num_heads=cfg.transformer.num_heads,
                mlp_dim=cfg.transformer.mlp_dim,
                attn_drop=cfg.transformer.attention_dropout_rate,
                drop=cfg.transformer.dropout_rate,
                vis=vis,
            )
            for _ in range(cfg.transformer.num_layers)
        ])
        self.encoder_norm = nn.LayerNorm(cfg.hidden_size, eps=1e-6)

    def forward(self, x):
        attn_weights = []
        for blk in self.layers:
            x, w = blk(x)
            if self.vis:
                attn_weights.append(w)
        x = self.encoder_norm(x)
        return x, attn_weights


class Transformer(nn.Module):
    def __init__(self, cfg: TransUNetConfig, img_size: Tuple[int, int], vis: bool = False):
        super().__init__()
        self.embeddings = Embeddings(cfg, img_size=img_size, in_channels=3)
        self.encoder = Encoder(cfg, vis=vis)

    def forward(self, x):
        x, features = self.embeddings(x)
        x, attn = self.encoder(x)
        return x, attn, features


# -------------------------
# Decoder
# -------------------------

class Conv2dReLU(nn.Sequential):
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1, use_batchnorm=True):
        conv = nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding, bias=not use_batchnorm)
        bn = nn.BatchNorm2d(out_ch) if use_batchnorm else nn.Identity()
        relu = nn.ReLU(inplace=True)
        super().__init__(conv, bn, relu)


class DecoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, skip_ch=0, use_batchnorm=True):
        super().__init__()
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv1 = Conv2dReLU(in_ch + skip_ch, out_ch, kernel_size=3, padding=1, use_batchnorm=use_batchnorm)
        self.conv2 = Conv2dReLU(out_ch, out_ch, kernel_size=3, padding=1, use_batchnorm=use_batchnorm)

    def forward(self, x, skip=None):
        x = self.up(x)

        if skip is not None:
            # --- IMPORTANT: align spatial size for safe concat ---
            if skip.shape[-2:] != x.shape[-2:]:
                skip = F.interpolate(skip, size=x.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)

        x = self.conv1(x)
        x = self.conv2(x)
        return x


class DecoderCup(nn.Module):
    def __init__(self, cfg: TransUNetConfig):
        super().__init__()
        self.cfg = cfg
        head_ch = 512

        self.conv_more = Conv2dReLU(cfg.hidden_size, head_ch, kernel_size=3, padding=1, use_batchnorm=True)

        decoder_channels = cfg.decoder_channels
        in_channels = [head_ch] + list(decoder_channels[:-1])
        out_channels = list(decoder_channels)

        # skip channel selection (repo logic)
        if cfg.n_skip != 0:
            skip_channels = list(cfg.skip_channels)
            for i in range(4 - cfg.n_skip):
                skip_channels[3 - i] = 0
        else:
            skip_channels = [0, 0, 0, 0]

        self.blocks = nn.ModuleList([
            DecoderBlock(in_ch, out_ch, sk_ch)
            for in_ch, out_ch, sk_ch in zip(in_channels, out_channels, skip_channels)
        ])

    def forward(self, hidden_states, features=None):
        # hidden_states: (B, N, hidden) -> (B, hidden, h, w)
        B, N, hidden = hidden_states.shape
        h = w = int(math.sqrt(N))
        x = hidden_states.permute(0, 2, 1).contiguous().view(B, hidden, h, w)

        x = self.conv_more(x)

        for i, blk in enumerate(self.blocks):
            skip = None
            if features is not None and i < self.cfg.n_skip:
                skip = features[i]
            x = blk(x, skip=skip)
        return x


class SegmentationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        up = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        super().__init__(conv, up)


class VisionTransformer(nn.Module):
    def __init__(self, cfg: TransUNetConfig, img_size=(256, 256), vis: bool = False):
        super().__init__()
        self.cfg = cfg
        self.transformer = Transformer(cfg, img_size=img_size, vis=vis)
        self.decoder = DecoderCup(cfg)
        self.segmentation_head = SegmentationHead(
            in_channels=cfg.decoder_channels[-1],
            out_channels=cfg.n_classes,
            kernel_size=3,
        )

    def forward(self, x):
        # TransUNet expects 3-channel for hybrid/patch embedding; repeat if grayscale
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        tokens, _, features = self.transformer(x)
        x = self.decoder(tokens, features)
        logits = self.segmentation_head(x)
        return logits


class TransUNet(nn.Module):
    """
    Simple wrapper: selects config + builds TransUNet VisionTransformer.
    """
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 3,
        img_size: Tuple[int, int] = (256, 256),
        variant: str = "R50-ViT-B_16",  # or "ViT-B_16"
        vis: bool = False,
    ):
        super().__init__()
        if variant.upper() in ("R50-VIT-B_16", "R50+VIT-B_16", "R50_VIT_B16", "R50-VIT-B16"):
            cfg = get_r50_vit_b16_config(n_classes=out_channels)
        else:
            cfg = get_vit_b16_config(n_classes=out_channels)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.img_size = img_size
        self.variant = variant

        self.net = VisionTransformer(cfg, img_size=img_size, vis=vis)

    def forward(self, x):
        # x must be (B, C, H, W)
        return self.net(x)