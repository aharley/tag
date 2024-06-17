import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import utils.basic
import utils.improc
import utils.samp
from utils.basic import print_stats
import utils.samp
import utils.misc
from torch import nn, einsum
from einops import rearrange, repeat
from einops.layers.torch import Rearrange, Reduce
import copy
import math
from torch import nn, Tensor
import time
from collections import OrderedDict
from typing import Any, Callable, Dict, List, NamedTuple, Optional


class MLP(torch.nn.Sequential):
    """This block implements the multi-layer perceptron (MLP) module.

    Args:
        in_channels (int): Number of channels of the input
        hidden_channels (List[int]): List of the hidden channel dimensions
        norm_layer (Callable[..., torch.nn.Module], optional): Norm layer that will be stacked on top of the linear layer. If ``None`` this layer won't be used. Default: ``None``
        activation_layer (Callable[..., torch.nn.Module], optional): Activation function which will be stacked on top of the normalization layer (if not None), otherwise on top of the linear layer. If ``None`` this layer won't be used. Default: ``torch.nn.ReLU``
        inplace (bool, optional): Parameter for the activation layer, which can optionally do the operation in-place.
            Default is ``None``, which uses the respective default values of the ``activation_layer`` and Dropout layer.
        bias (bool): Whether to use bias in the linear layer. Default ``True``
        dropout (float): The probability for the dropout layer. Default: 0.0
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: List[int],
        norm_layer: Optional[Callable[..., torch.nn.Module]] = None,
        activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU,
        inplace: Optional[bool] = None,
        bias: bool = True,
        dropout: float = 0.0,
    ):
        # The addition of `norm_layer` is inspired from the implementation of TorchMultimodal:
        # https://github.com/facebookresearch/multimodal/blob/5dec8a/torchmultimodal/modules/layers/mlp.py
        params = {} if inplace is None else {"inplace": inplace}

        layers = []
        in_dim = in_channels
        for hidden_dim in hidden_channels[:-1]:
            layers.append(torch.nn.Linear(in_dim, hidden_dim, bias=bias))
            if norm_layer is not None:
                layers.append(norm_layer(hidden_dim))
            layers.append(activation_layer(**params))
            layers.append(torch.nn.Dropout(dropout, **params))
            in_dim = hidden_dim

        layers.append(torch.nn.Linear(in_dim, hidden_channels[-1], bias=bias))
        layers.append(torch.nn.Dropout(dropout, **params))

        super().__init__(*layers)
        # _log_api_usage_once(self)

class MLPBlock(MLP):
    """Transformer MLP block."""

    _version = 2

    def __init__(self, in_dim: int, mlp_dim: int, dropout: float):
        super().__init__(in_dim, [mlp_dim, in_dim], activation_layer=nn.GELU, inplace=None, dropout=dropout)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        version = local_metadata.get("version", None)

        if version is None or version < 2:
            # Replacing legacy MLPBlock with MLP. See https://github.com/pytorch/vision/pull/6053
            for i in range(2):
                for type in ["weight", "bias"]:
                    old_key = f"{prefix}linear_{i+1}.{type}"
                    new_key = f"{prefix}{3*i}.{type}"
                    if old_key in state_dict:
                        state_dict[new_key] = state_dict.pop(old_key)

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


class EncoderBlock(nn.Module):
    """Transformer encoder block."""

    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        self.num_heads = num_heads

        # Attention block
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=attention_dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

        # MLP block
        self.ln_2 = norm_layer(hidden_dim)
        self.mlp = MLPBlock(hidden_dim, mlp_dim, dropout)


    def forward(self, input: torch.Tensor):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        B, K, C = input.shape
            
        x = self.ln_1(input)
        x, _ = self.self_attention(x, x, x, need_weights=False)
        x = self.dropout(x)
        x = x + input

        y = self.ln_2(x)
        y = self.mlp(y)
        out = x + y
            
        return out


class Encoder(nn.Module):
    """Transformer Model Encoder for sequence to sequence translation."""

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        # # Note that batch_size is on the first dim because
        # # we have batch_first=True in nn.MultiAttention() by default
        # self.pos_embedding = nn.Parameter(torch.empty(1, seq_length, hidden_dim).normal_(std=0.02))  # from BERT

        self.dropout = nn.Dropout(dropout)
        layers: OrderedDict[str, nn.Module] = OrderedDict()
        for i in range(num_layers):
            layers[f"encoder_layer_{i}"] = EncoderBlock(
                num_heads,
                hidden_dim,
                mlp_dim,
                dropout,
                attention_dropout,
                norm_layer,
            )
        self.layers = nn.Sequential(layers)

        self.ln = norm_layer(hidden_dim)

    def forward(self, x: torch.Tensor):
        torch._assert(x.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {x.shape}")
        # x = x + self.pos_embedding
        # # return self.ln(self.layers(self.dropout(input)))

        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x)
        return self.ln(x)

class VisionTransformer(nn.Module):
    """Vision Transformer as per https://arxiv.org/abs/2010.11929."""

    def __init__(
            self,
            image_size: int,
            patch_size: int,
            num_layers: int,
            num_heads: int,
            hidden_dim: int,
            mlp_dim: int,
            dropout: float = 0.0,
            attention_dropout: float = 0.0,
            num_classes: int = 1000,
            representation_size: Optional[int] = None,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            conv_stem_configs: None,
    ):
        super().__init__()
        # _log_api_usage_once(self)
        torch._assert(image_size % patch_size == 0, "Input shape indivisible by patch size!")
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim
        self.attention_dropout = attention_dropout
        self.dropout = dropout
        self.num_classes = num_classes
        self.representation_size = representation_size
        self.norm_layer = norm_layer

        self.vis_token = nn.Parameter(torch.zeros(1,1,hidden_dim))
        self.conf_token = nn.Parameter(torch.zeros(1,1,hidden_dim))

        self.encoder = Encoder(
            num_layers,
            num_heads,
            hidden_dim,
            mlp_dim,
            dropout,
            attention_dropout,
            norm_layer,
        )

    def forward(self, x, s_emb, B, S):
        # Reshape and permute the input tensor
        BS,C,H,W = x.shape
        assert(BS==B*S)

        x = x.reshape(B*S,C,H*W).permute(0,2,1) # BS,HW,C
        K = H*W
        # The self attention layer expects inputs in the format (N, S, E)
        # where S is the source sequence length, N is the batch size, E is the
        # embedding dimension
        
        vis_token = self.vis_token.repeat(BS,1,1) + s_emb # BS,1,C
        conf_token = self.conf_token.repeat(BS,1,1) + s_emb # BS,1,C

        x = torch.cat([vis_token, conf_token, x], dim=1)
        K = K + 2
        
        x = rearrange(x, '(b s) k c -> b (s k) c', b=B, s=S, k=K)
        x = self.encoder(x)
        x = rearrange(x, 'b (s k) c -> (b s) k c', b=B, s=S, k=K)
        
        return x

class ResidualBlock3d(nn.Module):
    def __init__(self, in_planes, planes, stride=1, transpose=False):
        super(ResidualBlock3d, self).__init__()
        
        if stride == 1:
            self.conv1 = nn.Conv3d(in_planes, planes, kernel_size=3, padding=1, stride=stride, padding_mode='zeros')
            self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, padding=1, padding_mode='zeros')
        else:
            # use stride as kernel size
            if transpose:
                self.conv1 = nn.ConvTranspose3d(in_planes, planes, kernel_size=stride, padding=0, stride=stride)
            else:
                self.conv1 = nn.Conv3d(in_planes, planes, kernel_size=stride, padding=0, stride=stride)
            self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, padding=1, padding_mode='zeros')
            
        self.relu = nn.LeakyReLU(inplace=True)
        self.norm1 = nn.InstanceNorm3d(planes)
        self.norm2 = nn.InstanceNorm3d(planes)
        if not stride == 1:
            self.norm3 = nn.InstanceNorm3d(planes)
        if stride == 1:
            self.resample = None
        else:
            if transpose:
                self.resample = nn.Sequential(
                    nn.ConvTranspose3d(in_planes, planes, kernel_size=stride, stride=stride),
                    self.norm3,
                )
            else:
                self.resample = nn.Sequential(
                    nn.Conv3d(in_planes, planes, kernel_size=stride, stride=stride),
                    self.norm3,
                )
                
    def forward(self, x):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))

        if self.resample is not None:
            x = self.resample(x)

        return self.relu(x+y)
    
def safe_resample(x, S, H, W):
    _, _, S0, H0, W0 = x.shape

    if S0==S and H0==H and W0==W:
        return x
    elif S0>=S and H0>=H and W0>=W:
        return F.interpolate(x, (S, H, W), mode='area')
    elif S0<=S and H0<=H and W0<=W:
        return F.interpolate(x, (S, H, W), mode='nearest-exact')
    else:
        return False # we don't mix downsampling and upsampling

class BasicEncoder3d(nn.Module):
    def __init__(self, input_dim=4, output_dim=128, tstride=8, sstride=32):
        super(BasicEncoder3d, self).__init__()
        self.tstride = tstride
        self.sstride = sstride

        self.in_planes = 32
            
        self.conv1 = nn.Conv3d(input_dim, self.in_planes, kernel_size=(1,7,7), stride=(1,2,2), padding=(0,3,3), padding_mode='zeros') # stride 1,2
        self.norm1 = nn.InstanceNorm3d(self.in_planes)
        self.relu1 = nn.LeakyReLU(inplace=True)

        dims = [2,4,8,16] 
        dim = 32*np.sum(dims) # 1472
        self.layer1 = self._make_layer(32*dims[0], stride=(1,2,2)) # stride 1,4
        self.layer2 = self._make_layer(32*dims[1], stride=(2,2,2)) # stride 2,8
        self.layer3 = self._make_layer(32*dims[2], stride=(2,2,2)) # stride 4,16
        self.layer4 = self._make_layer(32*dims[3], stride=(2,2,2)) # stride 8,32
        self.conv2 = nn.Conv3d(dim, output_dim, kernel_size=1)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, (nn.InstanceNorm3d, nn.InstanceNorm3d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1):
        layer1 = ResidualBlock3d(self.in_planes, dim, stride=stride)
        layer2 = ResidualBlock3d(dim, dim, stride=1)
        layers = (layer1, layer2)
        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x):
        B,C,T,H,W = x.shape
        
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        
        a = self.layer1(x)
        b = self.layer2(a)
        c = self.layer3(b)
        d = self.layer4(c)
        all_x = [a,b,c,d]

        all_x_resized = [safe_resample(x, T//self.tstride, H//self.sstride, W//self.sstride) for x in all_x]
        cat = torch.cat(all_x_resized, dim=1)
        
        x = self.conv2(cat)
        return x, [a,b,c]

class LayerNorm1d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None] * x + self.bias[:, None]
        return x

class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x

class LayerNorm3d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
        return x

    
def BasicMLP(in_dim, latent_dim, out_dim):
    return nn.Sequential(
        nn.LayerNorm(in_dim),
        nn.GELU(),
        nn.Linear(in_dim, latent_dim),
        nn.LayerNorm(latent_dim),
        nn.GELU(),
        nn.Linear(latent_dim, out_dim),
    )


def posemb_sincos_3d(B, S, H, W, C, offsets=None, temperature=10000, device='cuda:0', dtype=torch.float32):
    z, y, x = utils.basic.meshgrid3d(B,S,H,W, stack=False, norm=False, device=device)
    # these are each 1,S,H,W
    if offsets is not None:
        B1,S1,D = offsets.shape
        assert(B1==B)
        assert(S1==S)
        assert(D==2)
        off_x = offsets[:,:,0].reshape(B,S,1,1).repeat(1,1,H,W)
        off_y = offsets[:,:,1].reshape(B,S,1,1).repeat(1,1,H,W)
        # note we ADD
        # because: when we cropped at xy, that location became 00, so we need to add xy back
        x = x + off_x
        y = y + off_y
        
    assert (C % 2) == 0, 'feature dimension must be multiple of 2 for sincos emb'
    omega = torch.arange(C // 2, device=device) / (C // 2 - 1)
    omega = 1. / (temperature ** omega)
    z = z.flatten()[:, None] * omega[None, :]
    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    x = torch.cat([x.sin(), x.cos()], dim=1).type(dtype)
    y = torch.cat([y.sin(), y.cos()], dim=1).type(dtype)
    z = torch.cat([z.sin(), z.cos()], dim=1).type(dtype)
    pos = torch.cat([x,y,z], dim=1) # B*S*H*W,C*3
    pos = pos.reshape(B,S*H*W,C*3)
    return pos

def posemb_sincos_1d(S, C, temperature=10000, device='cuda:0', dtype=torch.float32):
    z = torch.arange(S, device=device, dtype=dtype).reshape(1,S)
    assert (C % 2) == 0, 'feature dimension must be multiple of 2 for sincos emb'
    omega = torch.arange(C // 2, device = device) / (C // 2 - 1)
    omega = 1. / (temperature ** omega)
    z = z.flatten()[:, None] * omega[None, :]
    z = torch.cat([z.sin(), z.cos()], dim=1).type(dtype)
    return z


class ConvDecoder3d(nn.Module):
    def __init__(self, in_dim, mid_dim, out_dim):
        super().__init__()

        C0 = in_dim
        C1 = max(mid_dim,in_dim//2)
        C2 = max(mid_dim,in_dim//4)
        C3 = max(mid_dim,in_dim//8)
        C4 = max(mid_dim,in_dim//16)
        C5 = max(mid_dim,in_dim//32)

        print('convdecoder channels:', C0, C1, C2, C3, C4, C5)

        # meta repo says:
        # conv, norm, act
        self.out_sing = nn.Sequential(
            nn.Conv3d(in_channels=C0, 
                      out_channels=C1,
                      kernel_size=1, stride=1),
            LayerNorm3d(C1),
            nn.GELU(),
            nn.Conv3d(in_channels=C1, 
                      out_channels=C2,
                      kernel_size=3, stride=1, padding=1, padding_mode='zeros'),
            LayerNorm3d(C2),
            nn.GELU(),
            nn.Conv3d(in_channels=C2, 
                      out_channels=mid_dim,
                      kernel_size=1, stride=1),
            LayerNorm3d(mid_dim),
            nn.GELU(),
        )
        self.out_doub = nn.Sequential(
            nn.ConvTranspose3d(in_channels=C0, 
                               out_channels=C1,
                               kernel_size=2, stride=2),
            LayerNorm3d(C1),
            nn.GELU(),
            nn.Conv3d(in_channels=C1, 
                      out_channels=C2,
                      kernel_size=3, stride=1, padding=1, padding_mode='zeros'),
            LayerNorm3d(C2),
            nn.GELU(),
            nn.Conv3d(in_channels=C2, 
                      out_channels=mid_dim,
                      kernel_size=1, stride=1),
            LayerNorm3d(mid_dim),
            nn.GELU(),
        )
        self.out_quad = nn.Sequential(
            nn.ConvTranspose3d(in_channels=C0, 
                               out_channels=C1,
                               kernel_size=2, stride=2),
            LayerNorm3d(C1),
            nn.GELU(),
            nn.ConvTranspose3d(in_channels=C1, 
                               out_channels=C2,
                               kernel_size=2, stride=2),
            LayerNorm3d(C2),
            nn.GELU(),
            nn.Conv3d(in_channels=C2, 
                      out_channels=C3,
                      kernel_size=3, stride=1, padding=1, padding_mode='zeros'),
            LayerNorm3d(C3),
            nn.GELU(),
            nn.Conv3d(in_channels=C3, 
                      out_channels=mid_dim,
                      kernel_size=1, stride=1),
            LayerNorm3d(mid_dim),
            nn.GELU(),
        )
        self.out_octo = nn.Sequential(
            nn.ConvTranspose3d(in_channels=C0, 
                               out_channels=C1,
                               kernel_size=2, stride=2),
            LayerNorm3d(C1),
            nn.GELU(),
            nn.ConvTranspose3d(in_channels=C1, 
                               out_channels=C2,
                               kernel_size=2, stride=2),
            LayerNorm3d(C2),
            nn.GELU(),
            nn.ConvTranspose3d(in_channels=C2, 
                               out_channels=C3,
                               kernel_size=2, stride=2), 
            LayerNorm3d(C3),
            nn.GELU(),
            nn.Conv3d(in_channels=C3, 
                      out_channels=C4,
                      kernel_size=3, stride=1, padding=1, padding_mode='zeros'),
            LayerNorm3d(C4),
            nn.GELU(),
            nn.Conv3d(in_channels=C4, 
                      out_channels=mid_dim,
                      kernel_size=1, stride=1),
            LayerNorm3d(mid_dim),
            nn.GELU(),
        )

        cat_dim = 576
        self.out_head = nn.Sequential(
            nn.Conv3d(in_channels=cat_dim, 
                      out_channels=256,
                      kernel_size=3, stride=1, padding=1, padding_mode='zeros'),
            LayerNorm3d(256),
            nn.GELU(),
            nn.ConvTranspose3d(in_channels=256, 
                               out_channels=128,
                               kernel_size=(1,2,2), stride=(1,2,2)),
            LayerNorm3d(128),
            nn.GELU(),
            nn.Conv3d(in_channels=128, 
                      out_channels=64,
                      kernel_size=(1,3,3), stride=1, padding=(0,1,1)),
            LayerNorm3d(64),
            nn.GELU(),
            nn.Conv3d(64, out_dim, kernel_size=1),
        )
        

    def forward(self, S, fH, fW, x, all_skips):
        B,C,sS,sH,sW = x.shape
        x_sing = self.out_sing(x)
        x_doub = self.out_doub(x)
        x_quad = self.out_quad(x)
        x_octo = self.out_octo(x)

        # the finest-level skip is at fH//2
        all_x = [x_sing, x_doub, x_quad, x_octo]
        all_x = [safe_resample(x, S, fH//2, fW//2) for x in all_x]
        all_x = torch.cat(all_x, dim=1)
        
        all_y = [safe_resample(y, S, fH//2, fW//2) for y in all_skips]
        all_y = torch.cat(all_y, dim=1)
        
        out = self.out_head(torch.cat([all_x, all_y], dim=1))

        return out
    
    
class Tag(nn.Module):
    def __init__(self, S, H, W, tstride=4, sstride=32, scales=[0.25,1.0]):
        super(Tip, self).__init__()
        
        self.S = S
        self.H = H
        self.W = W

        self.scales = scales
        
        self.tstride = tstride
        self.sstride = sstride
        
        self.final_stride = final_stride = sstride//16
        print('final_stride', final_stride)

        self.base_dim = 384
        self.latent_dim = self.base_dim*3 # for SHW

        sS, sH, sW = S//tstride, H//sstride, W//sstride
        fH, fW = H//final_stride, W//final_stride

        self.mean = torch.as_tensor([0.485, 0.456, 0.406]).reshape(1,3,1,1).float()
        self.std = torch.as_tensor([0.229, 0.224, 0.225]).reshape(1,3,1,1).float()

        self.conv_encoder = BasicEncoder3d(output_dim=self.latent_dim, tstride=tstride, sstride=sstride)

        self.processor = Encoder(
            num_layers=16,
            num_heads=8,
            hidden_dim=self.latent_dim,
            mlp_dim=self.latent_dim*4,
        )

        seg_dim = 32
        self.conv_decoder = ConvDecoder3d(self.latent_dim, seg_dim, 5)
        
        self.vis_head = nn.Sequential(
            nn.ConvTranspose1d(in_channels=self.latent_dim, 
                               out_channels=self.latent_dim//2,
                               kernel_size=2, stride=2),
            LayerNorm1d(self.latent_dim//2),
            nn.GELU(),
            nn.ConvTranspose1d(in_channels=self.latent_dim//2, 
                               out_channels=self.latent_dim//4,
                               kernel_size=2, stride=2),
            LayerNorm1d(self.latent_dim//4),
            nn.GELU(),
            nn.ConvTranspose1d(in_channels=self.latent_dim//4, 
                               out_channels=self.latent_dim//8,
                               kernel_size=2, stride=2),
            LayerNorm1d(self.latent_dim//8),
            nn.GELU(),
            nn.Conv1d(in_channels=self.latent_dim//8, 
                      out_channels=self.latent_dim//16,
                      kernel_size=3, stride=1, padding=1),
            LayerNorm1d(self.latent_dim//16),
            nn.GELU(),
            nn.Conv1d(in_channels=self.latent_dim//16, 
                      out_channels=1,
                      kernel_size=1, stride=1),
        )
        

    def decode_out(self, S, fH, fW, x, x_feats, skips, grid_xy=None, hard=False):
        B,C,sS,sH,sW = x.shape
        vis_e = self.vis_head(x_feats).squeeze(1) # B,S
        all_heats_e_ = self.conv_decoder(S, fH, fW, x, skips) # B,mC,S,fH,fW
        all_heats_e = all_heats_e_.permute(0,2,1,3,4) # B,S,mC,fH,fW
            
        if hard:
            xy_heats_e_ = all_heats_e[:,:,2].reshape(B*S,1,fH,fW)
            xy_heats_e_ = F.interpolate(xy_heats_e_, scale_factor=self.final_stride, mode='bilinear', align_corners=False)
            ys, xs = utils.basic.argmax2d(xy_heats_e_, hard=True)
            xys_e = torch.stack([xs, ys], dim=-1).reshape(B,S,2)

            lt_heats_e_ = all_heats_e[:,:,3].reshape(B*S,1,fH,fW)
            lt_heats_e_ = F.interpolate(lt_heats_e_, scale_factor=self.final_stride, mode='bilinear', align_corners=False)
            ys, xs = utils.basic.argmax2d(lt_heats_e_, hard=True)
            lts_e = torch.stack([xs, ys], dim=-1).reshape(B,S,2)

            rb_heats_e_ = all_heats_e[:,:,4].reshape(B*S,1,fH,fW)
            rb_heats_e_ = F.interpolate(rb_heats_e_, scale_factor=self.final_stride, mode='bilinear', align_corners=False)
            ys, xs = utils.basic.argmax2d(rb_heats_e_, hard=True)
            rbs_e = torch.stack([xs, ys], dim=-1).reshape(B,S,2)

        else:
            xy_heats_e_ = all_heats_e[:,:,0].reshape(B*S,fH*fW)
            xy_soft = F.softmax(xy_heats_e_, dim=1).reshape(B*S,fH*fW,1)
            xys_e_ = torch.sum(xy_soft*grid_xy, dim=1)
            xys_e = xys_e_.reshape(B,S,2)

            lt_heats_e_ = all_heats_e[:,:,3].reshape(B*S,fH*fW)
            lt_soft = F.softmax(lt_heats_e_, dim=1).reshape(B*S,fH*fW,1)
            lts_e_ = torch.sum(lt_soft*grid_xy, dim=1)
            lts_e = lts_e_.reshape(B,S,2)
            
            rb_heats_e_ = all_heats_e[:,:,4].reshape(B*S,fH*fW)
            rb_soft = F.softmax(rb_heats_e_, dim=1).reshape(B*S,fH*fW,1)
            rbs_e_ = torch.sum(rb_soft*grid_xy, dim=1)
            rbs_e = rbs_e_.reshape(B,S,2)

        ltrbs_e = torch.cat([lts_e, rbs_e], axis=2)
            
        outs_list = [xys_e, ltrbs_e, vis_e, all_heats_e]
        return outs_list
            
    def forward(self, rgbs, prompts, offsets=None, masks_g=None, pred_seg=True, xywhs_g=None, vis_g=None, xys_valid=None, whs_valid=None, vis_valid=None, sw=None, is_training=False, SI=None, hard=True):
        # print('rgbs, device', rgbs.shape, rgbs.device)
        device = rgbs.device
        dtype = rgbs.dtype

        B,S,C,cH,cW = rgbs.shape
        assert(C==3)

        C = self.latent_dim
        sS, sH, sW = S//self.tstride, cH//self.sstride, cW//self.sstride
        fH, fW = cH//self.final_stride, cW//self.final_stride

        assert(S <= self.S)

        inputs = torch.cat([rgbs, prompts], dim=2).reshape(B,S,4,cH,cW).permute(0,2,1,3,4) # B,C,S,H,W
        embeds, embed_skips = self.conv_encoder(inputs) # B,C,sS,sH,sW
        embeds_ = embeds.reshape(B,C,-1).permute(0,2,1) # B,-1,C
        shw_emb_ = posemb_sincos_3d(B,sS,sH,sW,self.base_dim,device=device) # B,S*sH*sW,C
        embeds_ = embeds_ + shw_emb_

        outs = self.processor(embeds_).permute(0,2,1) # B,C,sS*sH*sW

        x = outs.reshape(B,C,sS,sH,sW)

        x_feats = x.mean(dim=[3,4]) # B,C,S

        assert(x.shape[1] == self.latent_dim)
        
        grid_xy = utils.basic.gridcloud2d(1, fH, fW, norm=False, device=device) # 1,fH*fW,2
        grid_xy[:,:,0] /= (fW-1)
        grid_xy[:,:,1] /= (fH-1)
        grid_xy[:,:,0] *= (cW-1)
        grid_xy[:,:,1] *= (cH-1)
        
        all_outs = self.decode_out(S, fH, fW, x, x_feats, embed_skips, grid_xy=grid_xy, hard=hard)

        return all_outs
        
