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
# from natten import NeighborhoodAttention1D, NeighborhoodAttention2D
import copy
# from typing import Optional, List, Type, Tuple
import math
from torch import nn, Tensor
import time
# from nets.positional_encoding import PositionalEncoding
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

class ConvStemConfig(NamedTuple):
    out_channels: int
    kernel_size: int
    stride: int
    norm_layer: Callable[..., nn.Module] = nn.BatchNorm2d
    activation_layer: Callable[..., nn.Module] = nn.ReLU
    
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
        conv_stem_configs: Optional[List[ConvStemConfig]] = None,
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

        # if conv_stem_configs is not None:
        #     # As per https://arxiv.org/abs/2106.14881
        #     seq_proj = nn.Sequential()
        #     prev_channels = 3
        #     for i, conv_stem_layer_config in enumerate(conv_stem_configs):
        #         seq_proj.add_module(
        #             f"conv_bn_relu_{i}",
        #             Conv2dNormActivation(
        #                 in_channels=prev_channels,
        #                 out_channels=conv_stem_layer_config.out_channels,
        #                 kernel_size=conv_stem_layer_config.kernel_size,
        #                 stride=conv_stem_layer_config.stride,
        #                 norm_layer=conv_stem_layer_config.norm_layer,
        #                 activation_layer=conv_stem_layer_config.activation_layer,
        #             ),
        #         )
        #         prev_channels = conv_stem_layer_config.out_channels
        #     seq_proj.add_module(
        #         "conv_last", nn.Conv2d(in_channels=prev_channels, out_channels=hidden_dim, kernel_size=1)
        #     )
        #     self.conv_proj: nn.Module = seq_proj
        # else:
        #     self.conv_proj = nn.Conv2d(
        #         in_channels=3, out_channels=hidden_dim, kernel_size=patch_size, stride=patch_size
        #     )

        # pe = self.pos_enc(embeds_)
        # embeds_ = embeds_ + pe # B*S,C,sH,sW


        # # Add a class token
        # self.class_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))

        # self.cls_tokens = nn.Parameter(torch.zeros(1, T, hidden_dim))

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

        # heads_layers: OrderedDict[str, nn.Module] = OrderedDict()
        # if representation_size is None:
        #     heads_layers["head"] = nn.Linear(hidden_dim, num_classes)
        # else:
        #     heads_layers["pre_logits"] = nn.Linear(hidden_dim, representation_size)
        #     heads_layers["act"] = nn.Tanh()
        #     heads_layers["head"] = nn.Linear(representation_size, num_classes)

        # self.heads = nn.Sequential(heads_layers)

        # if isinstance(self.conv_proj, nn.Conv2d):
        #     # Init the patchify stem
        #     fan_in = self.conv_proj.in_channels * self.conv_proj.kernel_size[0] * self.conv_proj.kernel_size[1]
        #     nn.init.trunc_normal_(self.conv_proj.weight, std=math.sqrt(1 / fan_in))
        #     if self.conv_proj.bias is not None:
        #         nn.init.zeros_(self.conv_proj.bias)
        # elif self.conv_proj.conv_last is not None and isinstance(self.conv_proj.conv_last, nn.Conv2d):
        #     # Init the last 1x1 conv of the conv stem
        #     nn.init.normal_(
        #         self.conv_proj.conv_last.weight, mean=0.0, std=math.sqrt(2.0 / self.conv_proj.conv_last.out_channels)
        #     )
        #     if self.conv_proj.conv_last.bias is not None:
        #         nn.init.zeros_(self.conv_proj.conv_last.bias)

        # if hasattr(self.heads, "pre_logits") and isinstance(self.heads.pre_logits, nn.Linear):
        #     fan_in = self.heads.pre_logits.in_features
        #     nn.init.trunc_normal_(self.heads.pre_logits.weight, std=math.sqrt(1 / fan_in))
        #     nn.init.zeros_(self.heads.pre_logits.bias)

        # if isinstance(self.heads.head, nn.Linear):
        #     nn.init.zeros_(self.heads.head.weight)
        #     nn.init.zeros_(self.heads.head.bias)

    # def _process_input(self, x: torch.Tensor) -> torch.Tensor:
    #     n, c, h, w = x.shape
    #     p = self.patch_size
    #     # torch._assert(h == self.image_size, f"Wrong image height! Expected {self.image_size} but got {h}!")
    #     # torch._assert(w == self.image_size, f"Wrong image width! Expected {self.image_size} but got {w}!")
    #     n_h = h // p
    #     n_w = w // p

    #     # (n, c, h, w) -> (n, hidden_dim, n_h, n_w)
    #     x = self.conv_proj(x)
    #     # (n, hidden_dim, n_h, n_w) -> (n, hidden_dim, (n_h * n_w))
    #     x = x.reshape(n, self.hidden_dim, n_h * n_w)

    #     # (n, hidden_dim, (n_h * n_w)) -> (n, (n_h * n_w), hidden_dim)
    #     # The self attention layer expects inputs in the format (N, S, E)
    #     # where S is the source sequence length, N is the batch size, E is the
    #     # embedding dimension
    #     x = x.permute(0, 2, 1)

    #     return x

    def forward(self, x, s_emb, B, S):
        # Reshape and permute the input tensor
        BS,C,H,W = x.shape
        assert(BS==B*S)

        x = x.reshape(B*S,C,H*W).permute(0,2,1) # BS,HW,C
        K = H*W
        # The self attention layer expects inputs in the format (N, S, E)
        # where S is the source sequence length, N is the batch size, E is the
        # embedding dimension
        
        # # # Expand the class token to the full batch
        # cls = self.class_token.expand(BS, -1, -1).reshape(B,S,1,self.hidden_dim)
        # cls = cls.reshape(B*S,1,self.hidden_dim)
        # x = torch.cat([cls, x], dim=1)

        # vis = self.vis_token.expand(BS, -1, -1).reshape(B,S,self.hidden_dim) + s_emb
        # conf = self.conf_token.expand(BS, -1, -1).reshape(B,S,self.hidden_dim) + s_emb

        vis_token = self.vis_token.repeat(BS,1,1) + s_emb # BS,1,C
        conf_token = self.conf_token.repeat(BS,1,1) + s_emb # BS,1,C

        x = torch.cat([vis_token, conf_token, x], dim=1)
        K = K + 2
        
        # s_emb = s_emb.repeat(B,1,1).reshape(B*S,1,C) # B*S,1,C
        # x = x + s_emb

        # # Expand the class tokens to the full batch
        # cls = self.cls_tokens.expand(n, -1, -1).reshape(B,S,self.T,self.hidden_dim)
        # cls = (cls + s_emb).reshape(B*S,self.T,self.hidden_dim)
        # x = torch.cat([cls, x], dim=1)

        # x = rearrange(x, '(b s) n c -> b s n c', b=B, s=S)
        # # x = x + s_emb
        # x = rearrange(x, 'b s n c -> (b s) n c')

        x = rearrange(x, '(b s) k c -> b (s k) c', b=B, s=S, k=K)
        # print('x', x.shape)
        x = self.encoder(x)
        x = rearrange(x, 'b (s k) c -> (b s) k c', b=B, s=S, k=K)
        
        # # Classifier "token" as used by standard language architectures
        # x = x[:, 0]
        # x = self.heads(x)
        return x


# def nms(heat, kernel=11):
#     pad = (kernel - 1) // 2
#     hmax = F.max_pool2d(heat, (kernel, kernel), stride=1, padding=pad)
#     keep = (hmax == heat).float()
#     return heat * keep

# def exists(val):
#     return val is not None

# def default(val, d):
#     return val if exists(val) else d

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
        # print('-res block-')
        # print('res x', x.shape)
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))

        # print('res y', y.shape)
        if self.resample is not None:
            x = self.resample(x)
            # print('resamp x', x.shape)

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
        # self.norm2 = nn.InstanceNorm3d(dim)
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
        # print('x0', x.shape)
        
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        # print('x1', x.shape)
        
        a = self.layer1(x)
        b = self.layer2(a)
        c = self.layer3(b)
        d = self.layer4(c)
        # print('a', a.shape)
        # print('b', b.shape)
        # print('c', c.shape)
        # print('d', d.shape)
        # print('e', e.shape)
        all_x = [a,b,c,d]

        # print('strides', self.tstride, self.sstride)
        all_x_resized = [safe_resample(x, T//self.tstride, H//self.sstride, W//self.sstride) for x in all_x]
        # print('a_', a_.shape)
        # print('b_', b_.shape)
        # print('c_', c_.shape)
        # print('d_', d_.shape)
        # print('e_', e_.shape)

        cat = torch.cat(all_x_resized, dim=1)
        # print('cat', cat.shape)
        
        x = self.conv2(cat)
        # print('conv', x.shape)
        # x = self.norm2(x)
        # x = self.relu2(x)
        # x = self.conv3(x)
        return x, [a,b,c]


# def drop_path(x, drop_prob: float=0., training: bool=False, scale_by_keep: bool=True):
#     """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

#     This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
#     the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
#     See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
#     changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
#     'survival rate' as the argument.

#     """
#     if drop_prob == 0. or not training:
#         return x
#     keep_prob = 1 - drop_prob
#     shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
#     random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
#     if keep_prob > 0.0 and scale_by_keep:
#         random_tensor.div_(keep_prob)
#     return x * random_tensor


# class DropPath(nn.Module):
#     """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
#     """
#     def __init__(self, drop_prob: float=0., scale_by_keep: bool=True):
#         super(DropPath, self).__init__()
#         self.drop_prob = drop_prob
#         self.scale_by_keep = scale_by_keep
        
#     def forward(self, x):
#         return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)
    
#     def extra_repr(self):
#         return f'drop_prob={round(self.drop_prob,3):0.3f}'


# def _no_grad_trunc_normal_(tensor, mean, std, a, b):
#     def norm_cdf(x):
#         # Computes standard normal cumulative distribution function
#         return (1. + math.erf(x / math.sqrt(2.))) / 2.

#     if (mean < a - 2 * std) or (mean > b + 2 * std):
#         warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
#                       "The distribution of values may be incorrect.",
#                       stacklevel=2)

#     with torch.no_grad():
#         # Values are generated by using a truncated uniform distribution and
#         # then using the inverse CDF for the normal distribution.
#         # Get upper and lower cdf values
#         l = norm_cdf((a - mean) / std)
#         u = norm_cdf((b - mean) / std)

#         # Uniformly fill tensor with values from [l, u], then translate to
#         # [2l-1, 2u-1].
#         tensor.uniform_(2 * l - 1, 2 * u - 1)

#         # Use inverse cdf transform for normal distribution to get truncated
#         # standard normal
#         tensor.erfinv_()

#         # Transform to proper mean, std
#         tensor.mul_(std * math.sqrt(2.))
#         tensor.add_(mean)

#         # Clamp to ensure it's in the proper range
#         tensor.clamp_(min=a, max=b)
#         return tensor
    
# def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
#     # type: (Tensor, float, float, float, float) -> Tensor
#     r"""Fills the input Tensor with values drawn from a truncated
#     normal distribution. The values are effectively drawn from the
#     normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
#     with values outside :math:`[a, b]` redrawn until they are within
#     the bounds. The method used for generating the random values works
#     best when :math:`a \leq \text{mean} \leq b`.
#     Args:
#         tensor: an n-dimensional `torch.Tensor`
#         mean: the mean of the normal distribution
#         std: the standard deviation of the normal distribution
#         a: the minimum cutoff value
#         b: the maximum cutoff value
#     Examples:
#         >>> w = torch.empty(3, 5)
#         >>> nn.init.trunc_normal_(w)
#     """
#     return _no_grad_trunc_normal_(tensor, mean, std, a, b)

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



# def balanced_ce_loss(pred, gt):
#     # pred and gt are the same shape
#     for (a,b) in zip(pred.size(), gt.size()):
#         assert(a==b) # some shape mismatch!
#     pos = (gt > 0.95).float()
#     neg = (gt < 0.05).float()

#     label = pos*2.0 - 1.0
#     a = -label * pred
#     b = F.relu(a)
#     loss = b + torch.log(torch.exp(-b)+torch.exp(a-b))
    
#     pos_loss = utils.basic.reduce_masked_mean(loss, pos)
#     neg_loss = utils.basic.reduce_masked_mean(loss, neg)

#     balanced_loss = pos_loss + neg_loss

#     return balanced_loss

    
# def pair(t):
    # return t if isinstance(t, tuple) else (t, t)

# def posemb_sincos_2d(H, W, C, temperature=10000, device='cuda:0', dtype=torch.float32):
#     y, x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing = 'ij')
#     assert (C % 4) == 0, 'feature dimension must be multiple of 4 for sincos emb'
#     omega = torch.arange(C // 4, device = device) / (C // 4 - 1)
#     omega = 1. / (temperature ** omega)

#     y = y.flatten()[:, None] * omega[None, :]
#     x = x.flatten()[:, None] * omega[None, :] 
#     pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
#     return pe.type(dtype)

def posemb_sincos_3d(B, S, H, W, C, offsets=None, temperature=10000, device='cuda:0', dtype=torch.float32):
    z, y, x = utils.basic.meshgrid3d(B,S,H,W, stack=False, norm=False, device=device)
    # these are each 1,S,H,W
    if offsets is not None:
        # print('x', x.shape)
        # print('y', y.shape)
        # print('z', z.shape)
        # print('offsets', offsets.shape)
        B1,S1,D = offsets.shape
        assert(B1==B)
        assert(S1==S)
        assert(D==2)
        off_x = offsets[:,:,0].reshape(B,S,1,1).repeat(1,1,H,W)
        off_y = offsets[:,:,1].reshape(B,S,1,1).repeat(1,1,H,W)
        # print('off_x', off_x.shape)
        # print('off_y', off_y.shape)
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


# def posemb_sincos_2d_xy(x, y, C, temperature=10000, dtype=torch.float32):
#     device = x.device
#     dtype = x.dtype
#     B, S = x.shape

#     assert (C % 2) == 0, 'feature dimension must be multiple of 2 for sincos emb'
#     omega = torch.arange(C // 2, device = device) / (C // 2 - 1)
#     omega = 1. / (temperature ** omega)

#     y = y.flatten()[:, None] * omega[None, :]
#     x = x.flatten()[:, None] * omega[None, :] 
#     pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1) 
#     pe = pe.reshape(B,S,C*2)
#     return pe.type(dtype)

# def posemb_sincos_1d_x(x, C, temperature=10000, dtype=torch.float32):
#     device = x.device
#     dtype = x.dtype
#     B, S = x.shape

#     assert(C % 2) == 0, 'feature dimension must be multiple of 2 for sincos emb'
#     omega = torch.arange(C // 2, device = device) / (C // 2 - 1)
#     omega = 1. / (temperature ** omega)

#     x = x.flatten()[:, None] * omega[None, :] 
#     pe = torch.cat((x.sin(), x.cos()), dim=1)
#     pe = pe.reshape(B,S,C)
#     return pe.type(dtype)

# class NatFeedForward(nn.Module):
#     def __init__(self, dim, hidden_dim):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(dim, hidden_dim),
#             nn.GELU(),
#             nn.LayerNorm(hidden_dim),
#             nn.Linear(hidden_dim, dim),
#         )
#     def forward(self, x):
#         return self.net(x)

# class Attention(nn.Module):
#     def __init__(self, dim, heads = 8, dim_head = 64, dropout=0):
#         super().__init__()
#         inner_dim = dim_head *  heads
#         self.heads = heads
#         self.scale = dim_head ** -0.5
#         self.norm = nn.LayerNorm(dim)
#         self.qk_norm = nn.LayerNorm(inner_dim)

#         self.attend = nn.Softmax(dim = -1)

#         self.to_q = nn.Linear(dim, inner_dim, bias=True)
#         self.to_kv = nn.Linear(dim, inner_dim*2, bias=True)

#         self.to_out = nn.Sequential(
#             nn.Linear(inner_dim, dim, bias=True),
#             nn.Dropout(dropout)
#         )
        
        
#     def forward(self, x, context=None):
#         x = self.norm(x)

#         context = default(context, x)
#         q = self.to_q(x)
#         k, v = self.to_kv(context).chunk(2, dim=-1)
#         q = self.qk_norm(q)
#         k = self.qk_norm(k)
#         q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), (q, k, v))

#         # dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
#         # utils.basic.print_stats('dots', dots)
#         # 4_30_1e-5_bq_S48_dj11_165421
#         # dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
#         # attn = self.attend(dots)
#         # out = torch.matmul(attn, v)

#         out = F.scaled_dot_product_attention(q, k, v) # scale default is already dim^-0.5
#         # out = xops.memory_efficient_attention(q, k, v)

#         out = rearrange(out, 'b h n d -> b n (h d)')
#         return self.to_out(out)


class ConvDecoder3d(nn.Module):
    def __init__(self, in_dim, mid_dim, out_dim):
        super().__init__()

        C0 = in_dim
        # C1 = max(mid_dim,in_dim//4)
        # C2 = max(mid_dim,in_dim//8)
        # C3 = max(mid_dim,in_dim//16)
        # C4 = max(mid_dim,in_dim//32)
        # C5 = max(mid_dim,in_dim//64)
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
        # self.out_sixt = nn.Sequential(
        #     nn.ConvTranspose3d(in_channels=C0, 
        #                        out_channels=C1,
        #                        kernel_size=2, stride=2),
        #     LayerNorm3d(C1),
        #     nn.GELU(),
        #     nn.ConvTranspose3d(in_channels=C1, 
        #                        out_channels=C2,
        #                        kernel_size=2, stride=2),
        #     LayerNorm3d(C2),
        #     nn.GELU(),
        #     nn.ConvTranspose3d(in_channels=C2, 
        #                        out_channels=C3,
        #                        kernel_size=2, stride=2), 
        #     LayerNorm3d(C3),
        #     nn.GELU(),
        #     nn.ConvTranspose3d(in_channels=C3, 
        #                        out_channels=C4,
        #                        kernel_size=(1,2,2), stride=(1,2,2)),
        #     LayerNorm3d(C4),
        #     nn.GELU(),
        #     nn.Conv3d(in_channels=C4, 
        #               out_channels=C5,
        #               kernel_size=3, stride=1, padding=1),
        #     LayerNorm3d(C5),
        #     nn.GELU(),
        #     nn.Conv3d(in_channels=C5, 
        #               out_channels=mid_dim,
        #               kernel_size=1, stride=1),
        #     LayerNorm3d(mid_dim),
        #     nn.GELU(),
        # )

        # dims = [2,4,8] 
        # dim = 32*np.sum(dims)
        # self.out_skip = nn.Sequential(
        #     nn.Conv3d(dim, mid_dim, kernel_size=1, padding=0),
        #     LayerNorm3d(mid_dim),
        #     nn.GELU(),
        # )

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
        # print('x', x.shape)
        B,C,sS,sH,sW = x.shape
        
        x_sing = self.out_sing(x)
        # print('x_sing', x_sing.shape)
        x_doub = self.out_doub(x)
        # print('x_doub', x_doub.shape)
        x_quad = self.out_quad(x)
        # print('x_quad', x_quad.shape)
        x_octo = self.out_octo(x)
        # print('x_octo', x_octo.shape)

        # the finest-level skip is at fH//2
        all_x = [x_sing, x_doub, x_quad, x_octo]
        all_x = [safe_resample(x, S, fH//2, fW//2) for x in all_x]
        all_x = torch.cat(all_x, dim=1)
        
        # x_sixt = self.out_sixt(x)
        # # # print('x_sixt', x_sixt.shape)
        # # all_x = [x_sing, x_doub, x_quad, x_octo, x_sixt]
        # # all_x = [x_doub, x_quad, x_octo, x_sixt]
        # all_x = [x_quad, x_octo, x_sixt]
        
        # # for ax in all_x:
        # #     print('ax0', ax.shape)
        # for ax in all_skips:
        #     print('as0', ax.shape)
        
        all_y = [safe_resample(y, S, fH//2, fW//2) for y in all_skips]
        all_y = torch.cat(all_y, dim=1)

        # for ax in all_x:
        #     print('ax1', ax.shape)
        # for ax in all_skips:
        #     print('as1', ax.shape)
        # all_x = torch.cat(all_x, dim=1)

        # print('all_x', all_x.shape)
        # print('all_y', all_y.shape)
        # print('skip up', all_y.shape)
        # all_y = self.out_skip(all_y)
        # print('skip conv', all_y.shape)
        
        # # print('all_x', all_x.shape)
        out = self.out_head(torch.cat([all_x, all_y], dim=1))
        # print('out', out.shape)

        return out


# class Transformer(nn.Module):
#     def __init__(self, dim, depth, heads, dim_head, mlp_dim, drop_path=0.1):
#         super().__init__()
#         self.layers = nn.ModuleList([])
#         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
#         for _ in range(depth):
#             self.layers.append(nn.ModuleList([
#                 Attention(dim, heads = heads, dim_head = dim_head, dropout=0.1),
#                 FeedForward(dim, mlp_dim)
#             ]))
#     def forward(self, x):
#         for attn, ff in self.layers:
#             x = x + self.drop_path(attn(x))
#             x = x + self.drop_path(ff(x))
#         return x

class SpaceTimeTransformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, drop_path=0.1):
        super().__init__()
        self.depth = depth
        self.sparse_layers = nn.ModuleList([])
        self.dense_layers = nn.ModuleList([])
        self.time_layers = nn.ModuleList([])
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # # dilations = [1,2,1,3,1,4]#,1,5,1,6,1,7,1,8]
        # dilations = [1,1,1]#,1,5,1,6,1,7,1,8]
        # while len(dilations) < self.depth:
        #     dilations += dilations
        # dilations = dilations[:self.depth]
        # dilations[-1] = 1
        # print('dilations', dilations)
        # # assert(len(dilations) > self.depth)
        # # dilations = np.ones((self.depth))


        self.sca_layers = nn.ModuleList([])
        self.ssa_layers = nn.ModuleList([])
        self.sff_layers = nn.ModuleList([])
        self.dca_layers = nn.ModuleList([])
        self.dna_layers = nn.ModuleList([])
        self.dff_layers = nn.ModuleList([])
        
        for di in range(depth):

            self.sca0 = Attention(self.latent_dim)
            self.ssa0 = Attention(self.latent_dim)
            self.sff0 = FeedForward(self.latent_dim, self.latent_dim*4)
            self.dca0 = Attention(self.latent_dim)
            self.dna0 = NeighborhoodAttention2D(dim=self.latent_dim, kernel_size=7, dilation=1, num_heads=4)
            self.dff0 = FeedForward(self.latent_dim, self.latent_dim*4)

            self.sparse_layers.append(nn.ModuleList([
                nn.LayerNorm(dim),
                Attention(dim),
                nn.LayerNorm(dim),
                Attention(dim),
                NatFeedForward(dim, mlp_dim),
            ]))

            # self.broadcast_layers.append(nn.ModuleList([
            #     nn.LayerNorm(dim),
            #     nn.Linear(dim, 1, bias=False),
            #     nn.Softmax(dim=1),
            #     nn.LayerNorm(dim),
            # ]))

            # self.sca0 = Attention(self.latent_dim)
            # self.sf0 = FeedForward(self.latent_dim, self.latent_dim*4)
            # self.ssa0 = Attention(self.latent_dim)
            # self.sff0 = FeedForward(self.latent_dim, self.latent_dim*4)
            # self.dca0 = Attention(self.latent_dim)
            # self.df0 = FeedForward(self.latent_dim, self.latent_dim*4)
            # self.dna0 = NeighborhoodAttention2D(dim=self.latent_dim, kernel_size=7, dilation=1, num_heads=4)
            # self.dff0 = FeedForward(self.latent_dim, self.latent_dim*4)
            
            self.space_layers.append(nn.ModuleList([
                nn.LayerNorm(dim),
                NeighborhoodAttention2D(dim=dim, kernel_size=5, dilation=1, num_heads=heads),
                nn.LayerNorm(dim),
                NatFeedForward(dim, mlp_dim),
            ]))
            self.time_layers.append(nn.ModuleList([
                nn.LayerNorm(dim),
                NeighborhoodAttention1D(dim=dim, kernel_size=5, dilation=1, num_heads=heads),
                nn.LayerNorm(dim),
                NatFeedForward(dim, mlp_dim),
            ]))

    def forward(self, x, use_drop=True):
        B, S, H, W, C = x.shape
        
        x = rearrange(x, 'b s h w c -> (b s) h w c')
        
        for li in range(self.depth):
            space_norm1, space_attn, space_norm2, space_ff = self.space_layers[li]
            time_norm1, time_attn, time_norm2, time_ff = self.time_layers[li]
            # broad_norm1, broad_lin, broad_soft, broad_norm2 = self.broadcast_layers[li]

            # space
            shortcut = x
            x = space_norm1(x)
            x = space_attn(x)
            if use_drop:
                x = shortcut + self.drop_path(x)
                x = x + self.drop_path(space_ff(space_norm2(x)))
            else:
                x = shortcut + x
                x = x + space_ff(space_norm2(x))
                
            # time
            x = rearrange(x, '(b s) h w c -> (b h w) s c', b=B, s=S)
            shortcut = x
            x = time_norm1(x)
            x = time_attn(x)
            if use_drop:
                x = shortcut + self.drop_path(x)
                x = x + self.drop_path(time_ff(time_norm2(x)))
            else:
                x = shortcut + x
                x = x + time_ff(time_norm2(x))
            x = rearrange(x, '(b h w) s c -> (b s) h w c', b=B, h=H, w=W)

            # # full broadcast within each timestep
            # x = rearrange(x, '(b h w) s c -> (b s) (h w) c', b=B, h=H, w=W)
            # soft = broad_soft(broad_lin(broad_norm1(x))) # bs,hw,1; sums to 1 along hw
            # agg = torch.sum(x * soft, dim=1, keepdim=True) # bs,1,c
            # x = x + self.drop_path(broad_norm2(agg))
            # x = rearrange(x, '(b s) (h w) c -> (b s) h w c', b=B, h=H, w=W)
            

        x = rearrange(x, '(b s) h w c -> b s h w c', b=B, s=S)
            
        return x
    
def random_masking(x, mask_ratio):
    """
    Perform per-sample random masking by per-sample shuffling.
    Per-sample shuffling is done by argsort random noise.
    x: [N, L, D], sequence
    """
    N, L, D = x.shape  # batch, length, dim
    len_keep = int(L * (1 - mask_ratio))

    noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

    # sort noise for each sample
    ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    # keep the first subset
    ids_keep = ids_shuffle[:, :len_keep]
    x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

    # generate the binary mask: 0 is keep, 1 is remove
    mask = torch.ones([N, L], device=x.device)
    mask[:, :len_keep] = 0
    # unshuffle to get the binary mask
    mask = torch.gather(mask, dim=1, index=ids_restore)

    return x_masked, mask, ids_restore
    
class Tip(nn.Module):
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

        # self.base_dim = 256
        self.base_dim = 384
        self.latent_dim = self.base_dim*3 # SHW
        # self.latent_dim = 1024

        sS, sH, sW = S//tstride, H//sstride, W//sstride
        fH, fW = H//final_stride, W//final_stride

        # print('H, W', H, W)
        # print('sH, sW', sH, sW)
        # print('oH, oW', oH, oW)
        
        self.mean = torch.as_tensor([0.485, 0.456, 0.406]).reshape(1,3,1,1).float()
        self.std = torch.as_tensor([0.229, 0.224, 0.225]).reshape(1,3,1,1).float()

        # self.conv_encoder = ConvEncoder(latent_dim=self.latent_dim)
        self.conv_encoder = BasicEncoder3d(output_dim=self.latent_dim, tstride=tstride, sstride=sstride)
        # self.conv_encoder = BasicEncoder3d(output_dim=self.latent_dim, tstride=2, sstride=final_stride*2)

        # print('processor expects', sS, sH, sW)
        if False:
            self.processor = MLPMixer(
                S=sS*sH*sW,
                input_dim=self.latent_dim,
                dim=self.latent_dim,
                output_dim=self.latent_dim,
                depth=12,
            )
        else:
            self.processor = Encoder(
                num_layers=16,
                num_heads=8,
                hidden_dim=self.latent_dim,
                mlp_dim=self.latent_dim*4,
            )

            

        # self.transformer = VisionTransformer(
        #     image_size=512,
        #     patch_size=stride,
        #     num_layers=24,
        #     num_heads=16,
        #     hidden_dim=self.latent_dim,
        #     mlp_dim=self.latent_dim*4,
        # )
        # self.transformer = _vision_transformer(

        
        # self.pos_enc = PositionalEncoding3d(self.latent_dim,
        #                                   scale=32,
        #                                   temperature=128)

        # self.L = 3 # number of segmentation levels
        # # # 0: point/part
        # # # 1,2: object options (for ambiguity)

        # # 1. in addition to positions near argmax, we use positions near true gt, to ensure those pixels receive gradients
        # # 2. we use asymmetric pool sizes, so that in a lazy solution (e.g., uniform), xy is dominated by our estimate instead of the gt
        # self.wsize1 = 1 # wsize*2+1 is the real window
        # # self.wsize1 = 11 # wsize*2+1 is the real window
        # self.pool2d1 = nn.MaxPool2d((self.wsize1*2+1,self.wsize1*2+1),stride=1,padding=(self.wsize1,self.wsize1))
        # self.wsize2 = 7 # wsize*2+1 is the real window
        # # self.wsize2 = 11 # wsize*2+1 is the real window
        # # self.wsize2 = 21 # wsize*2+1 is the real window
        # self.pool2d2 = nn.MaxPool2d((self.wsize2*2+1,self.wsize2*2+1),stride=1,padding=(self.wsize2,self.wsize2))


        # patch_size=32,
        # num_layers=24,
        # num_heads=16,
        # hidden_dim=1024,
        # mlp_dim=4096,
        # weights=weights,
        # progress=progress,

        # self.s_emb = nn.Parameter(torch.randn(1, S, self.latent_dim))
        
        # self.s_emb = nn.Parameter(torch.empty(1, S, self.latent_dim).normal_(std=0.02))  # from BERT
        # # self.T = T = 8
        
        # weights = ViT_L_32_Weights.verify(weights)
        # self.transformer = VisionTransformer(
        #     image_size=512,
        #     patch_size=stride,
        #     num_layers=24,
        #     num_heads=16,
        #     hidden_dim=self.latent_dim,
        #     mlp_dim=self.latent_dim*4,
        # )
        # self.transformer = _vision_transformer(
        #     patch_size=32,
        #     num_layers=24,
        #     num_heads=16,
        #     hidden_dim=1024,
        #     mlp_dim=4096,
        #     weights=weights,
        #     progress=progress,
        #     **kwargs,
        # )
        # self.transformer = SpaceTimeTransformer(
        #     dim=self.latent_dim,
        #     depth=24,
        #     heads=16,
        #     mlp_dim=self.latent_dim*4,
        #     drop_path=0.2,
        # )
        
        # # self.center_head = MLP(self.latent_dim, self.latent_dim*4, 2)
        # # self.dwh_head = MLP(self.latent_dim, self.latent_dim*4, 2)
        # self.xy_head = MLP(self.latent_dim, self.latent_dim*4, 2)
        
        # self.wh_head = MLP(self.latent_dim, self.latent_dim, 2)
        # if False:
        #     self.vis_head = BasicMLP(self.latent_dim*2, self.latent_dim, 1)
        #     self.conf_head = BasicMLP(self.latent_dim*2, self.latent_dim, 1)
        # else:
        #     self.vis_head = nn.Linear(self.latent_dim, 1)
        #     self.conf_head = nn.Linear(self.latent_dim, 1)
            
        # self.conf_head = MLP(self.latent_dim, self.latent_dim, 1)
        
        # # self.seg_head = MLP(self.latent_dim, self.latent_dim*4, self.base_dim)
        
        # self.decoder = ConvDecoder(self.latent_dim, self.base_dim)
        # self.seg_head = MLP(self.base_dim*4, self.base_dim*4, 1)

        seg_dim = 32
        self.conv_decoder = ConvDecoder3d(self.latent_dim, seg_dim, 5)
        
        # self.vis_head = nn.Linear(self.latent_dim, 1)

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
        
        
    def get_multiscale_crops(self, images_, xywhs_, cH, cW):
        B,C,H,W = images_.shape
        B2,D = xywhs_.shape
        assert(B==B2)
        assert(D==4)
        all_crops = []
        for si, sc in enumerate(self.scales):
            boxlist_ = utils.geom.get_boxlist_from_centroid_and_size(
                xywhs_[:,1],
                xywhs_[:,0],
                xywhs_[:,3].clamp(min=64)/sc,
                xywhs_[:,2].clamp(min=64)/sc,
            ).unsqueeze(1) # B,1,4
            crops_ = utils.geom.crop_and_resize(images_, boxlist_, cH, cW) # B,3,cH,cW
            crops = crops_.reshape(B,C,cH,cW)
            all_crops.append(crops)
        all_crops = torch.stack(all_crops, dim=1) # B,N,C,cH,cW
        return all_crops

    # def decode_out(self, B,S,sH,sW,fH,fW, grid_xy, vis_feats, conf_feats, x_maps, masks_g=None):
    def decode_out(self, S, fH, fW, x, x_feats, skips, grid_xy=None, hard=False):
        B,C,sS,sH,sW = x.shape
        
        # x_maps_ = rearrange(x_maps, 'b s c h w -> (b s) c h w')
        vis_e = self.vis_head(x_feats).squeeze(1) # B,S

        # # conf_e = self.conf_head(conf_feats).squeeze(2) # B,S
        # if False:
            
        #     all_maps = self.decoder(x) # [list of different fmaps, each B*S,64,fH,fW
        #     # all_maps = self.conv_decoder(x)

        #     # all_maps_ = [F.interpolate(maps_, (fH, fW), mode='bilinear', align_corners=False) for maps_ in all_maps_] # B*S,64,fH,fW
        #     all_maps = [F.interpolate(maps_, (S, fH, fW), mode='nearest-exact') for maps_ in all_maps] # B*S,64,fH,fW
        #     maps_cat = torch.cat(all_maps, dim=1) # B,C,S,fH,fW
        #     C = maps_cat.shape[1]
        #     # print('maps_cat', maps_cat.shape)

        #     maps_cat_ = maps_cat.permute(0,2,1,3,4).reshape(B*S,C,fH,fW)
        #     xys_ = self.output_head(maps_cat_) # B*S,1,fH,fW
        #     # print('outs_', outs_.shape)
        #     # return outs_

        #     xy_heats_e = xys_.reshape(B,S,1,fH,fW)
        #     # return None
        # else:
            
        all_heats_e_ = self.conv_decoder(S, fH, fW, x, skips) # B,mC,S,fH,fW
        all_heats_e = all_heats_e_.permute(0,2,1,3,4) # B,S,mC,fH,fW
            

        # xys_ = outs_[:,0:1]
        # lts_ = outs_[:,1:2]
        # rbs_ = outs_[:,2:3]
        # segs_ = outs_[:,-self.L:]

        # lt_heats_e = lts_.reshape(B,S,1,fH,fW)
        # rb_heats_e = rbs_.reshape(B,S,1,fH,fW)
        # segs_e = segs_.reshape(B,S,self.L,fH,fW)

        # if masks_g is not None:
        #     masks_g_ = masks_g.reshape(B*S,1,fH,fW)
        #     mask_g = self.pool2d1(masks_g_)
        # else:
        #     mask_g = torch.zeros_like(xy_heats_e).reshape(B*S,1,fH,fW)

        # if True:
        #     xy_heats_e_ = xy_heats_e.reshape(B*S,fH*fW)
        #     mask_ = torch.zeros_like(xy_heats_e_).scatter(1,xy_heats_e_.argmax(1,True),value=1)
        #     mask = mask_.view(B*S,1,fH,fW)
        #     mask = (self.pool2d2(mask) + mask_g).clamp(0,1)
        #     masks_ret = mask.reshape(B,S,1,fH,fW)
        #     mask_ = mask.reshape(B*S,-1)
        #     xy_soft = xy_heats_e_.clone()
        #     xy_soft[mask_==0] = -10000 # don't affect softmax pls
        #     xy_soft = F.softmax(xy_soft, dim=1).reshape(B*S,fH*fW,1)
        #     xys_e_ = torch.sum(xy_soft*grid_xy, dim=1)
        #     xys_e = xys_e_.reshape(B,S,2)
        # else: # full window (don't use, bc then with 2 peaks far away, we 1 answer in between)

        if hard:
            # print('applying sig, and multiplying by amodal heats')
            # xy_heats_e_ = torch.sigmoid(all_heats_e[:,:,0].reshape(B*S,1,fH,fW))
            # xy_heats_e_ *= torch.sigmoid(all_heats_e[:,:,2]).reshape(B*S,1,fH,fW)
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
            

        # lt_heats_e_ = lt_heats_e.reshape(B*S,fH*fW)
        # mask_ = torch.zeros_like(lt_heats_e_).scatter(1,lt_heats_e_.argmax(1,True),value=1)
        # mask = mask_.view(B*S,1,fH,fW)
        # mask = (self.pool2d2(mask) + mask_g).clamp(0,1)
        # mask_ = mask.reshape(B*S,-1)
        # lt_soft = lt_heats_e_.clone()
        # lt_soft[mask_==0] = -10000 # don't affect softmax pls
        # lt_soft = F.softmax(lt_soft, dim=1).reshape(B*S,fH*fW,1)
        # lts_e_ = torch.sum(lt_soft*grid_xy, dim=1)
        # lts_e = lts_e_.reshape(B,S,2)

        # rb_heats_e_ = rb_heats_e.reshape(B*S,fH*fW)
        # mask_ = torch.zeros_like(rb_heats_e_).scatter(1,rb_heats_e_.argmax(1,True),value=1)
        # mask = mask_.view(B*S,1,fH,fW)
        # mask = (self.pool2d2(mask) + mask_g).clamp(0,1)
        # mask_ = mask.reshape(B*S,-1)
        # rb_soft = rb_heats_e_.clone()
        # rb_soft[mask_==0] = -10000 # don't affect softmax pls
        # rb_soft = F.softmax(rb_soft, dim=1).reshape(B*S,fH*fW,1)
        # rbs_e_ = torch.sum(rb_soft*grid_xy, dim=1)
        # rbs_e = rbs_e_.reshape(B,S,2)
        # outs_list = [xys_e, lts_e, rbs_e, vis_e, segs_e, xy_heats_e, lt_heats_e, rb_heats_e, masks_ret]
        # return outs_list

        # outs_list = [xys_e, vis_e, conf_e, xy_heats_e, masks_ret]
        outs_list = [xys_e, ltrbs_e, vis_e, all_heats_e]
        return outs_list
    
            
    def forward(self, rgbs, prompts, offsets=None, masks_g=None, pred_seg=True, xywhs_g=None, vis_g=None, xys_valid=None, whs_valid=None, vis_valid=None, sw=None, is_training=False, SI=None, hard=True):
        # print('rgbs, device', rgbs.shape, rgbs.device)
        device = rgbs.device
        dtype = rgbs.dtype

        B,S,C,cH,cW = rgbs.shape
        assert(C==3)
        # assert(cH==self.H)
        # assert(cW==self.W)

        # print('rgbs', rgbs.shape)

        # B,D = xywh_e.shape
        # assert(D==4)

        C = self.latent_dim
        # L = self.L
        # cH, cW = self.H, self.W
        sS, sH, sW = S//self.tstride, cH//self.sstride, cW//self.sstride
        fH, fW = cH//self.final_stride, cW//self.final_stride

        # if S > self.S:
        #     ss = torch.arange(S, dtype=torch.float32, device=device).reshape(1,S)
        #     ss = (ss/(S-1)) * (self.S-1) # rescale to [0,self.S-1]
        # else:
        #     ss = torch.arange(self.S, dtype=torch.float32, device=device).reshape(1,self.S)

        # print('sH, sW', sH, sW)
        
        assert(S <= self.S)

        # with torch.no_grad():

        inputs = torch.cat([rgbs, prompts], dim=2).reshape(B,S,4,cH,cW).permute(0,2,1,3,4) # B,C,S,H,W
        # print('inputs', inputs.shape)
        embeds, embed_skips = self.conv_encoder(inputs) # B,C,sS,sH,sW
        # print('embeds_skip', embeds_skip.shape)
        # embeds = safe_resample(embeds_skip, S//self.tstride, cH//self.sstride, cW//self.sstride)
        # print('embeds', embeds.shape)

        embeds_ = embeds.reshape(B,C,-1).permute(0,2,1) # B,-1,C
        # print('embeds_', embeds_.shape)
        shw_emb_ = posemb_sincos_3d(B,sS,sH,sW,self.base_dim,device=device) # B,S*sH*sW,C
        # print('shw_emb_', shw_emb_.shape)
        embeds_ = embeds_ + shw_emb_

        outs = self.processor(embeds_).permute(0,2,1) # B,C,sS*sH*sW
        # print('outs', outs.shape)

        x = outs.reshape(B,C,sS,sH,sW)
        # print('x', x.shape)

        x_feats = x.mean(dim=[3,4]) # B,C,S .permute(0,2,1) # B,S,C

        # # we only care about relative offset
        # offsets = offsets - offsets[:,0:1]
        # # and we want it in stride coords
        # offsets = offsets / self.stride

        # # shw_emb = posemb_sincos_3d(S,sH,sW,self.base_dim).unsqueeze(0).to(device) # 1,S*sH*sW,C
        # shw_emb = posemb_sincos_3d(B,S,sH,sW,self.base_dim,offsets=offsets,device=device) # B,S*sH*sW,C

        # s_emb = posemb_sincos_1d(S,self.latent_dim,device=device).unsqueeze(0) # 1,S,C
        # s_emb_ = s_emb.repeat(B,1,1).reshape(B*S,1,self.latent_dim)
        
        # # print('shw_emb', shw_emb.shape)
        # shw_emb = shw_emb.reshape(B,S,sH,sW,C).permute(0,1,4,2,3) # B,S,C,sH,sW

        # embeds = embeds_.reshape(B,S,self.latent_dim,sH,sW)
        # # print('embeds', embeds.shape)
        # embeds = embeds + shw_emb
        # # # pe = self.pos_enc(prompt_embeds_)
        # # prompt_embeds_ = prompt_embeds_ + pe # B*S,C,sH,sW
        # embeds_ = embeds.reshape(B*S,self.latent_dim,sH,sW)

        # # ss = torch.arange(self.S, dtype=torch.int32, device=device).reshape(self.S)
        # # perm = np.sort(np.random.permutation(self.S)[:S])
        # # ss = ss[perm]
        # # s_emb = self.s_emb[:,ss].reshape(1,S,self.latent_dim)

        # x = self.transformer(embeds_, s_emb_, B, S) # B*S,K,C

        # x_feat = x[:,:self.T].mean(dim=1).reshape(B,S,C)
        # x_maps = x[:,self.T:].permute(0,2,1).reshape(B,S,C,sH,sW)

        # x_feat = x[:,0].reshape(B,S,C)
        # x_maps = x[:,1:].permute(0,2,1).reshape(B,S,C,sH,sW)

        # if False:
        #     x_maps = x.permute(0,2,1).reshape(B,S,C,sH,sW)
        #     x_mean = x_maps.mean(dim=[3,4])
        #     x_var = ((x_maps - x_mean.reshape(B,S,C,1,1))**2).mean(dim=[3,4])
        #     x_feat = torch.cat([x_mean, x_var], dim=2)
        # else:
        assert(x.shape[1] == self.latent_dim)
        # K = x.shape[1]
        # x = x.reshape(B,S,K,self.latent_dim) # B,S,K,C
        # vis_feats = x[:,:,0] # B,S,C
        # conf_feats = x[:,:,1] # B,S,C
        # x_maps = x[:,:,2:].permute(0,1,3,2).reshape(B,S,C,sH,sW)

        # if sw is not None and sw.save_scalar:
        #     sw.summ_feats('1_model/x_maps', x_maps[0:1].unbind(1))
        
        grid_xy = utils.basic.gridcloud2d(1, fH, fW, norm=False, device=device) # 1,fH*fW,2
        grid_xy[:,:,0] /= (fW-1)
        grid_xy[:,:,1] /= (fH-1)
        grid_xy[:,:,0] *= (cW-1)
        grid_xy[:,:,1] *= (cH-1)
        
        # print_stats('grid_xy[:,:,0]', grid_xy[:,:,0])
        # print_stats('grid_xy[:,:,1]', grid_xy[:,:,1])
        
        # x_maps = rearrange(x, 'b s h w c -> b s c h w')
        
        # all_outs = self.decode_out(B,S,sH,sW,fH,fW, grid_xy, x_feat, x_maps, masks_g=masks_g)
        # all_outs = self.decode_out(B,sS,sH,sW,fH,fW, grid_xy, vis_feats, conf_feats, x_maps, masks_g=masks_g)
        all_outs = self.decode_out(S, fH, fW, x, x_feats, embed_skips, grid_xy=grid_xy, hard=hard)#B,sS,sH,sW,fH,fW, grid_xy, vis_feats, conf_feats, x_maps, masks_g=masks_g)

        return all_outs
        
        sparse_tokens_ = self.sca0(sparse_tokens_, context=dense_tokens_) + sparse_tokens_
        sparse_tokens_ = self.sf0(sparse_tokens_) + sparse_tokens_
        sparse_tokens_ = rearrange(sparse_tokens_, '(b s) n c -> b (s n) c', b=B, s=S, n=N)
        sparse_tokens_ = self.ssa0(sparse_tokens_) + sparse_tokens_
        sparse_tokens_ = self.sff0(sparse_tokens_) + sparse_tokens_
        sparse_tokens_ = rearrange(sparse_tokens_, 'b (s n) c -> (b s) n c', b=B, s=S, n=N)
        dense_tokens_ = self.dca0(dense_tokens_, context=sparse_tokens_) + dense_tokens_
        dense_tokens_ = self.df0(dense_tokens_) + dense_tokens_
        dense_tokens_ = rearrange(dense_tokens_, 'n (h w) c -> n h w c', h=sH, w=sW)
        dense_tokens_ = self.dna0(dense_tokens_) + dense_tokens_
        dense_tokens_ = self.dff0(dense_tokens_) + dense_tokens_
        dense_tokens_ = rearrange(dense_tokens_, 'n h w c -> n (h w) c', h=sH, w=sW)
        # all_outs.append(self.decode_out(B,S,sH,sW,fH,fW, grid_xy, sparse_tokens_, dense_tokens_))

        sparse_tokens_ = self.sca1(sparse_tokens_, context=dense_tokens_) + sparse_tokens_
        sparse_tokens_ = self.sf1(sparse_tokens_) + sparse_tokens_
        sparse_tokens_ = rearrange(sparse_tokens_, '(b s) n c -> b (s n) c', b=B, s=S, n=N)
        sparse_tokens_ = self.ssa1(sparse_tokens_) + sparse_tokens_
        sparse_tokens_ = self.sff1(sparse_tokens_) + sparse_tokens_
        sparse_tokens_ = rearrange(sparse_tokens_, 'b (s n) c -> (b s) n c', b=B, s=S, n=N)
        dense_tokens_ = self.dca1(dense_tokens_, context=sparse_tokens_) + dense_tokens_
        dense_tokens_ = self.df1(dense_tokens_) + dense_tokens_
        dense_tokens_ = rearrange(dense_tokens_, 'n (h w) c -> n h w c', h=sH, w=sW)
        dense_tokens_ = self.dna1(dense_tokens_) + dense_tokens_
        dense_tokens_ = self.dff1(dense_tokens_) + dense_tokens_
        dense_tokens_ = rearrange(dense_tokens_, 'n h w c -> n (h w) c', h=sH, w=sW)
        # all_outs.append(self.decode_out(B,S,sH,sW,fH,fW, grid_xy, sparse_tokens_, dense_tokens_))

        sparse_tokens_ = self.sca2(sparse_tokens_, context=dense_tokens_) + sparse_tokens_
        sparse_tokens_ = self.sf2(sparse_tokens_) + sparse_tokens_
        sparse_tokens_ = rearrange(sparse_tokens_, '(b s) n c -> b (s n) c', b=B, s=S, n=N)
        sparse_tokens_ = self.ssa2(sparse_tokens_) + sparse_tokens_
        sparse_tokens_ = self.sff2(sparse_tokens_) + sparse_tokens_
        sparse_tokens_ = rearrange(sparse_tokens_, 'b (s n) c -> (b s) n c', b=B, s=S, n=N)
        dense_tokens_ = self.dca2(dense_tokens_, context=sparse_tokens_) + dense_tokens_
        dense_tokens_ = self.df2(dense_tokens_) + dense_tokens_
        dense_tokens_ = rearrange(dense_tokens_, 'n (h w) c -> n h w c', h=sH, w=sW)
        dense_tokens_ = self.dna2(dense_tokens_) + dense_tokens_
        dense_tokens_ = self.dff2(dense_tokens_) + dense_tokens_
        dense_tokens_ = rearrange(dense_tokens_, 'n h w c -> n (h w) c', h=sH, w=sW)
        all_outs.append(self.decode_out(B,S,sH,sW,fH,fW, grid_xy, sparse_tokens_, dense_tokens_))

        sparse_tokens_ = self.sca3(sparse_tokens_, context=dense_tokens_) + sparse_tokens_
        sparse_tokens_ = self.sf3(sparse_tokens_) + sparse_tokens_
        sparse_tokens_ = rearrange(sparse_tokens_, '(b s) n c -> b (s n) c', b=B, s=S, n=N)
        sparse_tokens_ = self.ssa3(sparse_tokens_) + sparse_tokens_
        sparse_tokens_ = self.sff3(sparse_tokens_) + sparse_tokens_
        sparse_tokens_ = rearrange(sparse_tokens_, 'b (s n) c -> (b s) n c', b=B, s=S, n=N)
        dense_tokens_ = self.dca3(dense_tokens_, context=sparse_tokens_) + dense_tokens_
        dense_tokens_ = self.df3(dense_tokens_) + dense_tokens_
        dense_tokens_ = rearrange(dense_tokens_, 'n (h w) c -> n h w c', h=sH, w=sW)
        dense_tokens_ = self.dna3(dense_tokens_) + dense_tokens_
        dense_tokens_ = self.dff3(dense_tokens_) + dense_tokens_
        dense_tokens_ = rearrange(dense_tokens_, 'n h w c -> n (h w) c', h=sH, w=sW)
        # all_outs.append(self.decode_out(B,S,sH,sW,fH,fW, grid_xy, sparse_tokens_, dense_tokens_))

        sparse_tokens_ = self.sca4(sparse_tokens_, context=dense_tokens_) + sparse_tokens_
        sparse_tokens_ = self.sf4(sparse_tokens_) + sparse_tokens_
        sparse_tokens_ = rearrange(sparse_tokens_, '(b s) n c -> b (s n) c', b=B, s=S, n=N)
        sparse_tokens_ = self.ssa4(sparse_tokens_) + sparse_tokens_
        sparse_tokens_ = self.sff4(sparse_tokens_) + sparse_tokens_
        sparse_tokens_ = rearrange(sparse_tokens_, 'b (s n) c -> (b s) n c', b=B, s=S, n=N)
        dense_tokens_ = self.dca4(dense_tokens_, context=sparse_tokens_) + dense_tokens_
        dense_tokens_ = self.df4(dense_tokens_) + dense_tokens_
        dense_tokens_ = rearrange(dense_tokens_, 'n (h w) c -> n h w c', h=sH, w=sW)
        dense_tokens_ = self.dna4(dense_tokens_) + dense_tokens_
        dense_tokens_ = self.dff4(dense_tokens_) + dense_tokens_
        dense_tokens_ = rearrange(dense_tokens_, 'n h w c -> n (h w) c', h=sH, w=sW)
        # all_outs.append(self.decode_out(B,S,sH,sW,fH,fW, grid_xy, sparse_tokens_, dense_tokens_))

        sparse_tokens_ = self.sca5(sparse_tokens_, context=dense_tokens_) + sparse_tokens_
        sparse_tokens_ = self.sf5(sparse_tokens_) + sparse_tokens_
        sparse_tokens_ = rearrange(sparse_tokens_, '(b s) n c -> b (s n) c', b=B, s=S, n=N)
        sparse_tokens_ = self.ssa5(sparse_tokens_) + sparse_tokens_
        sparse_tokens_ = self.sff5(sparse_tokens_) + sparse_tokens_
        sparse_tokens_ = rearrange(sparse_tokens_, 'b (s n) c -> (b s) n c', b=B, s=S, n=N)
        dense_tokens_ = self.dca5(dense_tokens_, context=sparse_tokens_) + dense_tokens_
        dense_tokens_ = self.df5(dense_tokens_) + dense_tokens_
        dense_tokens_ = rearrange(dense_tokens_, 'n (h w) c -> n h w c', h=sH, w=sW)
        dense_tokens_ = self.dna5(dense_tokens_) + dense_tokens_
        dense_tokens_ = self.dff5(dense_tokens_) + dense_tokens_
        dense_tokens_ = rearrange(dense_tokens_, 'n h w c -> n (h w) c', h=sH, w=sW)
        all_outs.append(self.decode_out(B,S,sH,sW,fH,fW, grid_xy, sparse_tokens_, dense_tokens_))

        sparse_tokens_ = self.sca6(sparse_tokens_, context=dense_tokens_) + sparse_tokens_
        sparse_tokens_ = self.sf6(sparse_tokens_) + sparse_tokens_
        sparse_tokens_ = rearrange(sparse_tokens_, '(b s) n c -> b (s n) c', b=B, s=S, n=N)
        sparse_tokens_ = self.ssa6(sparse_tokens_) + sparse_tokens_
        sparse_tokens_ = self.sff6(sparse_tokens_) + sparse_tokens_
        sparse_tokens_ = rearrange(sparse_tokens_, 'b (s n) c -> (b s) n c', b=B, s=S, n=N)
        dense_tokens_ = self.dca6(dense_tokens_, context=sparse_tokens_) + dense_tokens_
        dense_tokens_ = self.df6(dense_tokens_) + dense_tokens_
        dense_tokens_ = rearrange(dense_tokens_, 'n (h w) c -> n h w c', h=sH, w=sW)
        dense_tokens_ = self.dna6(dense_tokens_) + dense_tokens_
        dense_tokens_ = self.dff6(dense_tokens_) + dense_tokens_
        dense_tokens_ = rearrange(dense_tokens_, 'n h w c -> n (h w) c', h=sH, w=sW)
        # all_outs.append(self.decode_out(B,S,sH,sW,fH,fW, grid_xy, sparse_tokens_, dense_tokens_))

        sparse_tokens_ = self.sca7(sparse_tokens_, context=dense_tokens_) + sparse_tokens_
        sparse_tokens_ = self.sf7(sparse_tokens_) + sparse_tokens_
        sparse_tokens_ = rearrange(sparse_tokens_, '(b s) n c -> b (s n) c', b=B, s=S, n=N)
        sparse_tokens_ = self.ssa7(sparse_tokens_) + sparse_tokens_
        sparse_tokens_ = self.sff7(sparse_tokens_) + sparse_tokens_
        sparse_tokens_ = rearrange(sparse_tokens_, 'b (s n) c -> (b s) n c', b=B, s=S, n=N)
        dense_tokens_ = self.dca7(dense_tokens_, context=sparse_tokens_) + dense_tokens_
        dense_tokens_ = self.df7(dense_tokens_) + dense_tokens_
        dense_tokens_ = rearrange(dense_tokens_, 'n (h w) c -> n h w c', h=sH, w=sW)
        dense_tokens_ = self.dna7(dense_tokens_) + dense_tokens_
        dense_tokens_ = self.dff7(dense_tokens_) + dense_tokens_
        dense_tokens_ = rearrange(dense_tokens_, 'n h w c -> n (h w) c', h=sH, w=sW)
        # all_outs.append(self.decode_out(B,S,sH,sW,fH,fW, grid_xy, sparse_tokens_, dense_tokens_))

        sparse_tokens_ = self.sca8(sparse_tokens_, context=dense_tokens_) + sparse_tokens_
        sparse_tokens_ = self.sf8(sparse_tokens_) + sparse_tokens_
        sparse_tokens_ = rearrange(sparse_tokens_, '(b s) n c -> b (s n) c', b=B, s=S, n=N)
        sparse_tokens_ = self.ssa8(sparse_tokens_) + sparse_tokens_
        sparse_tokens_ = self.sff8(sparse_tokens_) + sparse_tokens_
        sparse_tokens_ = rearrange(sparse_tokens_, 'b (s n) c -> (b s) n c', b=B, s=S, n=N)
        dense_tokens_ = self.dca8(dense_tokens_, context=sparse_tokens_) + dense_tokens_
        dense_tokens_ = self.df8(dense_tokens_) + dense_tokens_
        dense_tokens_ = rearrange(dense_tokens_, 'n (h w) c -> n h w c', h=sH, w=sW)
        dense_tokens_ = self.dna8(dense_tokens_) + dense_tokens_
        dense_tokens_ = self.dff8(dense_tokens_) + dense_tokens_
        dense_tokens_ = rearrange(dense_tokens_, 'n h w c -> n (h w) c', h=sH, w=sW)
        all_outs.append(self.decode_out(B,S,sH,sW,fH,fW, grid_xy, sparse_tokens_, dense_tokens_))

        sparse_tokens_ = self.sca9(sparse_tokens_, context=dense_tokens_) + sparse_tokens_
        sparse_tokens_ = self.sf9(sparse_tokens_) + sparse_tokens_
        sparse_tokens_ = rearrange(sparse_tokens_, '(b s) n c -> b (s n) c', b=B, s=S, n=N)
        sparse_tokens_ = self.ssa9(sparse_tokens_) + sparse_tokens_
        sparse_tokens_ = self.sff9(sparse_tokens_) + sparse_tokens_
        sparse_tokens_ = rearrange(sparse_tokens_, 'b (s n) c -> (b s) n c', b=B, s=S, n=N)
        dense_tokens_ = self.dca9(dense_tokens_, context=sparse_tokens_) + dense_tokens_
        dense_tokens_ = self.df9(dense_tokens_) + dense_tokens_
        dense_tokens_ = rearrange(dense_tokens_, 'n (h w) c -> n h w c', h=sH, w=sW)
        dense_tokens_ = self.dna9(dense_tokens_) + dense_tokens_
        dense_tokens_ = self.dff9(dense_tokens_) + dense_tokens_
        dense_tokens_ = rearrange(dense_tokens_, 'n h w c -> n (h w) c', h=sH, w=sW)
        # all_outs.append(self.decode_out(B,S,sH,sW,fH,fW, grid_xy, sparse_tokens_, dense_tokens_))

        sparse_tokens_ = self.sca10(sparse_tokens_, context=dense_tokens_) + sparse_tokens_
        sparse_tokens_ = self.sf10(sparse_tokens_) + sparse_tokens_
        sparse_tokens_ = rearrange(sparse_tokens_, '(b s) n c -> b (s n) c', b=B, s=S, n=N)
        sparse_tokens_ = self.ssa10(sparse_tokens_) + sparse_tokens_
        sparse_tokens_ = self.sff10(sparse_tokens_) + sparse_tokens_
        sparse_tokens_ = rearrange(sparse_tokens_, 'b (s n) c -> (b s) n c', b=B, s=S, n=N)
        dense_tokens_ = self.dca10(dense_tokens_, context=sparse_tokens_) + dense_tokens_
        dense_tokens_ = self.df10(dense_tokens_) + dense_tokens_
        dense_tokens_ = rearrange(dense_tokens_, 'n (h w) c -> n h w c', h=sH, w=sW)
        dense_tokens_ = self.dna10(dense_tokens_) + dense_tokens_
        dense_tokens_ = self.dff10(dense_tokens_) + dense_tokens_
        dense_tokens_ = rearrange(dense_tokens_, 'n h w c -> n (h w) c', h=sH, w=sW)
        # all_outs.append(self.decode_out(B,S,sH,sW,fH,fW, grid_xy, sparse_tokens_, dense_tokens_))

        sparse_tokens_ = self.sca11(sparse_tokens_, context=dense_tokens_) + sparse_tokens_
        sparse_tokens_ = self.sf11(sparse_tokens_) + sparse_tokens_
        sparse_tokens_ = rearrange(sparse_tokens_, '(b s) n c -> b (s n) c', b=B, s=S, n=N)
        sparse_tokens_ = self.ssa11(sparse_tokens_) + sparse_tokens_
        sparse_tokens_ = self.sff11(sparse_tokens_) + sparse_tokens_
        sparse_tokens_ = rearrange(sparse_tokens_, 'b (s n) c -> (b s) n c', b=B, s=S, n=N)
        dense_tokens_ = self.dca11(dense_tokens_, context=sparse_tokens_) + dense_tokens_
        dense_tokens_ = self.df11(dense_tokens_) + dense_tokens_
        dense_tokens_ = rearrange(dense_tokens_, 'n (h w) c -> n h w c', h=sH, w=sW)
        dense_tokens_ = self.dna11(dense_tokens_) + dense_tokens_
        dense_tokens_ = self.dff11(dense_tokens_) + dense_tokens_
        dense_tokens_ = rearrange(dense_tokens_, 'n h w c -> n (h w) c', h=sH, w=sW)
        all_outs.append(self.decode_out(B,S,sH,sW,fH,fW, grid_xy, sparse_tokens_, dense_tokens_))


        return all_outs
        
        # # x = rearrange(x, ' c h w -> b (s h w) c', b=B, s=S) # B,S*sH*sW,C
        # # print('x', x.shape)
        
        # # x = self.conv_proj(rgbs_).reshape(B*S,self.latent_dim,-1).permute(0,2,1) # B*S,sH*sW,C
        # # print('x conv', x.shape)
        # # x = x.reshape(B,S*sH*sW,self.latent_dim)

        
        # x[:,:,:self.base_dim*3] = x[:,:,:self.base_dim*3] + xyt_emb
        # # print('x add', x.shape)
        # x = x.reshape(B,S,sH*sW,self.latent_dim)
        # # print('x rgb', x.shape)
        # # x = torch.cat([out_tokens.unsqueeze(2), sink_tokens.unsqueeze(2), xywh_embs.unsqueeze(2), x], dim=2) # B,S,3+sH*sW,C
        # x = torch.cat([out_tokens.unsqueeze(2), sink_tokens.unsqueeze(2), x], dim=2) # B,S,3+sH*sW,C
        # # print('x cat', x.shape)
        
        # M = x.shape[2]


        
        # xys_ = utils.basic.gridcloud2d(B*S, sH, sW, norm=False, device=device) # B*S,sH*sW,2
        # xy_embs_ = posemb_sincos_2d_xy(xys_[:,:,0], xys_[:,:,1], self.base_dim) # B*S,sH*sW,bC*2
        # s_emb_ = s_emb.reshape(B*S,1,self.base_dim).repeat(1,sH*sW,1)
        # xy_embs_ = torch.cat([xy_embs_, s_emb_], dim=2) # B*S,sH*sW,bC*3

        
        # if False:
        #     # if is_training:
        #     # drop some amount from each frame
        #     x = x.reshape(B*S,-1,self.latent_dim)
        #     x, _, _ = random_masking(x, 0.8)
        #     M = x.shape[1]
        #     x = x.reshape(B,S*M,self.latent_dim)
        #     # else:
        #     #     # drop some amount from each frame
        #     #     x = x.reshape(B*S,-1,self.latent_dim)
        #     #     x, _, _ = random_masking(x, 0.7)
        #     #     M = x.shape[1]
        #     #     x = x.reshape(B,S*M,self.latent_dim)
        #     x = self.transformer(x.reshape(B,-1,self.latent_dim)).reshape(B,S,-1,self.latent_dim)
        # else:
            # spacetime version:
            # note we don't drop tokens bc right now they are aligned wrt spacetime
        # x = self.transformer(x)
                                
        # x_outs = x[:,:,0] # B,S,C
        # # seg_outs = x[:,:,1:1+nS] # B,S,nS,C

        # xys_e = self.xy_head(x_outs) # B,S,2
        # whs_e = self.wh_head(x_outs) # B,S,2
        # vis_e = self.vis_head(x_outs).squeeze(2) # B,S
        # conf_e = self.conf_head(x_outs.detach()).squeeze(2) # B,S
        
            #     # if pred_seg:
            # # add_pos = False
            # # if add_pos:
            # #     x_maps_ = x[:,:,-sH*sW:].reshape(B,S*sH*sW,self.latent_dim)
            # #     x_maps_[:,:,:self.base_dim*3] = x_maps_[:,:,:self.base_dim*3] + xyt_emb
            # #     x_maps_ = x_maps_.reshape(B*S,sH,sW,self.latent_dim) # B*S,sH,sW,C
            # #     # x_maps_[:,:,:,:self.base_dim*3] = x_maps_[:,:,:,:self.base_dim*3] + xyt_emb
            # # else:
            # x_maps_ = x[:,:,-sH*sW:].reshape(B*S,sH,sW,self.latent_dim) # B*S,sH,sW,C

            # # x_maps_[:,:,:,:self.base_dim*3] = x_maps_[:,:,:,:self.base_dim*3] + xyt_emb.reshape(B*S,sH,sW,self.base_dim*3)

            # # x_maps_ = x[:,:,-sH*sW:].reshape(B*S,sH,sW,self.latent_dim) # B*S,sH,sW,C
            # # x_maps_[:,:,:,:self.base_dim*3] = x_maps_[:,:,:,:self.base_dim*3] + xyt_emb.reshape(B*S,sH,sW,self.base_dim*3)


            # # seg_outs = self.seg_head(seg_outs) # B,S,3,bC
            # # seg_outs_ = seg_outs.reshape(B*S,nS,self.base_dim,1,1) # B*S,nS,bC,1,1

            # x_maps_ = x_maps_.permute(0,3,1,2) # B*S,C,sH,sW
            # all_maps_ = self.decoder(x_maps_) # [list of different fmaps, each B*S,64,H,W
            # # all_maps_ = [F.interpolate(maps_, (mH, mW), mode='bilinear', align_corners=False) for maps_ in all_maps_] # B*S,64,mH,mW
            # all_maps_ = [F.interpolate(maps_, (mH, mW), mode='nearest') for maps_ in all_maps_] # B*S,64,mH,mW
            # maps_cat_ = torch.cat(all_maps_, dim=1) # B*S,64*4,mH,mW
            # segs_ = self.seg_conv_head(maps_cat_) # B*S,L,fH,fW
            # segs_e = segs_.reshape(B,S,self.L,fH,fW)
            #     # else:
        #     segs_e = None
            
        return xys_e, whs_e, vis_e, conf_e, segs_e
    

