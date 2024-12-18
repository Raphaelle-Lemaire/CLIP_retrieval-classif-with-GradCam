"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

 Based on https://github.com/mlfoundations/open_clip
"""

""" CLIP Model
Adapted from https://github.com/openai/CLIP. Originally MIT License, Copyright (c) 2021 OpenAI.
"""

import datetime
import json
import logging
import os
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
import cv2

import re
import time
import warnings
from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
import torchvision.transforms as T
from PIL import Image

import csv
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from torch.nn.parameter import Parameter
from torch.nn.init import xavier_uniform_
from torch.nn.init import constant_
from torch.nn.init import xavier_normal_
from lavis.common.registry import registry
from lavis.common.utils import get_abs_path
from lavis.models.base_model import BaseModel
from lavis.models.clip_models.clip_outputs import ClipOutput, ClipOutputFeatures
from lavis.models.clip_models.timm_model import TimmModel
from lavis.models.clip_models.transform import image_transform
from lavis.models.clip_models.utils import freeze_batch_norm_2d
from lavis.tasks.multimodal_classification import MultimodalClassificationTask
from lavis.tasks.classRet import ClassRetTask
from torch import nn

from .pretrained import (
    download_pretrained,
    get_pretrained_url,
    list_pretrained_tag_models,
)

_MODEL_CONFIG_PATHS = [Path(__file__).parent.parent.parent / f"configs/models/clip/"]
_MODEL_CONFIGS = {}  # directory (model_name: config) of model architecture configs

class _LinearWithBias(torch.nn.Linear):
    bias: Tensor

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__(in_features, out_features, bias=True)

def multi_head_attention_forward(query: Tensor,
                                 key: Tensor,
                                 value: Tensor,
                                 embed_dim_to_check: int,
                                 num_heads: int,
                                 in_proj_weight: Tensor,
                                 in_proj_bias: Tensor,
                                 bias_k: Optional[Tensor],
                                 bias_v: Optional[Tensor],
                                 add_zero_attn: bool,
                                 dropout_p: float,
                                 out_proj_weight: Tensor,
                                 out_proj_bias: Tensor,
                                 training: bool = True,
                                 key_padding_mask: Optional[Tensor] = None,
                                 need_weights: bool = True,
                                 attn_mask: Optional[Tensor] = None,
                                 use_separate_proj_weight: bool = False,
                                 q_proj_weight: Optional[Tensor] = None,
                                 k_proj_weight: Optional[Tensor] = None,
                                 v_proj_weight: Optional[Tensor] = None,
                                 static_k: Optional[Tensor] = None,
                                 static_v: Optional[Tensor] = None,
                                 attention_probs_forward_hook = None,
                                 attention_probs_backwards_hook = None,
                                 ) -> Tuple[Tensor, Optional[Tensor]]:
    if not torch.jit.is_scripting():
        tens_ops = (query, key, value, in_proj_weight, in_proj_bias, bias_k, bias_v,
                    out_proj_weight, out_proj_bias)
        if any([type(t) is not Tensor for t in tens_ops]) and F.has_torch_function(tens_ops):
            return F.handle_torch_function(
                multi_head_attention_forward, tens_ops, query, key, value,
                embed_dim_to_check, num_heads, in_proj_weight, in_proj_bias,
                bias_k, bias_v, add_zero_attn, dropout_p, out_proj_weight,
                out_proj_bias, training=training, key_padding_mask=key_padding_mask,
                need_weights=need_weights, attn_mask=attn_mask,
                use_separate_proj_weight=use_separate_proj_weight,
                q_proj_weight=q_proj_weight, k_proj_weight=k_proj_weight,
                v_proj_weight=v_proj_weight, static_k=static_k, static_v=static_v)
    tgt_len, bsz, embed_dim = query.size()
    assert embed_dim == embed_dim_to_check
    # allow MHA to have different sizes for the feature dimension
    assert key.size(0) == value.size(0) and key.size(1) == value.size(1)

    head_dim = embed_dim // num_heads
    assert head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
    scaling = float(head_dim) ** -0.5

    if not use_separate_proj_weight:
        if torch.equal(query, key) and torch.equal(key, value):
            # self-attention
            q, k, v = F.linear(query, in_proj_weight, in_proj_bias).chunk(3, dim=-1)

        elif torch.equal(key, value):
            # encoder-decoder attention
            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = 0
            _end = embed_dim
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            q = F.linear(query, _w, _b)

            if key is None:
                assert value is None
                k = None
                v = None
            else:

                # This is inline in_proj function with in_proj_weight and in_proj_bias
                _b = in_proj_bias
                _start = embed_dim
                _end = None
                _w = in_proj_weight[_start:, :]
                if _b is not None:
                    _b = _b[_start:]
                k, v = F.linear(key, _w, _b).chunk(2, dim=-1)

        else:
            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = 0
            _end = embed_dim
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            q = F.linear(query, _w, _b)

            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = embed_dim
            _end = embed_dim * 2
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            k = F.linear(key, _w, _b)

            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = embed_dim * 2
            _end = None
            _w = in_proj_weight[_start:, :]
            if _b is not None:
                _b = _b[_start:]
            v = F.linear(value, _w, _b)
    else:
        q_proj_weight_non_opt = torch.jit._unwrap_optional(q_proj_weight)
        len1, len2 = q_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == query.size(-1)

        k_proj_weight_non_opt = torch.jit._unwrap_optional(k_proj_weight)
        len1, len2 = k_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == key.size(-1)

        v_proj_weight_non_opt = torch.jit._unwrap_optional(v_proj_weight)
        len1, len2 = v_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == value.size(-1)

        if in_proj_bias is not None:
            q = F.linear(query, q_proj_weight_non_opt, in_proj_bias[0:embed_dim])
            k = F.linear(key, k_proj_weight_non_opt, in_proj_bias[embed_dim:(embed_dim * 2)])
            v = F.linear(value, v_proj_weight_non_opt, in_proj_bias[(embed_dim * 2):])
        else:
            q = F.linear(query, q_proj_weight_non_opt, in_proj_bias)
            k = F.linear(key, k_proj_weight_non_opt, in_proj_bias)
            v = F.linear(value, v_proj_weight_non_opt, in_proj_bias)
    q = q * scaling

    if attn_mask is not None:
        assert attn_mask.dtype == torch.float32 or attn_mask.dtype == torch.float64 or \
            attn_mask.dtype == torch.float16 or attn_mask.dtype == torch.uint8 or attn_mask.dtype == torch.bool, \
            'Only float, byte, and bool types are supported for attn_mask, not {}'.format(attn_mask.dtype)
        if attn_mask.dtype == torch.uint8:
            warnings.warn("Byte tensor for attn_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.")
            attn_mask = attn_mask.to(torch.bool)

        if attn_mask.dim() == 2:
            attn_mask = attn_mask.unsqueeze(0)
            if list(attn_mask.size()) != [1, query.size(0), key.size(0)]:
                raise RuntimeError('The size of the 2D attn_mask is not correct.')
        elif attn_mask.dim() == 3:
            if list(attn_mask.size()) != [bsz * num_heads, query.size(0), key.size(0)]:
                raise RuntimeError('The size of the 3D attn_mask is not correct.')
        else:
            raise RuntimeError("attn_mask's dimension {} is not supported".format(attn_mask.dim()))
        # attn_mask's dim is 3 now.

    # convert ByteTensor key_padding_mask to bool
    if key_padding_mask is not None and key_padding_mask.dtype == torch.uint8:
        warnings.warn("Byte tensor for key_padding_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.")
        key_padding_mask = key_padding_mask.to(torch.bool)

    if bias_k is not None and bias_v is not None:
        if static_k is None and static_v is None:
            k = torch.cat([k, bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = pad(attn_mask, (0, 1))
            if key_padding_mask is not None:
                key_padding_mask = pad(key_padding_mask, (0, 1))
        else:
            assert static_k is None, "bias cannot be added to static key."
            assert static_v is None, "bias cannot be added to static value."
    else:
        assert bias_k is None
        assert bias_v is None

    q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    if k is not None:
        k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    if v is not None:
        v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)

    if static_k is not None:
        assert static_k.size(0) == bsz * num_heads
        assert static_k.size(2) == head_dim
        k = static_k

    if static_v is not None:
        assert static_v.size(0) == bsz * num_heads
        assert static_v.size(2) == head_dim
        v = static_v

    src_len = k.size(1)

    if key_padding_mask is not None:
        assert key_padding_mask.size(0) == bsz
        assert key_padding_mask.size(1) == src_len

    if add_zero_attn:
        src_len += 1
        k = torch.cat([k, torch.zeros((k.size(0), 1) + k.size()[2:], dtype=k.dtype, device=k.device)], dim=1)
        v = torch.cat([v, torch.zeros((v.size(0), 1) + v.size()[2:], dtype=v.dtype, device=v.device)], dim=1)
        if attn_mask is not None:
            attn_mask = pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = pad(key_padding_mask, (0, 1))

    attn_output_weights = torch.bmm(q, k.transpose(1, 2))
    assert list(attn_output_weights.size()) == [bsz * num_heads, tgt_len, src_len]

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_output_weights.masked_fill_(attn_mask, float('-inf'))
        else:
            attn_output_weights += attn_mask


    if key_padding_mask is not None:
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        attn_output_weights = attn_output_weights.masked_fill(
            key_padding_mask.unsqueeze(1).unsqueeze(2),
            float('-inf'),
        )
        attn_output_weights = attn_output_weights.view(bsz * num_heads, tgt_len, src_len)

    attn_output_weights = F.softmax(
        attn_output_weights, dim=-1)
    attn_output_weights = F.dropout(attn_output_weights, p=dropout_p, training=training)

    # use hooks for the attention weights if necessary
    if attention_probs_forward_hook is not None and attention_probs_backwards_hook is not None:
        attention_probs_forward_hook(attn_output_weights)
        attn_output_weights.register_hook(attention_probs_backwards_hook)

    attn_output = torch.bmm(attn_output_weights, v)
    assert list(attn_output.size()) == [bsz * num_heads, tgt_len, head_dim]
    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
    attn_output = F.linear(attn_output, out_proj_weight, out_proj_bias)

    if need_weights:
        # average attention weights over heads
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        return attn_output, attn_output_weights.sum(dim=1) / num_heads
    else:
        return attn_output, None

class MultiheadAttention(torch.nn.Module):
    r"""Allows the model to jointly attend to information
    from different representation subspaces.
    See reference: Attention Is All You Need

    .. math::
        \text{MultiHead}(Q, K, V) = \text{Concat}(head_1,\dots,head_h)W^O
        \text{where} head_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)

    Args:
        embed_dim: total dimension of the model.
        num_heads: parallel attention heads.
        dropout: a Dropout layer on attn_output_weights. Default: 0.0.
        bias: add bias as module parameter. Default: True.
        add_bias_kv: add bias to the key and value sequences at dim=0.
        add_zero_attn: add a new batch of zeros to the key and
                       value sequences at dim=1.
        kdim: total number of features in key. Default: None.
        vdim: total number of features in value. Default: None.

        Note: if kdim and vdim are None, they will be set to embed_dim such that
        query, key, and value have the same number of features.

    Examples::

        >>> multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
        >>> attn_output, attn_output_weights = multihead_attn(query, key, value)
    """
    bias_k: Optional[torch.Tensor]
    bias_v: Optional[torch.Tensor]

    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None, batch_first=False,device=None,dtype=None,):
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        if self._qkv_same_embed_dim is False:
            self.q_proj_weight = Parameter(torch.Tensor(embed_dim, embed_dim))
            self.k_proj_weight = Parameter(torch.Tensor(embed_dim, self.kdim))
            self.v_proj_weight = Parameter(torch.Tensor(embed_dim, self.vdim))
            self.register_parameter('in_proj_weight', None)
        else:
            self.in_proj_weight = Parameter(torch.empty(3 * embed_dim, embed_dim))
            self.register_parameter('q_proj_weight', None)
            self.register_parameter('k_proj_weight', None)
            self.register_parameter('v_proj_weight', None)

        if bias:
            self.in_proj_bias = Parameter(torch.empty(3 * embed_dim))
        else:
            self.register_parameter('in_proj_bias', None)
        self.out_proj = _LinearWithBias(embed_dim, embed_dim)

        if add_bias_kv:
            self.bias_k = Parameter(torch.empty(1, 1, embed_dim))
            self.bias_v = Parameter(torch.empty(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self._reset_parameters()

    def _reset_parameters(self):
        if self._qkv_same_embed_dim:
            xavier_uniform_(self.in_proj_weight)
        else:
            xavier_uniform_(self.q_proj_weight)
            xavier_uniform_(self.k_proj_weight)
            xavier_uniform_(self.v_proj_weight)

        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.)
            constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            xavier_normal_(self.bias_v)

    def __setstate__(self, state):
        # Support loading old MultiheadAttention checkpoints generated by v1.1.0
        if '_qkv_same_embed_dim' not in state:
            state['_qkv_same_embed_dim'] = True

        super(MultiheadAttention, self).__setstate__(state)

    def forward(self, query, key, value, key_padding_mask=None,
                need_weights=True, attn_mask=None, attention_probs_forward_hook=None, attention_probs_backwards_hook=None):
        r"""
    Args:
        query, key, value: map a query and a set of key-value pairs to an output.
            See "Attention Is All You Need" for more details.
        key_padding_mask: if provided, specified padding elements in the key will
            be ignored by the attention. When given a binary mask and a value is True,
            the corresponding value on the attention layer will be ignored. When given
            a byte mask and a value is non-zero, the corresponding value on the attention
            layer will be ignored
        need_weights: output attn_output_weights.
        attn_mask: 2D or 3D mask that prevents attention to certain positions. A 2D mask will be broadcasted for all
            the batches while a 3D mask allows to specify a different mask for the entries of each batch.

    Shape:
        - Inputs:
        - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - key_padding_mask: :math:`(N, S)` where N is the batch size, S is the source sequence length.
          If a ByteTensor is provided, the non-zero positions will be ignored while the position
          with the zero positions will be unchanged. If a BoolTensor is provided, the positions with the
          value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.
        - attn_mask: 2D mask :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
          3D mask :math:`(N*num_heads, L, S)` where N is the batch size, L is the target sequence length,
          S is the source sequence length. attn_mask ensure that position i is allowed to attend the unmasked
          positions. If a ByteTensor is provided, the non-zero positions are not allowed to attend
          while the zero positions will be unchanged. If a BoolTensor is provided, positions with ``True``
          is not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
          is provided, it will be added to the attention weight.

        - Outputs:
        - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
          E is the embedding dimension.
        - attn_output_weights: :math:`(N, L, S)` where N is the batch size,
          L is the target sequence length, S is the source sequence length.
        """
        if not self._qkv_same_embed_dim:
            return multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask, use_separate_proj_weight=True,
                q_proj_weight=self.q_proj_weight, k_proj_weight=self.k_proj_weight,
                v_proj_weight=self.v_proj_weight,
                attention_probs_forward_hook=attention_probs_forward_hook,
                attention_probs_backwards_hook=attention_probs_backwards_hook)
        else:
            return multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask,
                attention_probs_forward_hook=attention_probs_forward_hook,
                attention_probs_backwards_hook=attention_probs_backwards_hook)

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(
                OrderedDict(
                    [
                        ("-1", nn.AvgPool2d(stride)),
                        (
                            "0",
                            nn.Conv2d(
                                inplanes,
                                planes * self.expansion,
                                1,
                                stride=1,
                                bias=False,
                            ),
                        ),
                        ("1", nn.BatchNorm2d(planes * self.expansion)),
                    ]
                )
            )

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class AttentionPool2d(nn.Module):
    def __init__(
        self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None
    ):
        super().__init__()
        self.positional_embedding = nn.Parameter(
            torch.randn(spacial_dim**2 + 1, embed_dim) / embed_dim**0.5
        )
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(
            2, 0, 1
        )  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x,
            key=x,
            value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat(
                [self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]
            ),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False,
        )

        return x[0]


class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, image_size=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.image_size = image_size

        # the 3-layer stem
        self.conv1 = nn.Conv2d(
            3, width // 2, kernel_size=3, stride=2, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.conv2 = nn.Conv2d(
            width // 2, width // 2, kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.avgpool = nn.AvgPool2d(2)
        self.relu = nn.ReLU(inplace=True)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(image_size // 32, embed_dim, heads, output_dim)

        self.init_parameters()

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def init_parameters(self):
        if self.attnpool is not None:
            std = self.attnpool.c_proj.in_features**-0.5
            nn.init.normal_(self.attnpool.q_proj.weight, std=std)
            nn.init.normal_(self.attnpool.k_proj.weight, std=std)
            nn.init.normal_(self.attnpool.v_proj.weight, std=std)
            nn.init.normal_(self.attnpool.c_proj.weight, std=std)

        for resnet_block in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for name, param in resnet_block.named_parameters():
                if name.endswith("bn3.weight"):
                    nn.init.zeros_(param)

    def lock(self, unlocked_groups=0, freeze_bn_stats=False):
        assert (
            unlocked_groups == 0
        ), "partial locking not currently supported for this model"
        for param in self.parameters():
            param.requires_grad = False
        if freeze_bn_stats:
            freeze_batch_norm_2d(self)

    def stem(self, x):
        for conv, bn in [
            (self.conv1, self.bn1),
            (self.conv2, self.bn2),
            (self.conv3, self.bn3),
        ]:
            x = self.relu(bn(conv(x)))
        x = self.avgpool(x)
        return x

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)

        return x


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        return x.to(orig_type)

class ClassificationHead(nn.Linear):
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        x = F.linear(x, self.weight, self.bias)
        return x.to(orig_type)


class QuickGELU(nn.Module):
    # NOTE This is slower than nn.GELU or nn.SiLU and uses more GPU memory
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, act_layer: Callable = nn.GELU, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(d_model, d_model * 4)),
                    ("gelu", act_layer()),
                    ("c_proj", nn.Linear(d_model * 4, d_model)),
                ]
            )
        )
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

        self.attn_probs = None
        self.attn_grad = None

    def set_attn_probs(self, attn_probs):
        self.attn_probs = attn_probs

    def set_attn_grad(self, attn_grad):
        self.attn_grad = attn_grad

    def attention(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask, attention_probs_forward_hook=self.set_attn_probs,
                         attention_probs_backwards_hook=self.set_attn_grad)[0]

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        x = x + self.attention(self.ln_1(x), attn_mask=attn_mask)
        print(self.attn_grad, "....................")
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(
        self, width: int, layers: int, heads: int, act_layer: Callable = nn.GELU, attn_mask: torch.Tensor = None
    ):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.ModuleList(
            [
                ResidualAttentionBlock(width, heads, act_layer=act_layer, attn_mask=attn_mask)
                for _ in range(layers)
            ]
        )

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        for r in self.resblocks:
            x = r(x, attn_mask=attn_mask)
        return x

class VisualTransformer(nn.Module):
    def __init__(
        self,
        image_size: int,
        patch_size: int,
        width: int,
        layers: int,
        heads: int,
        output_dim: int,
        act_layer: Callable = nn.GELU,
    ):
        super().__init__()
        self.image_size = image_size
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=width,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False,
        )
        scale = width**-0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(
            scale * torch.randn((image_size // patch_size) ** 2 + 1, width)
        )
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads, act_layer=act_layer)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def lock(self, unlocked_groups=0, freeze_bn_stats=False):
        assert (
            unlocked_groups == 0
        ), "partial locking not currently supported for this model"
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor, model=None):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [
                self.class_embedding.to(x.dtype)
                + torch.zeros(
                    x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
                ),
                x,
            ],
            dim=1,
        )  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)
        
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj
        return x

@dataclass
class CLIPVisionCfg:
    layers: Union[Tuple[int, int, int, int], int] = 12
    width: int = 768
    patch_size: int = 16
    image_size: Union[Tuple[int, int], int] = 224
    timm_model_name: str = (
        None  # a valid model name overrides layers, width, patch_size
    )
    timm_model_pretrained: bool = (
        False  # use (imagenet) pretrained weights for named model
    )
    timm_pool: str = (
        "avg"  # feature pooling for timm model ('abs_attn', 'rot_attn', 'avg', '')
    )
    timm_proj: str = (
        "linear"  # linear projection for timm model output ('linear', 'mlp', '')
    )


@dataclass
class CLIPTextCfg:
    context_length: int
    vocab_size: int
    width: int
    heads: int
    layers: int


@registry.register_model("clip")
@registry.register_model("clip_feature_extractor")
class CLIP(BaseModel):
    PRETRAINED_MODEL_CONFIG_DICT = {
        "ViT-B-32": "configs/models/clip_vit_base32.yaml",
        "ViT-B-16": "configs/models/clip_vit_base16.yaml",
        "ViT-L-14": "configs/models/clip_vit_large14.yaml",
        "ViT-L-14-336": "configs/models/clip_vit_large14_336.yaml",
        "RN50": "configs/models/clip_resnet50.yaml",
    }

    def __init__(
        self,
        embed_dim: int,
        vision_cfg: CLIPVisionCfg,
        text_cfg: CLIPTextCfg,
        quick_gelu: bool = False,
        add_cls_token: bool = False,
        mode: bool=False,
    ):
        from .tokenizer import tokenize

        super().__init__()
        self.tokenizer = tokenize
        self._loss = None
        
        if isinstance(vision_cfg, dict):
            vision_cfg = CLIPVisionCfg(**vision_cfg)
        if isinstance(text_cfg, dict):
            text_cfg = CLIPTextCfg(**text_cfg)
        self.variable=vision_cfg
        self.variable2 = text_cfg
        self.context_length = text_cfg.context_length
        self.add_cls_token = add_cls_token
        self.mode = mode

        # OpenAI models are pretrained w/ QuickGELU but native nn.GELU is both faster and more
        # memory efficient in recent PyTorch releases (>= 1.10).
        # NOTE: timm models always use native GELU regardless of quick_gelu flag.
        self.act_layer = QuickGELU if quick_gelu else nn.GELU

        if vision_cfg.timm_model_name:
            self.visual = TimmModel(
                vision_cfg.timm_model_name,
                pretrained=vision_cfg.timm_model_pretrained,
                pool=vision_cfg.timm_pool,
                proj=vision_cfg.timm_proj,
                embed_dim=embed_dim,
                image_size=vision_cfg.image_size,
            )
            self.act_layer = (
                nn.GELU
            )  # so that text transformer doesn't use QuickGELU w/ timm models
        elif isinstance(vision_cfg.layers, (tuple, list)):
            vision_heads = vision_cfg.width * 32 // 64
            self.visual = ModifiedResNet(
                layers=vision_cfg.layers,
                output_dim=embed_dim,
                heads=vision_heads,
                image_size=vision_cfg.image_size,
                width=vision_cfg.width,
            )
        else:
            vision_heads = vision_cfg.width // 64
            self.visual = VisualTransformer(
                image_size=vision_cfg.image_size,
                patch_size=vision_cfg.patch_size,
                width=vision_cfg.width,
                layers=vision_cfg.layers,
                heads=vision_heads,
                output_dim=embed_dim,
                act_layer=self.act_layer,
            )

        self.transformer = Transformer(
            width=text_cfg.width,
            layers=text_cfg.layers,
            heads=text_cfg.heads,
            act_layer=self.act_layer,
            attn_mask=self.build_attention_mask(),
        )

        self.vocab_size = text_cfg.vocab_size
        self.token_embedding = nn.Embedding(text_cfg.vocab_size, text_cfg.width)
        self.positional_embedding = nn.Parameter(
            torch.empty(self.context_length, text_cfg.width)
        )
        self.ln_final = LayerNorm(text_cfg.width)
        self.classificationHead = ClassificationHead(text_cfg.width, 1, True)

        self.text_projection = nn.Parameter(torch.empty(text_cfg.width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.register_buffer("attn_mask", self.build_attention_mask(), persistent=False)

        self.prompt_templates = openai_imagenet_template
        self.classifier = None

        self.init_parameters()

    @property
    def loss(self):
        if self._loss is None:
            from lavis.models.clip_models.loss import ClipLoss
            from torch import distributed as dist
            self._loss = ClipLoss(
                world_size=dist.get_world_size(),
                rank=dist.get_rank(),
                local_loss=False,
                gather_with_grad=False,
                use_horovod=False,
            )

        return self._loss

    def init_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)
        nn.init.constant_(self.logit_scale, np.log(1 / 0.07))

        if hasattr(self.visual, "init_parameters"):
            self.visual.init_parameters()

        proj_std = (self.transformer.width**-0.5) * (
            (2 * self.transformer.layers) ** -0.5
        )
        attn_std = self.transformer.width**-0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width**-0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def lock_image_tower(self, unlocked_groups=0, freeze_bn_stats=False):
        # lock image tower as per LiT - https://arxiv.org/abs/2111.07991
        self.visual.lock(
            unlocked_groups=unlocked_groups, freeze_bn_stats=freeze_bn_stats
        )

    def encode_image(self, image):
        return self.visual(image, self)

    def encode_text(self, text):
        if self.add_cls_token:
            cls_token = torch.full((text.shape[0], 1), self.vocab_size-1, dtype=torch.long).to(self.device) 
            text = torch.cat([text, cls_token], dim=1) 

        x = self.token_embedding(text)  # [batch_size, n_ctx, d_model]
        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        try:
        	x = self.transformer(x, attn_mask=self.attn_mask)
        except:
        	x = self.transformer(x)       
        x = x.permute(1, 0, 2)  # LND -> NLD

        # ici si je veux faire du token par token à la place du cls
        x = self.ln_final(x)

        y=None
        if self.add_cls_token:
            y = self.classificationHead(x[:,-1,:])

        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x,y

    def forward(self, samples):
        image = samples.get("image")
        text = samples.get("text_input")
        targets=samples.get("label")
        
        if text is not None:
            text = self.tokenizer(text, add_cls_token=self.add_cls_token).to(self.device)
        if image is None:
            return self.encode_text(text)[0]
        elif text is None:
            return self.encode_image(image)
            
        image_embeds = self.encode_image(image)
        image_features = F.normalize(image_embeds, dim=-1)

        text_embeds, text_classif_features = self.encode_text(text)
        text_features = F.normalize(text_embeds, dim=-1)

        loss = self.loss(image_features, text_features, self.logit_scale.exp(), text_classif_features, targets, mode=self.mode)
        #if self.mode:
        #    image_embeds=image_embeds[self.loss.result]
        #    image_features=image_features[self.loss.result]
        #    text_embeds=text_embeds[self.loss.result]
        #    text_features=text_features[self.loss.result]
        return ClipOutput(
            intermediate_output=ClipOutputFeatures(
                image_embeds=image_embeds,
                image_embeds_proj=image_features,
                text_embeds=text_embeds,
                text_embeds_proj=text_features,
            ),
            loss=loss,
            logit_scale_exp=self.logit_scale.exp(),
        )

    def extract_features(self, samples):
        """
        Extract features from the model for samples.

        Keys allowed are "image" and "text_input" in samples.
        If either key is missing, the corresponding features are not extracted.

        Args:
            samples: dict of samples to extract features from.

        Returns:
            ClipOutputFeatures object with features for the samples.
        """
        image = samples.get("image")
        text = samples.get("text_input")

        if text is not None:
            text = self.tokenizer(text, add_cls_token=self.add_cls_token).to(self.device)

        if image is None:
            return self.encode_text(text)[0]
        elif text is None:
            return self.encode_image(image)

        image_embeds = self.encode_image(image)
        image_features = F.normalize(image_embeds, dim=-1)

        text_embeds, text_classif_embeds = self.encode_text(text)
        text_features = F.normalize(text_embeds, dim=-1)
        if self.add_cls_token:
            text_classif_features = F.normalize(text_classif_embeds, dim=-1)
        else:
            text_classif_features=None

        return ClipOutputFeatures(
            image_embeds=image_embeds,
            image_embeds_proj=image_features,
            text_embeds=text_embeds,
            text_embeds_proj=text_features,
            text_classif_embeds=text_classif_embeds,
            text_classif_embeds_proj=text_classif_features,
        )

    def predict(self, samples):
        image = samples["image"]
        targets = samples.get("label")
        text = samples.get("text_input")

        image_features = self.encode_image(image)
        image_features = F.normalize(image_features, dim=-1)

        logits = 100.0 * image_features @ self.classifier
        return {"predictions": logits, "targets": targets[targets!=2]}

    def before_evaluation(self, dataset, task_type, **kwargs):
        print("validation après")
        if task_type == MultimodalClassificationTask:
            self.classifier = self.zero_shot_classifier(
                classnames=dataset.classnames,
                templates=self.prompt_templates,
            )

    def zero_shot_classifier(self, classnames, templates):
        with torch.no_grad():
            zeroshot_weights = []
            for classname in classnames:
                texts = [
                    template(classname) for template in templates
                ]  # format with class
                texts = self.tokenizer(texts, add_cls_token=self.self.add_cls_token).to(self.device)  # tokenize

                class_embeddings = self.encode_text(texts)[0]
                class_embedding = F.normalize(class_embeddings, dim=-1).mean(dim=0)
                class_embedding /= class_embedding.norm()
                zeroshot_weights.append(class_embedding)
            zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(self.device)
        return zeroshot_weights

    @classmethod
    def default_config_path(cls, model_type="base"):
        model_type = "ViT-B-32" if model_type == "base" else model_type

        assert (
            model_type in cls.PRETRAINED_MODEL_CONFIG_DICT
        ), "Unknown model type {}. \n Available types: {}".format(
            model_type, cls.PRETRAINED_MODEL_CONFIG_DICT.keys()
        )
        return get_abs_path(cls.PRETRAINED_MODEL_CONFIG_DICT[model_type])

    @classmethod
    def from_config(cls, cfg=None):
        model_name = cfg.model_type
        pretrained = cfg.pretrained

        precision = cfg.get("precision", "fp32")

        return create_model(
            model_name=model_name, pretrained=pretrained, precision=precision
        )

    def zero_shot_predict(self, image_path, categories):
        assert isinstance(
            categories, list
        ), f"categories must be a list, got {type(categories)}."
        assert os.path.exists(image_path), f"File {image_path} does not exist."

        from lavis.processors.clip_processors import ClipImageEvalProcessor
        from PIL import Image

        image_preprocess = ClipImageEvalProcessor()
        image = image_preprocess(Image.open(image_path)).unsqueeze(0)

        text = self.tokenizer(categories, add_cls_token=self.add_cls_token)

        with torch.no_grad():
            image_features = self.encode_image(image)
            text_features = self.encode_text(text)[0]
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

    def compute_sim_matrix(self, data_loader, task_cfg):
        logging.info("Computing features for evaluation...")
        start_time = time.time()

        texts = data_loader.dataset.text
        num_text = len(texts)
        text_bs = 256
        text_features = []
        text_classif_features= []

        for i in range(0, num_text, text_bs):

            text = texts[i : min(num_text, i + text_bs)]
            text_input = self.tokenizer(text, add_cls_token=self.add_cls_token).to(self.device)

            text_feat, text_classif_feat = self.encode_text(text_input)
            text_feat = F.normalize(text_feat, dim=-1)

            text_features.append(text_feat)
            text_classif_features.append(text_classif_feat)

        text_features = torch.cat(text_features, dim=0)

        image_features = []
        for samples in data_loader:
            image = samples["image"]
            image = image.to(self.device)
            image_feat = self.encode_image(image)
            image_feat = F.normalize(image_feat, dim=-1)

            image_features.append(image_feat)

        image_features = torch.cat(image_features, dim=0)
        result=None
        if self.add_cls_token:
            text_classif_features = torch.cat(text_classif_features, dim=0)
            result = torch.sigmoid(text_classif_features)
            result =  (result >= 0.5).long()
            sims_matrix_i2t = image_features @ text_features[result.squeeze(1) ==1].t()
            sims_matrix_t2i = text_features @ image_features.t()
            result=result.cpu().numpy()
        else:
            sims_matrix_i2t = image_features @ text_features.t()
            sims_matrix_t2i = sims_matrix_i2t.t()

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logging.info("Evaluation time {}".format(total_time_str))

        return sims_matrix_i2t.cpu().numpy(), sims_matrix_t2i.cpu().numpy(), result

def convert_weights_to_fp16(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [
                *[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]],
                "in_proj_bias",
                "bias_k",
                "bias_v",
            ]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


def build_model_from_openai_state_dict(state_dict: dict):
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len(
            [
                k
                for k in state_dict.keys()
                if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")
            ]
        )
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round(
            (state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5
        )
        image_size = vision_patch_size * grid_size
    else:
        counts: list = [
            len(
                set(
                    k.split(".")[2]
                    for k in state_dict
                    if k.startswith(f"visual.layer{b}")
                )
            )
            for b in [1, 2, 3, 4]
        ]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round(
            (state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5
        )
        vision_patch_size = None
        assert (
            output_width**2 + 1
            == state_dict["visual.attnpool.positional_embedding"].shape[0]
        )
        image_size = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(
        set(
            k.split(".")[2]
            for k in state_dict
            if k.startswith(f"transformer.resblocks")
        )
    )

    vision_cfg = CLIPVisionCfg(
        layers=vision_layers,
        width=vision_width,
        patch_size=vision_patch_size,
        image_size=image_size,
    )
    text_cfg = CLIPTextCfg(
        context_length=context_length,
        vocab_size=vocab_size,
        width=transformer_width,
        heads=transformer_heads,
        layers=transformer_layers,
    )
    model = CLIP(
        embed_dim,
        vision_cfg=vision_cfg,
        text_cfg=text_cfg,
        quick_gelu=True,  # OpenAI models were trained with QuickGELU
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        state_dict.pop(key, None)

    convert_weights_to_fp16(model)
    missing_keys = ["classificationHead.weight", "classificationHead.bias"]
    for key in missing_keys:
        if key in state_dict:
            del state_dict[key]

    model.load_state_dict(state_dict, strict=False)

    torch.nn.init.xavier_uniform_(model.classificationHead.weight)
    torch.nn.init.zeros_(model.classificationHead.bias)
    return model.eval()


def trace_model(model, batch_size=256, device=torch.device("cpu")):
    model.eval()
    image_size = model.visual.image_size
    example_images = torch.ones((batch_size, 3, image_size, image_size), device=device)
    example_text = torch.zeros(
        (batch_size, model.context_length), dtype=torch.int, device=device
    )
    model = torch.jit.trace_module(
        model,
        inputs=dict(
            forward=(example_images, example_text),
            encode_text=(example_text,),
            encode_image=(example_images,),
        ),
    )
    model.visual.image_size = image_size
    return


def _natural_key(string_):
    return [int(s) if s.isdigit() else s for s in re.split(r"(\d+)", string_.lower())]


def _rescan_model_configs():
    global _MODEL_CONFIGS

    config_ext = (".json",)
    config_files = []
    for config_path in _MODEL_CONFIG_PATHS:
        if config_path.is_file() and config_path.suffix in config_ext:
            config_files.append(config_path)
        elif config_path.is_dir():
            for ext in config_ext:
                config_files.extend(config_path.glob(f"*{ext}"))

    for cf in config_files:
        with open(cf, "r") as f:
            model_cfg = json.load(f)
            if all(a in model_cfg for a in ("embed_dim", "vision_cfg", "text_cfg")):
                _MODEL_CONFIGS[cf.stem] = model_cfg

    _MODEL_CONFIGS = {
        k: v
        for k, v in sorted(_MODEL_CONFIGS.items(), key=lambda x: _natural_key(x[0]))
    }


_rescan_model_configs()  # initial populate of model config registry


def load_state_dict(checkpoint_path: str, map_location="cpu"):
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint
    if next(iter(state_dict.items()))[0].startswith("module"):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    return state_dict


def create_model(
    model_name: str,
    pretrained: str = "",
    precision: str = "fp32",
    device: torch.device = torch.device("cpu"),
    jit: bool = False,
    force_quick_gelu: bool = False,
    pretrained_image: bool = False,
):
    model_name = model_name.replace(
        "/", "-"
    )  # for callers using old naming with / in ViT names

    if pretrained.lower() == "openai":
        logging.info(f"Loading pretrained {model_name} from OpenAI.")
        model = load_openai_model(model_name, device=device, jit=jit)
        # See https://discuss.pytorch.org/t/valueerror-attemting-to-unscale-fp16-gradients/81372
        if precision == "amp" or precision == "fp32":
            model = model.float()
    else:
        logging.info(f"No pretrained weights loaded for {model_name} model.")
        if model_name in _MODEL_CONFIGS:
            logging.info(f"Loading {model_name} model config.")
            model_cfg = deepcopy(_MODEL_CONFIGS[model_name])
        else:
            logging.error(
                f"Model config for {model_name} not found; available models {list_models()}."
            )
            raise RuntimeError(f"Model config for {model_name} not found.")

        if force_quick_gelu:
            # override for use of QuickGELU on non-OpenAI transformer models
            model_cfg["quick_gelu"] = True

        if pretrained_image:
            if "timm_model_name" in model_cfg.get("vision_cfg", {}):
                # pretrained weight loading for timm models set via vision_cfg
                model_cfg["vision_cfg"]["timm_model_pretrained"] = True
            else:
                assert (
                    False
                ), "pretrained image towers currently only supported for timm models"

        model = CLIP(**model_cfg)

        if pretrained:
            checkpoint_path = ""
            url = get_pretrained_url(model_name, pretrained)
            if url:
                checkpoint_path = download_pretrained(url)
            elif os.path.exists(pretrained):
                checkpoint_path = pretrained


            if checkpoint_path:
                logging.info(f"Loading pretrained {model_name} weights ({pretrained}).")
                model.load_state_dict(load_state_dict(checkpoint_path))
            else:
                logging.warning(
                    f"Pretrained weights ({pretrained}) not found for model {model_name}."
                )
                raise RuntimeError(
                    f"Pretrained weights ({pretrained}) not found for model {model_name}."
                )

        model.to(device=device)
        if precision == "fp16":
            assert device.type != "cpu"
            convert_weights_to_fp16(model)

        if jit:
            model = torch.jit.script(model)

    return model


def create_model_and_transforms(
    model_name: str,
    pretrained: str = "",
    precision: str = "fp32",
    device: torch.device = torch.device("cpu"),
    jit: bool = False,
    force_quick_gelu: bool = False,
    pretrained_image: bool = False,
):
    model = create_model(
        model_name,
        pretrained,
        precision,
        device,
        jit,
        force_quick_gelu=force_quick_gelu,
        pretrained_image=pretrained_image,
    )
    preprocess_train = image_transform(model.visual.image_size, is_train=True)
    preprocess_val = image_transform(model.visual.image_size, is_train=False)
    return model, preprocess_train, preprocess_val


def list_models():
    """enumerate available model architectures based on config files"""
    return list(_MODEL_CONFIGS.keys())


def add_model_config(path):
    """add model config path or file and update registry"""
    if not isinstance(path, Path):
        path = Path(path)
    _MODEL_CONFIG_PATHS.append(path)
    _rescan_model_configs()


def list_openai_models() -> List[str]:
    """Returns the names of available CLIP models"""
    return list_pretrained_tag_models("openai")


def load_openai_model(
    name: str,
    device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu",
    jit=True,
):
    """Load a CLIP model
    Parameters
    ----------
    name : str
        A model name listed by `clip.available_models()`, or the path to a model checkpoint containing the state_dict
    device : Union[str, torch.device]
        The device to put the loaded model
    jit : bool
        Whether to load the optimized JIT model (default) or more hackable non-JIT model.
    Returns
    -------
    model : torch.nn.Module
        The CLIP model
    preprocess : Callable[[PIL.Image], torch.Tensor]
        A torchvision transform that converts a PIL image into a tensor that the returned model can take as its input
    """
    if get_pretrained_url(name, "openai"):
        model_path = download_pretrained(get_pretrained_url(name, "openai"))
    elif os.path.isfile(name):
        model_path = name
    else:
        raise RuntimeError(
            f"Model {name} not found; available models = {list_openai_models()}"
        )

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location=device if jit else "cpu").eval()
        state_dict = None
    except RuntimeError:
        # loading saved state dict
        if jit:
            warnings.warn(
                f"File {model_path} is not a JIT archive. Loading as a state dict instead"
            )
            jit = False
        state_dict = torch.load(model_path, map_location="cpu")

    if not jit:
        try:
            model = build_model_from_openai_state_dict(
                state_dict or model.state_dict()
            ).to(device)
        except KeyError:
            sd = {k[7:]: v for k, v in state_dict["state_dict"].items()}
            model = build_model_from_openai_state_dict(sd).to(device)

        if str(device) == "cpu":
            model.float()
        return model

    # patch the device names
    device_holder = torch.jit.trace(
        lambda: torch.ones([]).to(torch.device(device)), example_inputs=[]
    )
    device_node = [
        n
        for n in device_holder.graph.findAllNodes("prim::Constant")
        if "Device" in repr(n)
    ][-1]

    def patch_device(module):
        try:
            graphs = [module.graph] if hasattr(module, "graph") else []
        except RuntimeError:
            graphs = []

        if hasattr(module, "forward1"):
            graphs.append(module.forward1.graph)

        for graph in graphs:
            for node in graph.findAllNodes("prim::Constant"):
                if "value" in node.attributeNames() and str(node["value"]).startswith(
                    "cuda"
                ):
                    node.copyAttributes(device_node)

    model.apply(patch_device)
    patch_device(model.encode_image)
    patch_device(model.encode_text)

    # patch dtype to float32 on CPU
    if str(device) == "cpu":
        float_holder = torch.jit.trace(
            lambda: torch.ones([]).float(), example_inputs=[]
        )
        float_input = list(float_holder.graph.findNode("aten::to").inputs())[1]
        float_node = float_input.node()

        def patch_float(module):
            try:
                graphs = [module.graph] if hasattr(module, "graph") else []
            except RuntimeError:
                graphs = []

            if hasattr(module, "forward1"):
                graphs.append(module.forward1.graph)

            for graph in graphs:
                for node in graph.findAllNodes("aten::to"):
                    inputs = list(node.inputs())
                    for i in [
                        1,
                        2,
                    ]:  # dtype can be the second or third argument to aten::to()
                        if inputs[i].node()["value"] == 5:
                            inputs[i].node().copyAttributes(float_node)

        model.apply(patch_float)
        patch_float(model.encode_image)
        patch_float(model.encode_text)
        model.float()

    # ensure image_size attr available at consistent location for both jit and non-jit
    model.visual.image_size = model.input_resolution.item()
    return model


openai_imagenet_template = [
    lambda c: f"a bad photo of a {c}.",
    lambda c: f"a photo of many {c}.",
    lambda c: f"a sculpture of a {c}.",
    lambda c: f"a photo of the hard to see {c}.",
    lambda c: f"a low resolution photo of the {c}.",
    lambda c: f"a rendering of a {c}.",
    lambda c: f"graffiti of a {c}.",
    lambda c: f"a bad photo of the {c}.",
    lambda c: f"a cropped photo of the {c}.",
    lambda c: f"a tattoo of a {c}.",
    lambda c: f"the embroidered {c}.",
    lambda c: f"a photo of a hard to see {c}.",
    lambda c: f"a bright photo of a {c}.",
    lambda c: f"a photo of a clean {c}.",
    lambda c: f"a photo of a dirty {c}.",
    lambda c: f"a dark photo of the {c}.",
    lambda c: f"a drawing of a {c}.",
    lambda c: f"a photo of my {c}.",
    lambda c: f"the plastic {c}.",
    lambda c: f"a photo of the cool {c}.",
    lambda c: f"a close-up photo of a {c}.",
    lambda c: f"a black and white photo of the {c}.",
    lambda c: f"a painting of the {c}.",
    lambda c: f"a painting of a {c}.",
    lambda c: f"a pixelated photo of the {c}.",
    lambda c: f"a sculpture of the {c}.",
    lambda c: f"a bright photo of the {c}.",
    lambda c: f"a cropped photo of a {c}.",
    lambda c: f"a plastic {c}.",
    lambda c: f"a photo of the dirty {c}.",
    lambda c: f"a jpeg corrupted photo of a {c}.",
    lambda c: f"a blurry photo of the {c}.",
    lambda c: f"a photo of the {c}.",
    lambda c: f"a good photo of the {c}.",
    lambda c: f"a rendering of the {c}.",
    lambda c: f"a {c} in a video game.",
    lambda c: f"a photo of one {c}.",
    lambda c: f"a doodle of a {c}.",
    lambda c: f"a close-up photo of the {c}.",
    lambda c: f"a photo of a {c}.",
    lambda c: f"the origami {c}.",
    lambda c: f"the {c} in a video game.",
    lambda c: f"a sketch of a {c}.",
    lambda c: f"a doodle of the {c}.",
    lambda c: f"a origami {c}.",
    lambda c: f"a low resolution photo of a {c}.",
    lambda c: f"the toy {c}.",
    lambda c: f"a rendition of the {c}.",
    lambda c: f"a photo of the clean {c}.",
    lambda c: f"a photo of a large {c}.",
    lambda c: f"a rendition of a {c}.",
    lambda c: f"a photo of a nice {c}.",
    lambda c: f"a photo of a weird {c}.",
    lambda c: f"a blurry photo of a {c}.",
    lambda c: f"a cartoon {c}.",
    lambda c: f"art of a {c}.",
    lambda c: f"a sketch of the {c}.",
    lambda c: f"a embroidered {c}.",
    lambda c: f"a pixelated photo of a {c}.",
    lambda c: f"itap of the {c}.",
    lambda c: f"a jpeg corrupted photo of the {c}.",
    lambda c: f"a good photo of a {c}.",
    lambda c: f"a plushie {c}.",
    lambda c: f"a photo of the nice {c}.",
    lambda c: f"a photo of the small {c}.",
    lambda c: f"a photo of the weird {c}.",
    lambda c: f"the cartoon {c}.",
    lambda c: f"art of the {c}.",
    lambda c: f"a drawing of the {c}.",
    lambda c: f"a photo of the large {c}.",
    lambda c: f"a black and white photo of a {c}.",
    lambda c: f"the plushie {c}.",
    lambda c: f"a dark photo of a {c}.",
    lambda c: f"itap of a {c}.",
    lambda c: f"graffiti of the {c}.",
    lambda c: f"a toy {c}.",
    lambda c: f"itap of my {c}.",
    lambda c: f"a photo of a cool {c}.",
    lambda c: f"a photo of a small {c}.",
    lambda c: f"a tattoo of the {c}.",
]
