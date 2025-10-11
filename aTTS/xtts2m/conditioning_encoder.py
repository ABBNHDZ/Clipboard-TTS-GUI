import functools
import math
import random
from collections import namedtuple
from functools import wraps
from typing import TypeVar
from typing_extensions import TypeIs
from collections.abc import Callable

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import einsum, nn
from transformers import GPT2Config, GPT2Model,  GenerationMixin, GPT2PreTrainedModel, GenerationConfig
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

from xtts2m.utils import get_valid_dtype_for_device

import logging

logger = logging.getLogger(__name__)



class QKVAttentionLegacy(nn.Module):
    """Attention QKV héritée (implémentation adaptée pour compatibilité)."""

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv, mask=None, rel_pos=None):
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum("bct,bcs->bts", q * scale, k * scale)
        if rel_pos is not None:
            weight = rel_pos(weight.reshape(bs, self.n_heads, weight.shape[-2], weight.shape[-1])).reshape(
                bs * self.n_heads, weight.shape[-2], weight.shape[-1]
            )
        if mask is not None:
            mask = mask.repeat(self.n_heads, 1, 1)
            weight[mask.logical_not()] = -torch.inf
        weight = torch.softmax(weight.float(), dim=-1)#.type(v.dtype)
        a = torch.einsum("bts,bcs->bct", weight, v.float())

        return a.reshape(bs, -1, length)

class RelativePositionBias(nn.Module):
    """Fallback minimal pour RelativePositionBias.

    Cette implémentation est un no-op qui renvoie simplement l'entrée.
    Elle évite une erreur si l'option `relative_pos_embeddings` est activée
    mais qu'aucune implémentation plus complète n'est disponible.
    """
    def __init__(self, *_, **__):
        super().__init__()

    def forward(self, attn_weight: torch.Tensor) -> torch.Tensor:
        # attn_weight attendu: (batch, heads, i, j) — on renvoie tel quel
        return attn_weight

class GroupNorm32(nn.GroupNorm):
    def forward(self, input):
        return super().forward(input).type(input.dtype)


def normalization(channels):
    """Construit une couche de normalisation adaptée au nombre de canaux."""
    groups = 32
    if channels <= 16:
        groups = 8
    elif channels <= 64:
        groups = 16
    while channels % groups != 0:
        groups = int(groups / 2)
    assert groups > 2
    return GroupNorm32(groups, channels)


def zero_module(module):
    """Met à zéro tous les paramètres d'un module (utilisé pour init)."""
    for p in module.parameters():
        p.detach().zero_()
    return module

class AttentionBlock(nn.Module):
    """Bloc d'attention utilisé pour l'encodeur de conditionnement."""

    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        *,
        relative_pos_embeddings=False,
        tortoise_norm=False,
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert channels % num_head_channels == 0, (
                f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            )
            self.num_heads = channels // num_head_channels
        self.norm = normalization(channels)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        self.attention = QKVAttentionLegacy(self.num_heads)
        self.tortoise_norm = tortoise_norm

        self.proj_out = zero_module(nn.Conv1d(channels, channels, 1))
        if relative_pos_embeddings:
            self.relative_pos_embeddings = RelativePositionBias(
                scale=(channels // self.num_heads) ** 0.5,
                causal=False,
                heads=num_heads,
                num_buckets=32,
                max_distance=64,
            )
        else:
            self.relative_pos_embeddings = None

    def forward(self, x, mask=None):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        x_norm = self.norm(x)
        qkv = self.qkv(x_norm)
        h = self.attention(qkv, mask, self.relative_pos_embeddings)
        h = h.type(x.dtype)
        h = self.proj_out(h)
        if self.tortoise_norm:
            return (x + h).reshape(b, c, *spatial)
        return (x_norm + h).reshape(b, c, *spatial)


class ConditioningEncoder(nn.Module):
    def __init__(self, spec_dim, embedding_dim, attn_blocks=6, num_attn_heads=4, *, tortoise_norm=False):
        super().__init__()
        attn = []
        self.init = nn.Conv1d(spec_dim, embedding_dim, kernel_size=1)
        for a in range(attn_blocks):
            attn.append(AttentionBlock(embedding_dim, num_attn_heads, tortoise_norm=tortoise_norm))
        self.attn = nn.Sequential(*attn)
        self.dim = embedding_dim

    def forward(self, x):
        """x: (b, 80, s) -> renvoie (b, d, s)"""
        # Ensure input and weights are float16 if the model is in float16 mode
        target_dtype = next(self.parameters()).dtype
        if target_dtype == torch.float16:
            x = x.to(target_dtype)
            self.init = self.init.to(target_dtype)
            self.attn = self.attn.to(target_dtype)
        h = self.init(x)
        h = self.attn(h)
        return h
