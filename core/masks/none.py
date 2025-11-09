import torch
from .base import AttentionMask
from ..registries import register_mask

@register_mask("none")
class NoMask(AttentionMask):
    def __call__(self, attn, **kwargs):
        return torch.ones_like(attn)