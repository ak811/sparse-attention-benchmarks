import torch
from .base import AttentionBackend
from ..registries import register_backend

@register_backend("vit")
class ScaledDotProductBackend(AttentionBackend):
    def __call__(self, q, k, v, *, attn_drop=None, mask_fn=None, save_hook=None, **kwargs):
        # q, k, v: (B, H, N, D)
        attn = q @ k.transpose(-2, -1)
        d = q.size(-1)
        attn = attn / (d ** 0.5)

        if mask_fn is not None:
            # mask: (1 or B, H, N, N), 1=keep, 0=mask
            mask = mask_fn(attn, **kwargs).to(dtype=attn.dtype)
            attn = attn + (1.0 - mask) * (-1e9)

        attn = attn.softmax(dim=-1)
        if attn_drop is not None:
            attn = attn_drop(attn)
        x = attn @ v
        if save_hook is not None:
            save_hook(x)
        return x