import torch
from .base import AttentionBackend
from ..registries import register_backend

@register_backend("vit_qchunk")
class ScaledDotProductQChunkBackend(AttentionBackend):
    """
    Dense attention with query-chunking: processes queries in blocks to reduce peak memory.
    Compatible with static masks (e.g., local/strided/none) via mask_fn.
    """
    def __init__(self, query_chunk_size=1024):
        super().__init__()
        assert query_chunk_size is None or query_chunk_size > 0
        self.query_chunk_size = query_chunk_size

    def __call__(self, q, k, v, *, attn_drop=None, mask_fn=None, save_hook=None, **kwargs):
        # q,k,v: (B,H,N,D)
        B, H, N, D = q.shape
        scale = 1.0 / (D ** 0.5)
        C = self.query_chunk_size or N

        # Precompute a static mask once if possible (mask_fn that does NOT depend on logits)
        static_mask = None
        if mask_fn is not None:
            try:
                dummy = torch.zeros((B, H, N, N), device=q.device, dtype=q.dtype)
                static_mask = mask_fn(dummy, N=N, num_heads=H, device=q.device)
            except Exception:
                static_mask = None

        outs = []
        for s in range(0, N, C):
            e = min(N, s + C)
            logits = torch.matmul(q[:, :, s:e, :], k.transpose(-2, -1)) * scale  # (B,H,C,N)

            if mask_fn is not None:
                mask_block = static_mask[:, :, s:e, :] if static_mask is not None \
                             else mask_fn(logits, N=N, num_heads=H, device=logits.device)
                logits = logits + (1.0 - mask_block.to(dtype=logits.dtype)) * (-1e9)

            attn = torch.softmax(logits, dim=-1)
            if attn_drop is not None:
                attn = attn_drop(attn)
            outs.append(torch.matmul(attn, v))  # (B,H,C,D)

        out = torch.cat(outs, dim=2)  # (B,H,N,D)
        if save_hook is not None:
            save_hook(out)
        return out
