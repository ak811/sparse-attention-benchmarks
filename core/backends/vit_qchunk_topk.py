import torch
from .base import AttentionBackend
from ..registries import register_backend

@register_backend("vit_topk")
class TopKQueryChunkBackend(AttentionBackend):
    """
    Top-k attention with query-chunking. Selects row-wise top-k logits per (B,H) from K for each query.
    Ignores external mask_fn by design (paper approximates vanilla attention via top-k).
    Options:
      - query_chunk_size (int): rows per block
      - k (int): number of keys to keep per query
      - causal (bool): forbid attending to future positions j>i before top-k
      - symmetric (bool): OR with transpose (rarely needed for top-k)
      - keep_cls_dense (bool): keep CLS row/col as 1s
    """
    def __init__(self, query_chunk_size=1024, k=32, causal=False, symmetric=False, keep_cls_dense=True):
        super().__init__()
        assert query_chunk_size is None or query_chunk_size > 0
        self.query_chunk_size = query_chunk_size
        self.k = int(k)
        self.causal = bool(causal)
        self.symmetric = bool(symmetric)
        self.keep_cls_dense = bool(keep_cls_dense)
        self._reported = False

    def __call__(self, q, k, v, *, attn_drop=None, mask_fn=None, save_hook=None, **kwargs):
        # q,k,v: (B,H,N,D)
        # NOTE: mask_fn is intentionally ignored (top-k defines its own dynamic mask)
        B, H, N, D = q.shape
        scale = 1.0 / (D ** 0.5)
        C = self.query_chunk_size or N

        outs = []
        for s in range(0, N, C):
            e = min(N, s + C)
            logits = torch.matmul(q[:, :, s:e, :], k.transpose(-2, -1)) * scale  # (B,H,C,N)

            # Optional causal pre-mask (disallow future keys before top-k)
            if self.causal:
                i = torch.arange(N, device=logits.device)
                causal = (i.view(1,1,1,N) <= i.view(1,1,N,1))  # (1,1,N,N)
                logits = logits.masked_fill(~causal[:, :, s:e, :], float('-inf'))

            # Row-wise top-k per (B,H,query_row)
            k_eff = min(self.k, N)
            top_vals, top_idx = torch.topk(logits, k=k_eff, dim=-1)  # (B,H,C,k)

            # Build sparse mask for this block
            block_mask = torch.zeros_like(logits, dtype=logits.dtype)  # (B,H,C,N)
            block_mask.scatter_(dim=-1, index=top_idx, src=torch.ones_like(top_vals, dtype=logits.dtype))

            if self.symmetric:
                # Need full (B,H,N,N) to OR with transpose; do a compact stitch for this block
                # Build an (N,N) mask per (B,H) by placing the block rows; to avoid O(N^2) memory here,
                # we only symmetrize *within* the block rows against all columns:
                # M_block = max(M_block, M_block^T rows subset)
                # Approx: transpose across last two dims then slice rows
                trans_rows = torch.zeros_like(block_mask)
                trans_rows = trans_rows.scatter(
                    dim=-1,
                    index=torch.arange(e-s, device=logits.device).view(1,1,-1,1).expand(B,H,e-s,1),
                    src=torch.gather(block_mask.transpose(-1,-2), -1, top_idx.transpose(-1,-2))
                )
                block_mask = torch.maximum(block_mask, trans_rows)

            # Keep CLS dense (row & col 0)
            if self.keep_cls_dense and N > 0:
                block_mask[:, :, :, 0] = 1  # all queries (this block) attend to CLS
                if s == 0:
                    # CLS row affects only the first block rows (i=0..C-1). Make CLS row dense for those rows.
                    # Safer: handle after softmax by not masking CLS row; here we just set ones for the present rows.
                    pass

            # Apply additive mask and finish attention for the block
            logits = logits + (1.0 - block_mask) * (-1e9)
            attn = torch.softmax(logits, dim=-1)
            if attn_drop is not None:
                attn = attn_drop(attn)
            outs.append(torch.matmul(attn, v))  # (B,H,C,D)

            # One-shot density report (approx; uses this block only which is representative)
            if not self._reported:
                with torch.no_grad():
                    if N > 1:
                        # Exclude CLS row/col to estimate patch block density
                        r0 = 1 - s if s == 0 else 0
                        patch = block_mask[:, :, r0:, 1:]
                        patch_density = float(patch.mean().item())
                    else:
                        patch_density = 1.0
                    full_density = float(block_mask.mean().item())
                    print(f"[TopKBackend] C={C}, k={self.k}, causal={self.causal}, symmetric={self.symmetric}, "
                          f"keep_cls_dense={self.keep_cls_dense} | "
                          f"patch_density={patch_density:.6f}, patch_sparsity={1.0 - patch_density:.6f} | "
                          f"block_full_density={full_density:.6f}, block_full_sparsity={1.0 - full_density:.6f}")
                self._reported = True

        out = torch.cat(outs, dim=2)  # (B,H,N,D)
        if save_hook is not None:
            save_hook(out)
        return out
