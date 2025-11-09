# image_classification/core/masks/sparse.py
import os
import torch
from .base import AttentionMask
from ..registries import register_mask
from sparse_attention import get_attn_mask as sa_get_attn_mask


@register_mask("sparse")
class SparseMask(AttentionMask):
    """
    Patterns:
      - "local":   sliding window band (uses sa_get_attn_mask)
      - "strided": every l-th token (uses sa_get_attn_mask)
      - "fixed":   block-local + last-c positions of EVERY previous block (paper's fixed pattern)

    Returns a mask with shape (1, H, N, N); N includes the class token at index 0.
    For ViT tasks use symmetric=True (bidirectional). For causal AR, set symmetric=False.

    Fixed pattern knobs:
      - local_attn_ctx (l): block size
      - fixed_c (c):        summary width in each block, clamped to 0..l
      - per_head_distinct:  if True, split the c summary band across heads (distinct subbands)
    """

    def __init__(
        self,
        pattern: str = "local",
        local_attn_ctx: int = None,
        symmetric: bool = True,
        fixed_c: int = None,
        per_head_distinct: bool = False,
    ):
        self.pattern = (pattern or "local").lower()
        self.local_attn_ctx = local_attn_ctx
        self.symmetric = symmetric

        # fixed-specific
        self.fixed_c = fixed_c
        self.per_head_distinct = per_head_distinct

        # one-shot logging/plotting guards
        self._plotted = False
        self._reported = False

    def __call__(self, attn, *, N=None, num_heads=None, device=None, **kwargs):
        """
        Build an attention mask of shape (1, H, N, N) with 1=keep, 0=mask.
        N includes a [CLS] token; patch tokens are indices [1:].
        """
        if N is None or num_heads is None:
            raise ValueError("[SparseMask] Expected N and num_heads in kwargs")

        # number of patch tokens (exclude class token)
        n = int(N) - 1
        if n <= 0:
            raise ValueError(f"[SparseMask] Invalid N={N}; must be >= 2 when class token is present")

        dtype = attn.dtype
        device = device if device is not None else attn.device

        # ---------------------------------------------------------------------
        # LOCAL / STRIDED via helper
        # ---------------------------------------------------------------------
        if self.pattern in ("local", "strided"):
            lac = self.local_attn_ctx
            if lac is None:
                lac = min(64, n)  # sensible default
                self.local_attn_ctx = lac

            base = sa_get_attn_mask(n, self.pattern, lac).to(device=device, dtype=dtype)  # (1,1,n,n)

            if self.symmetric:
                base = torch.maximum(base, base.transpose(-1, -2))  # bidirectional

            base = base.expand(1, num_heads, n, n)  # (1,H,n,n)

            full = torch.ones((1, num_heads, N, N), device=device, dtype=dtype)
            full[:, :, 1:, 1:] = base

            self._maybe_report_and_plot(full, base)
            return full

        # ---------------------------------------------------------------------
        # FIXED (paper): local block + last-c of EVERY previous block
        # ---------------------------------------------------------------------
        if self.pattern == "fixed":
            # block size l
            l = self.local_attn_ctx
            if l is None or l <= 0:
                l = min(128, max(1, n))
            l = int(max(1, min(l, n - 1)))  # avoid l >= n fully filling mask
            self.local_attn_ctx = l

            # summary width c, allow 0..l
            c = self.fixed_c
            if c is None:
                c = min(16, l)
            c = int(max(0, min(c, l)))
            self.fixed_c = c

            # number of blocks
            B = (n + l - 1) // l

            # per-head mask: (1, H, n, n)
            base = torch.zeros((1, num_heads, n, n), device=device, dtype=dtype)

            # 1) local block attention
            for b in range(B):
                s = b * l
                e = min((b + 1) * l, n)
                base[:, :, s:e, s:e] = 1.0

            # 2) summaries: last-c columns of EVERY previous block (pb = 0..b-1)
            if c > 0:
                if not self.per_head_distinct:
                    # All heads share the same summary band
                    for b in range(1, B):
                        qs = slice(b * l, min((b + 1) * l, n))
                        for pb in range(0, b):
                            ps = pb * l
                            pe = min((pb + 1) * l, n)
                            summ_s = max(ps, pe - c)
                            if summ_s < pe:
                                base[:, :, qs, summ_s:pe] = 1.0
                else:
                    # Distinct sub-bands per head within the last-c region
                    sizes = self._split_sizes(c, num_heads)  # lengths sum to c
                    offsets = [sum(sizes[:h]) for h in range(num_heads)]
                    for b in range(1, B):
                        qs = slice(b * l, min((b + 1) * l, n))
                        for pb in range(0, b):
                            ps = pb * l
                            pe = min((pb + 1) * l, n)
                            summ_s = max(ps, pe - c)
                            if summ_s < pe:
                                for h in range(num_heads):
                                    length_h = sizes[h]
                                    if length_h <= 0:
                                        continue
                                    start_h = summ_s + offsets[h]
                                    end_h = min(start_h + length_h, pe)
                                    if start_h < end_h:
                                        base[:, h:h+1, qs, start_h:end_h] = 1.0

            # 3) symmetric (ViT) or keep as causal-style (no future summaries)
            if self.symmetric:
                base = torch.maximum(base, base.transpose(-1, -2))

            # 4) compose full mask with [CLS] passthrough
            full = torch.ones((1, num_heads, N, N), device=device, dtype=dtype)
            full[:, :, 1:, 1:] = base

            self._maybe_report_and_plot(full, base)
            return full

        raise ValueError(f"[SparseMask] Unknown pattern='{self.pattern}'. Expected one of: local, strided, fixed.")

    # ----------------------------
    # helpers
    # ----------------------------
    @staticmethod
    def _split_sizes(total, parts):
        """Split 'total' into 'parts' non-negative integers that sum to total, as evenly as possible."""
        if parts <= 0:
            return []
        base = total // parts
        rem = total % parts
        return [base + (1 if i < rem else 0) for i in range(parts)]

    def _maybe_report_and_plot(self, full, patch_only):
        # one-time sparsity report (both patch-only and full)
        if not self._reported:
            with torch.no_grad():
                patch_density = float(patch_only.mean().item())  # (1,H,n,n)
                patch_sparsity = 1.0 - patch_density
                full_density = float(full.mean().item())          # (1,H,N,N) incl. CLS row/col
                full_sparsity = 1.0 - full_density
                print(
                    "[SparseMask] "
                    f"pattern={self.pattern}, local_attn_ctx={self.local_attn_ctx}, "
                    f"fixed_c={self.fixed_c}, per_head_distinct={self.per_head_distinct}, "
                    f"symmetric={self.symmetric} | "
                    f"patch_density={patch_density:.6f}, patch_sparsity={patch_sparsity:.6f} | "
                    f"full_density={full_density:.6f}, full_sparsity={full_sparsity:.6f}"
                )
            self._reported = True

        # one-time mask plot for all heads (optional)
        if not self._plotted:
            try:
                from utils.plot import plot_attention_mask_for_all_heads
                from configs import config
                output_dir = getattr(config, "output_dir", "image_classification/plots")
                os.makedirs(output_dir, exist_ok=True)
                plot_attention_mask_for_all_heads(full, output_dir)
                print(f"[SparseMask] Saved mask PNG to: {output_dir}")
            except Exception as e:
                print(f"[SparseMask] plot skipped: {e}")
            self._plotted = True
