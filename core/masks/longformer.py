# image_classification/core/masks/longformer.py
import os
from typing import Optional, Sequence, Union

import torch

from .base import AttentionMask
from ..registries import register_mask


@register_mask("longformer")
class LongformerMask(AttentionMask):
    """
    Longformer-style mask: local sliding window (optionally dilated) + global tokens.

    Returns a float mask with shape (B, H, N, N) where 1 = keep, 0 = mask.

    Args
    ----
    window : int
        Half-window size 'w'. Each token i keeps keys j where |i-j| <= w * dilation
        and (|i-j| % dilation == 0). If dilation=1, this is the standard sliding window.
    dilation : int
        Dilation factor 'd' for the sliding window.
    per_head_dilation : Optional[Sequence[int]]
        Optional per-head dilation factors (length H). If provided, overrides 'dilation'
        per head; useful to mix local and longer context heads.
    num_global : int
        Number of initial tokens to mark as global (rows/cols fully kept). If
        'global_indices' is provided, that overrides this.
    global_indices : Optional[Sequence[int]]
        Explicit list of token indices to mark as global (e.g., [0] for CLS).
    symmetric : bool
        If True, OR with transpose for bidirectional attention (typical for ViT).
    keep_cls_dense : bool
        If True, force CLS (index 0) row/col to 1 in addition to globals.
    include_self : bool
        If False, the main diagonal is excluded from the sliding window (matches your
        previous "without main diagonal" behavior). Set True to include self.
    seed : Optional[int]
        Base RNG seed to stabilize any randomness if later extended; not used here
        (mask is deterministic), retained for API symmetry.
    reseed_each_epoch : bool
        Kept for API symmetry (not used here).
    plot_once : bool
        Best-effort one-time plot to config.output_dir/mask_viz.
    """

    def __init__(
        self,
        window: int = 1,
        dilation: int = 1,
        per_head_dilation: Optional[Sequence[int]] = None,
        num_global: int = 1,
        global_indices: Optional[Sequence[int]] = None,
        symmetric: bool = True,
        keep_cls_dense: bool = True,
        include_self: bool = False,
        seed: Optional[int] = None,
        reseed_each_epoch: bool = True,
        plot_once: bool = True,
    ):
        self.w = int(max(0, window))
        self.d = int(max(1, dilation))
        self.per_head_dilation = None if per_head_dilation is None else [max(1, int(x)) for x in per_head_dilation]
        self.num_global = int(max(0, num_global))
        self.global_indices = None if global_indices is None else list({int(i) for i in global_indices if i is not None})
        self.symmetric = bool(symmetric)
        self.keep_cls_dense = bool(keep_cls_dense)
        self.include_self = bool(include_self)

        self.seed = seed
        self.reseed_each_epoch = bool(reseed_each_epoch)
        self.plot_once = bool(plot_once)

        self._reported = False
        self._plotted = False

    @staticmethod
    def _sliding_mask(N: int, w: int, d: int, *, include_self: bool, device, dtype=torch.bool):
        """
        Build (N,N) bool mask with diagonals at offsets k*d for k=1..w (and k=0 if include_self).
        Vectorized over diagonals (loop only over w, typically very small).
        """
        if w <= 0:
            return torch.zeros((N, N), device=device, dtype=dtype)

        M = torch.zeros((N, N), device=device, dtype=dtype)
        # Decide which offsets to include
        start = 0 if include_self else 1
        for k in range(start, w + 1):
            off = k * d
            if off == 0:
                # main diagonal
                idx = torch.arange(N, device=device)
                M[idx, idx] = True
            else:
                # +off
                i = torch.arange(0, max(0, N - off), device=device)
                j = i + off
                if i.numel() > 0:
                    M[i, j] = True
                # -off
                j = torch.arange(0, max(0, N - off), device=device)
                i = j + off
                if j.numel() > 0:
                    M[i, j] = True
        return M

    @torch.no_grad()
    def __call__(self, attn, *, N: Optional[int] = None, num_heads: Optional[int] = None,
                 estep=None, device: Optional[Union[str, torch.device]] = None, **kwargs):
        """
        attn: (B, H, N, N) attention logits (only shape/device/dtype used)
        return: (B, H, N, N) float mask with 1=keep, 0=mask
        """
        B, H, N_attn, _ = attn.shape
        N = int(N_attn if N is None else N)
        device = device or attn.device
        dtype_out = attn.dtype

        # ---- Globals ----
        if self.global_indices is not None:
            g_idx = sorted({i for i in self.global_indices if 0 <= i < N})
        else:
            g = min(self.num_global, N)
            g_idx = list(range(g)) if g > 0 else []

        # ---- Per-head dilation config ----
        if self.per_head_dilation is not None:
            if len(self.per_head_dilation) != H:
                raise ValueError(f"[LongformerMask] per_head_dilation length {len(self.per_head_dilation)} "
                                 f"does not match num_heads={H}")
            per_head_d = self.per_head_dilation
        else:
            per_head_d = [self.d] * H

        # ---- Build per-head (N,N) masks and stack -> (H,N,N) ----
        head_masks = []
        for h in range(H):
            Mh = self._sliding_mask(N, self.w, per_head_d[h],
                                    include_self=self.include_self, device=device, dtype=torch.bool)

            # Globals: rows/cols fully kept
            if g_idx:
                gi = torch.tensor(g_idx, device=device, dtype=torch.long)
                Mh[gi, :] = True
                Mh[:, gi] = True

            # CLS dense
            if self.keep_cls_dense and N > 0:
                Mh[0, :] = True
                Mh[:, 0] = True

            head_masks.append(Mh)

        HN = torch.stack(head_masks, dim=0)  # (H,N,N)

        # Symmetric (useful if include_self=False but still want bidirectional band)
        if self.symmetric:
            HN = HN | HN.transpose(-1, -2)

        # Expand to (B,H,N,N)
        mask_bool = HN.unsqueeze(0).expand(B, H, N, N)

        mask = mask_bool.to(dtype_out)

        # ---- One-time sparsity print ----
        if not self._reported:
            with torch.no_grad():
                if N > 1:
                    patch = mask[:, :, 1:, 1:]
                    patch_density = float(patch.mean().item())
                else:
                    patch_density = 1.0
                full_density = float(mask.mean().item())
                print(
                    f"[LongformerMask] w={self.w}, dilation={self.d}, per_head_dilation={self.per_head_dilation is not None}, "
                    f"num_global={'len='+str(len(g_idx)) if g_idx else 0}, symmetric={self.symmetric}, "
                    f"keep_cls_dense={self.keep_cls_dense}, include_self={self.include_self} | "
                    f"patch_density={patch_density:.6f}, patch_sparsity={1.0 - patch_density:.6f} | "
                    f"full_density={full_density:.6f}, full_sparsity={1.0 - full_density:.6f}"
                )
            self._reported = True

        # ---- One-time plot ----
        if self.plot_once and (not self._plotted):
            try:
                from utils.plot import plot_attention_mask_for_all_heads
                from configs import config
                save_dir = os.path.join(config.output_dir, "mask_viz")
                os.makedirs(save_dir, exist_ok=True)
                plot_attention_mask_for_all_heads(mask[0:1], save_dir)
                print(f"[LongformerMask] Saved mask PNG to: {save_dir}")
            except Exception as e:
                print(f"[LongformerMask] plot skipped: {e}")
            self._plotted = True

        return mask
