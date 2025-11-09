# image_classification/core/masks/bigbird.py
import os
from typing import Optional, Sequence

import torch
from .base import AttentionMask
from ..registries import register_mask


@register_mask("bigbird")
class BigBirdMask(AttentionMask):
    """
    BigBird-style sparse attention mask with (global + local band + random) connectivity.

    Returns a float mask with shape (B, H, N, N) where 1 = keep, 0 = mask.
    """

    def __init__(
        self,
        window: int = 1,
        num_random: int = 1,
        num_global: int = 1,
        global_indices: Optional[Sequence[int]] = None,
        symmetric: bool = False,
        keep_cls_dense: bool = True,
        per_head_same: bool = False,
        per_batch_same: bool = False,
        seed: Optional[int] = 1234,
        reseed_each_epoch: bool = True,
        plot_once: bool = True,
    ):
        self.w = int(max(0, window))
        self.r = int(max(0, num_random))
        self.num_global = int(max(0, num_global))
        self.global_indices = None if global_indices is None else list(map(int, global_indices))
        self.symmetric = bool(symmetric)
        self.keep_cls_dense = bool(keep_cls_dense)
        self.per_head_same = bool(per_head_same)
        self.per_batch_same = bool(per_batch_same)
        self.seed = seed
        self.reseed_each_epoch = bool(reseed_each_epoch)
        self.plot_once = bool(plot_once)

        self._reported = False
        self._plotted = False

    def _make_generator(self, device, estep):
        if self.seed is None:
            return None
        seed_eff = int(self.seed)
        if self.reseed_each_epoch and isinstance(estep, (tuple, list)) and len(estep) >= 1:
            try:
                seed_eff = int(self.seed + int(estep[0]))
            except Exception:
                pass
        gen = torch.Generator(device=device)
        gen.manual_seed(seed_eff)
        return gen

    @staticmethod
    def _band_mask(N: int, w: int, device, dtype_bool):
        if w <= 0:
            return torch.zeros((N, N), device=device, dtype=dtype_bool)
        i = torch.arange(N, device=device)
        j = torch.arange(N, device=device)
        return (i.view(-1, 1) - j.view(1, -1)).abs().le(w)

    @torch.no_grad()
    def __call__(self, attn, *, N=None, num_heads=None, estep=None, device=None, **kwargs):
        B, H, N_attn, _ = attn.shape
        N = int(N_attn if N is None else N)
        device = device or attn.device
        dtype_out = attn.dtype

        # ---- Base keep mask (N,N): local band ----
        keep = self._band_mask(N, self.w, device, torch.bool)

        # ---- Globals ----
        if self.global_indices is not None:
            g_idx = torch.tensor(sorted({i for i in self.global_indices if 0 <= i < N}),
                                 device=device, dtype=torch.long)
        else:
            g = min(self.num_global, N)
            g_idx = torch.arange(g, device=device, dtype=torch.long) if g > 0 else None

        if g_idx is not None and g_idx.numel() > 0:
            keep[g_idx, :] = True
            keep[:, g_idx] = True

        # ---- CLS dense ----
        if self.keep_cls_dense and N > 0:
            keep[0, :] = True
            keep[:, 0] = True

        # ---- Random connections (vectorized) ----
        if self.r > 0:
            allowed = ~keep
            eye = torch.eye(N, device=device, dtype=torch.bool)
            allowed &= ~eye

            # noise shape depends on sharing options
            if self.per_batch_same and self.per_head_same:
                noise = torch.empty((1, 1, N, N), device=device).uniform_(0, 1)
            elif self.per_batch_same and not self.per_head_same:
                noise = torch.empty((1, H, N, N), device=device).uniform_(0, 1)
            elif (not self.per_batch_same) and self.per_head_same:
                noise = torch.empty((B, 1, N, N), device=device).uniform_(0, 1)
            else:
                noise = torch.empty((B, H, N, N), device=device).uniform_(0, 1)

            noise = noise.expand(B, H, N, N).clone()
            noise = noise.to(dtype_out)

            allowed_bh = allowed.view(1, 1, N, N).expand(B, H, N, N)
            neg_inf = torch.finfo(dtype_out).min
            noise = torch.where(allowed_bh, noise, torch.full_like(noise, neg_inf))

            _, idx = torch.topk(noise, k=self.r, dim=-1)  # (B,H,N,r)
            rand_keep = torch.zeros((B, H, N, N), device=device, dtype=torch.bool)
            row_idx = torch.arange(N, device=device).view(1, 1, N, 1).expand(B, H, N, self.r)
            rand_keep.scatter_(dim=-1, index=idx, src=torch.ones_like(idx, dtype=torch.bool))

            keep_bh = keep.view(1, 1, N, N).expand(B, H, N, N) | rand_keep
        else:
            keep_bh = keep.view(1, 1, N, N).expand(B, H, N, N)

        if self.symmetric:
            keep_bh = keep_bh | keep_bh.transpose(-1, -2)

        mask = keep_bh.to(dtype_out)

        # ---- One-time sparsity report ----
        if not self._reported:
            with torch.no_grad():
                if N > 1:
                    patch = mask[:, :, 1:, 1:]
                    patch_density = float(patch.mean().item())
                else:
                    patch_density = 1.0
                full_density = float(mask.mean().item())
                print(
                    f"[BigBirdMask] w={self.w}, r={self.r}, num_global={self.num_global}, "
                    f"symmetric={self.symmetric}, keep_cls_dense={self.keep_cls_dense}, "
                    f"per_head_same={self.per_head_same}, per_batch_same={self.per_batch_same} | "
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
                print(f"[BigBirdMask] Saved mask PNG to: {save_dir}")
            except Exception as e:
                print(f"[BigBirdMask] plot skipped: {e}")
            self._plotted = True

        return mask
