# core/masks/random.py
import os
import random
import torch
from .base import AttentionMask
from ..registries import register_mask

from configs import config

@register_mask("random")
class RandomMask(AttentionMask):
    """
    Random (Bernoulli) attention mask over the (N x N) token grid.
    """

    def __init__(
        self,
        mask_ratio: float = 0.0,
        keep_ratio: float = None,
        causal: bool = False,
        symmetric: bool = False,
        keep_cls_dense: bool = True,
        per_head_same: bool = False,
        seed: int = None,
        reseed_each_epoch: bool = True,
    ):
        if keep_ratio is not None:
            mask_ratio = 1.0 - float(keep_ratio)
        self.mask_ratio = float(mask_ratio)
        if not (0.0 <= self.mask_ratio < 1.0):
            raise ValueError(f"[RandomMask] mask_ratio must be in [0,1), got {self.mask_ratio}")
        self.causal = bool(causal)
        self.symmetric = bool(symmetric)
        self.keep_cls_dense = bool(keep_cls_dense)
        self.per_head_same = bool(per_head_same)
        
        if seed is None or str(seed).lower() == "none":
            self.seed = None
        else:
            self.seed = int(seed)
            
        self.reseed_each_epoch = bool(reseed_each_epoch)

        self._reported = False
        self._plotted = False

    @torch.no_grad()
    def __call__(self, attn, *, N=None, num_heads=None, estep=None, device=None, **kwargs):
        """
        attn: (B, H, N, N) logits
        return: (B, H, N, N) float mask with 1=keep, 0=mask
        """
        B, H, N_, _ = attn.shape
        N = N_ if (N is None) else N
        device = device or attn.device
        dtype = attn.dtype

        keep_p = 1.0 - self.mask_ratio

        # --- FIX: Ensure dynamic non-repeating randomness when seed is None ---
        gen = torch.Generator(device=device)
        if self.seed is not None:
            epoch = None
            if isinstance(estep, (tuple, list)) and len(estep) >= 1:
                epoch = estep[0]
            seed = int(self.seed if (epoch is None or not self.reseed_each_epoch) else (self.seed + int(epoch)))
            gen.manual_seed(seed)
        else:
            # Pull a genuinely random seed from Python's system entropy pool 
            # so DataLoader subprocesses don't sync up with identical states.
            gen.manual_seed(random.randint(0, 2**31 - 1))

        # --- Sample Bernoulli mask(s) ---
        if self.per_head_same:
            base = (torch.rand((B, 1, N, N), device=device, generator=gen) < keep_p).to(dtype)
            mask = base.expand(B, H, N, N).contiguous()
        else:
            mask = (torch.rand((B, H, N, N), device=device, generator=gen) < keep_p).to(dtype)

        # --- Causal constraint ---
        if self.causal:
            i = torch.arange(N, device=device)
            causal = (i.view(1, 1, N, 1) >= i.view(1, 1, 1, N)).to(dtype)
            mask = mask * causal

        # --- Symmetry ---
        if self.symmetric:
            mask = torch.maximum(mask, mask.transpose(-1, -2))

        # --- Keep CLS row/col dense ---
        if self.keep_cls_dense and N > 0:
            mask[:, :, 0, :] = 1
            mask[:, :, :, 0] = 1

        # --- One-time sparsity/density report ---
        if not self._reported:
            with torch.no_grad():
                if N > 1:
                    patch = mask[:, :, 1:, 1:]
                    patch_density = float(patch.mean().item())
                else:
                    patch_density = 1.0
                full_density = float(mask.mean().item())
                print(
                    f"[RandomMask] mask_ratio={self.mask_ratio:.4f}, keep_p={keep_p:.4f}, "
                    f"causal={self.causal}, symmetric={self.symmetric}, keep_cls_dense={self.keep_cls_dense}, "
                    f"per_head_same={self.per_head_same} | "
                    f"patch_density={patch_density:.6f}, patch_sparsity={1.0 - patch_density:.6f} | "
                    f"full_density={full_density:.6f}, full_sparsity={1.0 - full_density:.6f}"
                )
            self._reported = True

        # --- One-time mask plot ---
        if not self._plotted:
            try:
                from utils.plot import plot_attention_mask_for_all_heads
                save_dir = os.path.join(config.output_dir, "mask_viz")
                os.makedirs(save_dir, exist_ok=True)
                plot_attention_mask_for_all_heads(mask[0:1], save_dir)
                print(f"[RandomMask] Saved mask PNG to: {save_dir}")
            except Exception as e:
                print(f"[RandomMask] plot skipped: {e}")
            self._plotted = True

        return mask