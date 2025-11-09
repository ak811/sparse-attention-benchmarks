# image_classification/core/backends/efficient_local.py
import os
import torch
from .base import AttentionBackend
from ..registries import register_backend


@register_backend("efficient_local")
class EfficientLocalBackend(AttentionBackend):
    """
    Block-local Efficient Attention:
      - Partition sequence (including CLS at index 0) into non-overlapping blocks of length L.
      - Apply Efficient Attention within each block only (no cross-block mixing).
      - Effective patch density ~= L / N.
      - Optionally keep CLS as global (full row/col dense).

    Diagnostics:
      - Prints one-time density/sparsity report.
      - Saves one-time PNG of block mask to config.output_dir/mask_viz.
    """

    def __init__(self, block_size: int = 4, keep_cls_global: bool = True, plot_once: bool = True):
        super().__init__()
        if block_size <= 0:
            raise ValueError(f"[EfficientLocalBackend] block_size must be >=1, got {block_size}")
        self.block_size = int(block_size)
        self.keep_cls_global = bool(keep_cls_global)
        self.plot_once = bool(plot_once)

        self._reported = False
        self._plotted = False

    def __call__(self, q, k, v, *, attn_drop=None, save_hook=None, **kwargs):
        """
        q, k, v: (B, H, N, D)
        returns: (B, H, N, D_out) with block-local efficient attention
        """
        B, H, N, D = q.shape
        device, dtype = q.device, q.dtype

        # Optionally separate CLS
        has_cls = (N > 0)
        if self.keep_cls_global and has_cls:
            q_cls, q_tok = q[:, :, :1, :], q[:, :, 1:, :]
            k_cls, k_tok = k[:, :, :1, :], k[:, :, 1:, :]
            v_cls, v_tok = v[:, :, :1, :], v[:, :, 1:, :]
        else:
            q_tok, k_tok, v_tok = q, k, v

        rho_q = torch.softmax(q_tok, dim=-1)       # (B,H,NT,D)
        rho_k = torch.softmax(k_tok, dim=-2)       # (B,H,NT,D)

        NT = rho_q.size(-2)
        L = min(self.block_size, max(1, NT)) if NT > 0 else 1
        n_blocks = (NT + L - 1) // L

        out_tok = torch.empty((B, H, NT, D), device=device, dtype=dtype)

        for b in range(n_blocks):
            s, e = b * L, min((b + 1) * L, NT)
            if s >= e:
                continue

            rho_k_blk = rho_k[:, :, s:e, :]      # (B,H,Lb,D)
            v_blk     = v_tok[:, :, s:e, :]      # (B,H,Lb,D)

            C_blk = torch.einsum('b h l d, b h l v -> b h d v', rho_k_blk, v_blk)
            rho_q_blk = rho_q[:, :, s:e, :]      # (B,H,Lb,D)
            y_blk = torch.einsum('b h l d, b h d v -> b h l v', rho_q_blk, C_blk)

            if attn_drop is not None:
                y_blk = attn_drop(y_blk)

            out_tok[:, :, s:e, :] = y_blk

        if self.keep_cls_global and has_cls:
            rho_q_all = torch.softmax(q, dim=-1)
            rho_k_all = torch.softmax(k, dim=-2)
            C_all = torch.einsum('b h n d, b h n v -> b h d v', rho_k_all, v)
            y_all = torch.einsum('b h n d, b h d v -> b h n v', rho_q_all, C_all)

            out = torch.empty_like(q)
            out[:, :, :1, :] = y_all[:, :, :1, :]
            out[:, :, 1:, :] = out_tok
        else:
            out = out_tok

        if save_hook is not None:
            save_hook(out)

        # ---- One-time diagnostics ----
        if not self._reported:
            with torch.no_grad():
                # Build mask once (for reporting/plotting)
                mask = torch.zeros((1, H, N, N), device=device, dtype=torch.float32)
                for b in range(n_blocks):
                    s, e = b * L, min((b + 1) * L, N)
                    if s >= e:
                        continue
                    mask[:, :, s:e, s:e] = 1.0
                if self.keep_cls_global and has_cls:
                    mask[:, :, 0, :] = 1.0
                    mask[:, :, :, 0] = 1.0

                patch = mask[:, :, 1:, 1:]
                patch_density = float(patch.mean().item()) if N > 1 else 1.0
                full_density = float(mask.mean().item())
                print(
                    f"[EfficientLocalBackend] block_size={self.block_size}, keep_cls_global={self.keep_cls_global} | "
                    f"patch_density={patch_density:.6f}, patch_sparsity={1.0 - patch_density:.6f} | "
                    f"full_density={full_density:.6f}, full_sparsity={1.0 - full_density:.6f}"
                )

                if self.plot_once and not self._plotted:
                    try:
                        from utils.plot import plot_attention_mask_for_all_heads
                        from configs import config
                        save_dir = os.path.join(config.output_dir, "mask_viz")
                        os.makedirs(save_dir, exist_ok=True)
                        plot_attention_mask_for_all_heads(mask, save_dir)
                        print(f"[EfficientLocalBackend] Saved mask PNG to: {save_dir}")
                    except Exception as e:
                        print(f"[EfficientLocalBackend] plot skipped: {e}")
                    self._plotted = True

            self._reported = True

        return out
