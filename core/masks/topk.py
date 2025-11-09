import torch
import os
from .base import AttentionMask
from ..registries import register_mask
from configs import config  # <-- use experiment config


@register_mask("topk")
class TopKMask(AttentionMask):
    """
    Dynamic, content-based mask: per (B,H,row) keep the top-k logits along the key dim.
    Options:
      k (int): number of keys to keep per query row
      causal (bool): forbid j > i BEFORE topk
      symmetric (bool): OR with transpose after selection (increases kept entries)
      keep_cls_dense (bool): keep CLS row & col = 1s
    """
    def __init__(self, k=3, causal=False, symmetric=False, keep_cls_dense=True):
        self.k = int(k)
        self.causal = bool(causal)
        self.symmetric = bool(symmetric)
        self.keep_cls_dense = bool(keep_cls_dense)
        self._reported = False

    @torch.no_grad()
    def __call__(self, attn, *, N=None, num_heads=None, device=None, **kwargs):
        """
        attn: attention *logits* (B,H,N,N), BEFORE softmax (as provided by backend)
        return: binary mask (B,H,N,N) with 1=keep, 0=mask
        """
        B, H, Nlog, _ = attn.shape
        N = Nlog if (N is None) else N
        k_eff = min(self.k, N)

        logits = attn
        if self.causal:
            i = torch.arange(N, device=logits.device)
            causal = (i.view(1,1,N,1) >= i.view(1,1,1,N))  # (1,1,N,N)
            logits = logits.masked_fill(~causal, float('-inf'))

        # Row-wise topk per (B,H,row)
        top_vals, top_idx = torch.topk(logits, k=k_eff, dim=-1)  # (B,H,N,k)
        mask = torch.zeros_like(logits, dtype=logits.dtype)       # (B,H,N,N)
        mask.scatter_(dim=-1, index=top_idx, src=torch.ones_like(top_vals, dtype=logits.dtype))

        if self.symmetric:
            mask = torch.maximum(mask, mask.transpose(-1, -2))

        if self.keep_cls_dense and N > 0:
            mask[:, :, 0, :] = 1
            mask[:, :, :, 0] = 1

        if not self._reported:
            with torch.no_grad():
                if N > 1:
                    patch = mask[:, :, 1:, 1:]
                    patch_density = float(patch.mean().item())
                else:
                    patch_density = 1.0
                full_density = float(mask.mean().item())
                print(
                    f"[TopKMask] k={self.k}, causal={self.causal}, symmetric={self.symmetric}, "
                    f"keep_cls_dense={self.keep_cls_dense} | "
                    f"patch_density={patch_density:.6f}, patch_sparsity={1.0 - patch_density:.6f} | "
                    f"full_density={full_density:.6f}, full_sparsity={1.0 - full_density:.6f}"
                )

                # --- Save mask visualization once ---
                try:
                    import matplotlib.pyplot as plt
                    save_dir = os.path.join(config.output_dir, "mask_viz")
                    os.makedirs(save_dir, exist_ok=True)

                    mask_img = mask[0, 0].detach().cpu().numpy()  # batch=0, head=0
                    plt.imshow(mask_img, cmap="gray")
                    plt.title(f"TopKMask k={self.k}")

                    save_path = os.path.join(save_dir, f"topk_mask_k{self.k}.png")
                    plt.savefig(save_path)
                    plt.close()

                    print(f"[TopKMask] Saved mask visualization to {save_path}")
                except Exception as e:
                    print(f"[TopKMask] Warning: could not save mask viz ({e})")

            self._reported = True

        return mask
