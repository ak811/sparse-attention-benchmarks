# image_classification/core/backends/linformer.py
import os
import torch
import torch.nn as nn

from .base import AttentionBackend
from ..registries import register_backend


class LinformerProjector(nn.Module):
    def __init__(
        self,
        max_seq_len,
        r,
        num_heads,
        shared_heads: bool = True,
        share_kv: bool = True,
        separate_cls: bool = True,
        dtype=torch.float32,
    ):
        super().__init__()
        self.r = int(r)
        self.num_heads = int(num_heads)
        self.shared_heads = bool(shared_heads)
        self.share_kv = bool(share_kv)
        self.separate_cls = bool(separate_cls)

        # Shape of projection matrices E_k and E_v
        shape = (max_seq_len - 1, self.r) if self.shared_heads else (self.num_heads, max_seq_len - 1, self.r)

        def init_E():
            w = torch.empty(shape, dtype=dtype)
            nn.init.xavier_uniform_(w)
            return nn.Parameter(w)

        self.E_k = init_E()
        self.E_v = self.E_k if self.share_kv else init_E()

    def forward(self, k, v):
        B, H, N, Dh = k.shape

        if self.separate_cls:
            k_cls, k_tok = k[:, :, :1], k[:, :, 1:]
            v_cls, v_tok = v[:, :, :1], v[:, :, 1:]
            n = N - 1
        else:
            k_tok, v_tok, n = k, v, N

        if self.shared_heads:
            E_k, E_v = self.E_k[:n], self.E_v[:n]
            k_r = torch.einsum("b h n d, n r -> b h r d", k_tok, E_k)
            v_r = torch.einsum("b h n d, n r -> b h r d", v_tok, E_v)
        else:
            E_k, E_v = self.E_k[:, :n], self.E_v[:, :n]
            k_r = torch.einsum("b h n d, h n r -> b h r d", k_tok, E_k)
            v_r = torch.einsum("b h n d, h n r -> b h r d", v_tok, E_v)

        if self.separate_cls:
            k_r = torch.cat([k[:, :, :1], k_r], dim=2)
            v_r = torch.cat([v[:, :, :1], v_r], dim=2)

        return k_r, v_r


@register_backend("linformer")
class LinformerBackend(AttentionBackend):
    """
    Linformer backend with runtime device/dtype syncing for the projector.
    Adds one-time density/sparsity diagnostic print and visualization.
    """

    def __init__(self, projector: LinformerProjector, plot_once: bool = True):
        self.projector = projector
        self._synced_once = False
        self._reported = False
        self._plotted = False
        self.plot_once = plot_once

    def _sync_projector_device_dtype(self, ref_tensor: torch.Tensor):
        dev = ref_tensor.device
        dtype = ref_tensor.dtype
        if next(self.projector.parameters()).device != dev:
            self.projector.to(dev)
        try:
            self.projector.to(dtype=dtype)
        except TypeError:
            for p in self.projector.parameters():
                p.data = p.data.to(dtype)
        if not self._synced_once:
            print(f"[LinformerBackend] synced projector to device={dev}, dtype={dtype}")
            self._synced_once = True

    def _maybe_report_and_plot(self, k):
        if self._reported:
            return
        with torch.no_grad():
            N = k.shape[2]
            r = int(getattr(self.projector, "r", 0))
            separate_cls = bool(getattr(self.projector, "separate_cls", True))
            if N > 0 and r >= 0:
                N_patch = N - 1 if separate_cls else N
                patch_density = (float(r) / float(N_patch)) if N_patch > 0 else 1.0
                patch_sparsity = 1.0 - patch_density
                full_eff = r + (1 if separate_cls else 0)
                full_density = float(full_eff) / float(N)
                full_sparsity = 1.0 - full_density
                print(
                    f"[LinformerBackend] r={r}, separate_cls={separate_cls} | "
                    f"patch_density={patch_density:.6f}, patch_sparsity={patch_sparsity:.6f} | "
                    f"full_density={full_density:.6f}, full_sparsity={full_sparsity:.6f}"
                )

                # ---- One-time visualization ----
                if self.plot_once and not self._plotted:
                    try:
                        # Approximate mask: each row connects to r columns (+CLS if separate_cls)
                        mask = torch.zeros((1, self.projector.num_heads, N, N), device=k.device)
                        if separate_cls:
                            mask[:, :, 0, :] = 1
                            mask[:, :, :, 0] = 1
                        # simulate projection: all tokens map into r slots
                        if separate_cls:
                            cols = torch.arange(1, r + 1)
                        else:
                            cols = torch.arange(0, r)
                        mask[:, :, :, cols] = 1

                        from utils.plot import plot_attention_mask_for_all_heads
                        from configs import config
                        save_dir = os.path.join(config.output_dir, "mask_viz")
                        os.makedirs(save_dir, exist_ok=True)
                        plot_attention_mask_for_all_heads(mask, save_dir)
                        print(f"[LinformerBackend] Saved mask PNG to: {save_dir}")
                    except Exception as e:
                        print(f"[LinformerBackend] plot skipped: {e}")
                    self._plotted = True
        self._reported = True

    def __call__(self, q, k, v, *, attn_drop=None, save_hook=None, **kwargs):
        self._sync_projector_device_dtype(k)
        k_r, v_r = self.projector(k, v)
        attn = q @ k_r.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        if attn_drop is not None:
            attn = attn_drop(attn)
        x = attn @ v_r
        if save_hook is not None:
            save_hook(x)
        self._maybe_report_and_plot(k)
        return x
