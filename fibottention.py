# Copyright (c) Charlotte-CharMLab at University of North Carolina at Charlotte.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# https://github.com/Charlotte-CharMLab/Fibottention
# --------------------------------------------------------

import torch
import math
import random
from sparse_attention import get_attn_mask as sa_get_attn_mask

# ---------------- existing utilities ----------------

def get_mask_attn_wythoff(H, N, is_modified, is_shuffled, depth_id,
                          add_class_token=True, device=torch.device('cpu'),
                          dtype=torch.float32):
    headindices = generate_head_indices(N=N, h=H, wmin=5, is_modified=is_modified)
    if is_shuffled:
        headindices = shuffle(depth_id, headindices)

    mask = torch.zeros((H, N, N), device=device, dtype=dtype)
    for h in range(H):
        for i in headindices[h]:
            idx = torch.arange(max(-i, 0), min(N, N - i), device=device)
            mask[h, idx, idx + i] = 1
            idx = torch.arange(max(i, 0), min(N, N + i), device=device)
            mask[h, idx, idx - i] = 1

    if add_class_token:
        ext = torch.ones((H, N + 1, N + 1), device=device, dtype=dtype)
        ext[:, 1:, 1:] = mask
        return ext
    return mask

def generate_head_indices(N, h, wmin, is_modified):
    wmax = N // 3
    headindices = [[] for _ in range(h)]
    phi = (1 + math.sqrt(5)) / 2
    for i in range(1, h + 1):
        a = int(math.floor(math.floor(i * phi) * phi))
        b = int(math.floor(math.floor(i * phi) * (phi ** 2)))
        w = wmin + int((wmax - wmin) / (h - 1) * (i - 1))
        if is_modified:
            b_m = b - a
            a_m = a - b_m
            seq = get_fibonacci(a_m, b_m, w)
        else:
            seq = get_fibonacci(a, b, w)
        headindices[i - 1] = seq
    return [torch.tensor(seq, dtype=torch.int64) for seq in headindices]

def get_fibonacci(a, b, w):
    fib = [a, b]
    while fib[-1] <= w:
        fib.append(fib[-1] + fib[-2])
    return fib[:-1]

def shuffle(seed, array_of_sets):
    random.seed(seed)
    arr = list(array_of_sets)
    random.shuffle(arr)
    return arr

# ---------------- new: caching + dispatcher ----------------

# Global cache and one-shot plot flag (module-level)
__fib_cache = {"tensor": None, "last_epoch": None}
__mask_plotted_once = False

def __build_fibottention_cache(*, epoch, N, num_heads, depth_id, device, dtype, add_class_token=True):
    """
    Rebuild cached mask only when epoch changes (or cache is empty).
    Returns the (1, H, N, N) cache tensor.
    """
    global __fib_cache
    cache, last_epoch = __fib_cache["tensor"], __fib_cache["last_epoch"]

    if cache is None or last_epoch != epoch:
        m = get_mask_attn_wythoff(
            H=num_heads,
            N=N - 1,              # exclude class token; we add it back next
            is_modified=False,
            is_shuffled=True,
            depth_id=depth_id,
            add_class_token=add_class_token,
            device=device,
            dtype=dtype,
        )
        cache = m.unsqueeze(0)    # → (1, H, N, N) when add_class_token=True
        __fib_cache = {"tensor": cache, "last_epoch": epoch}
    return cache

def __plot_fibottention_mask_once(cache):
    """
    Always plot on first use of Fibottention. Requires the project-level
    plot function and config to be importable.
    """
    global __mask_plotted_once
    if __mask_plotted_once:
        return
    try:
        # Adjust these imports to match your project layout
        from utils.plot import plot_attention_mask_for_all_heads
        from configs import config
    except Exception as e:
        # If plotting must happen, fail fast with a clear error
        raise RuntimeError(
            "Fibottention plotting is required but the plotting utilities "
            "could not be imported. Ensure `plot_attention_mask_for_all_heads` "
            "and `config.output_dir` are available."
        ) from e

    plot_attention_mask_for_all_heads(cache, config.output_dir)
    __mask_plotted_once = True

def apply_mask(attn: torch.Tensor, mode: str, **kwargs) -> torch.Tensor:
    """
    Apply a masking scheme to attention scores (B,H,N,N).

    Known modes:
      - "none"
      - "fibottention"
      - "sparse_attention"  (no cache; mask built on the fly)
    """
    if mode is None or mode.lower() == "none":
        return attn

    if mode.lower() == "fibottention":
        required = ["epoch", "N", "num_heads", "depth_id", "device"]
        missing = [k for k in required if k not in kwargs]
        if missing:
            raise ValueError(f"Missing kwargs for fibottention: {missing}")

        cache = get_mask_attn_wythoff(
            H=kwargs["num_heads"],
            N=kwargs["N"] - 1,                 # exclude class token; we re-add below
            is_modified=False,
            is_shuffled=True,
            depth_id=kwargs["depth_id"],
            add_class_token=True,
            device=kwargs["device"],
            dtype=attn.dtype,
        ).unsqueeze(0)                          # (1,H,N,N)
        # First-use plotting stays as in your original:
        # (wrap in try/except if desired)
        try:
            from utils.plot import plot_attention_mask_for_all_heads
            from configs import config
            plot_attention_mask_for_all_heads(cache, config.output_dir)
        except Exception:
            pass
        return attn * cache

    if mode.lower() == "sparse_attention":
        # minimal required args
        required = ["N", "num_heads", "device"]
        missing = [k for k in required if k not in kwargs]
        if missing:
            raise ValueError(f"Missing kwargs for sparse_attention: {missing}")

        N = kwargs["N"]
        H = kwargs["num_heads"]
        device = kwargs["device"]

        pattern = kwargs.get("pattern", "local")        # "all" | "local" | "strided"
        local_attn_ctx = kwargs.get("local_attn_ctx", None)  # int or None
        symmetric = kwargs.get("symmetric", True)       # ViT default: bidirectional

        # Build mask for tokens *without* cls:
        n_wo_cls = N - 1
        base = sa_get_attn_mask(n_wo_cls, pattern, local_attn_ctx).to(
            device=device, dtype=attn.dtype
        )                                               # (1,1,n,n)

        if symmetric:
            base = torch.maximum(base, base.transpose(-1, -2))

        # Expand to heads and re-add class token as fully connected
        base = base.expand(1, H, n_wo_cls, n_wo_cls)    # (1,H,n,n)
        full = torch.ones((1, H, N, N), device=device, dtype=attn.dtype)
        full[:, :, 1:, 1:] = base                       # cls token rows/cols are 1s

        return attn * full

    raise ValueError(f"Unknown mask mode: {mode}")
