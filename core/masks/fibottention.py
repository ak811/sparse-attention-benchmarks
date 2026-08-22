import math, random, torch
from .base import AttentionMask
from ..registries import register_mask

_CACHE = {"tensor": None, "epoch": None}
_PLOTTED = False

@register_mask("fibottention")
class FibottentionMask(AttentionMask):
    def __init__(self, add_class_token=True, modified=False, shuffled=True):
        self.add_class_token = add_class_token
        self.modified = modified
        self.shuffled = shuffled
        self._reported_epoch = None  # for one-time-per-epoch sparsity print

    def __call__(self, attn, *, estep=None, N=None, num_heads=None, depth_id=None, device=None, **kwargs):
        epoch, _ = estep
        global _CACHE
        if _CACHE["tensor"] is None or _CACHE["epoch"] != epoch:
            m = _wythoff(
                num_heads, N - 1, self.modified, self.shuffled, depth_id,
                self.add_class_token, device, attn.dtype
            )
            _CACHE = {"tensor": m.unsqueeze(0), "epoch": epoch}
        _plot_once(_CACHE["tensor"])

        # one-time-per-epoch sparsity report
        if self._reported_epoch != epoch:
            with torch.no_grad():
                full_block = _CACHE["tensor"]
                patch_block = full_block[..., 1:, 1:] if self.add_class_token else full_block

                patch_density = float(patch_block.mean().item())
                full_density = float(full_block.mean().item())

            print(
                f"[FibottentionMask][epoch={epoch}] modified={self.modified}, shuffled={self.shuffled}, "
                f"add_class_token={self.add_class_token} | "
                f"patch_density={patch_density:.6f}, patch_sparsity={1.0 - patch_density:.6f} | "
                f"full_density={full_density:.6f}, full_sparsity={1.0 - full_density:.6f}"
            )
            self._reported_epoch = epoch

        return _CACHE["tensor"]

def _wythoff(H, N, is_modified, is_shuffled, depth_id, add_class_token, device, dtype):
    headindices = _head_indices(N, H, 5, is_modified)
    if is_shuffled:
        headindices = _shuffle(depth_id, headindices)
    mask = torch.zeros((H, N, N), device=device, dtype=dtype)
    for h in range(H):
        for i in headindices[h]:
            # main-diagonal bands (original behavior)
            idx = torch.arange(max(-i, 0), min(N, N - i), device=device)
            mask[h, idx, idx + i] = 1
            idx = torch.arange(max(i, 0), min(N, N + i), device=device)
            mask[h, idx, idx - i] = 1
    if add_class_token:
        ext = torch.ones((H, N + 1, N + 1), device=device, dtype=dtype)
        ext[:, 1:, 1:] = mask
        return ext
    return mask

def _head_indices(N, h, wmin, is_modified):
    wmax = N // 3
    headindices = [[] for _ in range(h)]
    phi = (1 + math.sqrt(5)) / 2
    for i in range(1, h + 1):
        a = int(math.floor(math.floor(i * phi) * phi))
        b = int(math.floor(math.floor(i * phi) * (phi ** 2)))
        w = wmin + int((wmax - wmin) / (h - 1) * (i - 1)) if h > 1 else wmin
        if is_modified:
            b_m = b - a
            a_m = a - b_m
            seq = _fib(a_m, b_m, w)
        else:
            seq = _fib(a, b, w)
        headindices[i - 1] = seq
    return [torch.tensor(seq, dtype=torch.int64) for seq in headindices]

def _fib(a, b, w):
    fib = [a, b]
    while fib[-1] <= w:
        fib.append(fib[-1] + fib[-2])
    return fib[:-1]

def _shuffle(seed, arr):
    random.seed(seed)
    arr = list(arr)
    random.shuffle(arr)
    return arr

def _plot_once(cache):
    global _PLOTTED
    if _PLOTTED:
        return
    try:
        from utils.plot import plot_attention_mask_for_all_heads
        from configs import config
        plot_attention_mask_for_all_heads(cache, config.output_dir)
    except Exception:
        pass
    _PLOTTED = True
