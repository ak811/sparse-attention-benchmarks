import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

def get_attn_mask(n, attn_mode, local_attn_ctx=None):
    """
    Returns a lower-triangular-like attention mask with shape (1, 1, n, n)
    where 1s indicate allowed attention.
    """
    if attn_mode == 'all':
        b = torch.tril(torch.ones([n, n], dtype=torch.float32))
    elif attn_mode == 'local':
        # Build a LOCAL *band* (not full lower triangle)
        if local_attn_ctx is None:
            local_attn_ctx = n
        L = int(local_attn_ctx)
        L = max(0, min(L, n - 1))

        idx = torch.arange(n, dtype=torch.int32)
        i = idx.view(n, 1)
        j = idx.view(1, n)

        # Causal band: allow current token and previous L tokens: 0 <= i - j <= L
        b = ((i - j) >= 0) & ((i - j) <= L)
        b = b.float()
    elif attn_mode == 'strided':
        if local_attn_ctx is None:
            raise ValueError("get_attn_mask(attn_mode='strided') requires local_attn_ctx (stride) to be set")
        stride = int(local_attn_ctx)
        x = torch.arange(n, dtype=torch.int32).reshape(n, 1)
        y = x.transpose(0, 1)
        z = torch.zeros([n, n], dtype=torch.int32)
        q = z + x
        k = z + y
        c1 = q >= k  # causal
        c2 = torch.eq(torch.remainder(q - k, stride), 0)
        c3 = torch.logical_and(c1, c2)
        b = c3.float()
    else:
        raise ValueError(f"get_attn_mask: attn_mode '{attn_mode}' not implemented")
    b = b.reshape(1, 1, n, n)
    return b

def strided_transpose(x, n_ctx, local_attn_ctx, blocksize):
    """
    Reorders time dimension into (blocks, local_ctx) and back for strided attention.
    """
    bT_ctx = n_ctx // local_attn_ctx
    if bT_ctx % blocksize != 0:
        raise AssertionError(f"strided_transpose: {bT_ctx} not divisible by blocksize {blocksize}")
    n, t, embd = x.size()
    x = x.reshape(n, bT_ctx, local_attn_ctx, embd)  # (B, Bc, Lc, E)
    x = x.permute(0, 2, 1, 3)                       # (B, Lc, Bc, E)
    x = x.reshape(n, t, embd)
    return x

def split_states(x, n_heads):
    """
    reshape (batch, tokens, dim) -> (batch, heads, tokens, head_dim)
    """
    b, t, d = x.size()
    return x.reshape(b, t, n_heads, d // n_heads).transpose(1, 2)

def merge_states(x):
    """
    reshape (batch, heads, tokens, head_dim) -> (batch, tokens, dim)
    """
    b, h, t, dh = x.size()
    return x.transpose(1, 2).reshape(b, t, h * dh)

def split_heads(x, n):
    return split_states(x, n)

def merge_heads(x):
    return merge_states(x)

def attention_impl(q, k, v, heads, attn_mode, local_attn_ctx=None):
    q = split_heads(q, heads)
    k = split_heads(k, heads)
    v = split_heads(v, heads)
    n_timesteps = k.size()[2]
    mask = get_attn_mask(n_timesteps, attn_mode, local_attn_ctx).float()
    w = torch.matmul(q, k.transpose(-2, -1))
    scale_amount = 1.0 / np.sqrt(q.size()[-1])
    w = w * scale_amount
    w = w * mask + -1e9 * (1 - mask)
    w = F.softmax(w, dim=-1)
    a = torch.matmul(w, v)
    a = merge_heads(a)
    return a

def blocksparse_attention_impl(q, k, v, heads, attn_mode, local_attn_ctx=None, blocksize=32, num_verts=None, vertsize=None):
    n_ctx = q.size()[1]
    if attn_mode == 'strided':
        q = strided_transpose(q, n_ctx, local_attn_ctx, blocksize)
        k = strided_transpose(k, n_ctx, local_attn_ctx, blocksize)
        v = strided_transpose(v, n_ctx, local_attn_ctx, blocksize)
    n_state = q.size()[-1] // heads
    scale_amount = 1.0 / np.sqrt(n_state)
    w = torch.matmul(q, k.transpose(-2, -1))
    w = F.softmax(w * scale_amount, dim=-1)
    a = torch.matmul(w, v)
    if attn_mode == 'strided':
        n, t, embd = a.size()
        bT_ctx = n_ctx // local_attn_ctx
        a = a.reshape(n, local_attn_ctx, bT_ctx, embd)
        a = a.permute(0, 2, 1, 3)
        a = a.reshape(n, t, embd)
    return a

class SparseAttention(nn.Module):
    def __init__(self, heads, attn_mode, local_attn_ctx=None, blocksize=32):
        super(SparseAttention, self).__init__()
        self.heads = heads
        self.attn_mode = attn_mode
        self.local_attn_ctx = local_attn_ctx
        self.blocksize = blocksize

    def forward(self, q, k, v):
        return blocksparse_attention_impl(q, k, v, self.heads, self.attn_mode, self.local_attn_ctx)
