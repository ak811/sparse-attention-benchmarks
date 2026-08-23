import torch
from .base import AttentionBackend
from ..registries import register_backend
from ..masks.fibottention import _head_indices, _shuffle

@register_backend("fibottention_sparse")
class FibottentionTrueSparseBackend(AttentionBackend):
    def __init__(self, add_class_token=True, modified=False, shuffled=True, shared_offsets=False, shuffle_mode=None):
        super().__init__()
        self.add_class_token = add_class_token
        self.modified = modified
        self.shuffled = shuffled
        self.shared_offsets = shared_offsets
        self.shuffle_mode = shuffle_mode
        self._cache = {}
        self._reported = False

    def _get_offsets(self, H, P, depth_id, device):
        if depth_id is None: depth_id = 0
        key = (H, P, depth_id)
        if key in self._cache:
            return self._cache[key]

        headindices = _head_indices(P, H, 5, self.modified)
        
        mode = self.shuffle_mode
        if mode is None:
            mode = 'both' if self.shuffled else 'none'
            
        if mode == 'heads':
            headindices = _shuffle(42, headindices)
        elif mode == 'layers':
            shift = depth_id % H
            headindices = headindices[shift:] + headindices[:shift]
        elif mode == 'both':
            headindices = _shuffle(depth_id, headindices)
            
        if self.shared_offsets:
            headindices = [headindices[0] for _ in range(H)]

        all_offsets = []
        max_M = 0
        for h in range(H):
            h_off = []
            for d in headindices[h]:
                h_off.extend([int(d), -int(d)])
            all_offsets.append(h_off)
            if len(h_off) > max_M:
                max_M = len(h_off)

        offsets = torch.zeros((H, max_M), dtype=torch.long, device=device)
        valid_offsets = torch.zeros((H, max_M), dtype=torch.bool, device=device)

        for h in range(H):
            M_h = len(all_offsets[h])
            if M_h > 0:
                offsets[h, :M_h] = torch.tensor(all_offsets[h], dtype=torch.long, device=device)
                valid_offsets[h, :M_h] = True

        idx_i = torch.arange(P, device=device).view(1, P, 1)  
        idx_j = idx_i + offsets.view(H, 1, max_M)             

        valid_idx = (idx_j >= 0) & (idx_j < P) & valid_offsets.view(H, 1, max_M)
        idx_safe = idx_j.clamp(0, P - 1)                      

        self._cache[key] = (valid_idx, idx_safe, max_M)
        return self._cache[key]

    def __call__(self, q, k, v, *, estep=None, N=None, num_heads=None, depth_id=None, save_hook=None, attn_drop=None, **kwargs):
        B, H, N_in, D = q.shape
        device = q.device
        P = N_in - 1  

        valid_idx, idx_safe, max_M = self._get_offsets(H, P, depth_id, device)

        q_cls, q_p = q[:, :, :1, :], q[:, :, 1:, :]
        k_cls, k_p = k[:, :, :1, :], k[:, :, 1:, :]
        v_cls, v_p = v[:, :, :1, :], v[:, :, 1:, :]

        scale = 1.0 / (D ** 0.5)

        scores_p2cls = (q_p * k_cls).sum(dim=-1, keepdim=True) * scale

        b_idx = torch.arange(B, device=device).view(B, 1, 1, 1)
        h_idx = torch.arange(H, device=device).view(1, H, 1, 1)
        
        k_gathered = k_p[b_idx, h_idx, idx_safe.unsqueeze(0)]            
        scores_p2p = (q_p.unsqueeze(3) * k_gathered).sum(dim=-1) * scale 

        scores_p2p = scores_p2p.masked_fill(~valid_idx.unsqueeze(0), float('-inf'))

        scores_p = torch.cat([scores_p2cls, scores_p2p], dim=-1)         
        attn_p = torch.softmax(scores_p, dim=-1)
        if attn_drop is not None:
            attn_p = attn_drop(attn_p)

        attn_p2cls = attn_p[..., 0:1]                                    
        attn_p2p = attn_p[..., 1:]                                       
        v_gathered = v_p[b_idx, h_idx, idx_safe.unsqueeze(0)]            
        
        out_p = attn_p2cls * v_cls + (attn_p2p.unsqueeze(-1) * v_gathered).sum(dim=-2)

        scores_cls = (q_cls @ k.transpose(-2, -1)) * scale               
        attn_cls = torch.softmax(scores_cls, dim=-1)
        if attn_drop is not None:
            attn_cls = attn_drop(attn_cls)
        out_cls = attn_cls @ v                                           

        out = torch.cat([out_cls, out_p], dim=2)                         

        if save_hook is not None:
            save_hook(out)

        if not self._reported:
            total_elements = N_in * N_in
            computed_elements = N_in + P * (1 + max_M)
            density = computed_elements / total_elements
            print(f"[Fibottention True Sparse] shuffle_mode={self.shuffle_mode} | Max Offsets (M)={max_M} | "
                  f"Matrix Elements: Dense O(N^2)={total_elements}, Sparse Computed={computed_elements} | "
                  f"Realized Compute Density={density:.6f}")
            self._reported = True

        return out