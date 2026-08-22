import torch
import yaml
import torch.nn.functional as F

from models.vision_transformer import Attention
from core.registries import register_backend
from core.backends.base import AttentionBackend
from core.intermediate_storage import save_intermediate_x

# 1. Dynamically register PyTorch's highly optimized SDPA (FlashAttention)
@register_backend("sdpa")
class SDPABackend(AttentionBackend):
    def __call__(self, q, k, v, *, attn_drop=None, save_hook=None, **kwargs):
        # The VisionTransformer Attention class pre-scales 'q' by (head_dim ** -0.5).
        # F.scaled_dot_product_attention scales it AGAIN by default. 
        # We unscale 'q' here to prevent double-scaling.
        d = q.size(-1)
        q_unscaled = q * (d ** 0.5)
        
        p = attn_drop.p if attn_drop is not None and self.training else 0.0
        
        # PyTorch 2.0+ optimized dense attention
        x = F.scaled_dot_product_attention(q_unscaled, k, v, dropout_p=p)
        
        if save_hook is not None:
            save_hook(x)
        return x

def run_benchmark(label, cfg_dict, seq_len, batch_size=32):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dim = 768
    num_heads = 12
    
    attn_layer = Attention(dim=dim, num_heads=num_heads, attn_cfg=cfg_dict).to(device)
    attn_layer.eval()

    # Dummy inputs
    x_lat = torch.randn(1, seq_len + 1, dim, device=device)
    x_tput = torch.randn(batch_size, seq_len + 1, dim, device=device)
    estep = (0, 0)
    
    # ==========================================
    # 1. Latency Benchmark (Batch Size = 1)
    # ==========================================
    torch.cuda.synchronize()
    # Warmup
    with torch.no_grad(), torch.cuda.amp.autocast():
        for _ in range(15):
            _ = attn_layer(x_lat, estep)
            
    iters_lat = 200
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    torch.cuda.synchronize()
    start.record()
    with torch.no_grad(), torch.cuda.amp.autocast():
        for _ in range(iters_lat):
            _ = attn_layer(x_lat, estep)
    end.record()
    torch.cuda.synchronize()
    
    avg_latency_ms = start.elapsed_time(end) / iters_lat

    # ==========================================
    # 2. Throughput & Peak Memory (Batch Size = 32)
    # ==========================================
    throughput_str = "OOM"
    mem_str = "OOM"
    
    try:
        # Reset memory tracking to isolate this specific forward pass
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # Warmup
        with torch.no_grad(), torch.cuda.amp.autocast():
            for _ in range(5):
                _ = attn_layer(x_tput, estep)
                
        iters_tput = 50
        torch.cuda.synchronize()
        start.record()
        with torch.no_grad(), torch.cuda.amp.autocast():
            for _ in range(iters_tput):
                _ = attn_layer(x_tput, estep)
        end.record()
        torch.cuda.synchronize()
        
        # Calculate Throughput (Sequences per second)
        elapsed_sec = start.elapsed_time(end) / 1000.0
        throughput = (batch_size * iters_tput) / elapsed_sec
        throughput_str = f"{throughput:.0f}"
        
        # Calculate Peak GPU Memory Allocation
        peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        mem_str = f"{peak_mem_mb:.0f}"
        
    except RuntimeError as e:
        if "out of memory" not in str(e).lower():
            raise e

    # Print the formatted row
    print(f"| {label:<25} | {seq_len:<6} | {avg_latency_ms:>10.3f} | {throughput_str:>12} | {mem_str:>10} |")


def load_cfg(path):
    with open(path, "r") as f:
        return yaml.safe_load(f).get("attention")

if __name__ == '__main__':
    lengths = [196, 512, 1024, 2048, 4096, 8192]
    
    # 1. Optimized Dense (Using PyTorch SDPA FlashAttention)
    sdpa_cfg = {
        "backend": "sdpa",
        "backend_kwargs": {},
        "mask": "none",
        "mask_kwargs": {}
    }
    
    # 2. Simulated Fibottention (O(N^2) Math + Sparse Mask)
    simulated_cfg = load_cfg("configs/attention/vit_fibottention_shuffled.yaml")
    
    # 3. True Sparse Fibottention (O(NM) Gather + Math)
    true_sparse_cfg = load_cfg("configs/attention/vit_fibottention_true_sparse.yaml")
    
    configs = [
        ("Optimized Dense (SDPA)", sdpa_cfg),
        ("Simulated Fibo (Masked)", simulated_cfg), 
        ("True Sparse Fibo (O(NM))", true_sparse_cfg)
    ]
    
    print("\n" + "="*77)
    print(" ATTENTION LAYER HARDWARE EFFICIENCY BENCHMARK")
    print("="*77)
    print(f"| {'Model':<25} | {'N':<6} | {'Lat (ms)':>10} | {'Tput (seq/s)':>12} | {'Mem (MB)':>10} |")
    print("-" * 77)
    
    for n in lengths:
        for label, cfg in configs:
            run_benchmark(label, cfg, n, batch_size=32)
        print("-" * 77)