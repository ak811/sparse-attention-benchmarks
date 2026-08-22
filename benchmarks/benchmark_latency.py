import torch
import time
import yaml
from models.vision_transformer import Attention

def run_benchmark(label, cfg_path, seq_len):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dim = 768
    num_heads = 12
    
    with open(cfg_path, "r") as f:
        attn_cfg = yaml.safe_load(f).get("attention")
        
    attn_layer = Attention(dim=dim, num_heads=num_heads, attn_cfg=attn_cfg).to(device)
    attn_layer.eval()

    # Create dummy input (Batch=1, SeqLen + 1 CLS token, Dim)
    x = torch.randn(1, seq_len + 1, dim, device=device)
    estep = (0, 0)

    # Warmup
    with torch.no_grad():
        for _ in range(50):
            _ = attn_layer(x, estep)
    
    torch.cuda.synchronize()

    # Benchmark
    iters = 1000
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    with torch.no_grad():
        for _ in range(iters):
            _ = attn_layer(x, estep)
    end_event.record()
    
    torch.cuda.synchronize()
    elapsed_time_ms = start_event.elapsed_time(end_event)
    
    avg_latency = elapsed_time_ms / iters
    print(f"[{label:<25}] N={seq_len:<4} | Latency: {avg_latency:.4f} ms")

if __name__ == '__main__':
    print("--- True Latency Benchmark (Batch=1) ---")
    lengths = [196, 512, 1024]
    
    # Tests all possibilities without altering original configs
    configs = [
        ("Dense (O(N^2))", "configs/attention/vit_none.yaml"),
        ("Simulated Fibo (O(N^2))", "configs/attention/vit_fibottention_shuffled.yaml"), 
        ("True Sparse Fibo (O(NM))", "configs/attention/vit_fibottention_true_sparse.yaml")
    ]
    
    for n in lengths:
        for label, cfg in configs:
            run_benchmark(label, cfg, n)
        print("-" * 65)