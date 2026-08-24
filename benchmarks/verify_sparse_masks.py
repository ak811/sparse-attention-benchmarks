import torch
import os
from core.masks.fibottention import FibottentionMask
from core.backends.fibottention_sparse import FibottentionTrueSparseBackend
from utils.plot import plot_attention_mask_for_all_heads
from configs import config

def main():
    H = 12
    N = 197 # 196 patches + 1 CLS
    depth_id = 0
    device = torch.device('cpu')

    print("Generating standard Fibottention mask...")
    dummy_attn = torch.zeros(1, H, N, N, device=device)
    # Set shuffled=False and shuffle_mode='none' for standard mask
    mask_generator = FibottentionMask(add_class_token=True, modified=False, shuffled=False, shuffle_mode='none')
    std_mask = mask_generator(dummy_attn, estep=(0,0), N=N, num_heads=H, depth_id=depth_id, device=device)

    print("Generating True Sparse backend offset equivalents...")
    # Set shuffled=False and shuffle_mode='none' for sparse backend
    backend = FibottentionTrueSparseBackend(add_class_token=True, modified=False, shuffled=False, shuffle_mode='none')
    valid_idx, idx_safe, max_M = backend._get_offsets(H, N - 1, depth_id, device)

    sparse_reconstructed = torch.zeros(1, H, N, N, device=device)
    P = N - 1

    # Apply class token logic: CLS attends to all, all attend to CLS
    sparse_reconstructed[:, :, 0, :] = 1
    sparse_reconstructed[:, :, :, 0] = 1

    # Map the gathered indices back to dense patch interactions for verification
    for h in range(H):
        for i in range(P):
            for m in range(max_M):
                if valid_idx[h, i, m]:
                    j = idx_safe[h, i, m]
                    sparse_reconstructed[0, h, i+1, j+1] = 1

    # Verify mathematical equivalence
    is_equal = torch.equal(std_mask, sparse_reconstructed)
    print(f"Masks are exactly equal across all 12 heads: {is_equal}")

    if not is_equal:
        diff = torch.abs(std_mask - sparse_reconstructed).sum().item()
        print(f"Differences found: {diff} elements")

    # Generate the 12-head plot using the existing plotting utility
    print("Plotting the standard fibottention mask over 12 heads...")
    config.output_dir = "runs/plots_verification/std_mask"
    os.makedirs(config.output_dir, exist_ok=True)
    plot_attention_mask_for_all_heads(std_mask, config.output_dir)

    print("Plotting the true sparse reconstructed mask over 12 heads...")
    config.output_dir = "runs/plots_verification/sparse_reconstructed"
    os.makedirs(config.output_dir, exist_ok=True)
    plot_attention_mask_for_all_heads(sparse_reconstructed, config.output_dir)

    print("Done! Plots saved to runs/plots_verification/")

if __name__ == "__main__":
    main()