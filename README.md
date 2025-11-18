# Sparse Attention Mechanisms for Vision Transformers

This repository implements **sparse multi-head self-attention (MHSA)** modules for Vision Transformers (ViTs), emphasizing **structured, head-diverse sparsity** that preserves critical long-range pathways while substantially reducing compute and memory. The library provides **drop-in attention backends** (Random, Local/Structured, Top-K, Pruning), established baselines (BigBird, Longformer, Linformer, Efficient Attention), and our proposed **Fibottention** family. Components are configured via YAML and integrate with standard pretraining and finetuning pipelines.

---

## Motivation: sparsity for visual tokens

Standard attention has **quadratic** time and memory in the number of tokens `N`. Visual tokens (images, video) display strong locality and redundancy; dense all-to-all attention allocates capacity to many low-utility interactions. **Sparse attention** constrains each query to a subset of keys, yielding:

- Lower **compute/memory** at training and inference,
- Useful **inductive biases** (locality, multiscale aggregation),
- **Head diversity** via complementary sparse patterns across heads,
- Better suitability for **resource-constrained** or **low-data** settings.

---

## Technical overview

We consider a single head with `Q, K, V` of shape `(N, d_h)`.

### Dense MHSA
Scores → softmax (row-wise) → value aggregation
```text
A = (Q @ K.T) / sqrt(d_h)   # scores
P = softmax(A)              # row-wise
Y = P @ V                   # output
```
Asymptotics:
```text
time  ~ Theta(N^2 * d_h)    # to form A
mem   ~ Theta(N^2)          # to store P
```

### Sparse MHSA (masked scores)

Let `Omega ⊆ [0..N-1] × [0..N-1]` be the **edge set** of permitted query–key pairs.
We compute/keep scores only for edges in `Omega` and mask the rest to `-inf` before softmax.

```text
A_Ω[j, k] =  (Q[j] · K[k]) / sqrt(d_h)   if (j, k) ∈ Omega
            -inf                         otherwise

P_Ω = softmax(A_Ω)        # row-wise
Y[j] = Σ_{k : (j,k)∈Omega}  P_Ω[j, k] * V[k]
```

Key points:
- We prune **entries of (Q @ K.T)** and the corresponding gathers from `V`.
- The **Q/K/V projection weights remain dense** and fully trainable.
- With `|Omega| = s << N^2`, costs scale as:
  ```text
  time ~ Theta(s * d_h)
  mem  ~ Theta(s)
  ```

---

## Implemented mechanisms

- **Random** — uniform edge sampling per query.  
- **Structured Local / Dilated** — sliding windows with optional dilation; supports fixed or depth-dependent windows.  
- **Top-K** — per-query selection by score; `K` may be static or scheduled.  
- **Pruning** — static sparsification to meet a specified attention FLOP budget.  
- **Baselines:**
  - **BigBird** — hybrid of local + random + global tokens.  
    [paper](https://arxiv.org/abs/2007.14062) · [code](https://github.com/google-research/bigbird)
  - **Longformer** — sliding windows with global tokens.  
    [paper](https://arxiv.org/abs/2004.05150) · [code](https://github.com/allenai/longformer)
  - **Linformer** — low-rank projections to linearize complexity.  
    [paper](https://arxiv.org/abs/2006.04768)
  - **Efficient Attention** — kernel/feature-mapped linearized attention.  
    [paper](https://arxiv.org/abs/1812.01243)
- **Fibottention (ours)** — Wythoff/Fibonacci-driven **head-diverse** dilations that cover local and long-range hops with **low inter-head overlap**.  
  [project code](https://github.com/Charlotte-CharMLab/Fibottention)

---

## Fibottention: head-diverse structured masks

### Construction

Each head `i ∈ {1..h}` uses a distinct support:
```text
Omega_i = { (j, k) | abs(j - k) ∈ F_i   and   abs(j - k) ≤ w_i }
```
- `F_i`: head-specific Fibonacci/Wythoff-style dilation sequence,  
- `w_i ∈ [w_min, w_max]`: per-head maximum hop distance,  
- Lower-index heads: predominantly **local** links; higher-index heads: **long-range** hops,  
- Sequences `{F_i}` chosen to **minimize pairwise overlap**, encouraging complementary features across heads.

Per head:
```text
A_Ωi = (Q @ K.T) ⊙ mask(Omega_i, fill=-inf outside)
Y_i  = softmax(A_Ωi) @ V
```

### Complexity

With Fibonacci/Wythoff dilations and maximum window `w_max`, the per-head edge count is `O(N * log w_max)`.
Hence attention cost becomes:
```text
dense:  O(h * N^2 * d_h)
sparse: O(h * N * log(w_max) * d_h)
```
In practice we target **~2–5%** of dense edges while maintaining strong accuracy.

### Rationale

- **Vision priors:** local context with sparse long hops matches spatial statistics and reduces spurious interactions.  
- **Head complementarity:** near-disjoint supports across heads increase representation diversity at fixed FLOPs.  
- **Resolution scaling:** logarithmic edge growth with `w_max` supports high-resolution inputs with controlled cost.

> Implementation focus: **edge sparsity** on attention scores/gathers. Weight pruning on Q/K/V is orthogonal and optional.

---

## Benchmarks (Image Classification)

| Method | CIFAR-10 | CIFAR-100 | Tiny-IN | IN-1K | Pruning Ratio ↑ |
|---|:--:|:--:|:--:|:--:|:--:|
| ViT-B (DeiT) | 84.2 | 59.4 | 75.2 | **75.9** | Dense |
| + Random Pruning | 80.7 | 56.5 | 69.4 | 68.7 | 98.52% |
| + Top-K Pruning | 81.1 | 57.1 | 72.9 | 73.4 | 98.48% |
| + Sparse Transformer (Child et al.) | 81.3 | 58.2 | 70.3 | 68.7 | 98.47% |
| + **BigBird** | 86.8 | 63.4 | 73.4 | 71.5 | 97.96% |
| + **Longformer** | 87.8 | 64.7 | 74.3 | 71.6 | 98.47% |
| + Linformer | 73.1 | 48.7 | 62.8 | 60.1 | 97.96% |
| + Efficient Attention | 84.4 | 62.6 | 73.7 | 70.1 | 97.98% |
| **+ Fibottention (Ours)** | **91.8** | **70.7** | **79.1** | 75.5 | **98.01%** |

**Protocol:** ViT-B across methods with **matched attention FLOPs** (~0.014 GFLOPs) vs dense (~0.72 GFLOPs). “Pruning ratio” denotes the percentage of dense attention edges removed.

---

## Installation

```bash
git clone https://github.com/<your-org>/<your-repo>.git
cd <your-repo>
conda create -n sparsevit python=3.10 -y
conda activate sparsevit
pip install -r requirements.txt
```

---

## Quick start

Fine-tune ViT with Fibottention:

```bash
python -m main_finetune \
  --dataset imagenet \
  --data_path /path/to/imagenet \
  --model vit_base_patch16 \
  --nb_classes 1000 \
  --epochs 50 --batch_size 64 --blr 1e-3 \
  --attn-cfg configs/attention/vit_fibottention.yaml \
  --output_dir runs/fibo_exp1 --log_dir runs/fibo_exp1
```

**Swap mechanisms** by changing `--attn-cfg`:
```bash
--attn-cfg configs/attention/vit_bigbird.yaml
--attn-cfg configs/attention/vit_longformer.yaml
--attn-cfg configs/attention/vit_topk.yaml
--attn-cfg configs/attention/vit_structured_local.yaml
```

---

## Configuration schema (YAML)

Minimal example:
```yaml
attention:
  backend: vit           # see core/backends
  mask: fibo             # see core/masks
  mask_kwargs:
    w_min: 5
    w_max: 65
    schedule: wythoff    # or: fibonacci
```

Common options:
- `backend`: attention implementation (e.g., ViT, linearized variants).  
- `mask`: mask generator identifier (local, topk, bigbird, longformer, fibo, …).  
- `mask_kwargs`: mechanism-specific parameters (window sizes, dilations, global tokens, K for Top-K, etc.).

---

## Repository layout

```
.
├── main_pretrain.py                # self-supervised / pretraining
├── main_finetune.py                # supervised training + eval
├── core/
│   ├── attention.py                # base attention interface
│   ├── factory.py                  # build attention from YAML
│   ├── registries.py               # registry for backends/masks
│   ├── backends/                   # attention backends (ViT, linearized, etc.)
│   └── masks/                      # mask generators (local, topk, bigbird, longformer, fibo, ...)
├── fibottention.py                 # Wythoff/Fibonacci sequences + helpers
├── configs/attention/              # YAMLs for all mechanisms
├── utils/                          # plotting, logging, datasets
└── script.sh                       # multi-GPU launcher (torchrun)
```

---

## Extensibility: adding a new sparse attention

1. **Mask pattern** → implement in `core/masks/` and register in `registries.py`.  
2. **Backend (optional)** → add to `core/backends/` if your method requires a distinct score or kernel path.  
3. **YAML** → create `configs/attention/your_method.yaml`:
   ```yaml
   attention:
     backend: vit
     mask: your_mask_name
     mask_kwargs:
       # parameters...
   ```
4. **Run** → `--attn-cfg configs/attention/your_method.yaml`.

---

## Practical guidance

- **Numerical masking:** write masked scores as `-inf` (or a sufficiently negative sentinel) *before* softmax. Prefer fused kernels to avoid materializing dense `N × N` tensors.  
- **Top-K selection:** select by scaled scores `(Q @ K.T) / sqrt(d_h)` per query; handle ties deterministically for stable gradients.  
- **Indexing & batching:** cache sparse indices per head/layer; avoid dense score matrices when possible.  
- **Reporting:** co-report **Top-1 accuracy** and **attention GFLOPs** for comparisons at matched attention budgets.  
- **Autocast:** mixed precision is supported; ensure masking values survive FP16/BF16 underflows (e.g., use `-1e9` in FP16 code paths as needed).

---

## References & resources

- **BigBird:** [paper](https://arxiv.org/abs/2007.14062) · [code](https://github.com/google-research/bigbird)  
- **Longformer:** [paper](https://arxiv.org/abs/2004.05150) · [code](https://github.com/allenai/longformer)  
- **Linformer:** [paper](https://arxiv.org/abs/2006.04768)  
- **Efficient Attention:** [paper](https://arxiv.org/abs/1812.01243)  
- **Fibottention (ours):** [arXiv](https://arxiv.org/abs/2406.19391)

For convenience:  
• BigBird — https://github.com/google-research/bigbird  
• Longformer — https://github.com/allenai/longformer  
• Fibottention — https://github.com/Charlotte-CharMLab/Fibottention

---

## Acknowledgement

This repository is built on top of **MAE**, **TimeSformer**, and **Crossway Diffusion**. We thank all contributors for their well-organized codebases.

---

## Citation

```
@article{rahimian2024fibottention,
  title = {Fibottention: Inceptive Visual Representation Learning with Diverse Attention Across Heads},
  author = {Rahimian, Ali Khaleghi and Govind, Manish Kumar and Maity, Subhajit and Reilly, Dominick and Kümmerle, Christian and Das, Srijan and Dutta, Aritra},
  journal = {arXiv preprint arXiv:2406.19391},
  year = {2024},
  url = {https://arxiv.org/abs/2406.19391}
}
```

---

## License

This project is licensed under the **Creative Commons Attribution 4.0 International (CC BY 4.0)** — see the LICENSE website/file for details.
