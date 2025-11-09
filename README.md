# Sparse Attention Mechanisms for Vision Transformers

This repository investigates and implements **sparse multi-head self-attention (MHSA)** for Vision Transformers (ViTs), with a focus on **structured, head-diverse sparsity** that preserves essential long-range connectivity while aggressively reducing compute. We provide **drop-in attention modules** (e.g., Random, Local/Structured, Top-K, Pruning), literature baselines (**BigBird**, **Longformer**, **Linformer**, **Efficient Attention**), and our proposed **Fibottention** family, all configurable via YAML and integrated into standard pretrain/finetune loops.

> **TL;DR:** We prune *attention scores* (the QKᵀ matrix entries) according to a mask Ω; we do **not** delete Q/K/V parameters. This yields **O(s)** time/memory with **s ≪ N²** active edges, while maintaining accuracy via head-wise complementary patterns.

---

## Why sparse attention for vision?

Transformer backbones (GPT [1–3], BERT [4], ALBERT [5], ViT [6], DETR [7], D-DETR [8], CLIP [9], etc.) excel at scale but incur **quadratic** attention cost in the number of tokens **N**. For images/videos, token neighborhoods are **highly redundant**, so dense all-to-all attention wastes compute on low-utility interactions. Sparse attention restricts each query to a subset of keys, cutting cost while encoding useful **inductive biases** (locality, multiscale aggregation, and head diversity). This is especially helpful **on edge/IoT hardware** and **in low-data regimes**, where strong priors can improve generalization.

---

## Technical formulation

Consider a single attention head with queries \(Q \in \mathbb{R}^{N \times d_h}\), keys \(K \in \mathbb{R}^{N \times d_h}\), and values \(V \in \mathbb{R}^{N \times d_h}\), where \(N\) is the token count and \(d_h\) is head dim.

### Dense MHSA
\[
A \;=\; \frac{QK^\top}{\sqrt{d_h}}, \quad
P \;=\; \operatorname{softmax}(A), \quad
Y \;=\; PV
\]

This costs \( \Theta(N^2 d_h) \) to form \(A\) and \( \Theta(N^2)\) memory for \(P\).

### Sparse MHSA (masking on QKᵀ)

Let \(\Omega \subseteq [N]\times[N]\) be the **support** (edges kept). Define a masked score matrix:
\[
A^{(\Omega)}_{jk} \;=\;
\begin{cases}
\frac{Q_j^\top K_k}{\sqrt{d_h}}, & (j,k)\in\Omega \\
-\infty, & \text{otherwise}
\end{cases}
\quad\Rightarrow\quad
P^{(\Omega)} \;=\; \operatorname{softmax}(A^{(\Omega)})
\]

The output is
\[
Y_j \;=\; \sum_{k : (j,k)\in\Omega} P^{(\Omega)}_{jk} \, V_k
\]

Thus, **we prune entries of \(QK^\top\)** (the attention *scores*) via \(\Omega\); this implicitly prunes the **gathers** from \(V\) to only those keys permitted by \(\Omega\). We **do not** prune or re-parameterize the Q/K/V projections themselves; gradients still flow to all Q/K/V parameters that contribute to any active edge.

With \(|\Omega| = s \ll N^2\), compute and memory scale as \(\Theta(sd_h)\) and \(\Theta(s)\), respectively. This is the core reason sparse attention is useful: maintain representational capacity while avoiding quadratic cost.

---

## What we implement

- **Random** (uniform edges)  
- **Structured Local / Dilated windows** (sliding windows, optionally dilated)
- **Top-K** (per-query selection by score)
- **Pruning** (static sparsification to a target FLOP budget)
- **Baselines from literature**  
  - **BigBird** — local + random + global hybrids  
    [[paper]](https://arxiv.org/abs/2007.14062) · [[code]](https://github.com/google-research/bigbird)
  - **Longformer** — sliding windows + global tokens  
    [[paper]](https://arxiv.org/abs/2004.05150) · [[code]](https://github.com/allenai/longformer)
  - **Linformer** — low-rank projections for linear complexity  
    [[paper]](https://arxiv.org/abs/2006.04768)
  - **Efficient Attention** — linearized attention variants  
    [[paper]](https://arxiv.org/abs/1812.01243)
- **Fibottention (ours)** — **head-diverse**, structured sparsity using **Wythoff/Fibonacci**-driven dilations to cover local and long-range interactions with **low head overlap**  
  [[project code]](https://github.com/Charlotte-CharMLab/Fibottention)

---

## Fibottention: head-diverse structured sparsity

### Mask family
Each head \(i \in \{1,\dots,h\}\) uses a distinct support set
\[
\Omega_i \;=\; \{(j,k) \;|\; |j-k| \in \mathcal{F}_i,\; |j-k|\le w_i\},
\]
where \(\mathcal{F}_i\) is a (generalized) **Fibonacci/Wythoff** dilation sequence and \(w_i \in [w_{\min}, w_{\max}]\) is a head-specific window bound. Lower-indexed heads favor **local** connectivity (small \(w_i\)), while higher-indexed heads include **long-range** hops (large \(w_i\)). Across heads, \(\{\mathcal{F}_i\}\) are chosen to **minimize overlap**, encouraging **complementary features**.

We then compute
\[
A^{(\Omega_i)} \;=\; (QK^\top) \odot \iota_{\Omega_i}\quad\text{with}\quad
(\iota_{\Omega_i})_{jk}=
\begin{cases}
1,&(j,k)\in\Omega_i\\
-\infty,&\text{otherwise}
\end{cases}
\]
and \(Y_i = \operatorname{softmax}(A^{(\Omega_i)})V\).

### Complexity
For suitable dilation schedules (e.g., Fibonacci/Wythoff sequences) and windows up to \(w_{\max}\), the **per-head** number of edges scales as \(O(N \log w_{\max})\), hence
\[
\text{cost} \;=\; O\!\big(h\, N \log w_{\max}\, d_h\big)
\]
versus \(O(h\,N^2 d_h)\) dense. In practice, we target **~2–5%** of dense edges.

### Why it helps
1. **Inductive bias** for vision: dense local + sparse long-range links mirrors spatial statistics; fewer spurious long-range interactions.  
2. **Head diversity**: near-disjoint \(\Omega_i\) yield higher **feature variance** across heads (measured via Frobenius distances on last-layer features), improving robustness under a fixed FLOP budget.  
3. **Graceful scaling**: log-growth in edges with \(w_{\max}\) supports high-resolution inputs.

> Note: We prune **entries** of \(QK^\top\) (and corresponding gathers from \(V\))—not the Q/K/V projection weights. If desired, you can add *weight* pruning on top, but that is orthogonal.

---

## Benchmarks (Image Classification)

| Method | CIFAR-10 | CIFAR-100 | Tiny-IN | IN-1K | Pruning Ratio ↑ |
|---|:--:|:--:|:--:|:--:|:--:|
| ViT-B (DeiT [6,43]) | 84.2 | 59.4 | 75.2 | **75.9** | Dense |
| + Random Pruning | 80.7 | 56.5 | 69.4 | 68.7 | 98.52% |
| + Top-K Pruning | 81.1 | 57.1 | 72.9 | 73.4 | 98.48% |
| + Sparse Transformer [Child19] | 81.3 | 58.2 | 70.3 | 68.7 | 98.47% |
| + **BigBird** [[paper]](https://arxiv.org/abs/2007.14062) [[code]](https://github.com/google-research/bigbird) | 86.8 | 63.4 | 73.4 | 71.5 | 97.96% |
| + **Longformer** [[paper]](https://arxiv.org/abs/2004.05150) [[code]](https://github.com/allenai/longformer) | 87.8 | 64.7 | 74.3 | 71.6 | 98.47% |
| + Linformer [[paper]](https://arxiv.org/abs/2006.04768) | 73.1 | 48.7 | 62.8 | 60.1 | 97.96% |
| + Efficient Attention [[paper]](https://arxiv.org/abs/1812.01243) | 84.4 | 62.6 | 73.7 | 70.1 | 97.98% |
| 🟢 **+ Fibottention (Ours)** [[code]](https://github.com/Charlotte-CharMLab/Fibottention) | **91.8** | **70.7** | **79.1** | 75.5 | **98.01%** |

*Setup:* all methods on ViT-B with matched attention FLOPs (~0.014 GFLOPs) vs dense (~0.72 GFLOPs). “Pruning ratio” = % of dense attention edges removed.

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

```bash
# Fine-tune ViT with Fibottention
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

## Adding a new sparse attention

1. **Mask pattern:** add to `core/masks/` and register.  
2. **Backend logic (optional):** add to `core/backends/` if you need a new score path.  
3. **YAML config:** `configs/attention/your_method.yaml`:
   ```yaml
   attention:
     backend: vit             # or your backend
     mask: fibo               # your mask name
     mask_kwargs:
       w_min: 5
       w_max: 65
       schedule: wythoff
   ```
4. **Run:** `--attn-cfg configs/attention/your_method.yaml`.

---

## Practical notes

- **Numerics:** Use \(-\infty\) (or a large negative) for masked scores prior to softmax; prefer fused kernels if available.  
- **Top-K:** selecting by raw scores \(QK^\top/\sqrt{d_h}\) per query often works; ensure stable backprop for ties.  
- **Batching:** store sparse indices per head/layer; avoid dense \(N\times N\) tensors where possible.  
- **Evaluation:** report both **Top-1** and **attention GFLOPs** to ensure fair comparisons at equal budgets.

---

## References & resources

- **Transformers & ViT:** Vaswani et al. [17,18], ViT [6], DeiT [43,44]  
- **Sparse attention theory & practice:** Child et al. (2019), BigBird (Zaheer et al., 2020) [[paper]](https://arxiv.org/abs/2007.14062) [[code]](https://github.com/google-research/bigbird),  
  Longformer (Beltagy et al., 2020) [[paper]](https://arxiv.org/abs/2004.05150) [[code]](https://github.com/allenai/longformer),  
  Linformer (Wang et al., 2020) [[paper]](https://arxiv.org/abs/2006.04768),  
  Efficient Attention (Shen et al., 2021) [[paper]](https://arxiv.org/abs/1812.01243)  
- **Fibottention (ours):** Fibonacci/Wythoff-driven head-diverse sparsity [[code]](https://github.com/Charlotte-CharMLab/Fibottention)  
- **Foundational models:** GPT [1–3], BERT [4], ALBERT [5], DETR [7], D-DETR [8], CLIP [9]  
- **Vision transformers & variants:** CvT [30], ConViT [40], Swin [39], MViTv2 [32], iFormer [38]  
- **Token sparsification:** DynamicViT [58], PS-ViT [59], Evo-ViT [60], EViT [61]

> For convenience, cite the BigBird, Longformer, and Fibottention GitHubs directly:  
> • BigBird — https://github.com/google-research/bigbird  
> • Longformer — https://github.com/allenai/longformer  
> • Fibottention — https://github.com/Charlotte-CharMLab/Fibottention

---

## License

This code builds on open-source implementations; see file headers and `LICENSE` for attribution and terms corresponding to each upstream dependency.

---

### Acknowledgments (optional)

Ali K. Rahimian, Manish K. Govind, Dominick Reilly, Christian Kümmerle†, Srijan Das† (UNC Charlotte);  
Subhajit Maity, Aritra Dutta† (UCF). †Equal contribution as Project Lead.
