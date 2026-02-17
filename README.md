# Benchmark Study: Gated DeltaNet (GDN) vs Gated Sparse Attention (GSA)

A comparative benchmark of two attention mechanism architectures measuring throughput (tokens/sec), latency (ms), and peak GPU memory (MB) across varying sequence lengths.

**Reference Implementations:**
- GDN: [NVlabs/GatedDeltaNet](https://github.com/NVlabs/GatedDeltaNet) - `lit_gpt/gated_delta_net.py`, `lit_gpt/gated_delta_rule_ops/chunk.py`
- GSA: [alfredcs/Gated-Sparse-Attention](https://github.com/alfredcs/Gated-Sparse-Attention) - `gsa/attention/gated_sparse_attention.py`, `gsa/kernels/`

The benchmark implementations are aligned with these reference repos. Triton/CUDA kernels are replaced with equivalent PyTorch fallbacks for portability.

## Table of Contents

- [Overview](#overview)
- [Architecture Summary](#architecture-summary)
- [Detailed Operations Per Iteration](#detailed-operations-per-iteration)
- [Bottlenecks and Critical Paths](#bottlenecks-and-critical-paths)
- [Theoretical Minimum Operations](#theoretical-minimum-operations)
- [Correctness Analysis](#correctness-analysis)
- [Benchmark Configuration](#benchmark-configuration)
- [How to Run](#how-to-run)

---

## Overview

| Model | Core Idea | Time Complexity | Memory Complexity |
|-------|-----------|-----------------|-------------------|
| **Gated DeltaNet (GDN)** | Recurrent state-space attention via delta rule | O(T * H * d_k * d_v) | O(H * d_k * d_v) state |
| **Gated Sparse Attention (GSA)** | Adaptive top-k sparse attention | O(T^2 * H_idx * d_idx) indexer + O(T * k * d) attention | O(T * k) indices |

Where: `B` = batch, `T` = sequence length, `D` = hidden dimension, `H` = attention heads (16), `d_k` = key head dim (`D*0.75/H` = 96), `d_v` = value head dim (`D*1.5/H` = 192), `H_idx` = indexer heads (4), `d_idx` = indexer dim (64), `k` = adaptive sparsity budget (256-4096).

**Key reference-aligned parameters:**
- GDN: `expand_k=0.75`, `expand_v=1.5`, Mamba-style gating, `recurrent_gated_delta_rule_ref` recurrence
- GSA: `GatedLightningIndexer(d_indexer=64)`, `AdaptiveTopKSelector(k_base=2048)`, `ValueGate` (G2) + `OutputGate` (G1)

---

## Architecture Summary

### Gated DeltaNet (GDN)

```
Input [B,T,D]
  |
  +---> Q_proj(D, D*0.75) ---> Conv1D(k=4,SiLU) ---> RoPE ---> L2 Normalize ---> scale by d_k^(-0.5)
  +---> K_proj(D, D*0.75) ---> Conv1D(k=4,SiLU) ---> RoPE ---> L2 Normalize
  +---> V_proj(D, D*1.5)  ---> Conv1D(k=4,SiLU)
  +---> G_proj(D, D*1.5)  (gate for output norm)
  +---> beta_proj(D, H) ---> sigmoid  (write gate)
  +---> gk_proj(D, H) ---> -A.exp() * softplus(gk + dt_bias)  (log-decay gate, Mamba-style)
  |
  v
  Delta Rule Recurrence (ref: recurrent_gated_delta_rule_ref, sequential over T):
    for t = 0..T-1:
      S = S * g_t.exp()                          # decay state
      v_new = (v_t - (S * k_t).sum(-2)) * beta_t # delta: subtract memory readout
      S = S + k_t @ v_new^T                      # rank-1 state update
      o_t = q_t @ S                              # query the state
  |
  v
  FusedRMSNormSwishGate(o, g) ---> O_proj(D*1.5, D) ---> Output [B,T,D]
```

### Gated Sparse Attention (GSA)

```
Input [B,T,D]
  |
  +===[GatedLightningIndexer]===+
  |  q_proj(D, H_idx*d_idx=256) -> q_I [B,T,4,64]   (per-head queries)
  |  k_proj(D, d_idx=64)        -> k_I [B,T,64]      (shared keys)
  |  weight_proj(D, H_idx=4)    -> w   [B,T,4]        (importance weights, +bias)
  |  bias                       -> b   [4]             (learnable threshold)
  |       |
  |       v
  |  scores = sum_h( sigmoid(w_h) * sigmoid(q_I_h @ k_I^T * scale + b_h) )
  |  Causal mask (torch.triu diagonal=1)
  |  -> indexer_scores [B, T, T]
  |
  +===[AdaptiveTopKSelector]===+
  |  Variance-based adaptive k (k_base=2048, k_min=256, k_max=4096)
  |  TopK selection -> indices [B, T, k_effective], mask [B, T, k_effective]
  |
  +===[Attention Phase]===+
  |  q_proj(D,D), k_proj(D,D), v_proj(D,D)
  |  ValueGate (G2): v = v * sigmoid(W_gv @ h + b),  bias_init=0.5
  |  RoPE on Q, K (standard _rotate_half)
  |  Sparse gather via _gather_along_seq (5D expand + torch.gather)
  |  Sparse SDPA: softmax(Q @ K_sel^T / sqrt(d)) @ V_sel
  |  OutputGate (G1): o = o * sigmoid(W_go @ h + b),  bias_init=0.5
  |  o_proj(D,D)
  |
  v
  Output [B,T,D]
```

---

## Detailed Operations Per Iteration

All counts are in FLOPs (1 multiply-add = 2 FLOPs). Default config: `B=1, T=variable, D=2048, H=16, d_k=96, d_v=192, conv_size=4` (GDN uses expand_k=0.75, expand_v=1.5). GSA uses `d=128, d_idx=64, H_idx=4, k_base=2048`.

### Gated DeltaNet: Operation Breakdown

| # | Operation | Shape Transformation | FLOPs |
|---|-----------|---------------------|-------|
| 1 | Q projection (`x @ W_q`) | [B,T,D] x [D,D] | 2BTD^2 |
| 2 | K projection (`x @ W_k`) | [B,T,D] x [D,D] | 2BTD^2 |
| 3 | V projection (`x @ W_v`) | [B,T,D] x [D,D] | 2BTD^2 |
| 4 | G projection (`x @ W_g`) | [B,T,D] x [D,D] | 2BTD^2 |
| 5 | Beta projection (`x @ W_b`) | [B,T,D] x [D,H] | 2BTDH |
| 6 | GK projection (`x @ W_gk`) | [B,T,D] x [D,H] | 2BTDH |
| 7 | Q Conv1D (depthwise, k=4) | [B,D,T] conv k=4 | 2BTD * 4 = 8BTD |
| 8 | K Conv1D (depthwise, k=4) | same | 8BTD |
| 9 | V Conv1D (depthwise, k=4) | same | 8BTD |
| 10 | SiLU activations (3x) | element-wise | ~9BTD |
| 11 | RoPE cos/sin compute | [T, d] | Td |
| 12 | RoPE apply Q | [B,T,H,d] | 6BTHd = 6BTD |
| 13 | RoPE apply K | [B,T,H,d] | 6BTD |
| 14 | L2 Normalize Q | [B,T,H,d] | 3BTD |
| 15 | L2 Normalize K | [B,T,H,d] | 3BTD |
| 16 | Sigmoid (beta) | [B,T,H] | BTH |
| 17 | Softplus + exp (alpha) | [B,T,H] | ~3BTH |
| **18** | **Delta recurrence per step t (ref: recurrent_gated_delta_rule_ref):** | | |
| 18a | `S = S * g_t.exp()` (state decay) | [B,H,d_k,d_v] | BH*d_k*d_v |
| 18b | `(S * k_t).sum(-2)` (memory readout) | [B,H,d_k,d_v] x [B,H,d_k] | 2BH*d_k*d_v |
| 18c | `v_new = (v_t - readout) * beta_t` | [B,H,d_v] | 2BH*d_v |
| 18d | `S += k_t @ v_new^T` (rank-1 update) | [B,H,d_k] x [B,H,d_v] | BH*d_k*d_v |
| 18e | `o_t = q_t @ S` (query state) | [B,H,d_k] x [B,H,d_k,d_v] | 2BH*d_k*d_v |
| | **Subtotal per step** | | **~6BH*d_k*d_v** |
| | **Total recurrence (x T steps)** | | **T * 6BH*d_k*d_v** |
| 19 | RMSNorm on output | [BTH, d] | ~5BTD |
| 20 | SiLU gate | [BTH, d] | ~2BTD |
| 21 | Output projection (`o @ W_o`) | [B,T,D] x [D,D] | 2BTD^2 |

#### GDN Total FLOPs (one forward pass)

```
Projections:      Q,K -> D*D*0.75; V,G -> D*D*1.5; O -> D*1.5*D; beta,gk -> D*H
                  = 2BT(D*D*0.75*2 + D*D*1.5*2 + D*1.5*D) + 4BTDH
                  ≈ 2BT * D^2 * (1.5 + 3.0 + 1.5) = 12BTD^2
Convolutions:     Q,K conv(d=D*0.75,k=4) + V conv(d=D*1.5,k=4)
RoPE + Norm:      ~25BT * D*0.75
Delta recurrence: T * 6BH * d_k * d_v  (rank-1 update, NOT d^3 matrix multiply)

DOMINANT TERM:  T * 6BH * d_k * d_v  (recurrence with rank-1 state updates)
```

**Numerical example (B=1, T=4096, D=2048, H=16, d_k=96, d_v=192):**

| Component | FLOPs | % of Total |
|-----------|-------|------------|
| Linear projections (7 total) | ~12 * 2048^2 * 4096 ≈ ~102.8 G | ~54% |
| Delta recurrence | 4096 * 6 * 16 * 96 * 192 = **~72.5 G** | **~38%** |
| Convolutions | ~16 * 4096 * 2048 = ~134 M | <0.1% |
| RoPE + Norm + Gates | ~20 * 4096 * 1536 = ~126 M | <0.1% |
| **Total** | **~175 GFLOPs** | |

### Gated Sparse Attention: Operation Breakdown

| # | Operation | Shape Transformation | FLOPs |
|---|-----------|---------------------|-------|
| **Indexer Phase** | | | |
| 1 | q_proj (indexer) | [B,T,D] x [D, H_idx*d_idx] | 2BT * D * (4*64) = 2BTD*256 |
| 2 | k_proj (indexer) | [B,T,D] x [D, d_idx] | 2BTD*64 |
| 3 | weight_proj (indexer) | [B,T,D] x [D, H_idx] | 2BTD*4 |
| 4 | Gated indexer QK scores | [B,T,4,64] x [B,T,64] -> [B,4,T,T] | 2BT^2 * H_idx * d_idx |
| 5 | Sigmoid (scores + bias) | [B,4,T,T] | B * 4 * T^2 |
| 6 | Sigmoid (weights) + multiply | [B,4,T,T] | B * 4 * T^2 |
| 7 | Sum across indexer heads | [B,4,T,T] -> [B,T,T] | B * 4 * T^2 |
| 8 | Causal masking | [B,T,T] | B * T^2 |
| 9 | Variance computation | [B,T,T] | 3BT^2 |
| 10 | TopK selection | [B,T,T] -> [B,T,k] | ~BT^2 * log(k) |
| 11 | Adaptive k_t computation | [B,T] | ~5BT |
| **Attention Phase** | | | |
| 12 | W_q projection | [B,T,D] x [D,D] | 2BTD^2 |
| 13 | W_k projection | same | 2BTD^2 |
| 14 | W_v projection | same | 2BTD^2 |
| 15 | W_gv projection + sigmoid | same | 2BTD^2 + BTD |
| 16 | v * g_v (value gating) | [B,T,D] | BTD |
| 17 | RoPE apply Q | [B,T,H,d] | 6BTD |
| 18 | RoPE apply K | same | 6BTD |
| 19 | Sparse gather K,V | [B,H,T,k,d] from [B,H,T,d] | BTHkd (memory-bound) |
| 20 | Sparse QK^T scores | [B,H,T,d] x [B,H,T,k,d] | 2BTHkd |
| 21 | Softmax over k | [B,H,T,k] | ~5BTHk |
| 22 | Attention @ V | [B,H,T,k] x [B,H,T,k,d] | 2BTHkd |
| 23 | W_go projection + sigmoid | [B,T,D] x [D,D] | 2BTD^2 + BTD |
| 24 | Output gating | [B,T,D] | BTD |
| 25 | O_proj | [B,T,D] x [D,D] | 2BTD^2 |

#### GSA Total FLOPs (one forward pass)

```
Indexer projections:  2BTD * (256 + 64 + 4)    = 648BTD
Indexer QK scores:    2BT^2 * 4 * 64            = 512BT^2
Indexer overhead:     ~12BT^2
TopK:                 ~BT^2 * log(k)
Attention projections: 4 * 2BTD^2               = 8BTD^2    (Q,K,V,O)
Gate projections:      2 * 2BTD^2               = 4BTD^2    (ValueGate, OutputGate)
Sparse attention:      4BTHkd                   = 4BTkD     (since Hd = D)
RoPE + gates:         ~15BTD

DOMINANT TERMS:  12BTD^2 (projections+gates)  +  512BT^2 (indexer)  +  4BTkD (sparse attn)
```

**Numerical example (B=1, T=4096, D=2048, H=16, d=128, k=2048):**

| Component | FLOPs | % of Total |
|-----------|-------|------------|
| Attn + gate projections (6x) | 12 * 2048^2 * 4096 = ~102.8 G | ~48.6% |
| Indexer QK scores | 512 * 4096^2 = ~8.6 G | ~4.1% |
| Indexer projections | 648 * 4096 * 2048 = ~5.4 G | ~2.6% |
| Sparse attention (QK + AV) | 4 * 4096 * 2048 * 2048 = ~68.7 G | ~32.5% |
| TopK selection | ~4096^2 * 11 = ~184 M | <0.1% |
| **Total** | **~186 GFLOPs** | |

---

## Bottlenecks and Critical Paths

### Gated DeltaNet - Critical Path

```
[Q,K,V,G projections]  ──parallel──>  [Conv1D x3]  ──parallel──>  [RoPE + Norm]
                                                                        |
                                                                        v
        t=0 ──> t=1 ──> t=2 ──> ... ──> t=T-1    (SEQUENTIAL - cannot parallelize)
        |       |       |               |
        Each step: S_{t+1} = f(S_t, q_t, k_t, v_t)    ← data dependency
                                                        |
                                                        v
                                                  [Output Norm + O_proj]
```

**Primary bottleneck: Delta rule recurrence (Step 18)**

- **Why:** Each timestep `t` requires the state matrix `S` from step `t-1`. This is an inherently sequential dependency chain of length `T`.
- **Dominating op within each step:** `S @ orthogonal_proj` is a `[d x d] @ [d x d]` matrix multiply per head, costing `2Hd^3` FLOPs per step. With d=128, that's **67M FLOPs per timestep** just for this one operation.
- **Total sequential work:** `T * 2BHd^3` = for T=4096, this is **275 GFLOPs** of strictly sequential computation.
- **GPU utilization:** Each step operates on small `[d x d]` matrices (128x128) which underutilize GPU parallelism. Modern GPUs need much larger matrices for peak throughput.

**Secondary bottleneck: Python loop overhead**

- The `for t in range(T)` loop in `_delta_rule_python` incurs Python interpreter overhead per step, adding significant latency compared to a fused CUDA/Triton kernel.

### Gated Sparse Attention - Critical Path

```
[Indexer Projections] ──> [QK Scores O(T^2)] ──> [TopK Selection] ──> [Sparse Indices]
       |                                                                      |
       |                                                                      v
[Attn Projections Q,K,V] ──parallel──> [RoPE] ──> [Sparse Gather] ──> [Sparse Attn]
       |                                                                      |
[Gate Projections gv,go] ──parallel──────────────────────────> [Gating] ──> [O_proj]
```

**Primary bottleneck: Indexer QK score computation (Step 4)**

- **Why:** Computing `q_I @ k_I^T` produces a `[B, H_idx, T, T]` score matrix. This is **O(T^2)** in both compute and memory.
- **For T=4096:** The score matrix is `[1, 4, 4096, 4096]` = 256 MB in float32. At T=8192 it grows to 1 GB.
- **This defeats the purpose of sparse attention** at long sequences since the indexer itself is quadratic.

**Secondary bottleneck: TopK selection**

- TopK on `[B, T, T]` tensors is memory-bandwidth-bound and scales as O(T^2 * log k).

**Tertiary bottleneck: Sparse gather**

- Gathering K,V based on indices is irregular memory access, causing poor cache utilization and low GPU occupancy.

### Comparative Bottleneck Summary

| Aspect | GDN | GSA |
|--------|-----|-----|
| **Time complexity** | O(T * H * d^3) | O(T^2 * H_idx * d_idx + T * k * D) |
| **Sequential dependency** | T-step recurrence (fully serial) | No serial dependency (fully parallelizable) |
| **GPU parallelism** | Poor (small mat-mul per step) | Good (large batched operations) |
| **Memory scaling** | O(H * d^2) state (constant in T) | O(T^2) indexer scores (quadratic in T) |
| **Practical crossover** | Faster at very long T (>16K) | Faster at moderate T (<8K) |

---

## Theoretical Minimum Operations

The minimum operations represent the irreducible compute required by each algorithm's mathematical formulation, assuming perfect fusion, zero overhead, and optimal hardware utilization.

### GDN - Theoretical Minimum

The current implementation already uses the **optimal rank-1 update** formulation (matching the NVlabs reference `recurrent_gated_delta_rule_ref`):

```
Per timestep (irreducible operations):
  - Decay state: S *= g_t.exp()                  BH * d_k * d_v  FLOPs
  - Memory readout: (S * k_t).sum(-2)             2BH * d_k * d_v  FLOPs
  - Delta: v_new = (v_t - readout) * beta_t       2BH * d_v  FLOPs
  - Rank-1 update: S += k_t @ v_new^T             BH * d_k * d_v  FLOPs
  - Query state: o_t = q_t @ S                    2BH * d_k * d_v  FLOPs

Minimum per step:   ~6BH * d_k * d_v  FLOPs
Minimum total:      6BTH * d_k * d_v  FLOPs
```

**The reference implementation is already at the theoretical minimum** for a sequential recurrence. The only further optimization is **chunk-parallel** execution (as in `chunk_gated_delta_rule` from the reference), which groups C timesteps into a chunk, computes intra-chunk attention in parallel, and propagates state between chunks.

| Version | Per-step FLOPs | Total (T=4096, d_k=96, d_v=192) |
|---------|---------------|----------------|
| **Recurrent (current, matches ref)** | 6BH*d_k*d_v = ~1.77M | **~72.5 G** |
| Chunk-parallel (ref: chunk_gated_delta_rule) | Same total FLOPs, but parallelizable | Same, ~10-50x faster wall-clock |

### GSA - Theoretical Minimum

```
Indexer (irreducible for dense scoring):
  - QK computation: 2BT^2 * H_idx * d_idx   (must compute all pairwise scores)
  - TopK:           O(BT^2)                  (must scan all scores)

Sparse Attention (irreducible):
  - Gather:        BTHkd reads (irregular, cannot avoid)
  - QK scores:     2BTHkd
  - Softmax:       5BTHk
  - Attn @ V:      2BTHkd
  - Total:         ~4BTHkd = 4BTkD

Projections (irreducible for the chosen dimensions):
  - 6 linear layers: 12BTD^2

Minimum total: 12BTD^2 + 256BT^2 + 4BTkD
```

**Potential optimizations to reduce the minimum:**

1. **Sub-quadratic indexer:** Replace the dense `q_I @ k_I^T` with locality-sensitive hashing (LSH) or learned routing to achieve O(T log T) indexer cost.
2. **Smaller indexer:** Reduce `H_idx` or `d_idx` (currently 4 and 32). The indexer only needs to identify top-k positions, not compute precise scores.
3. **Block-sparse indexer:** Compute scores only within local windows + random blocks, reducing from T^2 to T * (w + r) where w = window size, r = random blocks.

| Version | Total FLOPs (T=4096) | Speedup |
|---------|---------------------|---------|
| Current | ~127 G | 1x |
| With LSH indexer | ~120 G | ~1.06x |
| With block-sparse indexer (w=256) | ~108 G | ~1.18x |
| Projections-only lower bound | ~103 G | ~1.23x |

### Cross-Model Comparison: Minimum Ops

| Metric | GDN (ref-aligned) | GSA (ref-aligned) |
|--------|-----------------|---------------|
| FLOPs at T=4096 | ~175 G | ~186 G |
| Scaling with T | Linear (projections dominate) | Quadratic (indexer T^2) |
| Parallelism | Sequential (T steps) | Fully parallel |
| Memory (activation) | O(BH*d_k*d_v) constant | O(BT^2) quadratic |
| Reference kernel | `chunk_gated_delta_rule` (Triton) | `triton_sparse_attention` + `triton_gated_indexer` |

---

## Correctness Analysis

### Original Bugs (All Fixed via Reference Alignment)

The original implementation had 5 bugs that were resolved by rewriting the code to match the reference repositories:

| Bug | Severity | Root Cause | Resolution |
|-----|----------|------------|------------|
| `pytorch_sparse_attention` returned zeros | Critical | Stub implementation | Rewrote to match `gsa/kernels/triton_sparse_attn.py -> _pytorch_sparse_attention()` using 5D gather |
| Variance on masked values | Medium | Naive `-inf` -> `0.0` replacement | Rewrote as `AdaptiveTopKSelector._compute_adaptive_k()` matching reference |
| Redundant `torch.no_grad()` | Minor | Oversight | Removed; added `model.eval()` |
| RoPE concat instead of interleave | Minor | Non-standard rotation | Replaced with standard `_rotate_half` + `apply_rotary_pos_emb` matching both refs |
| `variance_ema` never used | Medium | Disconnected buffer | Eliminated; using `AdaptiveTopKSelector` module matching reference |

### Current Implementation Verification

Both models now match their reference repos:

**GDN verification points:**
- Delta rule formula matches `recurrent_gated_delta_rule_ref` exactly: `S *= g.exp(); v_new = (v - (S*k).sum(-2)) * beta; S += k @ v_new^T; o = q @ S`
- Mamba-style gating: `gk = -A_log.exp() * softplus(gk + dt_bias)`
- `expand_k=0.75`, `expand_v=1.5` asymmetric key/value dimensions
- `FusedRMSNormSwishGate` output normalization

**GSA verification points:**
- `GatedLightningIndexer` with `d_indexer=64`, xavier init (gain=1.0/0.1)
- `AdaptiveTopKSelector` with variance-based method, `k_base=2048`
- `ValueGate` (G2) and `OutputGate` (G1) with `bias_init=0.5`
- Sparse gather via `_gather_along_seq` pattern (5D expand + `torch.gather`)
- Indices shape `[B, T, k_selected]` shared across attention heads

---

## Benchmark Configuration

| Parameter | GDN | GSA |
|-----------|-----|-----|
| Batch size | 1 | 1 |
| Sequence lengths | 1024, 2048, 4096, 8192 | 1024, 2048, 4096, 8192 |
| Hidden dimension | 2048 | 2048 |
| Attention heads | 16 | 16 |
| Key head dim (d_k) | 96 (expand_k=0.75) | 128 (D/H) |
| Value head dim (d_v) | 192 (expand_v=1.5) | 128 (D/H) |
| Indexer heads | - | 4 |
| Indexer dimension | - | 64 |
| k_base / k_min / k_max | - | 2048 / 256 / 4096 |
| Conv kernel size | 4 | - |
| Value gate (G2) | - | Yes (bias_init=0.5) |
| Output gate (G1) | - | Yes (bias_init=0.5) |
| Warmup / Timed iterations | 5 / 10 | 5 / 10 |
| Dtype | bfloat16 | bfloat16 |
| Gradients | Disabled | Disabled |

### Metrics

- **Throughput:** `(B * T) / (elapsed_ms / 1000)` tokens/second
- **Latency:** Average ms per forward pass (over 10 runs)
- **Peak Memory:** `torch.cuda.max_memory_allocated()` in MB

---

## How to Run

### Prerequisites

```bash
pip install torch matplotlib numpy
```

- CUDA-capable GPU required for timing and memory benchmarks
- Triton is optional (PyTorch fallbacks are used when unavailable)

### Execution

Open `benchmark_models.ipynb` in Jupyter and run all cells. On CPU-only machines, the benchmark will print a skip message.

### Expected Output

```
Benchmarking GatedDeltaNet...
B    T      D      | Time (ms)  | Tokens/s   | Mem (MB)
------------------------------------------------------------
1    1024   2048   | XXX.XX     | XXXXX      | XXX
1    2048   2048   | XXX.XX     | XXXXX      | XXX
...

Benchmarking GatedSparseAttention...
...
```

Two plots are generated: Throughput vs Sequence Length and Memory vs Sequence Length.

### Important Caveats

1. **GDN uses a Python loop** (`recurrent_gated_delta_rule_ref`) for the recurrence. The reference repo uses `chunk_gated_delta_rule` with Triton kernels for 10-100x faster execution. The benchmark measures the algorithm's *relative* scaling behavior, not absolute production performance.
2. **GSA sparse attention** uses a PyTorch `gather`-based fallback matching the reference `_pytorch_sparse_attention`. The reference repo also provides `triton_sparse_attention` and `triton_gated_indexer` kernels for GPU-optimized execution.
3. **Both implementations are aligned** with their reference repos in terms of architecture, tensor shapes, gating formulas, and mathematical operations. Only the kernel backends differ (PyTorch vs Triton).