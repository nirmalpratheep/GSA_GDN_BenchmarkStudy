# Benchmark Study: Gated DeltaNet (GDN) vs Gated Sparse Attention (GSA)

A comparative benchmark of two attention mechanism architectures measuring throughput (tokens/sec), latency (ms), and peak GPU memory (MB) across varying sequence lengths.

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
| **Gated DeltaNet (GDN)** | Recurrent state-space attention via delta rule | O(T * H * d^3) | O(H * d^2) state |
| **Gated Sparse Attention (GSA)** | Adaptive top-k sparse attention | O(T^2 * H_idx * d_idx) indexer + O(T * k * d) attention | O(T * k) indices |

Where: `B` = batch, `T` = sequence length, `D` = hidden dimension, `H` = attention heads (16), `d` = head_dim (D/H = 128), `H_idx` = indexer heads (4), `d_idx` = indexer dim (32), `k` = adaptive sparsity budget (32-1024).

---

## Architecture Summary

### Gated DeltaNet (GDN)

```
Input [B,T,D]
  |
  +---> Q_proj(D,D) ---> Conv1D(k=4) ---> RoPE ---> L2 Normalize
  +---> K_proj(D,D) ---> Conv1D(k=4) ---> RoPE ---> L2 Normalize
  +---> V_proj(D,D) ---> Conv1D(k=4)
  +---> G_proj(D,D) (gate for output norm)
  +---> beta_proj(D,H) ---> sigmoid  (write gate)
  +---> gk_proj(D,H) ---> softplus ---> exp(-A * ...) = alpha  (decay gate)
  |
  v
  Delta Rule Recurrence (sequential over T):
    for t = 0..T-1:
      o_t = S @ q_t + D * (q_t . k_t) * v_t
      S   = alpha_t * S @ (I - beta_t * k_t k_t^T) + beta_t * v_t k_t^T
  |
  v
  RMSNorm + SiLU Gate ---> O_proj(D,D) ---> Output [B,T,D]
```

### Gated Sparse Attention (GSA)

```
Input [B,T,D]
  |
  +===[Indexer Phase]===+
  |  W_Iq(D, H_idx*d_idx)  -> q_I [B,T,H_idx,d_idx]
  |  W_Ik(D, d_idx)        -> k_I [B,T,d_idx]
  |  W_Iw(D, H_idx)        -> w   [B,T,H_idx]
  |  gate_bias              -> b   [H_idx]
  |       |
  |       v
  |  Gated Indexer: scores = sigmoid(q_I @ k_I^T + b) * sigmoid(w)
  |  Variance computation -> adaptive k_t
  |  TopK selection -> sparse indices [B,T,k_limit]
  |
  +===[Attention Phase]===+
  |  W_q(D,D), W_k(D,D), W_v(D,D)
  |  W_gv(D,D) -> sigmoid -> value gate
  |  RoPE on Q, K
  |  Sparse gather + attention using indices
  |  W_go(D,D) -> sigmoid -> output gate
  |  O_proj(D,D)
  |
  v
  Output [B,T,D]
```

---

## Detailed Operations Per Iteration

All counts are in FLOPs (1 multiply-add = 2 FLOPs). Default config: `B=1, T=variable, D=2048, H=16, d=128, conv_size=4`.

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
| **18** | **Delta recurrence (per step t):** | | |
| 18a | `o_t = S @ q_t` | [B,H,d,d] x [B,H,d] | 2BHd^2 |
| 18b | `(q_t . k_t)` dot product | [B,H,d] | 2BHd |
| 18c | `D * dot * v_t` (skip connection) | [B,H,d] | 2BHd |
| 18d | `v_t @ k_t^T` (outer product) | [B,H,d] x [B,H,d] | BHd^2 |
| 18e | `k_t @ k_t^T` (outer product) | same | BHd^2 |
| 18f | `I - beta * k_outer` | [B,H,d,d] | 2BHd^2 |
| 18g | `S @ orthogonal_proj` (mat-mat) | [B,H,d,d] x [B,H,d,d] | **2BHd^3** |
| 18h | `alpha * result + beta * v_outer` | [B,H,d,d] | 3BHd^2 |
| | **Subtotal per step** | | **2BHd^3 + ~9BHd^2** |
| | **Total recurrence (x T steps)** | | **T(2BHd^3 + 9BHd^2)** |
| 19 | RMSNorm on output | [BTH, d] | ~5BTD |
| 20 | SiLU gate | [BTH, d] | ~2BTD |
| 21 | Output projection (`o @ W_o`) | [B,T,D] x [D,D] | 2BTD^2 |

#### GDN Total FLOPs (one forward pass)

```
Projections:      5 * 2BTD^2           = 10BTD^2
                  + 2 * 2BTDH          = 4BTDH
Convolutions:     3 * 8BTD             = 24BTD
RoPE + Norm:      ~25BTD
Delta recurrence: T * (2BHd^3 + 9BHd^2)
Output norm+proj: ~7BTD + 2BTD^2       = 12BTD^2 total with projections

DOMINANT TERM:  T * 2BHd^3  (recurrence matrix multiply)
```

**Numerical example (B=1, T=4096, D=2048, H=16, d=128):**

| Component | FLOPs | % of Total |
|-----------|-------|------------|
| Linear projections (6 total) | 12 * 2048^2 * 4096 = ~102.8 G | ~6.0% |
| Delta recurrence | 4096 * 2 * 16 * 128^3 = **~275.4 G** | **~87.6%** |
| Convolutions | 24 * 4096 * 2048 = ~201 M | <0.1% |
| RoPE + Norm + Gates | ~25 * 4096 * 2048 = ~210 M | <0.1% |
| **Total** | **~378 GFLOPs** | |

### Gated Sparse Attention: Operation Breakdown

| # | Operation | Shape Transformation | FLOPs |
|---|-----------|---------------------|-------|
| **Indexer Phase** | | | |
| 1 | W_Iq projection | [B,T,D] x [D, H_idx*d_idx] | 2BT * D * (4*32) = 2BTD*128 |
| 2 | W_Ik projection | [B,T,D] x [D, d_idx] | 2BTD*32 |
| 3 | W_Iw projection | [B,T,D] x [D, H_idx] | 2BTD*4 |
| 4 | Gated indexer QK scores | [B,T,4,32] x [B,T,32] -> [B,4,T,T] | 2BT^2 * H_idx * d_idx |
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
Indexer projections:  2BTD * (128 + 32 + 4)   = 328BTD
Indexer QK scores:    2BT^2 * 4 * 32           = 256BT^2
Indexer overhead:     ~12BT^2
TopK:                 ~BT^2 * log(k)
Attention projections: 6 * 2BTD^2              = 12BTD^2
Sparse attention:      4BTHkd                  = 4BTkD  (since Hd = D)
RoPE + gates:         ~15BTD

DOMINANT TERMS:  12BTD^2 (projections)  +  256BT^2 (indexer)  +  4BTkD (sparse attn)
```

**Numerical example (B=1, T=4096, D=2048, H=16, d=128, k=512):**

| Component | FLOPs | % of Total |
|-----------|-------|------------|
| Attention projections (6x) | 12 * 2048^2 * 4096 = ~102.8 G | ~59.2% |
| Indexer QK scores | 256 * 4096^2 = ~4.3 G | ~2.5% |
| Indexer projections | 328 * 4096 * 2048 = ~2.75 G | ~1.6% |
| Sparse attention (QK + AV) | 4 * 4096 * 512 * 2048 = ~17.2 G | ~9.9% |
| TopK selection | ~4096^2 * 10 = ~168 M | <0.1% |
| **Total** | **~127 GFLOPs** | |

Note: GSA's sparse attention FLOPs are reported as theoretical. The current `pytorch_sparse_attention` returns zeros (see [Correctness Analysis](#correctness-analysis)), so actual measured benchmark numbers for GSA are invalid.

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

The delta rule recurrence is mathematically irreducible:

```
Per timestep (minimum required):
  - Read q_t, k_t, v_t, alpha_t, beta_t:      5 * BHd reads
  - o_t = S @ q_t:                              2BHd^2  FLOPs
  - S update (rank-1 + decay):                  2BHd^2  FLOPs (using Woodbury/rank-1 update)
  - Write o_t, S:                               BHd + BHd^2 writes

Minimum per step:   4BHd^2 FLOPs  (with optimized rank-1 state update)
Minimum total:      4BTHd^2 FLOPs
```

**Key insight:** The `S @ orthogonal_proj` (d x d matrix multiply, costing 2BHd^3 per step) in the current implementation is NOT the minimum. Using the **Woodbury identity** or **rank-1 update formulation**, the state update can be rewritten as:

```
S' = alpha * S - alpha * beta * S @ (k_t @ k_t^T) + beta * (v_t @ k_t^T)
   = alpha * S - alpha * beta * (S @ k_t) @ k_t^T + beta * v_t @ k_t^T
```

This avoids the d x d matrix multiply entirely, replacing it with two matrix-vector products (`S @ k_t` costs 2BHd^2) and two outer products (BHd^2 each).

| Version | Per-step FLOPs | Total (T=4096) | Speedup |
|---------|---------------|----------------|---------|
| Current (mat-mat) | 2BHd^3 + 9BHd^2 = ~70.6M | ~289 G | 1x |
| **Optimized (rank-1)** | ~8BHd^2 = ~33.6M | **~137 G** | **~2.1x** |

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

| Metric | GDN (optimized) | GSA (current) |
|--------|-----------------|---------------|
| FLOPs at T=4096 | ~137 G | ~127 G |
| FLOPs at T=16384 | ~549 G | ~567 G (indexer dominates) |
| **Crossover point** | ~T=12000 | ~T=12000 |
| Parallelism | Sequential (T steps) | Fully parallel |
| Memory (activation) | O(BHd^2) constant | O(BT^2) quadratic |

---

## Correctness Analysis (All Fixed)

The following bugs were identified and have been fixed in the notebook:

### BUG 1 (Critical) - FIXED: `pytorch_sparse_attention` Was Returning All Zeros

**What was wrong:** The sparse attention fallback returned `torch.zeros(...)` instead of computing actual attention, making all GSA outputs zero and benchmark results invalid.

**Fix applied:** Implemented full sparse gather-and-attend using `torch.gather` to select K,V by indices, `einsum` for QK scores and weighted sum, with proper masking and softmax.

### BUG 2 (Medium) - FIXED: Variance Computed on Masked Values

**What was wrong:** Replacing `-inf` with `0.0` before `var()` inflated variance for early positions where most keys are causally masked.

**Fix applied:** Variance is now computed only over valid (non-masked) entries using correct count-based denominator.

### BUG 3 (Minor) - FIXED: Redundant `torch.no_grad()` in Benchmark

**What was wrong:** `with torch.no_grad()` was used despite gradients already being globally disabled.

**Fix applied:** Removed redundant context manager. Added `model.eval()` for proper batch norm / dropout behavior.

### BUG 4 (Minor) - FIXED: RoPE Concatenation Instead of Interleaving

**What was wrong:** Rotated pairs were concatenated as `[all_first_halves, all_second_halves]` instead of interleaved `[pair0_a, pair0_b, pair1_a, pair1_b, ...]`.

**Fix applied:** Uses `torch.stack(..., dim=-1).flatten(-2)` to properly interleave the rotated dimensions. Splits input into first/second halves instead of even/odd indices.

### BUG 5 (Medium) - FIXED: GSA `variance_ema` Buffer Never Used

**What was wrong:** `variance_ema` buffer was registered but `None` was passed to `fused_indexer_topk`, so adaptive-k used global mean instead of a stabilized EMA.

**Fix applied:** `self.variance_ema` is now passed to `fused_indexer_topk`. Added EMA update logic during training with momentum=0.1.

---

## Benchmark Configuration

| Parameter | Value |
|-----------|-------|
| Batch size | 1 |
| Sequence lengths | 1024, 2048, 4096, 8192 |
| Hidden dimension | 2048 |
| Attention heads (both models) | 16 |
| Head dimension | 128 |
| Indexer heads (GSA) | 4 |
| Indexer dimension (GSA) | 32 |
| Conv kernel size (GDN) | 4 |
| Warmup iterations | 5 |
| Timed iterations | 10 |
| Dtype | bfloat16 |
| Gradients | Disabled |

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

1. **GDN uses a Python loop** for the recurrence. Real-world implementations use fused Triton/CUDA kernels that are 10-100x faster. The benchmark measures the algorithm's *relative* scaling behavior, not absolute production performance.
2. **The GDN recurrence can be optimized** from O(Td^3) to O(Td^2) per head using rank-1 state updates (see [Theoretical Minimum Operations](#theoretical-minimum-operations)).
3. **GSA sparse attention** uses a PyTorch gather-based fallback that is slower than a fused Triton kernel would be. The throughput numbers reflect this overhead.