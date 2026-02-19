# GSA Triton: Optimized Gated Sparse Attention

Triton-optimized implementation of [Gated Sparse Attention](https://github.com/alfredcs/Gated-Sparse-Attention) with a pure-PyTorch reference baseline. Both produce identical outputs and are benchmarked head-to-head.

## Quick Start

```bash
cd gsa_triton
uv run benchmark.py                    # verify correctness + full benchmark
uv run benchmark.py --skip-verify      # benchmark only (skip correctness check)
uv run benchmark.py --verify-T 256     # smaller verification seq len
uv run benchmark.py --dtype fp32       # use float32 instead of bfloat16
uv run benchmark.py --atol 0.2         # relax numerical tolerance
```

## Files

| File | Purpose |
|---|---|
| `gsa_ref.py` | Pure PyTorch reference GSA (correctness baseline) |
| `gsa_opt.py` | Triton-optimized GSA (two fused kernels + autotune) |
| `benchmark.py` | Numerical verification + throughput/memory benchmark + profiling entry point |

---

## Profiling

Both `gsa_ref.py` and `gsa_opt.py` contain NVTX range annotations around every logical step of the forward pass. These appear in both Nsight Systems (timeline rows) and Nsight Compute (kernel context panel).

### NVTX ranges emitted

| Range | ref | opt |
|---|---|---|
| `gsa_ref/qkv_proj` | yes | — |
| `gsa_ref/value_gate` | yes | — |
| `gsa_ref/rotary_emb` | yes | — |
| `gsa_ref/indexer` | yes | — |
| `gsa_ref/topk_select` | yes | — |
| `gsa_ref/sparse_attn` | yes | — |
| `gsa_ref/output_gate` | yes | — |
| `gsa_ref/o_proj` | yes | — |
| `gsa_opt/qkv_proj` | — | yes |
| `gsa_opt/value_gate` | — | yes |
| `gsa_opt/rotary_emb` | — | yes |
| `gsa_opt/indexer` | — | yes |
| `gsa_opt/topk_select` | — | yes |
| `gsa_opt/sparse_attn` | — | yes |
| `gsa_opt/output_gate` | — | yes |
| `gsa_opt/o_proj` | — | yes |
| `gsa_opt/triton_gated_indexer_kernel` | — | yes (inside indexer) |
| `gsa_opt/triton_sparse_attn_kernel` | — | yes (inside sparse_attn) |

---

### Nsight Systems (`nsys`)

`benchmark.py --profile` runs warmup outside the capture window, then calls `cudaProfilerStart()` / `cudaProfilerStop()` around `PROFILE_ITERS=3` forward passes per model.

```bash
# Both ref and opt (sequential, same report)
nsys profile --trace=cuda,nvtx --capture-range=cudaProfilerApi \
    -o gsa_profile \
    uv run benchmark.py --profile --profile-T 2048

# Reference only
nsys profile --trace=cuda,nvtx --capture-range=cudaProfilerApi \
    -o gsa_ref_profile \
    uv run benchmark.py --profile --profile-ref-only --profile-T 2048

# Optimised only
nsys profile --trace=cuda,nvtx --capture-range=cudaProfilerApi \
    -o gsa_opt_profile \
    uv run benchmark.py --profile --profile-opt-only --profile-T 2048
```

`--profile` flags:

| Flag | Default | Description |
|---|---|---|
| `--profile` | — | Enable nsys mode |
| `--profile-T` | 2048 | Sequence length |
| `--profile-B` | 1 | Batch size |
| `--profile-ref-only` | false | Capture ref model only |
| `--profile-opt-only` | false | Capture opt model only |

Open the resulting `.nsys-rep` in the Nsight Systems GUI.

---

### Nsight Compute (`ncu`)

`benchmark.py --ncu` performs the same warmup then captures **exactly one forward pass** per model. NCU replays each kernel many times internally to collect roofline / memory / compute metrics, so minimizing kernels in the capture window is critical.

```bash
# Reference only — all kernels
ncu --target-processes all --profile-from-start off \
    --replay-mode application --set full \
    -o gsa_ref_ncu \
    uv run benchmark.py --ncu --ncu-ref-only --ncu-T 2048

# Optimised only — all kernels
ncu --target-processes all --profile-from-start off \
    --replay-mode application --set full \
    -o gsa_opt_ncu \
    uv run benchmark.py --ncu --ncu-opt-only --ncu-T 2048

# Optimised only — Triton kernels only (faster, skips cuBLAS replays)
ncu --target-processes all --profile-from-start off \
    --replay-mode application --set full \
    --kernel-name regex:"_fused_gated_indexer|_fused_sparse_attn" \
    -o gsa_opt_triton_ncu \
    uv run benchmark.py --ncu --ncu-opt-only --ncu-T 2048

# Both ref and opt in one report
ncu --target-processes all --profile-from-start off \
    --replay-mode application --set full \
    -o gsa_both_ncu \
    uv run benchmark.py --ncu --ncu-T 2048
```

`--ncu` flags:

| Flag | Default | Description |
|---|---|---|
| `--ncu` | — | Enable ncu mode |
| `--ncu-T` | 2048 | Sequence length |
| `--ncu-B` | 1 | Batch size |
| `--ncu-ref-only` | false | Capture ref model only |
| `--ncu-opt-only` | false | Capture opt model only |

**Key ncu flags explained:**

| Flag | Purpose |
|---|---|
| `--profile-from-start off` | Wait for `cudaProfilerStart()` instead of capturing from process start. This is ncu's equivalent of nsys `--capture-range=cudaProfilerApi`. |
| `--replay-mode application` | Re-run the entire forward pass for each metric section. Required for Triton kernels — kernel-level replay can corrupt Triton's JIT state. |
| `--set full` | Collect the full set of hardware metrics (roofline, memory throughput, compute throughput, occupancy, warp stall reasons). Use `--set default` for a faster lighter pass. |
| `--kernel-name regex:...` | Restrict collection to matching kernel names. Use to focus on the two Triton kernels and skip cuBLAS. |

Open the resulting `.ncu-rep` in the Nsight Compute GUI. To focus on the Triton kernels, filter by `_fused_gated_indexer_kernel` or `_fused_sparse_attn_kernel` in the kernel list.

---

## Architecture: GSA Forward Pass

```
Input x [B, T, D=2048]
  |
  +---> q_proj(x) ---> reshape [B,T,H=16,d=128]  ---> RoPE --------+
  +---> k_proj(x) ---> reshape [B,T,H,d]          ---> RoPE --------+
  +---> v_proj(x) ---> reshape [B,T,H,d] ---> ValueGate(G2) -------+
  |                                                                  |
  +---> GatedLightningIndexer ---> scores [B,T,T] ---+               |
  |       idx_q [D->256], idx_k [D->64]              |               |
  |       idx_w [D->4],   idx_bias [4]               v               |
  |                                            AdaptiveTopK          |
  |                                         indices [B,T,k] ---------+
  |                                         mask [B,T,k] -----------+|
  |                                                                  ||
  |                                       Sparse Attention <---------+|
  |                                              |                    |
  +---> OutputGate(G1) --------------------------+                    |
  |         |                                                         |
  |     o_proj(out) ---> Output [B,T,D]                               |
```

Default parameters: `D=2048, H=16, d_head=128, d_indexer=64, n_idx_heads=4, k=2048`

---

## Optimization Analysis

### Bottleneck 1: Gated Lightning Indexer

**Reference (PyTorch) data flow:**

```
idx_q_proj(x) -> [B,T,4,64]     (Linear)
idx_k_proj(x) -> [B,T,64]       (Linear)
einsum('bqhd,bkd->bhqk')        -> [B,4,T,T]   <-- 256 MB at T=4096, fp32
sigmoid(scores + bias)           -> [B,4,T,T]   <-- another 256 MB
sigmoid(w) * gated_scores        -> [B,4,T,T]   <-- another 256 MB
sum(dim=1)                       -> [B,T,T]
causal_mask                      -> [B,T,T]
```

**Problem:** 4 materialized `[B,4,T,T]` intermediates = **768 MB** of HBM traffic at T=4096. Each is read and written once, bottlenecked by memory bandwidth, not compute.

**Triton kernel `_fused_gated_indexer_kernel`:**

```
Grid: (B, ceil(T/BLOCK_Q), ceil(T/BLOCK_K))

For each output tile [BLOCK_Q, BLOCK_K]:
  acc = zeros                              // in registers
  for h in range(N_HEADS=4):               // 4 iterations, unrolled
    q_block = load Q[b, q_tile, h, :]      // [BLOCK_Q, 64] from HBM
    k_block = load K[b, k_tile, :]         // [BLOCK_K, 64] from HBM
    dot = tl.dot(q_block, k_block^T)       // [BLOCK_Q, BLOCK_K] in SRAM
    gated = sigmoid(dot * scale + bias[h]) // fused elementwise
    w_sig = sigmoid(load W[b, q_tile, h])  // [BLOCK_Q] from HBM
    acc += w_sig * gated                   // accumulate in registers
  causal_mask(acc)                         // single branch
  store acc -> OUT[b, q_tile, k_tile]      // one write to HBM
```

**What is eliminated:**
- The `[B,4,T,T]` intermediate is never materialized (768 MB -> 0)
- `sigmoid`, bias addition, weight multiplication all fused into the dot-product loop
- Causal mask applied once to the final sum, not as a separate kernel
- Output `[B,T,T]` written exactly once (64 MB at T=4096)

**Memory reduction:** 768 MB intermediates -> 0 MB. Total HBM writes drop from ~1 GB to 64 MB.

**Autotune configs:**

| Config | BLOCK_Q | BLOCK_K | Warps | Rationale |
|--------|---------|---------|-------|-----------|
| A | 64 | 64 | 4 | Baseline, good for small T |
| B | 128 | 64 | 4 | Better Q-reuse, amortizes W loads |
| C | 64 | 128 | 8 | Better K-reuse for large T |
| D | 128 | 128 | 8 | Maximum tile, best for T >= 4096 |

`D_IDX=64` is processed in a single pass (no inner D-loop needed since 64 fits in registers).

---

### Bottleneck 2: Sparse Attention with Gather

**Reference (PyTorch) data flow:**

```
indices [B,T,k]                                // from top-k
k.unsqueeze(1).expand(B,T,T_kv,H,D)           -> [B,T,T,16,128]   <-- 34 GB virtual
torch.gather(k_expanded, dim=2, idx_exp)       -> [B,T,k,16,128]   <-- 16 GB at k=2048
(same for V)                                   -> [B,T,k,16,128]   <-- 16 GB
permute(0,1,3,2,4)                             -> [B,T,16,k,128]
einsum('bqhd,bqhkd->bqhk') * scale            -> [B,T,16,k]       <-- 256 MB
softmax(dim=-1)                                -> [B,T,16,k]       <-- 256 MB
einsum('bqhk,bqhkd->bqhd')                    -> [B,T,16,128]
```

**Problem:** The 5D expand + gather creates **32 GB** of materialized K/V gather buffers. Even with PyTorch's lazy expand, the `torch.gather` output is physically `[1,4096,2048,16,128] * 2 bytes = 34 GB` in bf16 for K and V combined. This dominates both memory and time.

**Triton kernel `_fused_sparse_attn_kernel`:**

```
Grid: (B * H, ceil(T / BLOCK_T))

For each program (one batch, one head, BLOCK_T queries):
  q_block = load Q[b, t_tile, h, :]           // [BLOCK_T, 128] from HBM
  m_prev = -inf, l_prev = 0, acc = 0          // online softmax state

  for k_start in range(0, K_SEL, BLOCK_K):    // iterate over selected keys
    // Phase 1: Compute scores
    for ki in static_range(BLOCK_K):
      idx = load IDX[b, t_tile, k_start+ki]   // [BLOCK_T] indices
      k_vec = load K[b, idx[:], h, :]          // [BLOCK_T, 128] gathered on-the-fly
      scores[:, ki] = dot(q_block, k_vec) * scale
      scores[:, ki] = mask_invalid(scores[:, ki])

    // Phase 2: Online softmax (FlashAttention-style)
    m_new = max(m_prev, max(scores))
    alpha = exp(m_prev - m_new)
    p = exp(scores - m_new)
    l_new = alpha * l_prev + sum(p)
    acc = acc * alpha                          // rescale previous accumulator

    // Phase 3: Accumulate V
    for ki in static_range(BLOCK_K):
      v_vec = load V[b, idx[:], h, :]          // [BLOCK_T, 128] gathered
      acc += p[:, ki] * v_vec

    m_prev = m_new; l_prev = l_new

  store acc / l_prev -> OUT[b, t_tile, h, :]   // one write
```

**What is eliminated:**
- The 5D expand `[B,T,T_kv,H,D]` is never created (34 GB -> 0)
- No gather buffer — K/V vectors loaded on-the-fly via index arithmetic
- Attention weights `[B,T,H,k]` never materialized — online softmax in registers
- Single output write per query position

**Memory reduction:** ~34 GB gather buffers -> 0. Only K/V in their original layout are read.

**Online softmax correctness:**
The kernel maintains `(m, l, acc)` per query where:
- `m` = running max of scores (for numerical stability)
- `l` = running sum of `exp(score - m)`
- `acc` = running weighted sum of V vectors

When a new chunk arrives with a higher max `m'`, the previous accumulator is rescaled by `exp(m_old - m_new)`. This is mathematically equivalent to standard softmax but computed in a single streaming pass. This is the same algorithm used by FlashAttention (Dao et al., 2022).

**Autotune configs:**

| Config | BLOCK_T | BLOCK_K | Warps | Rationale |
|--------|---------|---------|-------|-----------|
| A | 16 | 32 | 4 | Low register pressure, small T |
| B | 16 | 64 | 4 | More key parallelism per iteration |
| C | 32 | 32 | 4 | More query parallelism |
| D | 32 | 64 | 8 | Best throughput for large T and k |

---

## Theoretical FLOP Analysis

Using benchmark defaults: `B=1, T=4096, D=2048, H=16, d_head=128, d_idx=64, n_idx_heads=4, k=2048`

### Per-Stage FLOP Count

| Stage | Operation | FLOPs | % of Total |
|-------|-----------|-------|------------|
| **Linear Projections** | | | |
| QKV (3x D->D) | 3 x 2xTxDxD | 100.7B | 40.4% |
| Gate projs (2x D->D) | 2 x 2xTxDxD | 33.6B | 13.5% |
| Output proj (D->D) | 2xTxDxD | 33.6B | 13.5% |
| Indexer projs | 2xTxDx(256+64+4) | 5.4B | 2.2% |
| **Indexer Scoring** | 2xT^2 x d_idx x n_idx | 8.6B | 3.4% |
| **TopK** | O(T^2 log k) comparisons | 0.2B | 0.1% |
| **Sparse Attn QK** | 2xTxkxHxd | 34.4B | 13.8% |
| **Sparse Attn AV** | 2xTxkxHxd | 34.4B | 13.8% |
| **Elementwise** | RoPE, gates, sigmoid | ~0.3B | 0.1% |
| | | | |
| **Total** | | **~249B** | **100%** |

### Theoretical Minimum Latency

The theoretical floor is determined by max(compute_time, memory_time):

| GPU | BF16 TFLOPS | Compute Floor | HBM BW | Memory Floor | Theoretical Min |
|-----|-------------|---------------|--------|--------------|-----------------|
| RTX 4090 | 165 | 1.51 ms | 1.0 TB/s | 1.1 ms | **1.51 ms** (compute-bound) |
| A100 80GB | 312 | 0.80 ms | 2.0 TB/s | 0.55 ms | **0.80 ms** (compute-bound) |
| H100 | 990 | 0.25 ms | 3.35 TB/s | 0.33 ms | **0.33 ms** (memory-bound) |

GSA at these dimensions is **compute-bound on consumer/A100 GPUs** because the linear projections dominate (69.4% of FLOPs are GEMM). On H100, HBM bandwidth becomes the bottleneck.

### Where the Optimizations Matter

The Triton kernels target the **remaining 30.6%** of FLOPs (indexer + sparse attention), but more importantly they eliminate enormous memory intermediates:

| Component | Ref HBM Traffic | Opt HBM Traffic | Reduction |
|-----------|----------------|-----------------|-----------|
| Indexer scoring | ~832 MB (4 intermediates + output) | ~64 MB (output only) | **13x** |
| Sparse attention | ~34 GB (5D gather) | ~2.1 GB (stream K/V) | **16x** |
| Total non-GEMM | ~35 GB | ~2.2 GB | **16x** |

The GEMM operations (linear projections) are already handled optimally by cuBLAS and are not targeted by our kernels. Our optimization focuses on the memory-bound operations where the reference implementation wastes 10-16x more HBM bandwidth than necessary.

### Arithmetic Intensity Analysis

Arithmetic intensity = FLOPs / Bytes transferred

| Operation | Ref AI | Opt AI | GPU Ridge Point |
|-----------|--------|--------|-----------------|
| Indexer (T=4096) | 0.3 FLOP/B | 4.1 FLOP/B | ~100 FLOP/B (A100) |
| Sparse Attn (T=4096) | 0.5 FLOP/B | 8.2 FLOP/B | ~100 FLOP/B (A100) |

Both operations remain memory-bound even after optimization (AI << ridge point), but the optimized versions are **10-16x closer** to the roofline because they eliminate redundant data movement. The remaining gap to the ridge point reflects the inherent irregular access pattern of sparse attention (non-coalesced gathers via token indices).

---

## Why the Ref Implementation is Slow

### 1. Intermediate Tensor Explosion

The reference indexer materializes `[B, n_idx_heads, T, T]` in fp32 four times during its forward pass:

```python
raw_scores = einsum('bqhd,bkd->bhqk', q_idx, k_idx) * scale   # write [B,4,T,T]
gated_scores = sigmoid(raw_scores + bias_exp)                    # read + write [B,4,T,T]
weighted = gated_scores * w_sigmoid                              # read + write [B,4,T,T]
final_scores = weighted.sum(dim=1)                               # read [B,4,T,T], write [B,T,T]
```

Each intermediate is a separate CUDA kernel launch with a full HBM round-trip. At T=4096, each intermediate is `1 x 4 x 4096 x 4096 x 4 bytes = 256 MB`. The total traffic for just the indexer is 4 reads + 4 writes of 256 MB each = **2 GB**.

### 2. The 5D Gather Catastrophe

The reference sparse attention uses PyTorch's `expand + gather` pattern:

```python
k_expanded = k.unsqueeze(1).expand(B, T, T_kv, H, D)    # virtual: 34 GB
k_gathered = torch.gather(k_expanded, 2, idx_exp)        # physical: 17 GB
```

Even though `expand` is lazy (no copy), `gather` must physically write `[B, T, k, H, D]` to a new tensor. At T=4096, k=2048, H=16, D=128, bf16: `1 x 4096 x 2048 x 16 x 128 x 2 = 34 GB` for K and V combined. This exceeds most GPU memory, causing OOM at moderate sequence lengths.

### 3. Kernel Launch Overhead

The reference forward pass launches ~15+ separate CUDA kernels (projections, einsum, sigmoid, mask, topk, expand, gather, einsum, softmax, einsum, gate, projection). Each launch has 5-10 us overhead, adding up to ~100+ us of pure overhead — significant when the target latency is 1-2 ms.

---

## Correctness Verification

The benchmark verifies numerical equivalence between `gsa_ref` and `gsa_opt`:

1. Both models are constructed with identical parameters
2. `ref_model.state_dict()` is loaded into `opt_model` via `load_state_dict(strict=True)`
3. Same input tensor is passed through both
4. `torch.allclose(ref_out, opt_out, atol=0.1, rtol=0.1)` for bf16

The tolerance is set to 0.1 (not 1e-5) because:
- bf16 has only ~3 decimal digits of precision
- The Triton kernels accumulate in fp32 internally but the intermediate precision differs from PyTorch's kernel fusion choices
- Online softmax in the sparse attention kernel reorders operations vs standard softmax, introducing O(eps_fp32) differences that compound over k=2048 selected tokens

The `state_dict` keys match exactly between ref and opt because both use identical attribute names: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `indexer.idx_q_proj`, `indexer.idx_k_proj`, `indexer.idx_w_proj`, `indexer.idx_bias`, `value_gate.gate_proj`, `output_gate.gate_proj`, `rotary_emb.*`.

---

## Limitations and Future Work

### Current Limitations

1. **TopK remains on PyTorch.** `torch.topk` is O(T log k) and launches its own CUDA kernel. Fusing top-k with the indexer output would eliminate one HBM round-trip of the `[B,T,T]` score matrix, but implementing top-k in Triton is complex.

2. **Irregular gather pattern.** The sparse attention kernel loads K/V via `K[b, idx[t,ki], h, :]` where `idx` varies per query. This causes non-coalesced HBM reads. Sorting indices before the kernel could improve L2 hit rates.

3. **D_HEAD must be a power of 2.** The indexer kernel uses `tl.dot` which requires power-of-2 inner dimensions. `d_indexer=64` and `d_head=128` satisfy this. Non-power-of-2 dimensions would need padding.

4. **Linear projections not fused.** The 5 linear projections (QKV + 2 gates) could be merged into a single large GEMM, reducing kernel launch overhead. This is a cuBLAS-level optimization, not a Triton target.

### Potential Further Optimizations

| Optimization | Expected Impact | Complexity |
|---|---|---|
| Fused QKV projection (single GEMM) | -5 kernel launches, ~5% speedup | Low |
| Sort indices before sparse attn | +10-20% L2 hit rate | Low |
| Fused TopK + indexer output | Eliminate [B,T,T] HBM write/read | High |
| Fused RoPE Triton kernel | Eliminate 2 elementwise passes | Low |
| Persistent kernel for sparse attn | Better SM utilization | High |
| FP8 accumulation in indexer | 2x throughput on H100 | Medium |
