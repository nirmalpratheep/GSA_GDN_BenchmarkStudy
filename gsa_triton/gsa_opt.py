"""
Gated Sparse Attention — Triton-Optimized Implementation.

Drop-in replacement for gsa_ref.GatedSparseAttention with identical
constructor/forward signatures but using fused Triton kernels for:
  1. Gated Lightning Indexer  (eliminates [B, H_idx, T, T] intermediate)
  2. Sparse Attention          (eliminates 5D expand + gather)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
from typing import Optional, Tuple


# ═══════════════════════════════════════════════════════════════════════════
# Triton Kernel 1: Fused Gated Indexer
#
# Fuses: einsum(q_I, k_I) + sigmoid(+bias) + sigmoid(w) * scores + head
#        reduction + causal mask into a single kernel.
# Eliminates the [B, H_idx, T, T] float32 intermediate (~256 MB at T=4096).
# ═══════════════════════════════════════════════════════════════════════════

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_Q": 64, "BLOCK_K": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_Q": 128, "BLOCK_K": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_Q": 64, "BLOCK_K": 128}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_Q": 128, "BLOCK_K": 128}, num_warps=8, num_stages=2),
    ],
    key=["T", "D_IDX", "N_HEADS"],
)
@triton.jit
def _fused_gated_indexer_kernel(
    # Pointers
    Q_ptr,    # [B, T, N_HEADS, D_IDX]  — indexer queries
    K_ptr,    # [B, T, D_IDX]           — indexer keys (shared)
    W_ptr,    # [B, T, N_HEADS]         — importance weight logits
    BIAS_ptr, # [N_HEADS]               — per-head bias
    OUT_ptr,  # [B, T, T]               — output scores
    # Strides
    stride_qb, stride_qt, stride_qh, stride_qd,
    stride_kb, stride_kt, stride_kd,
    stride_wb, stride_wt, stride_wh,
    stride_ob, stride_oq, stride_ok,
    # Dims
    T: tl.constexpr,
    D_IDX: tl.constexpr,
    N_HEADS: tl.constexpr,
    scale: tl.constexpr,
    CAUSAL: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Grid: (B, ceil(T/BLOCK_Q), ceil(T/BLOCK_K))
    pid_b = tl.program_id(0)
    pid_q = tl.program_id(1)
    pid_k = tl.program_id(2)

    q_start = pid_q * BLOCK_Q
    k_start = pid_k * BLOCK_K

    q_offs = q_start + tl.arange(0, BLOCK_Q)  # [BLOCK_Q]
    k_offs = k_start + tl.arange(0, BLOCK_K)  # [BLOCK_K]
    d_offs = tl.arange(0, D_IDX)              # [D_IDX] — full dim in one pass

    q_mask = q_offs < T
    k_mask = k_offs < T

    # Accumulator for head-reduced scores: [BLOCK_Q, BLOCK_K]
    acc = tl.zeros([BLOCK_Q, BLOCK_K], dtype=tl.float32)

    for h in range(N_HEADS):
        # Load bias for this head
        bias_h = tl.load(BIAS_ptr + h).to(tl.float32)

        # Load Q block: [BLOCK_Q, D_IDX]
        q_ptrs = (
            Q_ptr
            + pid_b * stride_qb
            + q_offs[:, None] * stride_qt
            + h * stride_qh
            + d_offs[None, :] * stride_qd
        )
        q_block = tl.load(q_ptrs, mask=q_mask[:, None], other=0.0).to(tl.float32)

        # Load K block: [BLOCK_K, D_IDX]
        k_ptrs = (
            K_ptr
            + pid_b * stride_kb
            + k_offs[:, None] * stride_kt
            + d_offs[None, :] * stride_kd
        )
        k_block = tl.load(k_ptrs, mask=k_mask[:, None], other=0.0).to(tl.float32)

        # Dot product: [BLOCK_Q, BLOCK_K]
        dot = tl.dot(q_block, tl.trans(k_block)) * scale

        # Sigmoid(dot + bias)
        gated = tl.sigmoid(dot + bias_h)

        # Load importance weights for this head: [BLOCK_Q]
        w_ptrs = W_ptr + pid_b * stride_wb + q_offs * stride_wt + h * stride_wh
        w_vals = tl.load(w_ptrs, mask=q_mask, other=0.0).to(tl.float32)
        w_sig = tl.sigmoid(w_vals)  # [BLOCK_Q]

        # Accumulate: weight * gated_score
        acc += w_sig[:, None] * gated

    # Causal mask: positions where k > q get -inf
    if CAUSAL:
        causal_mask = k_offs[None, :] > q_offs[:, None]
        acc = tl.where(causal_mask, float("-inf"), acc)

    # Mask out-of-bounds
    valid_mask = q_mask[:, None] & k_mask[None, :]
    acc = tl.where(valid_mask, acc, float("-inf"))

    # Store output: [BLOCK_Q, BLOCK_K]
    out_ptrs = (
        OUT_ptr
        + pid_b * stride_ob
        + q_offs[:, None] * stride_oq
        + k_offs[None, :] * stride_ok
    )
    tl.store(out_ptrs, acc, mask=valid_mask)


def fused_gated_indexer(
    q_idx: torch.Tensor,  # [B, T, n_heads, d_idx]
    k_idx: torch.Tensor,  # [B, T, d_idx]
    w: torch.Tensor,      # [B, T, n_heads]
    bias: torch.Tensor,   # [n_heads]
    scale: float,
    causal: bool = True,
) -> torch.Tensor:
    B, T, N_H, D_IDX = q_idx.shape
    out = torch.empty(B, T, T, device=q_idx.device, dtype=torch.float32)

    grid = lambda meta: (
        B,
        triton.cdiv(T, meta["BLOCK_Q"]),
        triton.cdiv(T, meta["BLOCK_K"]),
    )

    _fused_gated_indexer_kernel[grid](
        q_idx, k_idx, w, bias, out,
        # Q strides
        q_idx.stride(0), q_idx.stride(1), q_idx.stride(2), q_idx.stride(3),
        # K strides
        k_idx.stride(0), k_idx.stride(1), k_idx.stride(2),
        # W strides
        w.stride(0), w.stride(1), w.stride(2),
        # Out strides
        out.stride(0), out.stride(1), out.stride(2),
        # Dims
        T=T, D_IDX=D_IDX, N_HEADS=N_H, scale=scale, CAUSAL=causal,
    )
    return out


# ═══════════════════════════════════════════════════════════════════════════
# Triton Kernel 2: Fused Sparse Attention
#
# Fuses: gather K/V by indices + QK dot + online softmax + AV output.
# Eliminates the 5D expand that would create ~34 GB at T=4096.
# Uses online softmax (FlashAttention-style) over K_SEL chunks.
# ═══════════════════════════════════════════════════════════════════════════

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_T": 16, "BLOCK_K": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_T": 16, "BLOCK_K": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_T": 32, "BLOCK_K": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_T": 32, "BLOCK_K": 64}, num_warps=8, num_stages=2),
    ],
    key=["T", "K_SEL", "D_HEAD"],
)
@triton.jit
def _fused_sparse_attn_kernel(
    # Pointers
    Q_ptr,    # [B, T, H, D_HEAD]
    K_ptr,    # [B, T_KV, H, D_HEAD]
    V_ptr,    # [B, T_KV, H, D_HEAD]
    IDX_ptr,  # [B, T, K_SEL]   int64 indices
    MASK_ptr, # [B, T, K_SEL]   int8 mask
    OUT_ptr,  # [B, T, H, D_HEAD]
    # Q strides
    stride_qb, stride_qt, stride_qh, stride_qd,
    # K strides
    stride_kb, stride_kt, stride_kh, stride_kd,
    # V strides
    stride_vb, stride_vt, stride_vh, stride_vd,
    # Index strides
    stride_ib, stride_it, stride_ik,
    # Mask strides
    stride_mb, stride_mt, stride_mk,
    # Out strides
    stride_ob, stride_ot, stride_oh, stride_od,
    # Dims
    T: tl.constexpr,
    T_KV: tl.constexpr,
    H: tl.constexpr,
    K_SEL: tl.constexpr,
    D_HEAD: tl.constexpr,
    scale: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Fused sparse attention with online softmax.

    Grid: (B * H, ceil(T / BLOCK_T))
    Each program handles BLOCK_T query positions for one (batch, head).

    Algorithm:
      For each query position, iterate over K_SEL selected keys in
      chunks of BLOCK_K. Uses online softmax (FlashAttention-style)
      to avoid materializing the full attention weight matrix.

      Phase 1: Compute scores for BLOCK_K keys (gather K, dot with Q)
      Phase 2: Update online softmax state (m, l) and accumulate V
    """
    pid_bh = tl.program_id(0)
    pid_t = tl.program_id(1)

    pid_b = pid_bh // H
    pid_h = pid_bh % H

    t_start = pid_t * BLOCK_T
    t_offs = t_start + tl.arange(0, BLOCK_T)  # [BLOCK_T]
    t_mask = t_offs < T

    d_offs = tl.arange(0, D_HEAD)  # [D_HEAD]

    # Load Q vectors: [BLOCK_T, D_HEAD]
    q_ptrs = (
        Q_ptr
        + pid_b * stride_qb
        + t_offs[:, None] * stride_qt
        + pid_h * stride_qh
        + d_offs[None, :] * stride_qd
    )
    q_block = tl.load(q_ptrs, mask=t_mask[:, None], other=0.0).to(tl.float32)

    # Online softmax state
    m_prev = tl.full([BLOCK_T], float("-inf"), dtype=tl.float32)  # running max
    l_prev = tl.zeros([BLOCK_T], dtype=tl.float32)                # running sum(exp)
    acc = tl.zeros([BLOCK_T, D_HEAD], dtype=tl.float32)           # running weighted V

    # Iterate over selected keys in chunks of BLOCK_K
    for k_start in range(0, K_SEL, BLOCK_K):
        # ── Phase 1: Compute scores for this chunk ──
        scores = tl.full([BLOCK_T, BLOCK_K], float("-inf"), dtype=tl.float32)

        for ki in tl.static_range(BLOCK_K):
            ki_abs = k_start + ki
            if ki_abs < K_SEL:
                # Load index for each query: idx[b, t, ki_abs]
                idx_col = tl.load(
                    IDX_ptr
                    + pid_b * stride_ib
                    + t_offs * stride_it
                    + ki_abs * stride_ik,
                    mask=t_mask,
                    other=0,
                )
                idx_col = tl.minimum(tl.maximum(idx_col, 0), T_KV - 1)

                # Gather K: K[b, idx, h, :] -> [BLOCK_T, D_HEAD]
                k_ptrs = (
                    K_ptr
                    + pid_b * stride_kb
                    + idx_col[:, None] * stride_kt
                    + pid_h * stride_kh
                    + d_offs[None, :] * stride_kd
                )
                k_vec = tl.load(k_ptrs, mask=t_mask[:, None], other=0.0).to(tl.float32)

                # Dot product: [BLOCK_T]
                dot = tl.sum(q_block * k_vec, axis=1) * scale

                # Apply validity mask
                mask_col = tl.load(
                    MASK_ptr
                    + pid_b * stride_mb
                    + t_offs * stride_mt
                    + ki_abs * stride_mk,
                    mask=t_mask,
                    other=0,
                )
                dot = tl.where(mask_col != 0, dot, float("-inf"))

                # Store score into column ki
                ki_selector = tl.arange(0, BLOCK_K) == ki
                scores = tl.where(ki_selector[None, :], dot[:, None], scores)

        # ── Phase 2: Online softmax update ──
        # Current block max
        m_cur = tl.max(scores, axis=1)  # [BLOCK_T]
        m_new = tl.maximum(m_prev, m_cur)

        # Correction factor for previous accumulator
        alpha = tl.exp(m_prev - m_new)

        # exp(scores - m_new): [BLOCK_T, BLOCK_K]
        p = tl.exp(scores - m_new[:, None])
        # Zero out entries that were -inf (now exp(-inf - m_new) = 0, but be safe)
        p = tl.where(scores > float("-inf") + 1.0, p, 0.0)

        # Update running sum
        l_new = alpha * l_prev + tl.sum(p, axis=1)

        # Rescale previous accumulator
        acc = acc * alpha[:, None]

        # ── Accumulate V weighted by p for this chunk ──
        for ki in tl.static_range(BLOCK_K):
            ki_abs = k_start + ki
            if ki_abs < K_SEL:
                # Reload index (compiler should optimize redundant loads)
                idx_col = tl.load(
                    IDX_ptr
                    + pid_b * stride_ib
                    + t_offs * stride_it
                    + ki_abs * stride_ik,
                    mask=t_mask,
                    other=0,
                )
                idx_col = tl.minimum(tl.maximum(idx_col, 0), T_KV - 1)

                # Gather V: V[b, idx, h, :] -> [BLOCK_T, D_HEAD]
                v_ptrs = (
                    V_ptr
                    + pid_b * stride_vb
                    + idx_col[:, None] * stride_vt
                    + pid_h * stride_vh
                    + d_offs[None, :] * stride_vd
                )
                v_vec = tl.load(v_ptrs, mask=t_mask[:, None], other=0.0).to(tl.float32)

                # Extract p[:, ki] -> [BLOCK_T]
                ki_selector = tl.arange(0, BLOCK_K) == ki
                p_ki = tl.sum(tl.where(ki_selector[None, :], p, 0.0), axis=1)

                acc += p_ki[:, None] * v_vec

        m_prev = m_new
        l_prev = l_new

    # Final normalization: acc / l
    safe_l = tl.where(l_prev > 0, l_prev, 1.0)
    acc = acc / safe_l[:, None]
    # Zero out positions where all keys were masked
    acc = tl.where(l_prev[:, None] > 0, acc, 0.0)

    # Store output
    out_ptrs = (
        OUT_ptr
        + pid_b * stride_ob
        + t_offs[:, None] * stride_ot
        + pid_h * stride_oh
        + d_offs[None, :] * stride_od
    )
    tl.store(out_ptrs, acc.to(OUT_ptr.dtype.element_ty), mask=t_mask[:, None])


def fused_sparse_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    indices: torch.Tensor,
    mask: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    B, T, H, D = q.shape
    K_SEL = indices.shape[2]
    out = torch.empty_like(q)

    # Ensure indices are int64 and mask is int8 for Triton
    indices = indices.to(torch.int64).contiguous()
    mask = mask.to(torch.int8).contiguous()
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()

    grid = lambda meta: (B * H, triton.cdiv(T, meta["BLOCK_T"]))

    _fused_sparse_attn_kernel[grid](
        q, k, v, indices, mask, out,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        indices.stride(0), indices.stride(1), indices.stride(2),
        mask.stride(0), mask.stride(1), mask.stride(2),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        T=T, T_KV=k.shape[1], H=H, K_SEL=K_SEL, D_HEAD=D, scale=scale,
    )
    return out


# ═══════════════════════════════════════════════════════════════════════════
# Reusable components (same as ref — no Triton needed for these)
# ═══════════════════════════════════════════════════════════════════════════

def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 8192, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self._set_cos_sin_cache(max_seq_len)

    def _set_cos_sin_cache(self, seq_len: int):
        t = torch.arange(seq_len, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, seq_len, device, dtype=None):
        if seq_len > self.cos_cached.shape[0]:
            self._set_cos_sin_cache(seq_len)
            self.cos_cached = self.cos_cached.to(device)
            self.sin_cached = self.sin_cached.to(device)
        cos = self.cos_cached[:seq_len].to(device)
        sin = self.sin_cached[:seq_len].to(device)
        if dtype is not None:
            cos = cos.to(dtype)
            sin = sin.to(dtype)
        return cos, sin


# ── Sub-modules with same state_dict keys as ref ──────────────────────────

class GatedLightningIndexer(nn.Module):
    def __init__(self, hidden_size: int, d_indexer: int = 64, n_idx_heads: int = 4):
        super().__init__()
        self.d_indexer = d_indexer
        self.n_idx_heads = n_idx_heads
        self.scale = 1.0 / math.sqrt(d_indexer)

        self.idx_q_proj = nn.Linear(hidden_size, n_idx_heads * d_indexer, bias=False)
        self.idx_k_proj = nn.Linear(hidden_size, d_indexer, bias=False)
        self.idx_w_proj = nn.Linear(hidden_size, n_idx_heads, bias=True)
        self.idx_bias = nn.Parameter(torch.zeros(n_idx_heads))

        nn.init.xavier_uniform_(self.idx_q_proj.weight, gain=1.0)
        nn.init.xavier_uniform_(self.idx_k_proj.weight, gain=1.0)
        nn.init.xavier_uniform_(self.idx_w_proj.weight, gain=0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        q_idx = self.idx_q_proj(x).view(B, T, self.n_idx_heads, self.d_indexer)
        k_idx = self.idx_k_proj(x)
        w = self.idx_w_proj(x)

        return fused_gated_indexer(
            q_idx.contiguous(),
            k_idx.contiguous(),
            w.contiguous(),
            self.idx_bias,
            self.scale,
            causal=True,
        )


class AdaptiveTopKSelector(nn.Module):
    def __init__(self, k_base: int = 2048, k_min: int = 256, k_max: int = 4096):
        super().__init__()
        self.k_base = k_base
        self.k_min = k_min
        self.k_max = k_max

    def forward(self, scores: torch.Tensor):
        B, T, T_kv = scores.shape
        k_effective = min(self.k_base, self.k_max, T_kv)

        scores_for_topk = scores.masked_fill(scores == float("-inf"), -1e9)
        _, indices = torch.topk(scores_for_topk, k_effective, dim=-1)

        gathered_scores = torch.gather(scores, -1, indices)
        mask = gathered_scores != float("-inf")

        return indices, mask, k_effective


class ValueGate(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        nn.init.constant_(self.gate_proj.bias, 0.5)

    def forward(self, v: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.gate_proj(x))
        B, T, H, d = v.shape
        return v * gate.view(B, T, H, d)


class OutputGate(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        nn.init.constant_(self.gate_proj.bias, 0.5)

    def forward(self, output: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.gate_proj(x))
        B, T, H, d = output.shape
        return output * gate.view(B, T, H, d)


# ── Main Module ────────────────────────────────────────────────────────────

class GatedSparseAttention(nn.Module):
    """
    Triton-optimized Gated Sparse Attention.
    Drop-in replacement for gsa_ref.GatedSparseAttention.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        k_max: int = 4096,
        d_indexer: int = 64,
        n_idx_heads: int = 4,
        k_base: int = 2048,
        k_min: int = 256,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.d_head = hidden_size // num_heads
        self.scale = self.d_head ** -0.5

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        self.indexer = GatedLightningIndexer(hidden_size, d_indexer, n_idx_heads)
        self.topk_selector = AdaptiveTopKSelector(k_base, k_min, k_max)
        self.value_gate = ValueGate(hidden_size)
        self.output_gate = OutputGate(hidden_size)
        self.rotary_emb = RotaryEmbedding(self.d_head)

    def forward(
        self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B, T, D = x.shape

        # Step 1: QKV projections
        q = self.q_proj(x).view(B, T, self.num_heads, self.d_head)
        k = self.k_proj(x).view(B, T, self.num_heads, self.d_head)
        v = self.v_proj(x).view(B, T, self.num_heads, self.d_head)

        # Step 2: Value Gate (G2)
        v = self.value_gate(v, x)

        # Step 3: RoPE
        cos, sin = self.rotary_emb(T, x.device, x.dtype)
        cos = cos.unsqueeze(0).unsqueeze(2)
        sin = sin.unsqueeze(0).unsqueeze(2)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Step 4: Fused Gated Indexer (Triton)
        indexer_scores = self.indexer(x)

        # Step 5: Top-K selection
        indices, mask, k_eff = self.topk_selector(indexer_scores)

        # Step 6: Fused Sparse Attention (Triton)
        attn_out = fused_sparse_attention(q, k, v, indices, mask, self.scale)

        # Step 7: Output Gate (G1)
        attn_out = self.output_gate(attn_out, x)

        # Step 8: Output projection
        return self.o_proj(attn_out.reshape(B, T, D))
