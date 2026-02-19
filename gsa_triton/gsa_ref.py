"""
Gated Sparse Attention — Pure PyTorch Reference Implementation.

Aligned with: https://github.com/alfredcs/Gated-Sparse-Attention

This module provides a self-contained GatedSparseAttention nn.Module
that uses only PyTorch ops (no Triton). It serves as the correctness
baseline for gsa_opt.py.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

# NVTX is bundled with PyTorch's CUDA build; fall back to no-ops if unavailable.
try:
    from torch.cuda import nvtx as _nvtx
    _nvtx_push = _nvtx.range_push
    _nvtx_pop  = _nvtx.range_pop
except Exception:
    def _nvtx_push(msg: str): pass
    def _nvtx_pop(): pass


# ── Rotary Position Embeddings ─────────────────────────────────────────────

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

    def forward(
        self, seq_len: int, device: torch.device, dtype: torch.dtype = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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


# ── Gated Lightning Indexer ────────────────────────────────────────────────

class GatedLightningIndexer(nn.Module):
    """
    Computes pairwise importance scores using cheap low-dim projections.

    score[q,k] = sum_h( sigmoid(w[q,h]) * sigmoid(q_I[q,h] . k_I[k] * scale + b[h]) )
    """

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

        q_idx = q_idx.float()
        k_idx = k_idx.float()

        # [B, n_idx_heads, T, T]
        raw_scores = torch.einsum("bqhd,bkd->bhqk", q_idx, k_idx) * self.scale

        # Sigmoid gating with learnable bias
        bias_exp = self.idx_bias.float().view(1, -1, 1, 1)
        gated_scores = torch.sigmoid(raw_scores + bias_exp)

        # Query-dependent importance weights
        w_sigmoid = torch.sigmoid(w.float()).permute(0, 2, 1).unsqueeze(-1)  # [B, H, T, 1]

        # Weighted sum across indexer heads
        final_scores = (gated_scores * w_sigmoid).sum(dim=1)  # [B, T, T]

        # Causal mask
        causal_mask = torch.triu(
            torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1
        )
        final_scores = final_scores.masked_fill(causal_mask.unsqueeze(0), float("-inf"))

        return final_scores


# ── Adaptive Top-K Selector ────────────────────────────────────────────────

class AdaptiveTopKSelector(nn.Module):
    def __init__(self, k_base: int = 2048, k_min: int = 256, k_max: int = 4096):
        super().__init__()
        self.k_base = k_base
        self.k_min = k_min
        self.k_max = k_max

    def forward(
        self, scores: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        B, T, T_kv = scores.shape
        k_effective = min(self.k_base, self.k_max, T_kv)

        scores_for_topk = scores.masked_fill(scores == float("-inf"), -1e9)
        _, indices = torch.topk(scores_for_topk, k_effective, dim=-1)

        gathered_scores = torch.gather(scores, -1, indices)
        mask = gathered_scores != float("-inf")

        return indices, mask, k_effective


# ── Gates ──────────────────────────────────────────────────────────────────

class ValueGate(nn.Module):
    """G2: v = v * sigmoid(W·x + b), bias_init=0.5"""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        nn.init.constant_(self.gate_proj.bias, 0.5)

    def forward(self, v: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.gate_proj(x))
        B, T, H, d = v.shape
        return v * gate.view(B, T, H, d)


class OutputGate(nn.Module):
    """G1: o = o * sigmoid(W·x + b), bias_init=0.5"""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        nn.init.constant_(self.gate_proj.bias, 0.5)

    def forward(self, output: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.gate_proj(x))
        B, T, H, d = output.shape
        return output * gate.view(B, T, H, d)


# ── Sparse Attention (reference) ──────────────────────────────────────────

def _gather_along_seq(
    x: torch.Tensor, indices: torch.Tensor
) -> torch.Tensor:
    """
    Gather k selected tokens for every query position.
    x:       [B, T_kv, H, d]
    indices: [B, T_q, k]
    returns: [B, T_q, k, H, d]
    """
    B, T_kv, H, d = x.shape
    T_q, k = indices.shape[1], indices.shape[2]
    idx = indices.clamp(0, T_kv - 1).long()
    idx_exp = idx.unsqueeze(-1).unsqueeze(-1).expand(B, T_q, k, H, d)
    x_exp = x.unsqueeze(1).expand(B, T_q, T_kv, H, d)
    return torch.gather(x_exp, 2, idx_exp)


def sparse_attention_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    indices: torch.Tensor,
    mask: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """
    q:       [B, T, H, D]
    k:       [B, T_kv, H, D]
    v:       [B, T_kv, H, D]
    indices: [B, T, k_sel]
    mask:    [B, T, k_sel] bool
    returns: [B, T, H, D]
    """
    # Gather selected K and V: [B, T, k, H, D]
    k_gathered = _gather_along_seq(k, indices)
    v_gathered = _gather_along_seq(v, indices)

    # Permute for attention: [B, T, H, k, D]
    k_gathered = k_gathered.permute(0, 1, 3, 2, 4)
    v_gathered = v_gathered.permute(0, 1, 3, 2, 4)

    # Attention scores: [B, T, H, k]
    scores = torch.einsum("bqhd,bqhkd->bqhk", q, k_gathered) * scale

    # Mask: [B, T, 1, k]
    mask_exp = mask.unsqueeze(2)
    scores = scores.masked_fill(~mask_exp, float("-inf"))

    attn_weights = torch.softmax(scores, dim=-1)
    attn_weights = attn_weights.masked_fill(~mask_exp, 0.0)
    attn_weights = attn_weights.nan_to_num(0.0)

    # Output: [B, T, H, D]
    return torch.einsum("bqhk,bqhkd->bqhd", attn_weights, v_gathered)


# ── Main Module ────────────────────────────────────────────────────────────

class GatedSparseAttention(nn.Module):
    """
    Reference PyTorch implementation of Gated Sparse Attention.
    Aligned with: github.com/alfredcs/Gated-Sparse-Attention
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
        _nvtx_push("gsa_ref/qkv_proj")
        q = self.q_proj(x).view(B, T, self.num_heads, self.d_head)
        k = self.k_proj(x).view(B, T, self.num_heads, self.d_head)
        v = self.v_proj(x).view(B, T, self.num_heads, self.d_head)
        _nvtx_pop()

        # Step 2: Value Gate (G2)
        _nvtx_push("gsa_ref/value_gate")
        v = self.value_gate(v, x)
        _nvtx_pop()

        # Step 3: RoPE on Q, K
        _nvtx_push("gsa_ref/rotary_emb")
        cos, sin = self.rotary_emb(T, x.device, x.dtype)
        cos = cos.unsqueeze(0).unsqueeze(2)  # [1, T, 1, d_head]
        sin = sin.unsqueeze(0).unsqueeze(2)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        _nvtx_pop()

        # Step 4: Indexer scoring
        _nvtx_push("gsa_ref/indexer")
        indexer_scores = self.indexer(x)
        _nvtx_pop()

        # Step 5: Top-K selection
        _nvtx_push("gsa_ref/topk_select")
        indices, mask, k_eff = self.topk_selector(indexer_scores)
        _nvtx_pop()

        # Step 6: Sparse attention
        _nvtx_push("gsa_ref/sparse_attn")
        attn_out = sparse_attention_ref(q, k, v, indices, mask, self.scale)
        _nvtx_pop()

        # Step 7: Output Gate (G1)
        _nvtx_push("gsa_ref/output_gate")
        attn_out = self.output_gate(attn_out, x)
        _nvtx_pop()

        # Step 8: Output projection
        _nvtx_push("gsa_ref/o_proj")
        out = self.o_proj(attn_out.reshape(B, T, D))
        _nvtx_pop()
        return out
