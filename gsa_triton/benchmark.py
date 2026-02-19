"""
Benchmark harness for GSA reference vs Triton-optimized implementations.

Usage:
    uv run benchmark.py                    # full verify + benchmark
    uv run benchmark.py --skip-verify      # benchmark only
    uv run benchmark.py --verify-T 256     # custom verification seq length
    uv run benchmark.py --dtype fp32       # use float32 instead of bfloat16
"""

import argparse
import sys
import time
import torch
import torch.nn as nn
from typing import List, Tuple, Dict

import gsa_ref
import gsa_opt


# ── Configuration ──────────────────────────────────────────────────────────

BENCH_CONFIGS: List[Tuple[int, int, int]] = [
    # (B, T, D)
    (1, 512, 2048),
    (1, 1024, 2048),
    (1, 2048, 2048),
    (1, 4096, 2048),
]

GSA_KWARGS = {
    "hidden_size": 2048,
    "num_heads": 16,
    "k_max": 4096,
    "d_indexer": 64,
    "n_idx_heads": 4,
    "k_base": 2048,
    "k_min": 256,
}

WARMUP_ITERS = 5
BENCH_ITERS = 10


# ── Numerical Verification ─────────────────────────────────────────────────

def verify_numerical_match(
    device: torch.device,
    dtype: torch.dtype,
    seq_len: int,
    atol: float,
    rtol: float,
) -> bool:
    """
    Build both models, copy weights from ref -> opt, compare outputs.
    Uses a small T to avoid OOM from the reference 5D expand.
    """
    print(f"\n{'='*60}")
    print(f"  Numerical Verification (T={seq_len}, dtype={dtype})")
    print(f"{'='*60}")

    torch.manual_seed(42)

    kwargs = {**GSA_KWARGS, "k_max": min(GSA_KWARGS["k_max"], seq_len)}
    ref_model = gsa_ref.GatedSparseAttention(**kwargs).to(device).to(dtype).eval()
    opt_model = gsa_opt.GatedSparseAttention(**kwargs).to(device).to(dtype).eval()

    # Copy weights: both modules have identical state_dict keys
    ref_sd = ref_model.state_dict()
    opt_model.load_state_dict(ref_sd, strict=True)

    x = torch.randn(1, seq_len, GSA_KWARGS["hidden_size"], device=device, dtype=dtype)

    with torch.no_grad():
        ref_out = ref_model(x)
        opt_out = opt_model(x)

    max_diff = (ref_out - opt_out).abs().max().item()
    mean_diff = (ref_out - opt_out).abs().mean().item()
    match = torch.allclose(ref_out, opt_out, atol=atol, rtol=rtol)

    print(f"  Max  |ref - opt|: {max_diff:.6f}")
    print(f"  Mean |ref - opt|: {mean_diff:.6f}")
    print(f"  atol={atol}, rtol={rtol}")
    print(f"  Result: {'PASS' if match else 'FAIL'}")

    del ref_model, opt_model, x, ref_out, opt_out
    torch.cuda.empty_cache()

    return match


# ── Benchmark Harness ──────────────────────────────────────────────────────

def benchmark_model(
    model_cls,
    module_name: str,
    label: str,
    configs: List[Tuple[int, int, int]],
    device: torch.device,
    dtype: torch.dtype,
) -> List[Dict]:
    results = []

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"{'B':<5} {'T':<6} {'D':<6} | {'ms':>8} | {'tok/s':>12} | {'MB':>8}")
    print(f"{'-'*55}")

    for B, T, D in configs:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

        kwargs = {**GSA_KWARGS, "k_max": min(GSA_KWARGS["k_max"], T)}
        kwargs["hidden_size"] = D

        try:
            model = model_cls(**kwargs).to(device).to(dtype).eval()
            x = torch.randn(B, T, D, device=device, dtype=dtype)

            # Warmup
            with torch.no_grad():
                for _ in range(WARMUP_ITERS):
                    _ = model(x)

            torch.cuda.synchronize(device)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            with torch.no_grad():
                start.record()
                for _ in range(BENCH_ITERS):
                    _ = model(x)
                end.record()

            torch.cuda.synchronize(device)
            elapsed_ms = start.elapsed_time(end) / BENCH_ITERS
            tok_per_s = (B * T) / (elapsed_ms / 1000.0)
            mem_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

            print(f"{B:<5} {T:<6} {D:<6} | {elapsed_ms:>8.2f} | {tok_per_s:>12,.0f} | {mem_mb:>8.0f}")
            results.append({
                "B": B, "T": T, "D": D,
                "ms": elapsed_ms, "tok_per_s": tok_per_s, "mem_mb": mem_mb,
            })

            del model, x

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"{B:<5} {T:<6} {D:<6} | {'OOM':>8} | {'-':>12} | {'-':>8}")
                results.append({
                    "B": B, "T": T, "D": D,
                    "ms": float("inf"), "tok_per_s": 0, "mem_mb": float("inf"),
                })
                torch.cuda.empty_cache()
            else:
                print(f"{B:<5} {T:<6} {D:<6} | {'ERR':>8} | {'-':>12} | {str(e)[:20]}")
                results.append({
                    "B": B, "T": T, "D": D,
                    "ms": float("inf"), "tok_per_s": 0, "mem_mb": float("inf"),
                })

    return results


# ── Summary ────────────────────────────────────────────────────────────────

def print_summary(
    ref_results: List[Dict], opt_results: List[Dict]
):
    print(f"\n{'='*72}")
    print(f"  Summary: Speedup and Memory Reduction")
    print(f"{'='*72}")
    print(
        f"{'T':<6} | {'ref ms':>8} | {'opt ms':>8} | {'speedup':>8} | "
        f"{'ref MB':>8} | {'opt MB':>8} | {'mem save':>8}"
    )
    print(f"{'-'*72}")

    for r, o in zip(ref_results, opt_results):
        if r["ms"] == float("inf"):
            ref_str = "OOM"
        else:
            ref_str = f"{r['ms']:.1f}"

        if o["ms"] == float("inf"):
            opt_str = "OOM"
            speedup_str = "-"
            mem_str = "-"
        else:
            opt_str = f"{o['ms']:.1f}"
            if r["ms"] != float("inf"):
                speedup = r["ms"] / o["ms"]
                speedup_str = f"{speedup:.2f}x"
            else:
                speedup_str = "N/A"

            if r["mem_mb"] != float("inf") and o["mem_mb"] > 0:
                mem_ratio = r["mem_mb"] / o["mem_mb"]
                mem_str = f"{mem_ratio:.2f}x"
            else:
                mem_str = "N/A"

        print(
            f"{r['T']:<6} | {ref_str:>8} | {opt_str:>8} | {speedup_str:>8} | "
            f"{r.get('mem_mb', 0):>8.0f} | {o.get('mem_mb', 0):>8.0f} | {mem_str:>8}"
        )


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark GSA reference vs Triton-optimized"
    )
    parser.add_argument(
        "--skip-verify", action="store_true",
        help="Skip numerical verification",
    )
    parser.add_argument(
        "--verify-T", type=int, default=512,
        help="Sequence length for verification (default: 512)",
    )
    parser.add_argument(
        "--dtype", choices=["bf16", "fp32"], default="bf16",
        help="Data type for benchmarking",
    )
    parser.add_argument(
        "--atol", type=float, default=1e-1,
        help="Absolute tolerance for verification (default: 1e-1 for bf16)",
    )
    parser.add_argument(
        "--rtol", type=float, default=1e-1,
        help="Relative tolerance for verification (default: 1e-1 for bf16)",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("ERROR: CUDA is required for benchmarking.")
        print("Triton kernels require a CUDA device.")
        sys.exit(1)

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32

    print(f"Device: {torch.cuda.get_device_name(device)}")
    print(f"Dtype:  {dtype}")

    # ── Verification ──
    if not args.skip_verify:
        ok = verify_numerical_match(
            device, dtype, args.verify_T, args.atol, args.rtol
        )
        if not ok:
            print("\nFATAL: Numerical mismatch. Aborting benchmark.")
            print("Try increasing --atol/--rtol or debugging kernels.")
            sys.exit(1)

    # ── Benchmark ──
    print("\n" + "#" * 60)
    print("#  Throughput & Memory Benchmark")
    print("#" * 60)

    torch.set_grad_enabled(False)

    ref_results = benchmark_model(
        gsa_ref.GatedSparseAttention, "gsa_ref", "gsa_ref (PyTorch)",
        BENCH_CONFIGS, device, dtype,
    )
    opt_results = benchmark_model(
        gsa_opt.GatedSparseAttention, "gsa_opt", "gsa_opt (Triton)",
        BENCH_CONFIGS, device, dtype,
    )

    print_summary(ref_results, opt_results)


if __name__ == "__main__":
    main()
