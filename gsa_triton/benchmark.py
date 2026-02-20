"""
Benchmark harness for GSA reference vs Triton-optimized implementations.

Usage:
    uv run benchmark.py                    # full verify + benchmark
    uv run benchmark.py --skip-verify      # benchmark only
    uv run benchmark.py --verify-T 256     # custom verification seq length
    uv run benchmark.py --dtype fp32       # use float32 instead of bfloat16

    uv run benchmark.py --profile          # Nsight Systems mode (ref + opt)
    uv run benchmark.py --profile --profile-ref-only --profile-T 2048
    uv run benchmark.py --profile --profile-opt-only --profile-T 2048

    uv run benchmark.py --ncu              # Nsight Compute mode (ref + opt)
    uv run benchmark.py --ncu --ncu-ref-only --ncu-T 2048
    uv run benchmark.py --ncu --ncu-opt-only  --ncu-T 2048

Nsight Systems examples:
    nsys profile --trace=cuda,nvtx --capture-range=cudaProfilerApi \\
        -o gsa_ref_profile \\
        uv run benchmark.py --profile --profile-ref-only --profile-T 2048

    nsys profile --trace=cuda,nvtx --capture-range=cudaProfilerApi \\
        -o gsa_opt_profile \\
        uv run benchmark.py --profile --profile-opt-only --profile-T 2048

Nsight Compute examples:
    # All kernels — reference PyTorch model
    ncu --target-processes all --profile-from-start off \\
        --replay-mode application --set full \\
        -o gsa_ref_ncu \\
        uv run benchmark.py --ncu --ncu-ref-only --ncu-T 2048

    # All kernels — Triton optimised model
    ncu --target-processes all --profile-from-start off \\
        --replay-mode application --set full \\
        -o gsa_opt_ncu \\
        uv run benchmark.py --ncu --ncu-opt-only --ncu-T 2048

    # Triton kernels only (faster, avoids cuBLAS replays)
    ncu --target-processes all --profile-from-start off \\
        --replay-mode application --set full \\
        --kernel-name regex:"_fused_gated_indexer|_fused_sparse_attn" \\
        -o gsa_opt_triton_ncu \\
        uv run benchmark.py --ncu --ncu-opt-only --ncu-T 2048

Notes on ncu flags:
    --profile-from-start off  tells ncu to wait for cudaProfilerStart() rather
                              than capturing from process start. This is the ncu
                              equivalent of nsys --capture-range=cudaProfilerApi.
    --replay-mode application re-runs the entire forward pass for each metric
                              section; required for Triton kernels whose JIT
                              state cannot survive kernel-level replay.
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

PROFILE_ITERS = 3  # iterations captured inside the nsys profiler window


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
    opt_model.fuse_projections()  # build fused mega-GEMM weight

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
            if hasattr(model, "fuse_projections"):
                model.fuse_projections()
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


# ── Nsight Systems Profiling Mode ──────────────────────────────────────────

def run_profile(
    device: torch.device,
    dtype: torch.dtype,
    B: int,
    T: int,
    ref_only: bool = False,
    opt_only: bool = False,
) -> None:
    """
    Nsight Systems profiling mode.

    1. Builds the requested model(s) and runs WARMUP_ITERS warm-up iterations
       (outside the profiler capture window).
    2. Calls cudaProfilerStart() to open the capture window.
    3. Runs PROFILE_ITERS iterations for each selected model, wrapped in
       per-iteration NVTX ranges so the nsys timeline is readable.
    4. Calls cudaProfilerStop() to close the window.

    Invoke with:
        nsys profile --trace=cuda,nvtx --capture-range=cudaProfilerApi \\
            -o gsa_ref_profile uv run benchmark.py --profile --profile-ref-only
        nsys profile --trace=cuda,nvtx --capture-range=cudaProfilerApi \\
            -o gsa_opt_profile uv run benchmark.py --profile --profile-opt-only
        nsys profile --trace=cuda,nvtx --capture-range=cudaProfilerApi \\
            -o gsa_profile     uv run benchmark.py --profile
    """
    run_ref = not opt_only
    run_opt = not ref_only

    D = GSA_KWARGS["hidden_size"]
    kwargs = {**GSA_KWARGS, "k_max": min(GSA_KWARGS["k_max"], T)}

    targets = ("ref" if run_ref else "") + ("+" if run_ref and run_opt else "") + ("opt" if run_opt else "")
    print(f"\n{'='*60}")
    print(f"  Nsight Systems Profile Mode  [{targets}]  B={B} T={T} D={D} dtype={dtype}")
    print(f"{'='*60}")

    torch.manual_seed(0)
    ref_model = gsa_ref.GatedSparseAttention(**kwargs).to(device).to(dtype).eval() if run_ref else None
    opt_model = gsa_opt.GatedSparseAttention(**kwargs).to(device).to(dtype).eval() if run_opt else None
    if run_ref and run_opt:
        # Share weights so the two models are equivalent
        opt_model.load_state_dict(ref_model.state_dict(), strict=True)
    if opt_model is not None:
        opt_model.fuse_projections()

    x = torch.randn(B, T, D, device=device, dtype=dtype)

    # ── Warm-up (outside capture window) ──────────────────────────────────
    print(f"  Warming up ({WARMUP_ITERS} iters each) …")
    with torch.no_grad():
        for _ in range(WARMUP_ITERS):
            if run_ref:
                ref_model(x)
            if run_opt:
                opt_model(x)
    torch.cuda.synchronize(device)

    # ── Open Nsight Systems capture window ────────────────────────────────
    print(f"  cudaProfilerStart → capturing {PROFILE_ITERS} iters …")
    torch.cuda.cudart().cudaProfilerStart()

    with torch.no_grad():
        if run_ref:
            for i in range(PROFILE_ITERS):
                torch.cuda.nvtx.range_push(f"gsa_ref/iter{i}")
                ref_model(x)
                torch.cuda.nvtx.range_pop()
            torch.cuda.synchronize(device)

        if run_opt:
            for i in range(PROFILE_ITERS):
                torch.cuda.nvtx.range_push(f"gsa_opt/iter{i}")
                opt_model(x)
                torch.cuda.nvtx.range_pop()
            torch.cuda.synchronize(device)

    torch.cuda.cudart().cudaProfilerStop()
    print("  cudaProfilerStop  → capture window closed.")
    print("  Open the .nsys-rep file in Nsight Systems GUI to inspect.")


# ── Nsight Compute Profiling Mode ──────────────────────────────────────────

def run_ncu(
    device: torch.device,
    dtype: torch.dtype,
    B: int,
    T: int,
    ref_only: bool = False,
    opt_only: bool = False,
) -> None:
    """
    Nsight Compute (ncu) profiling mode.

    Key differences from Nsight Systems (--profile):
      • NCU replays every kernel many times to gather roofline / memory /
        compute metrics → the capture window must contain EXACTLY ONE forward
        pass per model so the replay cost stays manageable.
      • Use --replay-mode application with Triton kernels; without it NCU's
        kernel-level replay can corrupt Triton's internal state.
      • NVTX ranges still appear as kernel context in the NCU GUI.

    Invoke with (see module docstring for full command examples):
        ncu --target-processes all --profile-from-start off \\
            --replay-mode application --set full \\
            -o gsa_opt_ncu \\
            uv run benchmark.py --ncu --ncu-opt-only --ncu-T 2048
    """
    run_ref = not opt_only
    run_opt = not ref_only

    D = GSA_KWARGS["hidden_size"]
    kwargs = {**GSA_KWARGS, "k_max": min(GSA_KWARGS["k_max"], T)}

    targets = ("ref" if run_ref else "") + ("+" if run_ref and run_opt else "") + ("opt" if run_opt else "")
    print(f"\n{'='*60}")
    print(f"  Nsight Compute (ncu) Mode  [{targets}]  B={B} T={T} D={D} dtype={dtype}")
    print(f"{'='*60}")

    torch.manual_seed(0)
    ref_model = gsa_ref.GatedSparseAttention(**kwargs).to(device).to(dtype).eval() if run_ref else None
    opt_model = gsa_opt.GatedSparseAttention(**kwargs).to(device).to(dtype).eval() if run_opt else None
    if run_ref and run_opt:
        opt_model.load_state_dict(ref_model.state_dict(), strict=True)
    if opt_model is not None:
        opt_model.fuse_projections()

    x = torch.randn(B, T, D, device=device, dtype=dtype)

    # ── Warm-up (outside capture window) ──────────────────────────────────
    # Triggers Triton JIT compilation so the capture window only sees
    # steady-state kernel launches, not compilation overhead.
    print(f"  Warming up ({WARMUP_ITERS} iters each) …")
    with torch.no_grad():
        for _ in range(WARMUP_ITERS):
            if run_ref:
                ref_model(x)
            if run_opt:
                opt_model(x)
    torch.cuda.synchronize(device)

    # ── Open NCU capture window — ONE forward pass per model ───────────────
    # NCU replays every kernel internally; multiple captured passes would
    # multiply the replay cost for no extra information.
    print("  cudaProfilerStart → capturing 1 forward pass per model …")
    torch.cuda.cudart().cudaProfilerStart()

    with torch.no_grad():
        if run_ref:
            torch.cuda.nvtx.range_push("ncu/gsa_ref")
            ref_model(x)
            torch.cuda.nvtx.range_pop()
            torch.cuda.synchronize(device)

        if run_opt:
            torch.cuda.nvtx.range_push("ncu/gsa_opt")
            opt_model(x)
            torch.cuda.nvtx.range_pop()
            torch.cuda.synchronize(device)

    torch.cuda.cudart().cudaProfilerStop()
    print("  cudaProfilerStop  → capture window closed.")
    print("  Open the .ncu-rep file in Nsight Compute GUI to inspect.")
    if run_opt:
        print("  Tip: filter by kernel name '_fused_gated_indexer' or")
        print("       '_fused_sparse_attn' to focus on Triton kernels.")


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
    # ── Nsight Systems profiling mode ──
    parser.add_argument(
        "--profile", action="store_true",
        help="Run Nsight Systems profiling mode instead of the full benchmark. "
             "Use with: nsys profile --capture-range=cudaProfilerApi ...",
    )
    parser.add_argument(
        "--profile-B", type=int, default=1,
        help="Batch size for profiling run (default: 1)",
    )
    parser.add_argument(
        "--profile-T", type=int, default=2048,
        help="Sequence length for profiling run (default: 2048)",
    )
    parser.add_argument(
        "--profile-ref-only", action="store_true",
        help="Profile the reference (PyTorch) model only",
    )
    parser.add_argument(
        "--profile-opt-only", action="store_true",
        help="Profile the optimised (Triton) model only",
    )
    # ── Nsight Compute (ncu) mode ──
    parser.add_argument(
        "--ncu", action="store_true",
        help="Run Nsight Compute profiling mode (single forward pass per model). "
             "Use with: ncu --capture-range cudaProfilerApi --replay-mode application ...",
    )
    parser.add_argument(
        "--ncu-B", type=int, default=1,
        help="Batch size for ncu run (default: 1)",
    )
    parser.add_argument(
        "--ncu-T", type=int, default=2048,
        help="Sequence length for ncu run (default: 2048)",
    )
    parser.add_argument(
        "--ncu-ref-only", action="store_true",
        help="ncu: profile the reference (PyTorch) model only",
    )
    parser.add_argument(
        "--ncu-opt-only", action="store_true",
        help="ncu: profile the optimised (Triton) model only",
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

    # ── Nsight Systems profile mode (early exit) ───────────────────────────
    if args.profile:
        run_profile(
            device, dtype,
            B=args.profile_B, T=args.profile_T,
            ref_only=args.profile_ref_only,
            opt_only=args.profile_opt_only,
        )
        return

    # ── Nsight Compute (ncu) mode (early exit) ─────────────────────────────
    if args.ncu:
        run_ncu(
            device, dtype,
            B=args.ncu_B, T=args.ncu_T,
            ref_only=args.ncu_ref_only,
            opt_only=args.ncu_opt_only,
        )
        return

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
