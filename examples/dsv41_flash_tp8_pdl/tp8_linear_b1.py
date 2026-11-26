#!/usr/bin/env python3
"""Per-layer TP8 linear benchmark at B=1 on B300: tcgen05 GEMM vs GEMV.

The TP8 shards of DSV4.1-Flash (from config.json: hidden 5120, q_lora 1280,
o_lora 1024, 64 heads x 512, o_groups 8):

    wq_a   5120 -> 160      (q_lora_rank 1280 / 8)
    wq_b   1280 -> 4096     (8 heads x 512)
    wo_a   4096 -> 1024     (one group of 8 per rank)
    wo_b   1024 -> 5120

At B=1 every one of these is a GEMV: one activation vector dotted with a weight
matrix. It is purely HBM-bandwidth bound, so the metric is achieved bytes/s, not
TFLOP/s. The activation is fp8 with its own ue8m0 scale folded in once.

Run: python tp8_linear_b1.py
"""

import sys
import time

import torch

from mxfp8_gemm_b300 import (expand_scales, pack_per_row_scales,
                             dequant_rows, mxfp8_gemm)

# TP8 per-rank shard shapes, (N, K) in weight layout
SHAPES = {
    "wq_a": (160, 5120),
    "wq_b": (4096, 1280),
    "wo_a": (1024, 4096),
    "wo_b": (5120, 1024),
}


def make(N, K, seed=0):
    torch.manual_seed(seed)
    w = (torch.randn(N, K, device="cuda") * 0.05).to(torch.float8_e4m3fn)
    sf = torch.randint(120, 133, (N // 32, K // 32), dtype=torch.uint8,
                       device="cuda")
    x = (torch.randn(1, K, device="cuda") * 0.5).to(torch.float8_e4m3fn)
    sfa = torch.full((1, K // 32), 127, dtype=torch.uint8, device="cuda")
    return w, sf, x, sfa


def pad_to_128(x):
    """tcgen05 MMA needs M = 128 lanes; replicate the one real row."""
    return x.expand(128, -1).contiguous()


def timeit(fn, iters=200):
    for _ in range(20):
        fn()
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(7):
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
        best = min(best, (time.perf_counter() - t0) / iters * 1e6)
    return best


def main():
    print(f"{'layer':8s} {'N':>6s} {'K':>6s} {'MB':>7s} "
          f"{'GEMM us':>9s} {'GB/s':>8s} {'ref err':>9s}")
    total_us = 0.0
    total_mb = 0.0
    for name, (N, K) in SHAPES.items():
        w, sf, x, sfa = make(N, K)
        wbytes = N * K                      # e4m3 = 1 byte each
        mb = wbytes / 1e6
        A = pad_to_128(x)
        SFA = pack_per_row_scales(sfa.expand(128, -1).contiguous())
        SFB = expand_scales(sf, N, K)
        k = mxfp8_gemm(128, N, K)
        out = k(A, w, SFA, SFB)
        # correctness: row 0 against the fp32 dequantised reference
        wd = dequant_rows(w, sf)
        ref0 = A[0].to(torch.float32) @ wd.T
        err = (out[0] - ref0).abs().max().item()
        us = timeit(lambda: k(A, w, SFA, SFB))
        gbs = wbytes / (us * 1e-6) / 1e9
        total_us += us
        total_mb += mb
        print(f"{name:8s} {N:6d} {K:6d} {mb:7.2f} {us:9.2f} {gbs:8.0f} {err:9.2e}")
    print(f"{'sum':8s} {'':6s} {'':6s} {total_mb:7.2f} {total_us:9.2f} "
          f"{total_mb*1e6/(total_us*1e-6)/1e9:8.0f}")
    print(f"\n40 layers of these projections at B=1: "
          f"{total_us*40/1000:.2f} ms/token")


if __name__ == "__main__":
    main()
