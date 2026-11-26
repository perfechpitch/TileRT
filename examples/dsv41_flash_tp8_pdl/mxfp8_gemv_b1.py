#!/usr/bin/env python3
"""MXFP8 GEMV for B=1 decode on B300, TP8 shard shapes.

At B=1 every DSV4.1-Flash linear is a GEMV: one fp8 activation vector against a
weight matrix. Nothing is reused, so this is purely HBM-bandwidth bound and the
tensor cores are useless (M=1 would occupy 1 of 128 MMA lanes). The whole design
follows from one fact: **each weight byte is read exactly once**, so routing it
through shared memory is pure overhead. Weights stream global -> registers.

Per-thread mapping, which is where the bandwidth comes from:

    W is viewed as [N, K/4, 4] e4m3 so one "word" is 4 consecutive fp8 values.
    A CTA owns BN rows and a K slice. Thread (r, l) walks words
        w = w_base + step*32 + l
    so a warp covers 32 words = 128 contiguous bytes per step, and each thread
    keeps ONE scalar accumulator. The 4 bytes of a word are accumulated with
    compile-time offsets 0..3, which the compiler folds into a single 32-bit
    load, and all 4 share one ue8m0 group (a group is 32 bytes = 8 words).

The activation is folded once on the host into `xhat` (fp32, its own ue8m0 scale
already applied), so the inner loop is 4 converts + 4 FMAs + 1 scale multiply per
4 bytes. That is ~3 instructions per byte, well under the bandwidth roof.

`SK` splits K across CTAs. It exists for the small-N shards: wq_a is only 160
rows, so a plain row-striped grid is 20 CTAs on a 148-SM part. SK raises the CTA
count and the SK partial sums are combined with atomic_add.

Measured on B300 (see bench output): the fp8-byte-per-lane version of this kernel
reached only 407 GB/s; the 4-byte-word version below is what made it bandwidth
bound. Launch overhead (~4 us) is a real fraction of every one of these shapes
because the whole TP8 attention projection set is only ~20 MB per layer.

Run: python mxfp8_gemv_b1.py [check|bench|sweep]
"""

import sys
import time

import torch
import tilelang
import tilelang.language as T

FP8 = "float8_e4m3fn"
GRAN = 32          # one ue8m0 scale per 32 elements along K
WPB = 4            # bytes ("word") per lane per step -> 128 contiguous B/warp


@tilelang.jit
def mxfp8_gemv(N: int, K: int, BN: int, SK: int):
    """out[N] = (W[N,K] e4m3 * sf[N,K/32] ue8m0) @ xhat[K].

    W:    [N, K] e4m3, row-major, viewed here as [N, K/4, 4].
    SF:   [N, K/32] ue8m0 bytes.
    XH:   [K] fp32, activation with its own scale already folded in.
    OUT:  [N] fp32, accumulated (zeros required when SK > 1).
    """
    KS = K // SK
    assert K % SK == 0 and KS % (WPB * 32) == 0, "K slice must be a whole word stripe"
    assert KS % GRAN == 0
    threads = BN * 32
    assert threads <= 1024
    steps = KS // (WPB * 32)          # words per lane = KS/4/32
    acc_dtype = "float32"

    @T.prim_func
    def main(W: T.Tensor((N, K // WPB, WPB), FP8),
             SF: T.Tensor((N, K // GRAN), "uint8"),
             XH: T.Tensor((K,), "float32"),
             OUT: T.Tensor((N,), acc_dtype)):
        with T.Kernel((N // BN) * SK, threads=threads) as blk:
            stripe = blk // SK
            slice_id = blk % SK
            k0 = slice_id * KS          # element offset of this slice

            xs = T.alloc_shared((KS,), "float32")
            scl = T.alloc_shared((BN, KS // GRAN), "float32")
            acc = T.alloc_fragment((BN, 32), acc_dtype)
            rowv = T.alloc_shared((BN,), acc_dtype)

            for i in T.Parallel(KS):
                xs[i] = XH[k0 + i]
            for r, g in T.Parallel(BN, KS // GRAN):
                scl[r, g] = T.exp2(T.Cast("float32",
                                          SF[stripe * BN + r,
                                             k0 // GRAN + g]) - 127.0)
            T.sync_threads()
            for r, l in T.Parallel(BN, 32):
                acc[r, l] = 0.0
            T.sync_threads()

            # ---- the bandwidth-critical loop
            # thread (r, l) owns words w = step*32 + l of row r, i.e. 4 bytes at
            # element offset (step*32 + l)*4. All 4 bytes share one scale group.
            for step in T.serial(steps):
                for r, l in T.Parallel(BN, 32):
                    wi = step * 32 + l                 # word index in the slice
                    e = wi * WPB                       # element index in the slice
                    gg = wi // (GRAN // WPB)           # group index (8 words per group)
                    acc[r, l] += (
                        xs[e + 0] * T.Cast("float32", W[stripe * BN + r, k0 // WPB + wi, 0])
                        + xs[e + 1] * T.Cast("float32", W[stripe * BN + r, k0 // WPB + wi, 1])
                        + xs[e + 2] * T.Cast("float32", W[stripe * BN + r, k0 // WPB + wi, 2])
                        + xs[e + 3] * T.Cast("float32", W[stripe * BN + r, k0 // WPB + wi, 3])
                    ) * scl[r, gg]
            T.sync_threads()

            T.reduce_sum(acc, rowv, dim=1)
            T.sync_threads()

            if SK == 1:
                for r in T.Parallel(BN):
                    OUT[stripe * BN + r] = rowv[r]
            else:
                for r in T.Parallel(BN):
                    T.atomic_add(OUT[stripe * BN + r], rowv[r])

    return main


# --------------------------------------------------------------- host helpers
def fold_act_scale(x: torch.Tensor, sfa: torch.Tensor) -> torch.Tensor:
    """x [1, K] e4m3 (one ue8m0 per 32) -> fp32 [K] with the scale folded in."""
    return (x[0].to(torch.float32)
            * torch.exp2(sfa[0].to(torch.float32) - 127.0)
            .repeat_interleave(32)[:x.shape[1]])


def dequant_rows(w: torch.Tensor, sf: torch.Tensor) -> torch.Tensor:
    return (w.to(torch.float32)
            * torch.exp2(sf.to(torch.float32) - 127.0).repeat_interleave(32, dim=1))


def make(N, K, seed=0):
    torch.manual_seed(seed)
    w = (torch.randn(N, K, device="cuda") * 0.05).to(torch.float8_e4m3fn)
    sf = torch.randint(120, 133, (N, K // GRAN), dtype=torch.uint8, device="cuda")
    x = (torch.randn(1, K, device="cuda") * 0.5).to(torch.float8_e4m3fn)
    sfa = torch.full((1, K // GRAN), 127, dtype=torch.uint8, device="cuda")
    return w, sf, x, sfa


# TP8 per-rank shard shapes (N, K) taken from config.json:
#   hidden 5120, q_lora_rank 1280/8=160, 8 heads x 512 = 4096, o_lora 1024
SHAPES = {
    "wq_a": (160, 5120),
    "wq_b": (4096, 1280),
    "wo_a": (1024, 4096),
    "wo_b": (5120, 1024),
}


def pick(N, K):
    """Choose (BN, SK) for the CTA geometry.

    Measured on B300 for wq_b (4096x1280), so the rule is empirical, not guessed:
        BN=32 SK=1  128 ctas  3.51 us  1492 GB/s   <- best
        BN= 8 SK=1  512 ctas  3.90 us  1343 GB/s
        BN=16 SK=1  256 ctas  4.45 us  1179 GB/s
        BN=32 SK=2  256 ctas  5.32 us   986 GB/s
    Bigger BN wins because threads = BN*32, and 32 warps per CTA hide the global
    load latency of the dependent accumulate chain. SK>1 loses to the atomic_add
    it requires, so it is kept only for shards too skinny to fill the part
    (wq_a is 160 rows: at BN=8 that is 20 CTAs on 148 SMs).
    """
    threads = {8: 256, 16: 512, 32: 1024}
    best = None
    for BN in (32, 16, 8):
        if N % BN:
            continue
        for SK in (1, 2, 4, 8):
            if K % SK or (K // SK) % (WPB * 32):
                continue
            ctas = (N // BN) * SK
            if ctas > 4096:
                continue
            # Measurements say the biggest tile wins (BN=32 beat BN=16 on wq_b
            # even at only 128 of 148 CTAs), so the wave check is a loose floor,
            # not a target, and SK=1 beats anything that needs atomic_add.
            score = (0 if ctas >= 118 else 1, 0 if SK == 1 else 1,
                     -threads[BN], abs(ctas - 296))
            if best is None or score < best[0]:
                best = (score, BN, SK, ctas)
    _, BN, SK, ctas = best
    return BN, SK, ctas


def graph_time(fn, iters=200, warmup=20):
    """Per-call time with the CPU launch path removed by a CUDA graph.

    At B=1 these GEMVs are a few microseconds of real work, so eager timing
    measures Python + launch overhead (~13 us flat across every shape, which is
    how the first version of this file read). A graph is also what serving
    actually runs, so it is the honest number to optimise.
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(iters):
            fn()
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(10):
        t0 = time.perf_counter()
        g.replay()
        torch.cuda.synchronize()
        best = min(best, (time.perf_counter() - t0) / iters * 1e6)
    return best


def run_shape(N, K, BN, SK, iters=300):
    w, sf, x, sfa = make(N, K)
    xh = fold_act_scale(x, sfa)
    wv = w.view(N, K // WPB, WPB)
    k = mxfp8_gemv(N, K, BN, SK)
    out = torch.zeros(N, device="cuda")
    k(wv, sf, xh, out)
    torch.cuda.synchronize()

    ref = dequant_rows(w, sf) @ xh
    rel = (out - ref).abs().max().item() / ref.abs().max().item()

    def call():
        if SK > 1:
            out.zero_()
        k(wv, sf, xh, out)

    best = graph_time(call)
    return best, rel, N * K


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "bench"

    if what == "check":
        for name, (N, K) in SHAPES.items():
            BN, SK, _ = pick(N, K)
            us, rel, _ = run_shape(N, K, BN, SK)
            ok = "MATCH" if rel < 5e-3 else "MISMATCH"
            print(f"  {name:6s} {N:5d}x{K:5d} BN={BN:2d} SK={SK} "
                  f"rel={rel:.2e} {us:6.2f} us  {ok}")
        sys.exit(0)

    if what == "sweep":
        N, K = SHAPES["wq_b"]
        print(f"sweep wq_b {N}x{K}")
        for BN in (8, 16, 32):
            for SK in (1, 2, 4):
                try:
                    us, rel, nbytes = run_shape(N, K, BN, SK)
                    print(f"  BN={BN:2d} SK={SK} ctas={(N//BN)*SK:4d} "
                          f"{us:6.2f} us  {nbytes/(us*1e-6)/1e9:6.0f} GB/s")
                except Exception as e:
                    print(f"  BN={BN:2d} SK={SK} FAIL {str(e).splitlines()[0][:70]}")
        sys.exit(0)

    print(f"{'layer':6s} {'N':>5s} {'K':>5s} {'MB':>6s} {'BN':>3s} {'SK':>3s} "
          f"{'ctas':>5s} {'us':>7s} {'GB/s':>7s} {'relerr':>9s}")
    tot_us = tot_mb = 0.0
    for name, (N, K) in SHAPES.items():
        BN, SK, ctas = pick(N, K)
        us, rel, nbytes = run_shape(N, K, BN, SK)
        mb = nbytes / 1e6
        tot_us += us
        tot_mb += mb
        print(f"{name:6s} {N:5d} {K:5d} {mb:6.2f} {BN:3d} {SK:3d} {ctas:5d} "
              f"{us:7.2f} {nbytes/(us*1e-6)/1e9:7.0f} {rel:9.2e}")
    print(f"{'sum':6s} {'':5s} {'':5s} {tot_mb:6.2f} {'':3s} {'':3s} {'':5s} "
          f"{tot_us:7.2f} {tot_mb*1e6/(tot_us*1e-6)/1e9:7.0f}")
    print(f"\nattention projections, 40 layers, B=1: {tot_us*40/1000:.2f} ms/token")
