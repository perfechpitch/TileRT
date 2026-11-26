#!/usr/bin/env python3
"""MXFP4 GEMV for the DSV4.1-Flash MoE at B=1 on B300.

The MoE is the dominant cost of a B=1 decode step, by a wide margin. Per layer,
with TP8/EP8 (config.json: 384 experts, top-6, moe_intermediate 2304, hidden 5120):

    gate_up  4608 x 5120   @ 4 bit = 11.8 MB per expert
    down     5120 x 2304   @ 4 bit =  5.9 MB per expert
    -> 17.7 MB per expert, x 6 routed experts = 106 MB per layer
    -> 4.2 GB per token over 40 layers

So the MoE alone moves ~4.2 GB/token and is purely bandwidth bound; at B=1 the
tensor cores are irrelevant (M=1). This kernel is therefore a streaming FP4 GEMV,
the same shape as `mxfp8_gemv_b1.py` but with two differences that matter:

  * one ue8m0 scale covers 32 ELEMENTS = 16 BYTES, so a 4-byte word (8 values)
    sits inside a single group, and a warp's 32 words span 8 groups;
  * each byte holds two e2m1 values, so a word is 8 values: 8 nibbles decoded per
    4 bytes. That is ~4x the per-byte arithmetic of the fp8 path, which is why
    the e2m1 decode below (mask before shifting) has to be tight.

e2m1 decode hazards, both of which produced wrong answers while this was written:
  * `nib >> 1` folds the sign bit into the exponent -> mask first: `code = nib & 7`
  * exponent 0 is subnormal with magnitude 0.5 (not 0.75), so `e == 0` needs its
    own branch rather than the general 2^(e-1) * (1 + m/2) formula

Run: python mxfp4_gemv_b1.py [check|bench]
"""

import sys
import time

import torch
import tilelang
import tilelang.language as T

GRAN = 32          # one ue8m0 scale per 32 elements = 16 bytes
WPB = 4            # bytes per lane per step
VALS = WPB * 2     # e2m1 values per word


def _e2m1(nib):
    """4-bit e2m1 pattern -> float32. Caller must pass the magnitude-masked code."""
    code = nib & 7
    e = code >> 1
    m = code & 1
    mag = T.if_then_else(
        e == 0,
        T.Cast("float32", 0.5 * m),
        T.exp2(T.Cast("float32", e - 1)) * T.Cast("float32", 1.0 + 0.5 * m))
    return T.if_then_else(nib >= 8, -mag, mag)


@tilelang.jit
def mxfp4_gemv(N: int, K: int, BN: int, SK: int):
    """out[N] = (W[N,K] e2m1 * sf[N,K/32] ue8m0) @ xhat[K].

    W is passed as uint32 [N, K/8] (four bytes = eight e2m1 per word) and the
    16-entry code book as `LUT`. Both of those matter:

      * one uint32 load per 8 values instead of four byte loads;
      * the e2m1 value comes from a shared-memory LUT indexed by the nibble,
        so the decode is a load, not arithmetic. The first version of this
        kernel computed `exp2(e-1) * (1 + m/2)` with two branches per value and
        ran at 247 GB/s; the arithmetic, not the memory system, was the limit.
        A LUT is also the only way to get the e==0 subnormal right without a
        branch.
    """
    KS = K // SK
    WSL = KS // VALS                       # words in this slice
    # A TP8 `down` shard is 5120x288, so the slice is 36 words: NOT a multiple of
    # the 32 words a step sweeps. Tail-guard instead of requiring alignment.
    step_words = (WSL + 31) // 32
    exact = (WSL % 32) == 0
    threads = BN * 32
    assert threads <= 1024
    WPS = 32                               # words swept per step (one per lane)
    KT = K // VALS                         # words along K (8 e2m1 per word)

    @T.prim_func
    def main(W: T.Tensor((N, KT), "uint32"),
             SF: T.Tensor((N, K // GRAN), "uint8"),
             LUT: T.Tensor((16,), "float32"),
             XH: T.Tensor((K,), "float32"),
             OUT: T.Tensor((N,), "float32")):
        with T.Kernel((N // BN) * SK, threads=threads) as blk:
            stripe = blk // SK
            slice_id = blk % SK
            k0 = slice_id * KS

            xs = T.alloc_shared((KS,), "float32")
            scl = T.alloc_shared((BN, KS // GRAN), "float32")
            lut = T.alloc_shared((16,), "float32")
            acc = T.alloc_fragment((BN, 32), "float32")
            rowv = T.alloc_shared((BN,), "float32")

            for i in T.Parallel(KS):
                xs[i] = XH[k0 + i]
            for i in T.Parallel(16):
                lut[i] = LUT[i]
            for r, g in T.Parallel(BN, KS // GRAN):
                scl[r, g] = T.exp2(T.Cast("float32",
                                          SF[stripe * BN + r,
                                             k0 // GRAN + g]) - 127.0)
            T.sync_threads()
            for r, l in T.Parallel(BN, 32):
                acc[r, l] = 0.0
            T.sync_threads()

            for step in T.serial(step_words):
                for r, l in T.Parallel(BN, 32):
                    wi = step * WPS + l               # word index within the slice
                    if exact or wi < WSL:
                        e = wi * VALS             # element index within the slice
                        gg = wi * VALS // GRAN    # one group per 4 words
                        wd = W[stripe * BN + r, k0 // VALS + wi]
                        b0 = T.Cast("int32", wd & T.Cast("uint32", 0xFF))
                        b1 = T.Cast("int32", (wd >> T.Cast("uint32", 8))
                                    & T.Cast("uint32", 0xFF))
                        b2 = T.Cast("int32", (wd >> T.Cast("uint32", 16))
                                    & T.Cast("uint32", 0xFF))
                        b3 = T.Cast("int32", wd >> T.Cast("uint32", 24))
                        acc[r, l] += scl[r, gg] * (
                            xs[e + 0] * lut[b0 & 0x0F]
                            + xs[e + 1] * lut[b0 >> 4]
                            + xs[e + 2] * lut[b1 & 0x0F]
                            + xs[e + 3] * lut[b1 >> 4]
                            + xs[e + 4] * lut[b2 & 0x0F]
                            + xs[e + 5] * lut[b2 >> 4]
                            + xs[e + 6] * lut[b3 & 0x0F]
                            + xs[e + 7] * lut[b3 >> 4])
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


# ------------------------------------------------------------------- reference
CODE = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
        0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]


def dequant_fp4(w_packed: torch.Tensor, sf: torch.Tensor) -> torch.Tensor:
    """[N, K/2] uint8 + [N, K/32] ue8m0 -> [N, K] fp32."""
    N, KB = w_packed.shape
    K = KB * 2
    tbl = torch.tensor(CODE, dtype=torch.float32, device=w_packed.device)
    lo = tbl[(w_packed & 0x0F).long()]
    hi = tbl[(w_packed >> 4).long()]
    vals = torch.empty(N, K, device=w_packed.device)
    vals[:, 0::2] = lo
    vals[:, 1::2] = hi
    return vals * torch.exp2(sf.to(torch.float32) - 127.0).repeat_interleave(GRAN, dim=1)


def make(N, K, seed=0):
    torch.manual_seed(seed)
    w = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device="cuda")
    sf = torch.randint(120, 133, (N, K // GRAN), dtype=torch.uint8, device="cuda")
    xh = torch.randn(K, device="cuda")
    return w, sf, xh


def pick(N, K):
    """Choose (BN, SK). Measured on B300, so the table below is data, not taste.

    gate_up 4608x5120:  BN=16 10.33 us 1142 GB/s   BN=32 10.36 us 1138 GB/s
                        BN= 8 10.91 us 1082 GB/s   any SK>1 loses
    down    5120x2304:  BN= 8  6.78 us  870 GB/s   BN=16  7.57 us 780 GB/s
                        BN=32  9.84 us  599 GB/s

    The fp4 kernel wants MORE CTAs than the fp8 one (which peaked at BN=32): the
    per-value LUT decode makes it compute-heavy, so spreading across SMs beats
    giving each CTA 32 warps. SK>1 always loses to the atomic_add it needs.
    """
    known = {(4608, 5120): (16, 1), (5120, 2304): (8, 1),
             (4096, 2048): (8, 1), (2048, 4096): (8, 1)}
    if (N, K) in known:
        BN, SK = known[(N, K)]
        return BN, SK, (N // BN) * SK
    best = None
    for BN in (8, 16, 32):
        if N % BN:
            continue
        for SK in (1, 2, 4, 8):
            if K % SK:
                continue
            ctas = (N // BN) * SK
            if ctas > 8192:
                continue
            score = (0 if SK == 1 else 1, -min(ctas, 512), BN)
            if best is None or score < best[0]:
                best = (score, BN, SK, ctas)
    _, BN, SK, ctas = best
    return BN, SK, ctas


def graph_time(fn, iters=100, warmup=20):
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


def run_shape(N, K, BN, SK):
    w, sf, xh = make(N, K)
    wv = w.view(torch.int32).view(torch.uint32)
    lut = torch.tensor(CODE, dtype=torch.float32, device="cuda")
    k = mxfp4_gemv(N, K, BN, SK)
    out = torch.zeros(N, device="cuda")
    k(wv, sf, lut, xh, out)
    torch.cuda.synchronize()

    ref = dequant_fp4(w, sf) @ xh
    rel = (out - ref).abs().max().item() / ref.abs().max().item()

    def call():
        if SK > 1:
            out.zero_()
        k(wv, sf, lut, xh, out)

    us = graph_time(call)
    return us, rel, N * K // 2      # bytes of packed fp4 payload


# MoE shapes per rank at EP8 (48 experts), and one expert's two GEMMs
MOE = {
    # Whole-expert shapes (what a TP1 / non-sharded MoE would run).
    "moe_gate_up_1exp": (4608, 5120),
    "moe_down_1exp":    (5120, 2304),
    # The actual TP8 per-rank shards: intermediate 2304 -> 288 per rank.
    "tp8_gate_up":      (576, 5120),
    "tp8_down":         (5120, 288),
}

if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "check"
    if what == "check":
        for name, (N, K) in MOE.items():
            BN, SK, ctas = pick(N, K)
            us, rel, nbytes = run_shape(N, K, BN, SK)
            print(f"  {name:18s} {N:5d}x{K:5d} BN={BN:2d} SK={SK} "
                  f"rel={rel:.2e} {us:6.2f} us  "
                  f"{nbytes/(us*1e-6)/1e9:6.0f} GB/s")
        sys.exit(0)

    print(f"{'layer':20s} {'N':>5s} {'K':>5s} {'MB':>7s} {'BN':>3s} {'SK':>3s} "
          f"{'us':>7s} {'GB/s':>7s} {'relerr':>9s}")
    tot = 0.0
    for name, (N, K) in MOE.items():
        BN, SK, ctas = pick(N, K)
        us, rel, nbytes = run_shape(N, K, BN, SK)
        tot += us
        print(f"{name:20s} {N:5d} {K:5d} {nbytes/1e6:7.2f} {BN:3d} {SK:3d} "
              f"{us:7.2f} {nbytes/(us*1e-6)/1e9:7.0f} {rel:9.2e}")
    print(f"\nper expert (gate_up + down): {tot:.2f} us")
    print(f"per layer, 6 routed experts: {tot*6:.2f} us")
    print(f"40 layers, B=1:              {tot*6*40/1000:.2f} ms/token")
    print(f"weight bytes/token:          {tot*6*40/1e3:.2f} GB")
