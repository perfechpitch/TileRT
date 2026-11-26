#!/usr/bin/env python3
"""DSV4.1-Flash B=1 decode attention (dual KV), specialised for single-request serving.

Two implementations live here:

  decode_kernel_ref(nw, nc)    one CTA, provably correct, used as the oracle
  decode_kernel(nw, nc)        multi-CTA split over the key axis (the B=1 fast path)

Formats (from models/DeepSeek-V4.1-Flash-1017 and SGLang's dsv4 kv_layout):
  window row     528 B = 512 E4M3 + 16 UE8M0 scales (1 per 32 values)
  compressed row 288 B = 512 E2M1 packed 2/byte + 32 E4M3 scales (1 per 16)
The compressed rows are FP4 for exactly this model: its compress_ratios are all
in {0,1,2}, and SGLang maps ratio 1/2 to V41_FP4 because those latents are
already e2m1 (lossless).

Hard-won TileLang rules baked in here (each was a silent wrong-answer bug):
  1. Never accumulate into a scalar `alloc_local` or use a lane-dependent scalar
     inside `T.Parallel` -- the trailing axis is vectorised, so the value is
     hoisted to a uniform scalar and a subset of lanes silently goes wrong.
     Keep the reduction axis as the OUTER `T.serial` loop.
  2. e2m1 exponent must come from `code & 7`; `nib >> 1` folds the sign bit in.
  3. Nibble shifts must be compile-time constants (write low/high separately).
  4. No plain-function helpers containing `T.Parallel` -- only AST-rewritten
     prim_func bodies can iterate it.

Usage:
  python dsv41_mla_decode.py          # correctness of both, then timing
"""

import time

import torch
import tilelang
import tilelang.language as T

from dsv41_mla_ref import (HEAD_DIM, WINDOW, TOP_K, HEADS,
                           WIN_DATA_B, WIN_GRAN, WIN_ROW_B,
                           CMP_DATA_B, CMP_GRAN, CMP_ROW_B,
                           quant_window_row, quant_compressed_row,
                           decode_attention_ref, rope)

BK = 16           # keys per CTA block
THREADS = 256


def _e2m1_val(nib):
    """Decode an e2m1 code (0..15) to float32. e2m1 is sign|exp(2)|mant(1)."""
    code = nib & 7
    e = code >> 1
    m = code & 1
    mag = T.if_then_else(
        e == 0,
        T.Cast("float32", 0.5 * m),                                    # subnormal
        T.exp2(T.Cast("float32", e - 1)) * T.Cast("float32", 1.0 + 0.5 * m))
    return T.if_then_else(nib >= 8, -mag, mag)


# --------------------------------------------------------------- reference CTA
@tilelang.jit(out_idx=[-1])
def decode_kernel_ref(nw: int, nc: int):
    assert nw % BK == 0 and nc % BK == 0
    N = nw + nc
    D = HEAD_DIM
    H = HEADS

    @T.prim_func
    def main(Q: T.Tensor((H, D), "float32"),
             WIN: T.Tensor((nw, WIN_ROW_B), "uint8"),
             CMP: T.Tensor((nc, CMP_ROW_B), "uint8"),
             O: T.Tensor((H, D), "float32")):
        with T.Kernel(1, threads=THREADS):
            qs = T.alloc_shared((H, D), "float32")
            kv = T.alloc_shared((BK, D), "float32")
            sc = T.alloc_shared((H, N), "float32")
            os = T.alloc_shared((H, D), "float32")

            for h, d in T.Parallel(H, D):
                qs[h, d] = Q[h, d]
            T.sync_threads()

            for nb in T.serial(nw // BK):
                for j, d in T.Parallel(BK, D):
                    v = T.Cast("float32",
                               T.reinterpret(WIN[nb * BK + j, d], "float8_e4m3fn"))
                    sb = WIN[nb * BK + j, WIN_DATA_B + d // WIN_GRAN]
                    kv[j, d] = v * T.exp2(T.Cast("float32", sb) - 127.0)
                T.sync_threads()
                for h, j in T.Parallel(H, BK):
                    sc[h, nb * BK + j] = 0.0
                T.sync_threads()
                for d in T.serial(D):
                    for h, j in T.Parallel(H, BK):
                        sc[h, nb * BK + j] += qs[h, d] * kv[j, d]
                T.sync_threads()

            for nb in T.serial(nc // BK):
                for j, d2 in T.Parallel(BK, D // 2):
                    byte = CMP[nb * BK + j, d2]
                    lo = T.Cast("int32", byte & T.Cast("uint8", 0x0F))
                    hi = T.Cast("int32", byte >> T.Cast("uint8", 4))
                    sbyte = CMP[nb * BK + j, CMP_DATA_B + (2 * d2) // CMP_GRAN]
                    scl = T.Cast("float32", T.reinterpret(sbyte, "float8_e4m3fn"))
                    kv[j, 2 * d2] = _e2m1_val(lo) * scl
                    kv[j, 2 * d2 + 1] = _e2m1_val(hi) * scl
                T.sync_threads()
                for h, j in T.Parallel(H, BK):
                    sc[h, nw + nb * BK + j] = 0.0
                T.sync_threads()
                for d in T.serial(D):
                    for h, j in T.Parallel(H, BK):
                        sc[h, nw + nb * BK + j] += qs[h, d] * kv[j, d]
                T.sync_threads()

            for h in T.Parallel(H):
                mx = T.alloc_local((1,), "float32")
                sm = T.alloc_local((1,), "float32")
                mx[0] = -3.0e38
                for j in T.serial(N):
                    mx[0] = T.max(mx[0], sc[h, j])
                sm[0] = 0.0
                for j in T.serial(N):
                    sm[0] += T.exp(sc[h, j] - mx[0])
                for j in T.serial(N):
                    sc[h, j] = T.exp(sc[h, j] - mx[0]) / sm[0]
            T.sync_threads()

            for h, d in T.Parallel(H, D):
                os[h, d] = 0.0
            T.sync_threads()

            for nb in T.serial(nw // BK):
                for j, d in T.Parallel(BK, D):
                    v = T.Cast("float32",
                               T.reinterpret(WIN[nb * BK + j, d], "float8_e4m3fn"))
                    sb = WIN[nb * BK + j, WIN_DATA_B + d // WIN_GRAN]
                    kv[j, d] = v * T.exp2(T.Cast("float32", sb) - 127.0)
                T.sync_threads()
                for j in T.serial(BK):
                    for h, d in T.Parallel(H, D):
                        os[h, d] += sc[h, nb * BK + j] * kv[j, d]
                T.sync_threads()

            for nb in T.serial(nc // BK):
                for j, d2 in T.Parallel(BK, D // 2):
                    byte = CMP[nb * BK + j, d2]
                    lo = T.Cast("int32", byte & T.Cast("uint8", 0x0F))
                    hi = T.Cast("int32", byte >> T.Cast("uint8", 4))
                    sbyte = CMP[nb * BK + j, CMP_DATA_B + (2 * d2) // CMP_GRAN]
                    scl = T.Cast("float32", T.reinterpret(sbyte, "float8_e4m3fn"))
                    kv[j, 2 * d2] = _e2m1_val(lo) * scl
                    kv[j, 2 * d2 + 1] = _e2m1_val(hi) * scl
                T.sync_threads()
                for j in T.serial(BK):
                    for h, d in T.Parallel(H, D):
                        os[h, d] += sc[h, nw + nb * BK + j] * kv[j, d]
                T.sync_threads()

            for h, d in T.Parallel(H, D):
                O[h, d] = os[h, d]

    return main


# ------------------------------------------------------------ B=1 fast path
@tilelang.jit(out_idx=[-1])
def score_kernel(nw: int, nc: int):
    """Scores for every key block in parallel: SC[H, nw+nc]."""
    N = nw + nc
    NWB = nw // BK
    D = HEAD_DIM
    H = HEADS

    @T.prim_func
    def main(Q: T.Tensor((H, D), "float32"),
             WIN: T.Tensor((nw, WIN_ROW_B), "uint8"),
             CMP: T.Tensor((nc, CMP_ROW_B), "uint8"),
             SC: T.Tensor((H, N), "float32")):
        with T.Kernel(NWB + nc // BK, threads=THREADS) as blk:
            qs = T.alloc_shared((H, D), "float32")
            kv = T.alloc_shared((BK, D), "float32")
            scs = T.alloc_shared((H, BK), "float32")
            scp = T.alloc_shared((H, BK, 4), "float32")

            for h, d in T.Parallel(H, D):
                qs[h, d] = Q[h, d]

            if blk < NWB:
                for j, d in T.Parallel(BK, D):
                    v = T.Cast("float32", T.reinterpret(WIN[blk * BK + j, d],
                                                       "float8_e4m3fn"))
                    sb = WIN[blk * BK + j, WIN_DATA_B + d // WIN_GRAN]
                    kv[j, d] = v * T.exp2(T.Cast("float32", sb) - 127.0)
            else:
                for j, d2 in T.Parallel(BK, D // 2):
                    byte = CMP[(blk - NWB) * BK + j, d2]
                    lo = T.Cast("int32", byte & T.Cast("uint8", 0x0F))
                    hi = T.Cast("int32", byte >> T.Cast("uint8", 4))
                    sbyte = CMP[(blk - NWB) * BK + j,
                                CMP_DATA_B + (2 * d2) // CMP_GRAN]
                    scl = T.Cast("float32", T.reinterpret(sbyte, "float8_e4m3fn"))
                    kv[j, 2 * d2] = _e2m1_val(lo) * scl
                    kv[j, 2 * d2 + 1] = _e2m1_val(hi) * scl
            T.sync_threads()

            for h, j, u in T.Parallel(H, BK, 4):
                scp[h, j, u] = 0.0
            T.sync_threads()
            for d4 in T.serial(D // 4):
                for h, j, u in T.Parallel(H, BK, 4):
                    scp[h, j, u] += qs[h, d4 * 4 + u] * kv[j, d4 * 4 + u]
            T.sync_threads()
            for h, j in T.Parallel(H, BK):
                scs[h, j] = 0.0
            T.sync_threads()
            for u in T.serial(4):
                for h, j in T.Parallel(H, BK):
                    scs[h, j] += scp[h, j, u]
            T.sync_threads()
            for h, j in T.Parallel(H, BK):
                SC[h, blk * BK + j] = scs[h, j]

    return main


@tilelang.jit(out_idx=[-1])
def softmax_kernel(nw: int, nc: int):
    """In-place softmax over the key axis of SC[H, N], and zero O.

    Chunked reduction: a flat serial sweep gives only H=8 active threads, which
    made this kernel dominate the whole layer. Splitting N into C chunks puts
    H*C items on the threads instead.
    """
    N = nw + nc
    D = HEAD_DIM
    H = HEADS
    C = 32
    assert N % C == 0, f"N={N} must be divisible by {C}"
    CH = N // C

    @T.prim_func
    def main(SC: T.Tensor((H, N), "float32"),
             O: T.Tensor((H, D), "float32")):
        with T.Kernel(1, threads=THREADS):
            mx = T.alloc_shared((H,), "float32")
            sm = T.alloc_shared((H,), "float32")
            pm = T.alloc_shared((H, C), "float32")
            ps = T.alloc_shared((H, C), "float32")
            s = T.alloc_shared((H, N), "float32")

            for h, j in T.Parallel(H, N):
                s[h, j] = SC[h, j]
            T.sync_threads()

            # ---- per-chunk max, then reduce the C chunks per head
            for h, c in T.Parallel(H, C):
                pm[h, c] = -3.0e38
            T.sync_threads()
            for jj in T.serial(CH):
                for h, c in T.Parallel(H, C):
                    pm[h, c] = T.max(pm[h, c], s[h, c * CH + jj])
            T.sync_threads()
            for h in T.Parallel(H):
                mx[h] = -3.0e38
            T.sync_threads()
            for c in T.serial(C):
                for h in T.Parallel(H):
                    mx[h] = T.max(mx[h], pm[h, c])
            T.sync_threads()

            # ---- per-chunk sum of exp, then reduce
            for h, c in T.Parallel(H, C):
                ps[h, c] = 0.0
            T.sync_threads()
            for jj in T.serial(CH):
                for h, c in T.Parallel(H, C):
                    ps[h, c] += T.exp(s[h, c * CH + jj] - mx[h])
            T.sync_threads()
            for h in T.Parallel(H):
                sm[h] = 0.0
            T.sync_threads()
            for c in T.serial(C):
                for h in T.Parallel(H):
                    sm[h] += ps[h, c]
            T.sync_threads()

            for h, j in T.Parallel(H, N):
                SC[h, j] = T.exp(s[h, j] - mx[h]) / sm[h]
            for h, d in T.Parallel(H, D):
                O[h, d] = 0.0

    return main


@tilelang.jit
def pv_kernel(nw: int, nc: int):
    """P @ V for every key block in parallel, atomically accumulated into O."""
    N = nw + nc
    NWB = nw // BK
    D = HEAD_DIM
    H = HEADS

    @T.prim_func
    def main(SC: T.Tensor((H, N), "float32"),
             WIN: T.Tensor((nw, WIN_ROW_B), "uint8"),
             CMP: T.Tensor((nc, CMP_ROW_B), "uint8"),
             O: T.Tensor((H, D), "float32")):
        with T.Kernel(NWB + nc // BK, threads=THREADS) as blk:
            kv = T.alloc_shared((BK, D), "float32")
            os = T.alloc_shared((H, D), "float32")

            if blk < NWB:
                for j, d in T.Parallel(BK, D):
                    v = T.Cast("float32", T.reinterpret(WIN[blk * BK + j, d],
                                                       "float8_e4m3fn"))
                    sb = WIN[blk * BK + j, WIN_DATA_B + d // WIN_GRAN]
                    kv[j, d] = v * T.exp2(T.Cast("float32", sb) - 127.0)
            else:
                for j, d2 in T.Parallel(BK, D // 2):
                    byte = CMP[(blk - NWB) * BK + j, d2]
                    lo = T.Cast("int32", byte & T.Cast("uint8", 0x0F))
                    hi = T.Cast("int32", byte >> T.Cast("uint8", 4))
                    sbyte = CMP[(blk - NWB) * BK + j,
                                CMP_DATA_B + (2 * d2) // CMP_GRAN]
                    scl = T.Cast("float32", T.reinterpret(sbyte, "float8_e4m3fn"))
                    kv[j, 2 * d2] = _e2m1_val(lo) * scl
                    kv[j, 2 * d2 + 1] = _e2m1_val(hi) * scl
            T.sync_threads()

            for h, d in T.Parallel(H, D):
                os[h, d] = 0.0
            T.sync_threads()
            for j in T.serial(BK):
                for h, d in T.Parallel(H, D):
                    os[h, d] += SC[h, blk * BK + j] * kv[j, d]
            T.sync_threads()
            for h, d in T.Parallel(H, D):
                T.atomic_add(O[h, d], os[h, d])

    return main


def decode_kernel(nw, nc):
    """Multi-CTA B=1 path: returns a callable(q, win, cmp) -> O."""
    sk = score_kernel(nw, nc)
    mk = softmax_kernel(nw, nc)
    pk = pv_kernel(nw, nc)

    def run(q, win, cmp_):
        sc = sk(q, win, cmp_)
        o = mk(sc)
        pk(sc, win, cmp_, o)
        return o

    return run


# ---------------------------------------------------------------- harness
def make_inputs(nw=WINDOW, nc=TOP_K, seed=0, dev="cuda"):
    torch.manual_seed(seed)
    D = HEAD_DIM
    q = rope(torch.randn(1, HEADS, D, device=dev),
             torch.zeros(1, dtype=torch.long, device=dev)).squeeze(0)
    win = torch.stack([quant_window_row(torch.randn(D, device=dev))
                       for _ in range(nw)])
    cmp_ = torch.stack([quant_compressed_row(torch.randn(D, device=dev))
                        for _ in range(nc)])
    return q, win, cmp_


def check(nw=WINDOW, nc=TOP_K, seed=0):
    q, win, cmp_ = make_inputs(nw, nc, seed)
    ref = decode_attention_ref(q.unsqueeze(0), win.unsqueeze(0),
                               cmp_.unsqueeze(0))[0]

    r = decode_kernel_ref(nw, nc)(q, win, cmp_)
    f = decode_kernel(nw, nc)(q, win, cmp_)
    e_ref = (r - ref).abs().max().item()
    e_fast = (f - ref).abs().max().item()
    scale = ref.abs().max().item()
    print(f"B=1  nw={nw} nc={nc}   ref absmax {scale:.4f}")
    print(f"  single-CTA : max abs err {e_ref:.3e}  rel {e_ref/scale:.2e}")
    print(f"  multi-CTA  : max abs err {e_fast:.3e}  rel {e_fast/scale:.2e}")
    return e_ref, e_fast


def bench(nw=WINDOW, nc=TOP_K, iters=100):
    q, win, cmp_ = make_inputs(nw, nc)
    for name, fn in (("single-CTA", decode_kernel_ref(nw, nc)),
                     ("multi-CTA", decode_kernel(nw, nc))):
        for _ in range(5):
            fn(q, win, cmp_)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn(q, win, cmp_)
        torch.cuda.synchronize()
        print(f"  {name:11s} {(time.perf_counter()-t0)/iters*1e6:8.1f} us")


if __name__ == "__main__":
    check()
    bench()
