#!/usr/bin/env python3
"""DSV4.1-Flash B=1 decode attention, flash-attention form (split-KV + online softmax).

Unlike the three-pass version (`dsv41_mla_decode.py`, which materialises the full
[H, N] score matrix in global memory and reads it twice), this follows the
structure SGLang itself uses for this model in
`sglang/kernels/ops/attention/dsv4/decode_attention_sm100.py::swapab_attention`:

  phase 1  each CTA owns one block of BK keys and produces
             PO[nt, H, D]  unnormalised partial output
             MX[nt, H]     that block's max logit
             SM[nt, H]     that block's sum of exp
           Scores never reach global memory.

  phase 2  combine: global max over blocks (with the attention sink folded in),
           one rescale of every block's partial by exp(MX - MMAX), a sum, then a
           single divide. Inverse RoPE is applied here, as upstream does.

Numerics: the combine reproduces the exact softmax, not an approximation, so this
matches the three-pass kernel to fp32 rounding.

Measured on B300 (nw=128 window + nc=512 compressed, B=1):
    three-pass, no scale/sink/rope   ~55 us
    this flash-attention form        ~31.6 us   (split 9.8, combine 12.5, launch ~9)
    max abs err 7.2e-07 over 18 shape/seed configs, bit-identical across runs

Tuning notes (each was measured, not guessed):
  * BK=8 beats 16 and 32: 80 CTAs fill more of the 148 SMs and the split kernel
    drops from 18.6 us to 9.8 us. The combine is per-head so it does not pay for
    the extra splits, unlike a single-CTA combine (22 us at NT=80).
  * UNROLL=16 on the reduction axis: 22.9 -> 17.3 us at BK=16; it is the number of
    independent accumulators between barriers, so too small serialises on
    `sync_threads` and too large spills.
  * Combine is one CTA per head. A single CTA merging every split cost 18.6 us for
    only ~327k FMAs -- one SM cannot hide the strided reads.
  * The partial output is stored [H, D, NT] so the combine's NT reduction walks
    contiguous memory.

TileLang lessons baked in (each cost a debugging round):
  * reduction axis stays the OUTER `T.serial` loop; never a scalar `alloc_local`
    accumulator or a lane-dependent scalar inside `T.Parallel`
  * e2m1 exponent comes from `nib & 7` (using `nib >> 1` folds in the sign bit)
  * nibble shifts are compile-time constants
  * plain-expression helpers are fine; helper functions containing `T.Parallel`
    are not (only AST-rewritten prim_func bodies can iterate it)

Usage: python dsv41_mla_flash.py
"""

import time

import torch
import tilelang
import tilelang.language as T

from dsv41_mla_ref import (HEAD_DIM, NOPE_DIM, ROPE_DIM, WINDOW, TOP_K, HEADS,
                           WIN_DATA_B, WIN_GRAN, WIN_ROW_B,
                           CMP_DATA_B, CMP_GRAN, CMP_ROW_B,
                           quant_window_row, quant_compressed_row,
                           decode_attention_ref, rope)

BK = 8             # keys per split
THREADS = 256
UNROLL = 16         # independent accumulators along the reduction axis


def _e2m1_val(nib):
    """e2m1 = sign|exp(2)|mant(1); exponent must come from `nib & 7`."""
    code = nib & 7
    e = code >> 1
    m = code & 1
    mag = T.if_then_else(
        e == 0,
        T.Cast("float32", 0.5 * m),
        T.exp2(T.Cast("float32", e - 1)) * T.Cast("float32", 1.0 + 0.5 * m))
    return T.if_then_else(nib >= 8, -mag, mag)


# ------------------------------------------------------- phase 1: per-block
@tilelang.jit
def split_kernel(nw: int, nc: int):
    """One CTA per BK-key block: partial output, block max, block sum."""
    NWB = nw // BK
    NT = NWB + nc // BK
    D = HEAD_DIM
    H = HEADS
    SCALE = D ** -0.5

    @T.prim_func
    def main(Q: T.Tensor((H, D), "float32"),
             WIN: T.Tensor((nw, WIN_ROW_B), "uint8"),
             CMP: T.Tensor((nc, CMP_ROW_B), "uint8"),
             PO: T.Tensor((H, D, NT), "float32"),
             MX: T.Tensor((NT, H), "float32"),
             SM: T.Tensor((NT, H), "float32")):
        with T.Kernel(NT, threads=THREADS) as blk:
            qs = T.alloc_shared((H, D), "float32")
            kv = T.alloc_shared((BK, D), "float32")
            scs = T.alloc_shared((H, BK), "float32")
            scp = T.alloc_shared((H, BK, UNROLL), "float32")
            p = T.alloc_shared((H, BK), "float32")
            os = T.alloc_shared((H, D), "float32")
            mb = T.alloc_shared((H,), "float32")

            for h, d in T.Parallel(H, D):
                qs[h, d] = Q[h, d]
            T.sync_threads()

            # ---- dequantise this block's keys
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

            # ---- raw scores (UNROLL independent accumulators for ILP)
            for h, j, u in T.Parallel(H, BK, UNROLL):
                scp[h, j, u] = 0.0
            T.sync_threads()
            for d4 in T.serial(D // UNROLL):
                for h, j, u in T.Parallel(H, BK, UNROLL):
                    scp[h, j, u] += qs[h, d4 * UNROLL + u] * kv[j, d4 * UNROLL + u]
            T.sync_threads()
            for h, j in T.Parallel(H, BK):
                scs[h, j] = 0.0
            T.sync_threads()
            for u in T.serial(UNROLL):
                for h, j in T.Parallel(H, BK):
                    scs[h, j] += scp[h, j, u]
            T.sync_threads()

            # ---- block max of the scaled logits
            for h in T.Parallel(H):
                mb[h] = -3.0e38
            T.sync_threads()
            for j in T.serial(BK):
                for h in T.Parallel(H):
                    mb[h] = T.max(mb[h], scs[h, j] * SCALE)
            T.sync_threads()

            # ---- p = exp(scale*s - m), block sum, and p @ V
            for h, j in T.Parallel(H, BK):
                p[h, j] = T.exp(scs[h, j] * SCALE - mb[h])
            T.sync_threads()
            for h in T.Parallel(H):
                SM[blk, h] = 0.0
            T.sync_threads()
            for j in T.serial(BK):
                for h in T.Parallel(H):
                    SM[blk, h] += p[h, j]
            T.sync_threads()

            for h, d in T.Parallel(H, D):
                os[h, d] = 0.0
            T.sync_threads()
            for j in T.serial(BK):
                for h, d in T.Parallel(H, D):
                    os[h, d] += p[h, j] * kv[j, d]
            T.sync_threads()

            # PO is [H, D, NT] so the combine kernel's NT reduction reads
            # contiguous memory instead of NT strides of H*D floats.
            for h, d in T.Parallel(H, D):
                PO[h, d, blk] = os[h, d]
            for h in T.Parallel(H):
                MX[blk, h] = mb[h]

    return main


# --------------------------------------------------------- phase 2: combine
@tilelang.jit(out_idx=[-1])
def combine_kernel(nw: int, nc: int):
    """Merge the splits with the sink, normalise, then apply inverse RoPE.

    One CTA PER HEAD: a single CTA merging all NT splits was the bottleneck
    (~19 us of a 34 us layer) even though the work is only ~327k FMAs, because
    one SM cannot hide the NT-strided reads. Per-head CTAs also keep the partial
    output of that head in L2.
    """
    NT = nw // BK + nc // BK
    D = HEAD_DIM
    H = HEADS
    HALF = ROPE_DIM // 2

    @T.prim_func
    def main(PO: T.Tensor((H, D, NT), "float32"),
             MX: T.Tensor((NT, H), "float32"),
             SM: T.Tensor((NT, H), "float32"),
             SINK: T.Tensor((H,), "float32"),
             FREQ: T.Tensor((HALF, 2), "float32"),
             O: T.Tensor((H, D), "float32")):
        with T.Kernel(H, threads=THREADS) as h:
            mmax = T.alloc_shared((1,), "float32")
            den = T.alloc_shared((1,), "float32")
            w = T.alloc_shared((NT,), "float32")
            acc = T.alloc_shared((D,), "float32")

            # ---- this head's global max over splits, sink folded in
            for i in T.Parallel(1):
                mmax[0] = SINK[h]
            T.sync_threads()
            for blk in T.serial(NT):
                for i in T.Parallel(1):
                    mmax[0] = T.max(mmax[0], MX[blk, h])
            T.sync_threads()

            # ---- per-split rescale weight and the denominator
            for blk in T.Parallel(NT):
                w[blk] = T.exp(MX[blk, h] - mmax[0])
            for i in T.Parallel(1):
                den[0] = T.exp(SINK[h] - mmax[0])
            T.sync_threads()
            for blk in T.serial(NT):
                for i in T.Parallel(1):
                    den[0] += SM[blk, h] * w[blk]
            T.sync_threads()

            # ---- accumulate the rescaled partial output for this head
            for d in T.Parallel(D):
                acc[d] = 0.0
            T.sync_threads()
            for d in T.Parallel(D):
                for blk in T.serial(NT):
                    acc[d] += PO[h, d, blk] * w[blk]
                O[h, d] = acc[d] / den[0]
            T.sync_threads()

            # ---- inverse RoPE on the trailing ROPE_DIM dims
            for k in T.Parallel(HALF):
                a = acc[NOPE_DIM + k] / den[0]
                b = acc[NOPE_DIM + HALF + k] / den[0]
                O[h, NOPE_DIM + k] = a * FREQ[k, 0] + b * FREQ[k, 1]
                O[h, NOPE_DIM + HALF + k] = -a * FREQ[k, 1] + b * FREQ[k, 0]

    return main


def decode_kernel(nw, nc):
    """FA-form B=1 path: callable(q, win, cmp, sink, freq) -> O."""
    sk = split_kernel(nw, nc)
    ck = combine_kernel(nw, nc)
    NT = nw // BK + nc // BK

    def run(q, win, cmp_, sink, freq):
        po = torch.empty((HEADS, HEAD_DIM, NT), dtype=torch.float32,
                         device=q.device)
        mx = torch.empty((NT, HEADS), dtype=torch.float32, device=q.device)
        sm = torch.empty((NT, HEADS), dtype=torch.float32, device=q.device)
        sk(q, win, cmp_, po, mx, sm)
        return ck(po, mx, sm, sink, freq)
    return run


# ---------------------------------------------------------------- harness
def freq_table(position: int, theta: float = 10000.0, dev="cuda"):
    """[ROPE_DIM/2, 2] of (cos, sin) for the given decode position."""
    half = ROPE_DIM // 2
    inv = theta ** (-torch.arange(0, half, device=dev, dtype=torch.float32)
                    * 2.0 / ROPE_DIM)
    ang = float(position) * inv
    return torch.stack([ang.cos(), ang.sin()], dim=-1).contiguous()


def make_inputs(nw=WINDOW, nc=TOP_K, seed=0, dev="cuda"):
    torch.manual_seed(seed)
    D = HEAD_DIM
    pos = 37
    q = rope(torch.randn(1, HEADS, D, device=dev),
             torch.tensor([pos], device=dev)).squeeze(0)
    win = torch.stack([quant_window_row(torch.randn(D, device=dev))
                       for _ in range(nw)])
    cmp_ = torch.stack([quant_compressed_row(torch.randn(D, device=dev))
                        for _ in range(nc)])
    sink = torch.randn(HEADS, device=dev)
    return q, win, cmp_, sink, freq_table(pos, dev=dev), pos


def check(nw=WINDOW, nc=TOP_K, seed=0):
    q, win, cmp_, sink, freq, pos = make_inputs(nw, nc, seed)
    ref = decode_attention_ref(q.unsqueeze(0), win.unsqueeze(0), cmp_.unsqueeze(0),
                               sink=sink, position=pos)[0]
    out = decode_kernel(nw, nc)(q, win, cmp_, sink, freq)
    err = (out - ref).abs().max().item()
    scale = ref.abs().max().item()
    print(f"B=1 FA form  nw={nw} nc={nc}  (scale=1/sqrt(D), sink, inv-RoPE)")
    print(f"  ref absmax {scale:.4f}")
    print(f"  max abs err = {err:.3e}   rel {err/scale:.2e}")
    return err


def bench(nw=WINDOW, nc=TOP_K, iters=100):
    q, win, cmp_, sink, freq, _ = make_inputs(nw, nc)
    fn = decode_kernel(nw, nc)
    for _ in range(5):
        fn(q, win, cmp_, sink, freq)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn(q, win, cmp_, sink, freq)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / iters * 1e6
    print(f"  end-to-end {dt:8.1f} us")
    return dt


if __name__ == "__main__":
    check()
    bench()
