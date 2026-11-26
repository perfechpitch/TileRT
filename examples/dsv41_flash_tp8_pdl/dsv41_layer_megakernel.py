#!/usr/bin/env python3
"""One full DSV4.1-Flash decoder layer as a SINGLE cooperative launch.

B=1, TP8 per rank, B300. Includes MLA decode attention, so the layer really is one
kernel and not "everything except attention".

         rmsnorm1 -> wq_a -> rmsnormQL -> wq_b
              -> MLA split (NT CTAs) -> MLA combine (8 CTAs)
              -> wo_a -> wo_b -> rmsnorm2
              -> moe gate_up x6 -> swiglu -> moe down x6

16 stages, 15 `T.sync_grid()` barriers, one launch. Verified stage by stage against
a torch replica that includes the fp8/fp4 quantisation and the MLA reference.

Why a megakernel helps at B=1
-----------------------------
Every stage is a memory-bound GEMV: a few microseconds each, so the gap between
launches is a comparable fraction of the work. The MoE is six independent small
GEMMs; launched separately each runs at 300-360 GB/s purely from launch and
parallelism overhead (see `mxfp4_gemv_b1.py`).

Grid size, and why it is not the tile count
-------------------------------------------
A cooperative launch requires every block to be resident. These stages fully unroll
their K sweeps, so register pressure caps residency at 2 blocks per SM = 296 blocks
-- far below the 640 tiles that `wo_b` / MoE down need. Hence **persistent tile
loops**: each stage walks `blk + it*G` until the tiles run out. `MEGAK_G` overrides
the grid for sweeps; 444 already fails to launch.

A single threads-per-block for all stages is the price of fusion: the standalone
kernels each picked their own tile shape (BN=32 best for fp8, 16/8 for fp4). Here
everything is BN=8, which is what an 8-warp block allows with one warp per row.

What the fusion bought, and what it cost
----------------------------------------
* **gate_up is the big win.** Six 2.95 MB GEMMs that ran at 358 GB/s each when
  launched separately now sweep as one stage at **1424 GB/s**.
* **The grid is capped at 296 blocks.** `wo_b`/`down` (640 tiles) need 3 waves with
  the last only 16% full. Occupancy, not bandwidth, is the ceiling.
* **Barriers are free.** 15 x `T.sync_grid()` is inside the noise (0.02 us each).
* **MLA is cheap to fuse.** Its NT split CTAs and 8 combine CTAs fit inside the same
  grid, and its dequantised keys fit in the shared-memory budget.

Three measurement-driven rewrites (see the per-stage table in `bench`)
---------------------------------------------------------------------
1. **moe down 21.0 -> 13.1 us.** Reducing the six experts one at a time, each with
   its own AllReduce and syncthreads, ran at 211 GB/s. Concatenating the experts
   along K into a single accumulator turns six bursts of 1.125 warp loads into 7
   back-to-back loads. It was latency, not bandwidth.
2. **rmsnorms 17.5 -> 14.1 us** by splitting each into a partials stage over
   K/256 blocks plus a parallel apply.
3. **SwiGLU inlined into the down stage was WORSE** (20.0 vs 13.1 + 2.4): every
   down block would re-read all 6x576 gate_up values, 8.8 MB of redundant traffic
   against only 4.4 MB of down weights. Reverted.

Run: python dsv41_layer_megakernel.py [check|bench]     (MEGAK_G=n to sweep)
"""

import os
import sys
import time

import torch
import tilelang
import tilelang.language as T

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "..", "dsv41_mla"))

FP8 = "float8_e4m3fn"
FP8_MAX = 448.0
AMAX_FLOOR = 1e-10
GRAN = 32

H = 5120          # hidden
QLR = 1280        # q_lora_rank, full width (union of the 8 TP8 shards)
QN = 4096         # 8 heads x head_dim 512 (this rank)
WOA = 1024        # o_lora_rank; exactly one o_group per rank at TP8
E_GU = 576        # per-expert gate||up: intermediate 2304 / 8 = 288 per side
E_K = 288
NEXP = 384
TOPK = 6
NSH = 1           # shared expert (TP8-sharded, not degraded -> same 288 per side)
NGU = TOPK + NSH  # 7 "slots" in the MoE: 6 routed + the shared expert
RSCALE = 1.5      # routed_scaling_factor, applied to the routed weights only
SWIGLU_LIMIT = 10.0
EPS = 1e-20

# ---- MLA decode geometry (TP8-local: 8 heads, one shared latent KV per token)
HEAD_DIM = 512
NOPE_DIM = 448
ROPE_DIM = 64
HALF = ROPE_DIM // 2
HEADS = 8                   # TP8-local heads; QN == HEADS * HEAD_DIM
MLA_BK = 8                  # keys per split CTA
NWIN = 128                  # sliding-window keys
NCMP = 512                  # compressed / sparse keys
NWB = NWIN // MLA_BK        # window split blocks
NT = NWB + NCMP // MLA_BK   # total split blocks (16 + 64 = 80)
WIN_DATA_B = 512
WIN_GRAN = 32
WIN_ROW_B = WIN_DATA_B + WIN_DATA_B // WIN_GRAN       # 528
CMP_DATA_B = 256
CMP_GRAN = 16
CMP_ROW_B = CMP_DATA_B + HEAD_DIM // CMP_GRAN         # 288
MLA_UNROLL = 16             # independent accumulators along the reduction axis
QB_SCALE = HEAD_DIM ** -0.5
NGQ = HEAD_DIM // GRAN      # quant groups in one head's attention output (16)

G = int(os.environ.get("MEGAK_G", "296"))
THREADS = 256
BN = 8            # one warp per output row; 32 lanes sweep K
C_GU = 10         # gate_up reduction steps staged per shared-memory chunk
NB_MAX = 32       # most norm-partial blocks any stage uses (H/THREADS = 20)

CODE = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
        0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]


def _ceil_exp_byte(raw_bits):
    """float32 bits -> ue8m0 byte, exponent rounded UP (see fused_ew_b300.py)."""
    expo = (raw_bits >> 23) & 0xFF
    mant = raw_bits & 0x7FFFFF
    return expo + T.if_then_else(mant != 0, 1, 0)


def _e2m1_val(nib):
    """e2m1 = sign|exp(2)|mant(1); the exponent must come from `nib & 7`.

    Using `nib >> 1` folds the sign bit into the exponent -- that was a real bug
    in the first MLA version.
    """
    code = nib & 7
    e = code >> 1
    m = code & 1
    mag = T.if_then_else(
        e == 0,
        T.Cast("float32", 0.5 * m),
        T.exp2(T.Cast("float32", e - 1)) * T.Cast("float32", 1.0 + 0.5 * m))
    return T.if_then_else(nib >= 8, -mag, mag)


@tilelang.jit
def decode_layer(segs=("a", "b", "c"), resid_in_c=False, moe_sep=False,
                 wqa_rows=QLR, wqa_off=0):
    """The layer as one cooperative launch, or as the three pieces a TP8
    deployment has to split it into.

    At TP8 the layer needs cross-GPU collectives in the middle, and a single
    cooperative launch cannot contain an NCCL call, so the layer is cut at
    exactly the points where the tensor-parallel algebra requires a sync:

        seg a  rmsnorm1 + wq_a        -> all_gather(q_lora, 1280)
        seg b  rmsnormQL + wq_b + MLA + wo_a + wo_b
                                      -> all_reduce(attn out, 5120)
        seg c  rmsnorm2 + router + MoE -> all_reduce(moe out, 5120)

    `resid_in_c` moves the `X +` residual add out of wo_b's store and into the
    head of seg c: after an all_reduce the partial sums are what must be summed,
    and X is already identical on every rank, so folding X in beforehand would
    add it TP times.

    `moe_sep` sends the MoE result to `MO` instead of accumulating it into OUT.
    This is required whenever a collective follows: the all_reduce sums whatever
    the tensor holds, so a residual folded in beforehand would be summed TP
    times too (X once per rank) and the already-reduced attention output would be
    summed a second time. With moe_sep the reduce sees only the MoE partial.

    `wqa_rows` / `wqa_off` give wq_a its real TP8 shape: q_lora_rank 1280 is
    split over the ranks, so each rank multiplies a 160 x 5120 shard and writes
    into its 160-row slot of the shared 1280-wide q_lora buffer. The all_gather
    that follows is 5 KB per token, i.e. free next to the 5120-wide reductions.
    Running wqa_rows=1280 (the default) is the same numerics on one GPU with no
    gather, and costs ~4 us more: it streams 8x the weight bytes.
    """
    SEGA, SEGB, SEGC = "a" in segs, "b" in segs, "c" in segs
    NT_QA, NT_QB, NT_OA, NT_OB = wqa_rows // BN, QN // BN, WOA // BN, H // BN
    NT_GU = NGU * (E_GU // BN)
    NT_DN = H // BN
    NT_RT = NEXP // BN               # router logits: 384 rows / 8
    NROW_GU = E_GU // BN
    W_DN = E_K // 8
    S_DN = (W_DN + 31) // 32
    assert H % THREADS == 0 and QLR % THREADS == 0
    NBH = H // THREADS               # norm-partial blocks for a 5120-vector
    NBQ = QLR // THREADS
    NGP = THREADS // GRAN            # groups per norm-partial block (8)
    assert NT <= G and HEADS <= G and NBH <= G, "MLA must fit in one wave"
    assert QN == HEADS * HEAD_DIM
    assert (H // 2) % 32 == 0
    S_RT = H // 64                   # router: 2 bf16 per lane per step

    def waves(n):
        return (n + G - 1) // G

    IT = dict(qa=waves(NT_QA), qb=waves(NT_QB), oa=waves(NT_OA),
              ob=waves(NT_OB), gu=waves(NT_GU), dn=waves(NT_DN))

    # 4-byte words per lane per step -> serial step count. fp8 packs 4 values per
    # word, fp4 packs 8, so a fp4 stage needs HALF the steps for the same K.
    def s8(k):
        assert (k // 4) % 32 == 0
        return (k // 4) // 32

    def s4(k):
        assert (k // 8) % 32 == 0
        return (k // 8) // 32

    S_QA, S_QB, S_OA, S_OB = s8(H), s8(QLR), s8(QN), s8(WOA)
    S_GU = s4(H)

    @T.prim_func
    def main(
        X: T.Tensor((H,), "float32"),
        GN: T.Tensor((H,), "float32"),
        QNORM: T.Tensor((QLR,), "float32"),
        GN2: T.Tensor((H,), "float32"),
        WQA: T.Tensor((wqa_rows, H // 4, 4), FP8),
        SQA: T.Tensor((wqa_rows, H // GRAN), "uint8"),
        WQB: T.Tensor((QN, QLR // 4, 4), FP8),
        SQB: T.Tensor((QN, QLR // GRAN), "uint8"),
        WOAW: T.Tensor((WOA, QN // 4, 4), FP8),
        SOA: T.Tensor((WOA, QN // GRAN), "uint8"),
        WOBW: T.Tensor((H, WOA // 4, 4), FP8),
        SOB: T.Tensor((H, WOA // GRAN), "uint8"),
        WIN: T.Tensor((NWIN, WIN_ROW_B), "uint8"),
        CMP: T.Tensor((NCMP, CMP_ROW_B), "uint8"),
        SINK: T.Tensor((HEADS,), "float32"),
        FREQ: T.Tensor((HALF, 2), "float32"),
        WGU: T.Tensor((NEXP + NSH, E_GU, H // 8), "uint32"),
        SGU: T.Tensor((NEXP + NSH, E_GU, H // GRAN), "uint8"),
        WDN: T.Tensor((NEXP + NSH, H, E_K // 8), "uint32"),
        SDN: T.Tensor((NEXP + NSH, H, E_K // GRAN), "uint8"),
        GW: T.Tensor((NEXP, H), "bfloat16"),
        GB: T.Tensor((NEXP,), "float32"),   # ffn.gate.bias: selection only
        SHGW: T.Tensor((E_GU, H // 4, 4), FP8),   # shared expert w1||w3, fp8 e4m3
        SHGS: T.Tensor((E_GU, H // GRAN), "uint8"),
        SHDW: T.Tensor((H, E_K), FP8),            # shared expert w2, fp8 e4m3
        SHDS: T.Tensor((H, E_K // GRAN), "uint8"),
        LUT: T.Tensor((16,), "float32"),
        EXPID: T.Tensor((TOPK,), "int32"),
        WGT: T.Tensor((NGU,), "float32"),
        QL: T.Tensor((QLR,), "float32"),
        Q: T.Tensor((QN,), "float32"),
        PO: T.Tensor((HEADS, NT, HEAD_DIM), "float32"),
        MXA: T.Tensor((NT, HEADS), "float32"),
        SMA: T.Tensor((NT, HEADS), "float32"),
        AQ: T.Tensor((QN,), "float32"),
        HN: T.Tensor((H,), "float32"),
        LG: T.Tensor((NEXP,), "float32"),
        GU: T.Tensor((NGU, E_GU), "float32"),
        ME: T.Tensor((NGU, E_K), "float32"),
        OUT: T.Tensor((H,), "float32"),
        MO: T.Tensor((H,), "float32"),
        XF: T.Tensor((H,), "float32"),
        QLF: T.Tensor((QLR,), "float32"),
        HF: T.Tensor((H,), "float32"),
        WOAO: T.Tensor((WOA,), "float32"),
        PART: T.Tensor((NB_MAX,), "float32"),
        TS: T.Tensor((18,), "int64"),
    ):
        with T.Kernel(G, threads=THREADS) as blk:
            # Scratch is allocated once and reused by every stage: separate
            # allocations per stage would blow the 232 KB per-SM budget. `xs` also
            # serves as the MLA q buffer and the combine accumulator.
            xs = T.alloc_shared((H,), "float32")
            scl = T.alloc_shared((BN, H // GRAN), "float32")
            amx = T.alloc_shared((H // GRAN,), "float32")
            eby = T.alloc_shared((H // GRAN,), "uint32")
            part = T.alloc_shared((THREADS,), "float32")
            ss = T.alloc_shared((1,), "float32")
            rowv = T.alloc_shared((BN,), "float32")
            acc = T.alloc_fragment((BN, 32), "float32")
            lut = T.alloc_shared((16,), "float32")
            # ---- MLA scratch
            kv = T.alloc_shared((MLA_BK, HEAD_DIM), "float32")
            os = T.alloc_shared((HEADS, HEAD_DIM), "float32")
            scs = T.alloc_shared((HEADS, MLA_BK), "float32")
            scp = T.alloc_shared((HEADS, MLA_BK, MLA_UNROLL), "float32")
            pp = T.alloc_shared((HEADS, MLA_BK), "float32")
            mb = T.alloc_shared((HEADS,), "float32")
            wb = T.alloc_shared((NT,), "float32")
            mmax = T.alloc_shared((1,), "float32")
            den = T.alloc_shared((1,), "float32")
            # ---- router / top-k scratch
            lval = T.alloc_shared((NEXP,), "float32")   # the unbiased score (routing weights)
            selv = T.alloc_shared((NEXP,), "float32")   # score + gate bias (expert choice)
            lpart = T.alloc_shared((512,), "float32")   # >= NEXP, reduce buffer
            smt = T.alloc_shared((TOPK,), "float32")
            sd = T.alloc_shared((1,), "float32")
            bv = T.alloc_shared((1,), "float32")
            bi = T.alloc_shared((1,), "int32")
            deid = T.alloc_shared((NGU,), "int32")
            # One tile's MoE down weights, staged in shared memory. The accumulate
            # loop folds NGU experts into ONE acc[r, l], so each `acc +=` is a
            # read-modify-write and reading global inline makes the 14 loads a serial
            # chain of HBM round trips (~10.4 us of pure latency, measured 13.16).
            # Staging breaks the chain: the 14 addresses are independent, so they are
            # all in flight before the first FMA needs one. Per-thread registers for
            # the same job cost 14 regs, which dropped occupancy to 1 block/SM and
            # made the cooperative launch of G=296 fail outright.
            wsh = T.alloc_shared((NGU, BN, W_DN), "uint32")
            wsh8 = T.alloc_shared((BN, E_K), FP8)   # shared expert w2 tile (2.3 KB)
            # Same idea for the MoE gate_up sweep, which is the largest stage: 10
            # reduction steps per chunk staged in shared, so 10 loads are in flight
            # instead of 1. Ten rather than all 20 because each in-flight value
            # occupies a register between the LDG and the STS, and the budget for
            # 2 blocks/SM is ~13.
            wgu = T.alloc_shared((BN, C_GU * 32), "uint32")

            for i in T.Parallel(16):
                lut[i] = LUT[i]
            T.sync_threads()
            if blk == 0:
                if T.get_thread_binding() == 0:
                    TS[0] = T.call_extern("int64", "clock64")
            if SEGA:

                # ============ S1a: rmsnorm(H) partial sums, 20 blocks in parallel.
                # Each block owns exactly THREADS elements, so the per-thread partial
                # IS the block partial and T.reduce_sum over `part` works.
                if blk < NBH:
                    for t0 in T.Parallel(THREADS):
                        part[t0] = X[blk * THREADS + t0] * X[blk * THREADS + t0]
                    T.sync_threads()
                    T.reduce_sum(part, ss, dim=0)
                    T.sync_threads()
                    if T.get_thread_binding() == 0:
                        PART[blk] = ss[0]
                T.sync_grid()
                if blk == 0:
                    if T.get_thread_binding() == 0:
                        TS[1] = T.call_extern("int64", "clock64")

                # ============ S1b: finish rmsnorm + fp8 quant -> XF.
                # Normalisation is done in the reference's exact association,
                # (x * rsqrt) * w, BEFORE the group absmax. Taking absmax(x*w) and
                # multiplying by rsqrt afterwards differs by one rounding, which flips
                # the ue8m0 ceil exponent on some groups -> 1.9e-2 error.
                if blk < NBH:
                    for t0 in T.Parallel(THREADS):
                        part[t0] = T.if_then_else(t0 < NBH,
                                                  PART[T.min(t0, NB_MAX - 1)], 0.0)
                    T.sync_threads()
                    T.reduce_sum(part, ss, dim=0)
                    T.sync_threads()
                    for t0 in T.Parallel(THREADS):
                        xs[t0] = (X[blk * THREADS + t0]
                                  * T.ieee_frsqrt(ss[0] * (1.0 / H) + EPS)) * GN[blk * THREADS + t0]
                    for g in T.Parallel(NGP):
                        amx[g] = 0.0
                    T.sync_threads()
                    for j in T.serial(GRAN):
                        for g in T.Parallel(NGP):
                            amx[g] = T.max(amx[g], T.abs(xs[g * GRAN + j]))
                    T.sync_threads()
                    for g in T.Parallel(NGP):
                        eby[g] = _ceil_exp_byte(T.reinterpret(
                            T.max(amx[g], AMAX_FLOOR) / FP8_MAX, "uint32"))
                    T.sync_threads()
                    for t0 in T.Parallel(THREADS):
                        sc = T.exp2(T.Cast("float32", eby[t0 // GRAN]) - 127.0)
                        XF[blk * THREADS + t0] = T.Cast(
                            "float32", T.Cast(FP8, xs[t0] / sc)) * sc
                T.sync_grid()
                if blk == 0:
                    if T.get_thread_binding() == 0:
                        TS[2] = T.call_extern("int64", "clock64")

                # ============ S2: wq_a [QLR, H] fp8 -> QL
                for it in T.serial(IT["qa"]):
                    tq = blk + it * G
                    if tq < NT_QA:
                        for i in T.Parallel(H):
                            xs[i] = XF[i]
                        for r, g in T.Parallel(BN, H // GRAN):
                            scl[r, g] = T.exp2(T.Cast("float32",
                                                      SQA[tq * BN + r, g]) - 127.0)
                        T.sync_threads()
                        for r, l in T.Parallel(BN, 32):
                            acc[r, l] = 0.0
                        T.sync_threads()
                        for step in T.serial(S_QA):
                            for r, l in T.Parallel(BN, 32):
                                wi = step * 32 + l
                                acc[r, l] += (
                                    xs[wi * 4 + 0] * T.Cast("float32", WQA[tq * BN + r, wi, 0])
                                    + xs[wi * 4 + 1] * T.Cast("float32", WQA[tq * BN + r, wi, 1])
                                    + xs[wi * 4 + 2] * T.Cast("float32", WQA[tq * BN + r, wi, 2])
                                    + xs[wi * 4 + 3] * T.Cast("float32", WQA[tq * BN + r, wi, 3])
                                ) * scl[r, wi // 8]
                        T.sync_threads()
                        T.reduce_sum(acc, rowv, dim=1)
                        T.sync_threads()
                        for r in T.Parallel(BN):
                            QL[wqa_off + tq * BN + r] = rowv[r]
                T.sync_grid()
                if blk == 0:
                    if T.get_thread_binding() == 0:
                        TS[3] = T.call_extern("int64", "clock64")
            if SEGB:

                # ============ S3a: rmsnorm(QLR) partials, 5 blocks
                if blk < NBQ:
                    for t0 in T.Parallel(THREADS):
                        part[t0] = QL[blk * THREADS + t0] * QL[blk * THREADS + t0]
                    T.sync_threads()
                    T.reduce_sum(part, ss, dim=0)
                    T.sync_threads()
                    if T.get_thread_binding() == 0:
                        PART[blk] = ss[0]
                T.sync_grid()
                if blk == 0:
                    if T.get_thread_binding() == 0:
                        TS[4] = T.call_extern("int64", "clock64")

                # ============ S3b: finish rmsnorm(QLR) + quant -> QLF
                if blk < NBQ:
                    for t0 in T.Parallel(THREADS):
                        part[t0] = T.if_then_else(t0 < NBQ,
                                                  PART[T.min(t0, NB_MAX - 1)], 0.0)
                    T.sync_threads()
                    T.reduce_sum(part, ss, dim=0)
                    T.sync_threads()
                    for t0 in T.Parallel(THREADS):
                        xs[t0] = (QL[blk * THREADS + t0]
                                  * T.ieee_frsqrt(ss[0] * (1.0 / QLR) + EPS)) * QNORM[blk * THREADS + t0]
                    for g in T.Parallel(NGP):
                        amx[g] = 0.0
                    T.sync_threads()
                    for j in T.serial(GRAN):
                        for g in T.Parallel(NGP):
                            amx[g] = T.max(amx[g], T.abs(xs[g * GRAN + j]))
                    T.sync_threads()
                    for g in T.Parallel(NGP):
                        eby[g] = _ceil_exp_byte(T.reinterpret(
                            T.max(amx[g], AMAX_FLOOR) / FP8_MAX, "uint32"))
                    T.sync_threads()
                    for t0 in T.Parallel(THREADS):
                        sc = T.exp2(T.Cast("float32", eby[t0 // GRAN]) - 127.0)
                        QLF[blk * THREADS + t0] = T.Cast(
                            "float32", T.Cast(FP8, xs[t0] / sc)) * sc
                T.sync_grid()
                if blk == 0:
                    if T.get_thread_binding() == 0:
                        TS[5] = T.call_extern("int64", "clock64")

                # ============ S4: wq_b [QN, QLR] fp8 -> Q  (QN = 8 heads x 512)
                for it in T.serial(IT["qb"]):
                    tq = blk + it * G
                    if tq < NT_QB:
                        for i in T.Parallel(QLR):
                            xs[i] = QLF[i]
                        for r, g in T.Parallel(BN, QLR // GRAN):
                            scl[r, g] = T.exp2(T.Cast("float32",
                                                      SQB[tq * BN + r, g]) - 127.0)
                        T.sync_threads()
                        for r, l in T.Parallel(BN, 32):
                            acc[r, l] = 0.0
                        T.sync_threads()
                        for step in T.serial(S_QB):
                            for r, l in T.Parallel(BN, 32):
                                wi = step * 32 + l
                                acc[r, l] += (
                                    xs[wi * 4 + 0] * T.Cast("float32", WQB[tq * BN + r, wi, 0])
                                    + xs[wi * 4 + 1] * T.Cast("float32", WQB[tq * BN + r, wi, 1])
                                    + xs[wi * 4 + 2] * T.Cast("float32", WQB[tq * BN + r, wi, 2])
                                    + xs[wi * 4 + 3] * T.Cast("float32", WQB[tq * BN + r, wi, 3])
                                ) * scl[r, wi // 8]
                        T.sync_threads()
                        T.reduce_sum(acc, rowv, dim=1)
                        T.sync_threads()
                        for r in T.Parallel(BN):
                            Q[tq * BN + r] = rowv[r]
                T.sync_grid()
                if blk == 0:
                    if T.get_thread_binding() == 0:
                        TS[6] = T.call_extern("int64", "clock64")

                # ============ S5: MLA decode, phase 1 (split over NT blocks).
                # One block per MLA_BK keys: partial output, block max, block sum.
                # Scores never reach global memory. Same structure as the standalone
                # `dsv41_mla_flash.py` kernel, which matched the reference to 7.2e-07.
                if blk < NT:
                    # q lives in `xs` as [HEADS, HEAD_DIM]; H=5120 >= 4096 floats
                    for h, d in T.Parallel(HEADS, HEAD_DIM):
                        xs[h * HEAD_DIM + d] = Q[h * HEAD_DIM + d]
                    T.sync_threads()
                    if blk < NWB:
                        # window keys: fp8 data + ue8m0 per 32-element group
                        for j, d in T.Parallel(MLA_BK, HEAD_DIM):
                            v = T.Cast("float32", T.reinterpret(WIN[blk * MLA_BK + j, d],
                                                               "float8_e4m3fn"))
                            sb = WIN[blk * MLA_BK + j, WIN_DATA_B + d // WIN_GRAN]
                            kv[j, d] = v * T.exp2(T.Cast("float32", sb) - 127.0)
                    else:
                        # compressed keys: e2m1 nibbles + fp8 scale per 16 elements
                        for j, d2 in T.Parallel(MLA_BK, HEAD_DIM // 2):
                            byte = CMP[(blk - NWB) * MLA_BK + j, d2]
                            lo = T.Cast("int32", byte & T.Cast("uint8", 0x0F))
                            hi = T.Cast("int32", byte >> T.Cast("uint8", 4))
                            sbyte = CMP[(blk - NWB) * MLA_BK + j,
                                        CMP_DATA_B + (2 * d2) // CMP_GRAN]
                            sc2 = T.Cast("float32", T.reinterpret(sbyte, FP8))
                            kv[j, 2 * d2] = _e2m1_val(lo) * sc2
                            kv[j, 2 * d2 + 1] = _e2m1_val(hi) * sc2
                    T.sync_threads()
                    # raw scores, MLA_UNROLL independent accumulators for ILP
                    for h, j, u in T.Parallel(HEADS, MLA_BK, MLA_UNROLL):
                        scp[h, j, u] = 0.0
                    T.sync_threads()
                    for d4 in T.serial(HEAD_DIM // MLA_UNROLL):
                        for h, j, u in T.Parallel(HEADS, MLA_BK, MLA_UNROLL):
                            scp[h, j, u] += (xs[h * HEAD_DIM + d4 * MLA_UNROLL + u]
                                             * kv[j, d4 * MLA_UNROLL + u])
                    T.sync_threads()
                    for h, j in T.Parallel(HEADS, MLA_BK):
                        scs[h, j] = 0.0
                    T.sync_threads()
                    for u in T.serial(MLA_UNROLL):
                        for h, j in T.Parallel(HEADS, MLA_BK):
                            scs[h, j] += scp[h, j, u]
                    T.sync_threads()
                    # block max of the scaled logits
                    for h in T.Parallel(HEADS):
                        mb[h] = -3.0e38
                    T.sync_threads()
                    for j in T.serial(MLA_BK):
                        for h in T.Parallel(HEADS):
                            mb[h] = T.max(mb[h], scs[h, j] * QB_SCALE)
                    T.sync_threads()
                    # p = exp(scale*s - m), block sum, then p @ V
                    for h, j in T.Parallel(HEADS, MLA_BK):
                        pp[h, j] = T.exp(scs[h, j] * QB_SCALE - mb[h])
                    T.sync_threads()
                    for h in T.Parallel(HEADS):
                        SMA[blk, h] = 0.0
                    T.sync_threads()
                    for j in T.serial(MLA_BK):
                        for h in T.Parallel(HEADS):
                            SMA[blk, h] += pp[h, j]
                    T.sync_threads()
                    for h, d in T.Parallel(HEADS, HEAD_DIM):
                        os[h, d] = 0.0
                    T.sync_threads()
                    for j in T.serial(MLA_BK):
                        for h, d in T.Parallel(HEADS, HEAD_DIM):
                            os[h, d] += pp[h, j] * kv[j, d]
                    T.sync_threads()
                    # PO is [H, NT, D], NOT [H, D, NT]. With lanes varying over `d`
                    # both this store and the combine's read are fully coalesced. The
                    # [H, D, NT] layout the standalone kernel uses makes each lane's
                    # run contiguous but every warp load touch 32 sectors 320B apart,
                    # i.e. 1/8 of each sector used -- survivable only while L1 has
                    # room to hold it. Here shared memory is ~69 KB per block, so L1
                    # cannot, and the combine measured 32.6 us against 12.5 standalone.
                    for h, d in T.Parallel(HEADS, HEAD_DIM):
                        PO[h, blk, d] = os[h, d]
                    for h in T.Parallel(HEADS):
                        MXA[blk, h] = mb[h]
                T.sync_grid()
                if blk == 0:
                    if T.get_thread_binding() == 0:
                        TS[7] = T.call_extern("int64", "clock64")

                # ============ S6: MLA phase 2 (combine), one CTA per head.
                # A single CTA merging all NT splits was the bottleneck in the
                # standalone kernel (18.6 us): one SM cannot hide the strided reads.
                # The attention output is then quantised to fp8 per 32-element group,
                # because that is what upstream feeds to o_proj.
                if blk < HEADS:
                    for i in T.Parallel(1):
                        mmax[0] = SINK[blk]
                    T.sync_threads()
                    for s2 in T.serial(NT):
                        for i in T.Parallel(1):
                            mmax[0] = T.max(mmax[0], MXA[s2, blk])
                    T.sync_threads()
                    for s2 in T.Parallel(NT):
                        wb[s2] = T.exp(MXA[s2, blk] - mmax[0])
                    for i in T.Parallel(1):
                        den[0] = T.exp(SINK[blk] - mmax[0])
                    T.sync_threads()
                    for s2 in T.serial(NT):
                        for i in T.Parallel(1):
                            den[0] += SMA[s2, blk] * wb[s2]
                    T.sync_threads()
                    for d in T.Parallel(HEAD_DIM):
                        xs[d] = 0.0
                    T.sync_threads()
                    for d in T.Parallel(HEAD_DIM):
                        for s2 in T.serial(NT):
                            xs[d] += PO[blk, s2, d] * wb[s2]
                        xs[d] = xs[d] * (1.0 / den[0])
                    T.sync_threads()
                    # inverse RoPE on the trailing ROPE_DIM dims. Each thread owns the
                    # two indices it reads, so in-place is safe.
                    for k in T.Parallel(HALF):
                        a = xs[NOPE_DIM + k]
                        b = xs[NOPE_DIM + HALF + k]
                        xs[NOPE_DIM + k] = a * FREQ[k, 0] + b * FREQ[k, 1]
                        xs[NOPE_DIM + HALF + k] = -a * FREQ[k, 1] + b * FREQ[k, 0]
                    T.sync_threads()
                    # fp8 quantisation of this head's 512 values, per 32-group
                    for g in T.Parallel(NGQ):
                        amx[g] = 0.0
                    T.sync_threads()
                    for j in T.serial(GRAN):
                        for g in T.Parallel(NGQ):
                            amx[g] = T.max(amx[g], T.abs(xs[g * GRAN + j]))
                    T.sync_threads()
                    for g in T.Parallel(NGQ):
                        eby[g] = _ceil_exp_byte(T.reinterpret(
                            T.max(amx[g], AMAX_FLOOR) / FP8_MAX, "uint32"))
                    T.sync_threads()
                    for d in T.Parallel(HEAD_DIM):
                        sc = T.exp2(T.Cast("float32", eby[d // GRAN]) - 127.0)
                        AQ[blk * HEAD_DIM + d] = T.Cast(
                            "float32", T.Cast(FP8, xs[d] / sc)) * sc
                T.sync_grid()
                if blk == 0:
                    if T.get_thread_binding() == 0:
                        TS[8] = T.call_extern("int64", "clock64")

                # ============ S7: wo_a [WOA, QN] fp8 -> WOAO
                for it in T.serial(IT["oa"]):
                    tq = blk + it * G
                    if tq < NT_OA:
                        for i in T.Parallel(QN):
                            xs[i] = AQ[i]
                        for r, g in T.Parallel(BN, QN // GRAN):
                            scl[r, g] = T.exp2(T.Cast("float32",
                                                      SOA[tq * BN + r, g]) - 127.0)
                        T.sync_threads()
                        for r, l in T.Parallel(BN, 32):
                            acc[r, l] = 0.0
                        T.sync_threads()
                        for step in T.serial(S_OA):
                            for r, l in T.Parallel(BN, 32):
                                wi = step * 32 + l
                                acc[r, l] += (
                                    xs[wi * 4 + 0] * T.Cast("float32", WOAW[tq * BN + r, wi, 0])
                                    + xs[wi * 4 + 1] * T.Cast("float32", WOAW[tq * BN + r, wi, 1])
                                    + xs[wi * 4 + 2] * T.Cast("float32", WOAW[tq * BN + r, wi, 2])
                                    + xs[wi * 4 + 3] * T.Cast("float32", WOAW[tq * BN + r, wi, 3])
                                ) * scl[r, wi // 8]
                        T.sync_threads()
                        T.reduce_sum(acc, rowv, dim=1)
                        T.sync_threads()
                        for r in T.Parallel(BN):
                            WOAO[tq * BN + r] = rowv[r]
                T.sync_grid()
                if blk == 0:
                    if T.get_thread_binding() == 0:
                        TS[9] = T.call_extern("int64", "clock64")

                # ============ S8: wo_b [H, WOA] fp8. Residual folded into the store,
                # which saves both a kernel and a full read of X.
                for it in T.serial(IT["ob"]):
                    tq = blk + it * G
                    if tq < NT_OB:
                        for i in T.Parallel(WOA):
                            xs[i] = WOAO[i]
                        for r, g in T.Parallel(BN, WOA // GRAN):
                            scl[r, g] = T.exp2(T.Cast("float32",
                                                      SOB[tq * BN + r, g]) - 127.0)
                        T.sync_threads()
                        for r, l in T.Parallel(BN, 32):
                            acc[r, l] = 0.0
                        T.sync_threads()
                        for step in T.serial(S_OB):
                            for r, l in T.Parallel(BN, 32):
                                wi = step * 32 + l
                                acc[r, l] += (
                                    xs[wi * 4 + 0] * T.Cast("float32", WOBW[tq * BN + r, wi, 0])
                                    + xs[wi * 4 + 1] * T.Cast("float32", WOBW[tq * BN + r, wi, 1])
                                    + xs[wi * 4 + 2] * T.Cast("float32", WOBW[tq * BN + r, wi, 2])
                                    + xs[wi * 4 + 3] * T.Cast("float32", WOBW[tq * BN + r, wi, 3])
                                ) * scl[r, wi // 8]
                        T.sync_threads()
                        T.reduce_sum(acc, rowv, dim=1)
                        T.sync_threads()
                        if resid_in_c:
                            # pure TP partial: the residual is added after the
                            # all_reduce, never before it (X is identical on all
                            # ranks, so summing it 8 times would scale it by TP)
                            for r in T.Parallel(BN):
                                OUT[tq * BN + r] = rowv[r]
                        else:
                            for r in T.Parallel(BN):
                                OUT[tq * BN + r] = X[tq * BN + r] + rowv[r]
                T.sync_grid()
                if blk == 0:
                    if T.get_thread_binding() == 0:
                        TS[10] = T.call_extern("int64", "clock64")
            if SEGC:
                if resid_in_c:
                    # OUT arrived as the TP partial of the attention output. The
                    # residual is folded in here rather than by a host-side add:
                    # one pass over 5120 inside a launch we already pay for.
                    for j, t0 in T.Parallel(NBH, THREADS):
                        OUT[j * THREADS + t0] = OUT[j * THREADS + t0] + X[j * THREADS + t0]
                    T.sync_grid()

                # ============ S9a: rmsnorm(H) on the post-attention residual, partials
                if blk < NBH:
                    for t0 in T.Parallel(THREADS):
                        part[t0] = OUT[blk * THREADS + t0] * OUT[blk * THREADS + t0]
                    T.sync_threads()
                    T.reduce_sum(part, ss, dim=0)
                    T.sync_threads()
                    if T.get_thread_binding() == 0:
                        PART[blk] = ss[0]
                T.sync_grid()
                if blk == 0:
                    if T.get_thread_binding() == 0:
                        TS[11] = T.call_extern("int64", "clock64")

                # ============ S9b: finish rmsnorm(H) + quant -> HF
                if blk < NBH:
                    for t0 in T.Parallel(THREADS):
                        part[t0] = T.if_then_else(t0 < NBH,
                                                  PART[T.min(t0, NB_MAX - 1)], 0.0)
                    T.sync_threads()
                    T.reduce_sum(part, ss, dim=0)
                    T.sync_threads()
                    for t0 in T.Parallel(THREADS):
                        xs[t0] = (OUT[blk * THREADS + t0]
                                  * T.ieee_frsqrt(ss[0] * (1.0 / H) + EPS)) * GN2[blk * THREADS + t0]
                    for g in T.Parallel(NGP):
                        amx[g] = 0.0
                    T.sync_threads()
                    for j in T.serial(GRAN):
                        for g in T.Parallel(NGP):
                            amx[g] = T.max(amx[g], T.abs(xs[g * GRAN + j]))
                    T.sync_threads()
                    for g in T.Parallel(NGP):
                        eby[g] = _ceil_exp_byte(T.reinterpret(
                            T.max(amx[g], AMAX_FLOOR) / FP8_MAX, "uint32"))
                    T.sync_threads()
                    for t0 in T.Parallel(THREADS):
                        sc = T.exp2(T.Cast("float32", eby[t0 // GRAN]) - 127.0)
                        HN[blk * THREADS + t0] = xs[t0]
                        HF[blk * THREADS + t0] = T.Cast(
                            "float32", T.Cast(FP8, xs[t0] / sc)) * sc
                T.sync_grid()
                if blk == 0:
                    if T.get_thread_binding() == 0:
                        TS[12] = T.call_extern("int64", "clock64")

                # ============ S10a: router logits. gate is [NEXP, H] bf16 and its input
                # is the f32 normed hidden (HN), NOT the fp8-quantised HF the experts
                # see: sqrtsoftplus scores the unquantised logit, so quantising first
                # would move the routing decision itself.
                #
                # At TP8 without EP every rank holds all 384 experts' gate rows, so the
                # top-k needs no communication.
                for it in T.serial(waves(NT_RT)):
                    tq = blk + it * G
                    if tq < NT_RT:
                        for i in T.Parallel(H):
                            xs[i] = HN[i]
                        T.sync_threads()
                        for r, l in T.Parallel(BN, 32):
                            acc[r, l] = 0.0
                        T.sync_threads()
                        # two bf16 per lane per step, so a warp still moves 128 B
                        for step in T.serial(S_RT):
                            for r, l in T.Parallel(BN, 32):
                                k0 = step * 64 + l * 2
                                acc[r, l] += (
                                    xs[k0] * T.Cast("float32", GW[tq * BN + r, k0])
                                    + xs[k0 + 1] * T.Cast("float32", GW[tq * BN + r, k0 + 1]))
                        T.sync_threads()
                        T.reduce_sum(acc, rowv, dim=1)
                        T.sync_threads()
                        for r in T.Parallel(BN):
                            LG[tq * BN + r] = rowv[r]
                T.sync_grid()
                if blk == 0:
                    if T.get_thread_binding() == 0:
                        TS[13] = T.call_extern("int64", "clock64")

                # ============ S10b: sqrtsoftplus + top-6 + renormalise, one block.
                # score = sqrt(log1p(exp(x))) written in the numerically stable form
                # sqrt(max(x,0) + log(1 + exp(-|x|))), which is what
                # sgl_kernel::activation::sqrt_softplus computes.
                #
                # Selection is done with a parallel max-reduce plus an index scan rather
                # than a single-threaded scan of 384: the naive form is 2304 serial
                # shared-memory round trips, which would cost more than the MMA stages.
                if blk == 0:
                    for j in T.Parallel(NEXP):
                        lval[j] = T.sqrt(T.max(LG[j], 0.0)
                                         + T.log(1.0 + T.exp(-T.abs(LG[j]))))
                    T.sync_threads()
                    # The bias moves which experts are picked, never what they are
                    # worth (inference/model.py:823-825): keep both forms so the
                    # weights below stay unbiased.
                    for j in T.Parallel(NEXP):
                        selv[j] = lval[j] + GB[j]
                    T.sync_threads()
                    for i in T.Parallel(1):
                        sd[0] = 0.0
                    T.sync_threads()
                    for rr in T.serial(TOPK):
                        for j in T.Parallel(512):
                            lpart[j] = T.if_then_else(
                                j < NEXP, selv[T.min(j, NEXP - 1)], -3.0e38)
                        T.sync_threads()
                        T.reduce_max(lpart, bv, dim=0)
                        T.sync_threads()
                        for i in T.Parallel(1):
                            bi[0] = NEXP      # atomic_min starts from just past the end
                        T.sync_threads()
                        for j in T.Parallel(NEXP):
                            if selv[j] == bv[0]:
                                # atomic_min instead of a plain store: on an exact tie
                                # several j match, and an unordered store would make the
                                # route nondeterministic run to run (and trip the data
                                # race checker). Taking the smallest id is also what a
                                # stable top-k produces.
                                T.atomic_min(bi[0], T.Cast("int32", j))
                        T.sync_threads()
                        for i in T.Parallel(1):
                            EXPID[rr] = bi[0]
                            # bv[0] is the *biased* maximum: the routing weight must come
                            # from the unbiased score at the same expert
                            smt[rr] = lval[bi[0]]
                            sd[0] += lval[bi[0]]
                            lval[bi[0]] = -3.0e38
                            selv[bi[0]] = -3.0e38
                        T.sync_threads()
                    # norm_topk_prob: weights sum to 1, then routed_scaling_factor. The
                    # shared expert is slot NGU-1 with weight 1.0 and takes no part in
                    # the routing, exactly as upstream's separate shared branch.
                    for i in T.Parallel(1):
                        for rr in T.serial(TOPK):
                            WGT[rr] = RSCALE * smt[rr] / sd[0]
                        WGT[NGU - 1] = 1.0
                T.sync_grid()
                if blk == 0:
                    if T.get_thread_binding() == 0:
                        TS[14] = T.call_extern("int64", "clock64")

                # ============ S11: MoE gate_up, all NGU slots (6 routed + shared) in one
                # stage. Six 2.95 MB GEMMs that each ran at ~350 GB/s standalone become
                # one tile sweep at 1424 GB/s, and the shared expert rides along for
                # free: it is the same shape (2304/8 = 288 per side at TP8), so it is
                # just slot 6 reading expert row NEXP of the weight tensor.
                for it in T.serial(IT["gu"]):
                    tq = blk + it * G
                    if tq < NT_GU:
                        s = tq // NROW_GU            # which slot (0..5 routed, 6 shared)
                        ti = tq % NROW_GU
                        for i in T.Parallel(H):
                            xs[i] = HF[i]
                        eid = T.if_then_else(s < TOPK, EXPID[T.min(s, TOPK - 1)], NEXP)
                        if s < TOPK:
                            for r, g in T.Parallel(BN, H // GRAN):
                                scl[r, g] = T.exp2(T.Cast("float32",
                                                          SGU[eid, ti * BN + r, g]) - 127.0)
                        else:
                            # shared expert: fp8 e4m3, one ue8m0 per 32 values
                            for r, g in T.Parallel(BN, H // GRAN):
                                scl[r, g] = T.exp2(T.Cast("float32",
                                                          SHGS[ti * BN + r, g]) - 127.0)
                        T.sync_threads()
                        for r, l in T.Parallel(BN, 32):
                            acc[r, l] = 0.0
                        T.sync_threads()
                        if s >= TOPK:
                            # four fp8 per lane per step: H/4/32 = 40 steps
                            for step in T.serial((H // 4) // 32):
                                for r, l in T.Parallel(BN, 32):
                                    wi = step * 32 + l
                                    acc[r, l] += (
                                        xs[wi * 4 + 0] * T.Cast("float32", SHGW[ti * BN + r, wi, 0])
                                        + xs[wi * 4 + 1] * T.Cast("float32", SHGW[ti * BN + r, wi, 1])
                                        + xs[wi * 4 + 2] * T.Cast("float32", SHGW[ti * BN + r, wi, 2])
                                        + xs[wi * 4 + 3] * T.Cast("float32", SHGW[ti * BN + r, wi, 3])
                                    ) * scl[r, wi // 8]
                        else:
                          for c in T.serial(S_GU // C_GU):
                            for r, m in T.Parallel(BN, C_GU * 32):
                                wgu[r, m] = WGU[eid, ti * BN + r, c * C_GU * 32 + m]
                            T.sync_threads()
                            for step in T.serial(C_GU):
                              for r, l in T.Parallel(BN, 32):
                                wi = (c * C_GU + step) * 32 + l
                                wd = wgu[r, step * 32 + l]
                                b0 = T.Cast("int32", wd & T.Cast("uint32", 0xFF))
                                b1 = T.Cast("int32", (wd >> T.Cast("uint32", 8))
                                            & T.Cast("uint32", 0xFF))
                                b2 = T.Cast("int32", (wd >> T.Cast("uint32", 16))
                                            & T.Cast("uint32", 0xFF))
                                b3 = T.Cast("int32", wd >> T.Cast("uint32", 24))
                                acc[r, l] += (
                                    xs[wi * 8 + 0] * lut[b0 & 0x0F]
                                    + xs[wi * 8 + 1] * lut[b0 >> 4]
                                    + xs[wi * 8 + 2] * lut[b1 & 0x0F]
                                    + xs[wi * 8 + 3] * lut[b1 >> 4]
                                    + xs[wi * 8 + 4] * lut[b2 & 0x0F]
                                    + xs[wi * 8 + 5] * lut[b2 >> 4]
                                    + xs[wi * 8 + 6] * lut[b3 & 0x0F]
                                    + xs[wi * 8 + 7] * lut[b3 >> 4]) * scl[r, wi // 4]
                        T.sync_threads()
                        T.reduce_sum(acc, rowv, dim=1)
                        T.sync_threads()
                        for r in T.Parallel(BN):
                            GU[s, ti * BN + r] = rowv[r]
                T.sync_grid()
                if blk == 0:
                    if T.get_thread_binding() == 0:
                        TS[15] = T.call_extern("int64", "clock64")

                # ============ S11: clamped SwiGLU per expert, one block per expert.
                # Clamp in bf16 before widening to fp32, matching sglang's
                # silu_and_mul_clamp. The routing weight is folded in so the down stage
                # needs no per-expert scaling.
                #
                # Inlining this into the down stage instead was measured WORSE
                # (down+swiglu 20.0 us vs 13.1 + 2.4): every down block would re-read
                # all 6x576 gate_up values, 8.8 MB of redundant traffic against only
                # 4.4 MB of down weights.
                if blk < NGU:
                    for i in T.Parallel(E_K):
                        g = T.Cast("float32", T.min(T.Cast("bfloat16", GU[blk, i]),
                                                    T.Cast("bfloat16", SWIGLU_LIMIT)))
                        u = T.Cast("float32", T.max(
                            T.min(T.Cast("bfloat16", GU[blk, E_K + i]),
                                  T.Cast("bfloat16", SWIGLU_LIMIT)),
                            T.Cast("bfloat16", -SWIGLU_LIMIT)))
                        ME[blk, i] = g / (1.0 + T.exp(-g)) * u * WGT[blk]
                T.sync_grid()
                if blk == 0:
                    if T.get_thread_binding() == 0:
                        TS[16] = T.call_extern("int64", "clock64")

                # ============ S12: MoE down. The six experts are concatenated along K
                # into ONE accumulator. Reducing them one at a time (each with its own
                # AllReduce and syncthreads) cost 21 us at 211 GB/s; this form is 13.
                # The lane index stays division-free on purpose: `wg // 36` on a
                # vectorised lane makes TVM's simplifier build a Ramp with a non-scalar
                # base and abort.
                for s2 in T.Parallel(NGU):
                    deid[s2] = T.if_then_else(s2 < TOPK, EXPID[T.min(s2, TOPK - 1)], NEXP)
                T.sync_threads()
                for it in T.serial(IT["dn"]):
                    tq = blk + it * G
                    if tq < NT_DN:
                        for s2, i in T.Parallel(NGU, E_K):
                            xs[s2 * E_K + i] = ME[s2, i]
                        for s2, r, g in T.Parallel(NGU, BN, E_K // GRAN):
                            scl[r, s2 * (E_K // GRAN) + g] = T.if_then_else(
                                s2 < TOPK,
                                T.exp2(T.Cast("float32",
                                              SDN[deid[s2], tq * BN + r, g]) - 127.0),
                                T.exp2(T.Cast("float32",
                                              SHDS[tq * BN + r, g]) - 127.0))
                        T.sync_threads()
                        for r, l in T.Parallel(BN, 32):
                            acc[r, l] = 0.0
                        T.sync_threads()
                        # pass 1: every independent load, straight into shared.
                        # The shared slot is fp8 (one value per byte, 288 wide), so
                        # it gets its own buffer instead of a row of wsh.
                        for r, i8 in T.Parallel(BN, E_K):
                            wsh8[r, i8] = SHDW[tq * BN + r, i8]
                        for es, r, wi in T.Parallel(NGU - 1, BN, W_DN):
                            wsh[es, r, wi] = WDN[deid[es], tq * BN + r, wi]
                        T.sync_threads()
                        # pass 2: consume from shared (~30cy, not ~700)
                        for es in T.serial(NGU):
                          if es < TOPK:
                            for step in T.serial(S_DN):
                                for r, l in T.Parallel(BN, 32):
                                    wi = step * 32 + l
                                    if wi < W_DN:
                                        wd = wsh[es, r, wi]
                                        b0 = T.Cast("int32", wd & T.Cast("uint32", 0xFF))
                                        b1 = T.Cast("int32", (wd >> T.Cast("uint32", 8))
                                                    & T.Cast("uint32", 0xFF))
                                        b2 = T.Cast("int32", (wd >> T.Cast("uint32", 16))
                                                    & T.Cast("uint32", 0xFF))
                                        b3 = T.Cast("int32", wd >> T.Cast("uint32", 24))
                                        acc[r, l] += (
                                            xs[es * E_K + wi * 8 + 0] * lut[b0 & 0x0F]
                                            + xs[es * E_K + wi * 8 + 1] * lut[b0 >> 4]
                                            + xs[es * E_K + wi * 8 + 2] * lut[b1 & 0x0F]
                                            + xs[es * E_K + wi * 8 + 3] * lut[b1 >> 4]
                                            + xs[es * E_K + wi * 8 + 4] * lut[b2 & 0x0F]
                                            + xs[es * E_K + wi * 8 + 5] * lut[b2 >> 4]
                                            + xs[es * E_K + wi * 8 + 6] * lut[b3 & 0x0F]
                                            + xs[es * E_K + wi * 8 + 7] * lut[b3 >> 4]
                                        ) * scl[r, es * (E_K // GRAN) + wi // 4]
                          else:
                            # shared expert down: fp8, 288 = 9 x 32 so one value per lane
                            for step in T.serial(E_K // 32):
                                for r, l in T.Parallel(BN, 32):
                                    i8 = step * 32 + l
                                    acc[r, l] += (
                                        xs[(NGU - 1) * E_K + i8]
                                        * T.Cast("float32", wsh8[r, i8])
                                    ) * scl[r, (NGU - 1) * (E_K // GRAN) + i8 // GRAN]
                        T.sync_threads()
                        T.reduce_sum(acc, rowv, dim=1)
                        T.sync_threads()
                        if moe_sep:
                            for r in T.Parallel(BN):
                                MO[tq * BN + r] = rowv[r]
                        else:
                            for r in T.Parallel(BN):
                                OUT[tq * BN + r] = OUT[tq * BN + r] + rowv[r]
            if blk == 0:
                if T.get_thread_binding() == 0:
                    TS[17] = T.call_extern("int64", "clock64")

    return main


NOM_GHZ = 2.032   # B300 SM clock; us are always cycles/NOM_GHZ, never cycles/wall

STAGES = ["rmsnorm1", "wq_a", "rmsnormQL", "wq_b", "mla split", "mla combine",
          "wo_a", "wo_b", "rmsnorm2", "router logits", "router topk",
          "moe gate_up", "swiglu", "moe down"]
# each entry is (from, to) in the 18-slot timestamp array
STAMP_PAIRS = [(0, 2), (2, 3), (3, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 10),
               (10, 12), (12, 13), (13, 14), (14, 15), (15, 16), (16, 17)]


# --------------------------------------------------------------- host tensors
def make_host_weights(seed=0):
    dev = "cuda"
    torch.manual_seed(seed)

    def fp8_pair(N, K):
        w = (torch.randn(N, K, device=dev) * 0.05).to(torch.float8_e4m3fn)
        sf = torch.randint(120, 133, (N, K // GRAN), dtype=torch.uint8, device=dev)
        return w, sf

    def fp4_table(NE, N, K):
        # NE + NSH rows: the last one is the shared expert, which at TP8 is sharded
        # the same way (2304/8 = 288 per side) and so has an identical footprint.
        w = torch.randint(0, 256, (NE + NSH, N, K // 2), dtype=torch.uint8, device=dev)
        sf = torch.randint(120, 133, (NE + NSH, N, K // GRAN), dtype=torch.uint8,
                           device=dev)
        return w, sf

    qa_w, qa_s = fp8_pair(QLR, H)
    qb_w, qb_s = fp8_pair(QN, QLR)
    oa_w, oa_s = fp8_pair(WOA, QN)
    ob_w, ob_s = fp8_pair(H, WOA)
    gw = (torch.randn(NEXP, H, device=dev) * 0.02).to(torch.bfloat16)
    # ffn.gate.bias: selects experts only, never scales them (inference/model.py:824)
    gb = (torch.randn(NEXP, device=dev) * 0.5).float()
    gu_w, gu_s = fp4_table(NEXP, E_GU, H)
    dn_w, dn_s = fp4_table(NEXP, H, E_K)
    # the shared expert: fp8 e4m3 with one ue8m0 per 32 values, its own tensors
    sh_gw, sh_gs = fp8_pair(E_GU, H)
    sh_dw, sh_ds = fp8_pair(H, E_K)
    return dict(qa_w=qa_w, qa_s=qa_s, qb_w=qb_w, qb_s=qb_s,
                oa_w=oa_w, oa_s=oa_s, ob_w=ob_w, ob_s=ob_s, gw=gw, gb=gb,
                gu_w=gu_w, gu_s=gu_s, dn_w=dn_w, dn_s=dn_s,
                sh_gw=sh_gw, sh_gs=sh_gs, sh_dw=sh_dw, sh_ds=sh_ds)


def _quant_groups(v):
    """fp8 quantise a 1-D f32 vector per GRAN group, return the dequantised value.

    Mirrors the kernel exactly: group absmax, ue8m0 exponent rounded UP, round to
    fp8, multiply the scale back.
    """
    G_ = v.numel() // GRAN
    vg = v.view(G_, GRAN)
    amax = vg.abs().amax(dim=1).clamp_min(AMAX_FLOOR)
    bits = (amax / FP8_MAX).view(torch.int32)
    byte = (((bits >> 23) & 0xFF).to(torch.int64)
            + ((bits & 0x7FFFFF) != 0).to(torch.int64))
    scale = torch.exp2(byte.to(torch.float32) - 127.0)
    q = (vg / scale[:, None]).clamp(-FP8_MAX, FP8_MAX).to(
        torch.float8_e4m3fn).float()
    return q.view(-1) * scale.repeat_interleave(GRAN)


def _rmsnorm_f32(v, w):
    """The pre-quantisation normed value: what the router scores."""
    var = v.pow(2).mean()
    return v * torch.rsqrt(var + EPS) * w


def ref_layer(t, x, gn, qnorm, gn2, win, cmp_, sink, freq, pos):
    """Stage-for-stage torch replica, including fp8/fp4 quantisation and MLA."""
    from fused_ew_b300 import ref_rmsnorm_quant
    from mxfp8_gemv_b1 import dequant_rows as dq8
    from mxfp4_gemv_b1 import dequant_fp4 as dq4
    from dsv41_mla_ref import decode_attention_ref

    def norm_quant(v, w):
        y, byte = ref_rmsnorm_quant(v, w, EPS)
        scale = torch.exp2(byte.to(torch.float32) - 127.0)
        return y.float() * scale.repeat_interleave(GRAN)

    xf = norm_quant(x, gn)
    ql = dq8(t["qa_w"], t["qa_s"]) @ xf
    qlf = norm_quant(ql, qnorm)
    q = dq8(t["qb_w"], t["qb_s"]) @ qlf

    # MLA decode from this layer's own Q, then the same fp8 quantisation of the
    # output that the fused combine stage performs before o_proj.
    attn = decode_attention_ref(q.view(1, HEADS, HEAD_DIM), win.unsqueeze(0),
                                cmp_.unsqueeze(0), sink=sink, position=pos)[0]
    aq = _quant_groups(attn.reshape(-1))
    woa_o = dq8(t["oa_w"], t["oa_s"]) @ aq
    out = x + dq8(t["ob_w"], t["ob_s"]) @ woa_o
    hf_f32 = _rmsnorm_f32(out, gn2)     # the router scores the unquantised norm
    hf = norm_quant(out, gn2)

    # ---- router: fp32 logits from the f32 normed hidden, sqrtsoftplus, top-6 by
    # score (no correction bias, no group routing for this model), renormalised,
    # then routed_scaling_factor. Same as moe_fused_gate / route_warp.cuh.
    logits = (t["gw"].float() @ hf_f32)
    score = torch.sqrt(torch.nn.functional.softplus(logits))
    # ffn.gate.bias selects experts but does not scale them (inference/model.py:824);
    # the checkpoint carries one per layer even though topk_method is noaux_tc.
    topv, topi = (score + t["gb"].float()).topk(TOPK)
    wts = RSCALE * score[topi] / score[topi].sum()

    def swiglu(gu):
        lim = SWIGLU_LIMIT
        g = torch.minimum(gu[:E_K].to(torch.bfloat16),
                          torch.tensor(lim, dtype=torch.bfloat16))
        u = torch.maximum(torch.minimum(gu[E_K:].to(torch.bfloat16),
                                        torch.tensor(lim, dtype=torch.bfloat16)),
                          torch.tensor(-lim, dtype=torch.bfloat16))
        return torch.nn.functional.silu(g.float()) * u.float()

    moe = torch.zeros(H, device="cuda")
    for s, e in enumerate(topi.tolist()):
        gu = dq4(t["gu_w"][e], t["gu_s"][e]) @ hf
        moe += wts[s] * (dq4(t["dn_w"][e], t["dn_s"][e]) @ swiglu(gu))
    # shared expert: its own **fp8 e4m3** matrices (32x32 ue8m0 blocks), weight
    # 1.0, untouched by the routing scale. The checkpoint's
    # `ffn.shared_experts.w1/w3/w2` are F8_E4M3 while the routed experts are MXFP4,
    # so this is a different format at twice the bytes -- not slot NEXP of the fp4
    # table.
    gu = dq8(t["sh_gw"], t["sh_gs"]) @ hf
    moe += dq8(t["sh_dw"], t["sh_ds"]) @ swiglu(gu)
    return ql, q, aq, out + moe, topi


def build(seed=0):
    from dsv41_mla_flash import make_inputs as mla_inputs

    t = make_host_weights(seed)
    dev = "cuda"
    torch.manual_seed(seed + 1)
    x = torch.randn(H, device=dev)
    gn = torch.randn(H, device=dev) * 0.1 + 1.0
    qnorm = torch.randn(QLR, device=dev) * 0.1 + 1.0
    gn2 = torch.randn(H, device=dev) * 0.1 + 1.0
    # KV cache / sink / rope table: reuse the standalone MLA harness so both
    # kernels see identical inputs.
    _q, win, cmp_, sink, freq, pos = mla_inputs(NWIN, NCMP, seed=seed)
    # EXPID / WGT are outputs now: the kernel runs the router itself.
    expids = torch.zeros(TOPK, dtype=torch.int32, device=dev)
    wgts = torch.zeros(NGU, device=dev)
    lut = torch.tensor(CODE, dtype=torch.float32, device=dev)

    def z(*shape):
        return torch.zeros(*shape, dtype=torch.float32, device=dev)

    args = dict(
        X=x, GN=gn, QNORM=qnorm, GN2=gn2,
        WQA=t["qa_w"].view(QLR, H // 4, 4), SQA=t["qa_s"],
        WQB=t["qb_w"].view(QN, QLR // 4, 4), SQB=t["qb_s"],
        WOAW=t["oa_w"].view(WOA, QN // 4, 4), SOA=t["oa_s"],
        WOBW=t["ob_w"].view(H, WOA // 4, 4), SOB=t["ob_s"],
        WIN=win, CMP=cmp_, SINK=sink, FREQ=freq,
        WGU=t["gu_w"].view(torch.int32).view(torch.uint32).view(
            NEXP + NSH, E_GU, H // 8),
        SGU=t["gu_s"],
        WDN=t["dn_w"].view(torch.int32).view(torch.uint32).view(
            NEXP + NSH, H, E_K // 8),
        SDN=t["dn_s"],
        GW=t["gw"], GB=t["gb"],
        SHGW=t["sh_gw"].view(E_GU, H // 4, 4), SHGS=t["sh_gs"],
        SHDW=t["sh_dw"], SHDS=t["sh_ds"],
        LUT=lut, EXPID=expids, WGT=wgts,
        QL=z(QLR), Q=z(QN),
        PO=z(HEADS, NT, HEAD_DIM), MXA=z(NT, HEADS), SMA=z(NT, HEADS),
        AQ=z(QN), HN=z(H), LG=z(NEXP),
        GU=z(NGU, E_GU), ME=z(NGU, E_K), OUT=z(H), MO=z(H),
        XF=z(H), QLF=z(QLR), HF=z(H), WOAO=z(WOA),
        PART=z(NB_MAX),
        TS=torch.zeros(18, dtype=torch.int64, device=dev),
    )
    return t, args, (x, gn, qnorm, gn2, win, cmp_, sink, freq, pos)


ORDER = ["X", "GN", "QNORM", "GN2", "WQA", "SQA", "WQB", "SQB", "WOAW",
         "SOA", "WOBW", "SOB", "WIN", "CMP", "SINK", "FREQ", "WGU", "SGU",
         "WDN", "SDN", "GW", "GB", "SHGW", "SHGS", "SHDW", "SHDS", "LUT",
         "EXPID", "WGT", "QL", "Q", "PO", "MXA",
         "SMA", "AQ", "HN", "LG", "GU", "ME", "OUT", "MO", "XF", "QLF", "HF",
         "WOAO", "PART", "TS"]

if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "check"
    print(f"megakernel: G={G} threads={THREADS} BN={BN}  MLA: {HEADS} heads, "
          f"NT={NT} splits")
    k = decode_layer()
    t, args, ref_in = build()
    k(*[args[n] for n in ORDER])
    torch.cuda.synchronize()

    if what == "check":
        rql, rq, raq, rout, rtopi = ref_layer(t, *ref_in)
        e_ql = (args["QL"] - rql).abs().max().item() / rql.abs().max().item()
        e_aq = (args["AQ"] - raq).abs().max().item() / raq.abs().max().item()
        e_q = (args["OUT"] - rout).abs().max().item() / rout.abs().max().item()
        print(f"  q_lora           rel={e_ql:.3e}")
        print(f"  attn out (quant) rel={e_aq:.3e}")
        print(f"  layer out        rel={e_q:.3e}   "
              f"{'MATCH' if e_q < 5e-3 else 'MISMATCH'}")

        # The router may legitimately pick a different expert set when two scores are
        # a hair apart, so report the overlap rather than demanding equality.
        ktopi = torch.sort(args["EXPID"].long())[0]
        rtopi = torch.sort(rtopi.long())[0]
        same = int((ktopi == rtopi).sum().item())
        print(f"  routed experts   {same}/{TOPK} identical "
              f"(kernel {ktopi.tolist()} ref {rtopi.tolist()})")
        sys.exit(0)

    def call():
        k(*[args[n] for n in ORDER])

    for _ in range(10):
        call()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(50):
            call()
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(10):
        t0 = time.perf_counter()
        g.replay()
        torch.cuda.synchronize()
        best = min(best, (time.perf_counter() - t0) / 50 * 1e6)

    # The timestamps are overwritten by every launch, so a single read is a single
    # sample -- on a shared GPU that is a +-8% scatter per stage, which is wider
    # than anything we are trying to measure. Replay a few times and keep the
    # per-slot minimum: the minimum is the sample that ran with the least
    # interference, and every slot's minimum comes from the same best-case launch
    # ordering closely enough to keep the stage deltas consistent.
    tmin = None
    for _ in range(12):
        g.replay()
        torch.cuda.synchronize()
        cur = args["TS"].cpu()
        tmin = cur if tmin is None else torch.minimum(tmin, cur)
    args["TS"].copy_(tmin)

    # Absolute us come from the raw clock64 deltas at the nominal SM clock, NOT
    # from cycles/wall: that ratio silently absorbs cooperative-launch queueing and
    # would make sum(stages) equal the wall time by construction. Reported
    # separately so the difference between the two is visible.
    ts = args["TS"].cpu()
    cyc = (ts[-1] - ts[0]).item()
    eff = cyc / best / 1e3                          # effective clock incl. the gaps
    tm = {}
    for nm, (a, b) in zip(STAGES, STAMP_PAIRS):
        c = (ts[b] - ts[a]).item()
        us = c / NOM_GHZ / 1e3                      # cycles at a fixed 2.032 GHz
        tm[nm] = us
        print(f"    {nm:16s} {us:7.2f} us   {c/1e3:7.1f} kcyc")
    tot = sum(tm.values())
    print(f"    {'sum(stages)':16s} {tot:7.2f} us   {cyc/1e3:7.1f} kcyc")
    print(f"    {'effective clk':16s} {eff:7.2f} GHz  "
          f"(= cycles/wall; a gap means launch queueing, not kernel time)")

    print(f"  megakernel, whole layer, ONE launch: {best:7.2f} us   (G={G})")
    print(f"  kernel-internal (cycles @ {NOM_GHZ} GHz):   {tot:7.2f} us"
          f"   [{100.0*(best-tot)/best:5.1f}% of wall is outside the kernel]")
    print(f"  x40 layers: {best*40/1000:7.3f} ms/token -> {1e6/(best*40):6.1f} tok/s")
    print(f"  x40 layers, kernel-internal only:      "
          f"{tot*40/1000:6.3f} ms/token -> {1e6/(tot*40):6.1f} tok/s")

    # wq_a is the one stage where a single GPU stands in for all 8 TP8 ranks: it
    # runs full-width (1280 rows) because sharding it would need a cross-GPU
    # all-gather (2.5 KB, but not expressible inside one kernel). A real rank reads
    # 1/8 of those rows; 2.7 us is the measured standalone time for the 160-row
    # shard.
    adj = best - tm["wq_a"] + 2.7
    print(f"  TP8-adjusted for the sharded wq_a:  {adj:7.2f} us/layer"
          f"  -> {1e6/(adj*40):6.1f} tok/s  (estimate)")
