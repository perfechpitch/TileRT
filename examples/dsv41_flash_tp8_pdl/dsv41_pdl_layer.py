#!/usr/bin/env python3
"""The DSV4.1-Flash decoder layer as a PDL chain of specialised kernels.

Replaces `dsv41_layer_megakernel.py`'s single cooperative launch. Same math, same
stage order, but each stage is its own kernel and consecutive kernels are joined
by Programmatic Dependent Launch:

    kernel i     [ prefetch: weight tile -> smem ]  pdl_sync  [ consume ]  pdl_trigger
                                                                             |
    kernel i+1   [ prefetch ] .................. pdl_sync [ consume ] ........+

`pdl_sync()` is `griddepcontrol.wait`, `pdl_trigger()` is
`griddepcontrol.launch_dependents`. Everything before the wait overlaps the
predecessor's tail; the trigger releases the successor before this kernel's tail.

Why this beats one launch
-------------------------
The single launch forced every stage to share one shape: BN=8, 256 threads, and a
296-block grid cap because a cooperative launch needs every block resident. That
cost a lot:

  * `wo_b` and MoE `down` need 640 tiles, so they ran 3 waves with the last 16%
    full -- occupancy, not bandwidth, was the ceiling. The layer streamed 37.6 MB
    in 63.6 us = 591 GB/s against ~8 TB/s of hardware.
  * `moe down` had to use BN=8 where the standalone kernel's best was BN=32.
  * Registers were so tight (2 blocks/SM) that staging 14 weight values in
    registers dropped it to 1 block/SM and made the launch fail outright.

None of that applies once each stage is its own launch: no residency requirement,
no shared shape, no cooperative ceiling.

Why PDL is the right join, and `sync_grid` was not
--------------------------------------------------
Every stage here is a small latency-bound GEMV (the MoE's per-expert shard is only
1.47 MB, i.e. 10 KB per block -- nowhere near enough to saturate HBM). A chain of
latency-bound stages is exactly the case PDL exists for: the next stage's weight
stream can be in flight while this one finishes. `sync_grid` is a hard barrier and
cannot express that.

Measured on B300 (bench/pdl_graph_probe.py, single-wave grid, CUDA graph): PDL
survives graph capture in this stack and buys 3.34 us/kernel there, more than in
eager mode -- so no manual programmatic-edge installation is needed the way
NaiveRT's runtime does it.

Where the prefetch actually goes
--------------------------------
A stage can only hide work that does not depend on its predecessor. Two groups:

  * **Static weight tiles** -- wq_a, wq_b, wo_a, wo_b, router. The tile index is
    derived from blockIdx alone, so the entire tile streams in before the sync.
  * **The shared expert** -- its row is the fixed index NEXP, so gate_up and down
    can prefetch it while the top-k is still being computed. (This is the trick
    sgl-nvidia-mega names `load_shared_expert_before_pdl`.)

The routed experts cannot be prefetched: which experts they are is the top-k's
output. That is the one dependency the chain cannot break, and it is why the MoE
is the last thing to become fast.

Run:
    python dsv41_pdl_layer.py check     # numerics against the torch replica
    python dsv41_pdl_layer.py bench     # CUDA-graph timing, no-PDL vs PDL
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

from dsv41_layer_megakernel import (  # noqa: E402
    H, QLR, QN, WOA, E_GU, E_K, NEXP, TOPK, NSH, NGU, RSCALE, SWIGLU_LIMIT,
    EPS, HEAD_DIM, NOPE_DIM, ROPE_DIM, HALF, HEADS, MLA_BK, NWIN, NCMP, NWB, NT,
    WIN_DATA_B, WIN_GRAN, WIN_ROW_B, CMP_DATA_B, CMP_GRAN, CMP_ROW_B,
    MLA_UNROLL, QB_SCALE, NGQ, GRAN, FP8, FP8_MAX, AMAX_FLOOR, NB_MAX,
    _ceil_exp_byte, _e2m1_val, build, ref_layer, ORDER, NOM_GHZ,
)

THREADS = 256
BN = 8


# ------------------------------------------------------------------ norms
def k_norm_partial(src="X", pdl=True):
    """PART[i] = sum of squares of block i's slice of `src` (X / QL / OUT)."""
    NB = (QLR if src == "QL" else H) // THREADS

    @T.prim_func
    def main(X: T.Tensor((H,), "float32"),
             QL: T.Tensor((QLR,), "float32"),
             OUT: T.Tensor((H,), "float32"),
             PART: T.Tensor((NB_MAX,), "float32")):
        with T.Kernel(NB, threads=THREADS) as blk:
            part = T.alloc_shared((THREADS,), "float32")
            ss = T.alloc_shared((1,), "float32")
            if pdl:
                T.pdl_sync()
            for t0 in T.Parallel(THREADS):
                v = {"X": X, "QL": QL, "OUT": OUT}[src][blk * THREADS + t0]
                part[t0] = v * v
            T.sync_threads()
            T.reduce_sum(part, ss, dim=0)
            T.sync_threads()
            if T.get_thread_binding() == 0:
                PART[blk] = ss[0]
            if pdl:
                T.pdl_trigger()

    return main, ["X", "QL", "OUT", "PART"]


def k_norm_apply(n, src="X", wn="GN", dst="XF", keep_norm=False, pdl=True):
    """Finish the norm from the partials, fp8-quantise, write `dst`.

    `keep_norm` also writes the unquantised value: only rmsnorm2 needs it, because
    the router scores the f32 normed hidden, not the fp8 payload the experts see.
    """
    NB = n // THREADS
    NGP = THREADS // GRAN

    @T.prim_func
    def main(X: T.Tensor((H,), "float32"),
             QL: T.Tensor((QLR,), "float32"),
             OUT: T.Tensor((H,), "float32"),
             GN: T.Tensor((H,), "float32"),
             QNORM: T.Tensor((QLR,), "float32"),
             GN2: T.Tensor((H,), "float32"),
             PART: T.Tensor((NB_MAX,), "float32"),
             XF: T.Tensor((H,), "float32"),
             QLF: T.Tensor((QLR,), "float32"),
             HF: T.Tensor((H,), "float32"),
             HN: T.Tensor((H,), "float32")):
        with T.Kernel(NB, threads=THREADS) as blk:
            part = T.alloc_shared((THREADS,), "float32")
            ss = T.alloc_shared((1,), "float32")
            xs = T.alloc_shared((THREADS,), "float32")
            amx = T.alloc_shared((NGP,), "float32")
            eby = T.alloc_shared((NGP,), "uint32")
            if pdl:
                T.pdl_sync()
            for t0 in T.Parallel(THREADS):
                part[t0] = T.if_then_else(t0 < NB, PART[T.min(t0, NB_MAX - 1)], 0.0)
            T.sync_threads()
            T.reduce_sum(part, ss, dim=0)
            T.sync_threads()
            for t0 in T.Parallel(THREADS):
                v = {"X": X, "QL": QL, "OUT": OUT}[src][blk * THREADS + t0]
                w = {"GN": GN, "QNORM": QNORM, "GN2": GN2}[wn][blk * THREADS + t0]
                xs[t0] = (v * T.ieee_frsqrt(ss[0] * (1.0 / n) + EPS)) * w
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
                if keep_norm:
                    HN[blk * THREADS + t0] = xs[t0]
                q = T.Cast("float32", T.Cast(FP8, xs[t0] / sc)) * sc
                {"XF": XF, "QLF": QLF, "HF": HF}[dst][blk * THREADS + t0] = q
            if pdl:
                T.pdl_trigger()

    return main, ["X", "QL", "OUT", "GN", "QNORM", "GN2", "PART",
                   "XF", "QLF", "HF", "HN"]


# ------------------------------------------------------------------ fp8 GEMV
def k_gemv(which, N, K, off=0, pdl=True, pf=False, resid=True):
    """One fp8 GEMV stage -- wq_a / wq_b / wo_a / wo_b -- selected by `which`.

    `pf` stages the weight tile through shared memory before `pdl_sync`; without it
    the weights stream straight from HBM to registers, as the fused kernel did.

    Both paths exist because the prefetch is not free and not always a win. These
    stages already run near their bandwidth ceiling (~1000 GB/s for wq_a), so moving
    the tile earlier just relocates traffic that was going to happen anyway, and the
    smem round trip costs an extra STS+LDS pass plus a barrier. Measured: wq_a 6.5 ->
    9.4 us and wq_b 4.7 -> 5.6 with `pf`, so it defaults off and the wins come from
    the grid no longer being capped instead.

    Arithmetic is identical to the fused version's S2/S4/S7/S8 either way.
    """
    S_K = (K // 4) // 32

    @T.prim_func
    def main(WQA: T.Tensor((QLR, H // 4, 4), FP8),
             SQA: T.Tensor((QLR, H // GRAN), "uint8"),
             WQB: T.Tensor((QN, QLR // 4, 4), FP8),
             SQB: T.Tensor((QN, QLR // GRAN), "uint8"),
             WOAW: T.Tensor((WOA, QN // 4, 4), FP8),
             SOA: T.Tensor((WOA, QN // GRAN), "uint8"),
             WOBW: T.Tensor((H, WOA // 4, 4), FP8),
             SOB: T.Tensor((H, WOA // GRAN), "uint8"),
             XF: T.Tensor((H,), "float32"),
             QLF: T.Tensor((QLR,), "float32"),
             AQ: T.Tensor((QN,), "float32"),
             QL: T.Tensor((QLR,), "float32"),
             Q: T.Tensor((QN,), "float32"),
             WOAO: T.Tensor((WOA,), "float32"),
             X: T.Tensor((H,), "float32"),
             OUT: T.Tensor((H,), "float32")):
        with T.Kernel(N // BN, threads=THREADS) as blk:
            xs = T.alloc_shared((K,), "float32")
            scl = T.alloc_shared((BN, K // GRAN), "float32")
            acc = T.alloc_fragment((BN, 32), "float32")
            rowv = T.alloc_shared((BN,), "float32")
            wsh = T.alloc_shared((BN, K // 4, 4), FP8)
            Wt = {"qa": WQA, "qb": WQB, "oa": WOAW, "ob": WOBW}[which]
            St = {"qa": SQA, "qb": SQB, "oa": SOA, "ob": SOB}[which]
            src = {"qa": XF, "qb": QLF, "oa": AQ, "ob": WOAO}[which]
            if pf:
                for r, i4, j in T.Parallel(BN, K // 4, 4):
                    wsh[r, i4, j] = Wt[off + blk * BN + r, i4, j]
            if pdl:
                T.pdl_sync()
            for i in T.Parallel(K):
                xs[i] = src[i]
            T.sync_threads()
            for r, g in T.Parallel(BN, K // GRAN):
                scl[r, g] = T.exp2(T.Cast("float32", St[off + blk * BN + r, g]) - 127.0)
            T.sync_threads()
            for r, l in T.Parallel(BN, 32):
                acc[r, l] = 0.0
            T.sync_threads()
            for step in T.serial(S_K):
                for r, l in T.Parallel(BN, 32):
                    wi = step * 32 + l
                    if pf:
                        acc[r, l] += (
                            xs[wi * 4 + 0] * T.Cast("float32", wsh[r, wi, 0])
                            + xs[wi * 4 + 1] * T.Cast("float32", wsh[r, wi, 1])
                            + xs[wi * 4 + 2] * T.Cast("float32", wsh[r, wi, 2])
                            + xs[wi * 4 + 3] * T.Cast("float32", wsh[r, wi, 3])
                        ) * scl[r, wi // 8]
                    else:
                        acc[r, l] += (
                            xs[wi * 4 + 0] * T.Cast("float32", Wt[off + blk * BN + r, wi, 0])
                            + xs[wi * 4 + 1] * T.Cast("float32", Wt[off + blk * BN + r, wi, 1])
                            + xs[wi * 4 + 2] * T.Cast("float32", Wt[off + blk * BN + r, wi, 2])
                            + xs[wi * 4 + 3] * T.Cast("float32", Wt[off + blk * BN + r, wi, 3])
                        ) * scl[r, wi // 8]
            T.sync_threads()
            T.reduce_sum(acc, rowv, dim=1)
            T.sync_threads()
            for r in T.Parallel(BN):
                if which == "ob":
                    if resid:
                        OUT[blk * BN + r] = X[blk * BN + r] + rowv[r]
                    else:
                        OUT[blk * BN + r] = rowv[r]
                elif which == "qa":
                    QL[off + blk * BN + r] = rowv[r]
                elif which == "qb":
                    Q[blk * BN + r] = rowv[r]
                else:
                    WOAO[blk * BN + r] = rowv[r]
            if pdl:
                T.pdl_trigger()

    return main, ["WQA", "SQA", "WQB", "SQB", "WOAW", "SOA", "WOBW", "SOB",
                  "XF", "QLF", "AQ", "QL", "Q", "WOAO", "X", "OUT"]


# ------------------------------------------------------------------ attention
def k_mla_split(pdl=True):
    """One CTA per MLA_BK keys: dequantise them, then a flash-style partial.

    The key tile is prefetched before the sync -- its address depends only on the
    block index, so it is independent of the query wq_b is still writing.
    """

    @T.prim_func
    def main(WIN: T.Tensor((NWIN, WIN_ROW_B), "uint8"),
             CMP: T.Tensor((NCMP, CMP_ROW_B), "uint8"),
             Q: T.Tensor((QN,), "float32"),
             PO: T.Tensor((HEADS, NT, HEAD_DIM), "float32"),
             MXA: T.Tensor((NT, HEADS), "float32"),
             SMA: T.Tensor((NT, HEADS), "float32")):
        with T.Kernel(NT, threads=THREADS) as blk:
            xs = T.alloc_shared((H,), "float32")
            kv = T.alloc_shared((MLA_BK, HEAD_DIM), "float32")
            scs = T.alloc_shared((HEADS, MLA_BK), "float32")
            scp = T.alloc_shared((HEADS, MLA_BK, MLA_UNROLL), "float32")
            pp = T.alloc_shared((HEADS, MLA_BK), "float32")
            mb = T.alloc_shared((HEADS,), "float32")
            os_ = T.alloc_shared((HEADS, HEAD_DIM), "float32")
            if blk < NWB:
                for j, d in T.Parallel(MLA_BK, HEAD_DIM):
                    v = T.Cast("float32", T.reinterpret(WIN[blk * MLA_BK + j, d],
                                                       "float8_e4m3fn"))
                    sb = WIN[blk * MLA_BK + j, WIN_DATA_B + d // WIN_GRAN]
                    kv[j, d] = v * T.exp2(T.Cast("float32", sb) - 127.0)
            else:
                for j, d2 in T.Parallel(MLA_BK, HEAD_DIM // 2):
                    byte = CMP[(blk - NWB) * MLA_BK + j, d2]
                    lo = T.Cast("int32", byte & T.Cast("uint8", 0x0F))
                    hi = T.Cast("int32", byte >> T.Cast("uint8", 4))
                    sbyte = CMP[(blk - NWB) * MLA_BK + j,
                                CMP_DATA_B + (2 * d2) // CMP_GRAN]
                    sc2 = T.Cast("float32", T.reinterpret(sbyte, FP8))
                    kv[j, 2 * d2] = _e2m1_val(lo) * sc2
                    kv[j, 2 * d2 + 1] = _e2m1_val(hi) * sc2
            if pdl:
                T.pdl_sync()
            for h, d in T.Parallel(HEADS, HEAD_DIM):
                xs[h * HEAD_DIM + d] = Q[h * HEAD_DIM + d]
            T.sync_threads()
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
            for h in T.Parallel(HEADS):
                mb[h] = -3.0e38
            T.sync_threads()
            for j in T.serial(MLA_BK):
                for h in T.Parallel(HEADS):
                    mb[h] = T.max(mb[h], scs[h, j] * QB_SCALE)
            T.sync_threads()
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
                os_[h, d] = 0.0
            T.sync_threads()
            for j in T.serial(MLA_BK):
                for h, d in T.Parallel(HEADS, HEAD_DIM):
                    os_[h, d] += pp[h, j] * kv[j, d]
            T.sync_threads()
            for h, d in T.Parallel(HEADS, HEAD_DIM):
                PO[h, blk, d] = os_[h, d]
            for h in T.Parallel(HEADS):
                MXA[blk, h] = mb[h]
            if pdl:
                T.pdl_trigger()

    return main, ["WIN", "CMP", "Q", "PO", "MXA", "SMA"]


def k_mla_combine(pdl=True):
    """One CTA per head: merge the NT splits, inverse-RoPE, re-quantise to fp8.

    No prefetch is possible: every input is the split kernel's output.
    """

    @T.prim_func
    def main(PO: T.Tensor((HEADS, NT, HEAD_DIM), "float32"),
             MXA: T.Tensor((NT, HEADS), "float32"),
             SMA: T.Tensor((NT, HEADS), "float32"),
             SINK: T.Tensor((HEADS,), "float32"),
             FREQ: T.Tensor((HALF, 2), "float32"),
             AQ: T.Tensor((QN,), "float32")):
        with T.Kernel(HEADS, threads=THREADS) as blk:
            xs = T.alloc_shared((HEAD_DIM,), "float32")
            wb = T.alloc_shared((NT,), "float32")
            mmax = T.alloc_shared((1,), "float32")
            den = T.alloc_shared((1,), "float32")
            amx = T.alloc_shared((NGQ,), "float32")
            eby = T.alloc_shared((NGQ,), "uint32")
            if pdl:
                T.pdl_sync()
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
            for k in T.Parallel(HALF):
                a = xs[NOPE_DIM + k]
                b = xs[NOPE_DIM + HALF + k]
                xs[NOPE_DIM + k] = a * FREQ[k, 0] + b * FREQ[k, 1]
                xs[NOPE_DIM + HALF + k] = -a * FREQ[k, 1] + b * FREQ[k, 0]
            T.sync_threads()
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
            if pdl:
                T.pdl_trigger()

    return main, ["PO", "MXA", "SMA", "SINK", "FREQ", "AQ"]


# ------------------------------------------------------------------ router
def k_router(pdl=True):
    """logits = GW @ HN: bf16 gate against the f32 normed hidden.

    The gate tile is prefetched before the sync (row index from blockIdx only).
    """
    NROW = NEXP // BN
    S_RT = H // 64

    @T.prim_func
    def main(GW: T.Tensor((NEXP, H), "bfloat16"),
             HN: T.Tensor((H,), "float32"),
             LG: T.Tensor((NEXP,), "float32")):
        with T.Kernel(NROW, threads=THREADS) as blk:
            xs = T.alloc_shared((H,), "float32")
            acc = T.alloc_fragment((BN, 32), "float32")
            rowv = T.alloc_shared((BN,), "float32")
            gsh = T.alloc_shared((BN, H), "bfloat16")
            for r, i in T.Parallel(BN, H):
                gsh[r, i] = GW[blk * BN + r, i]
            if pdl:
                T.pdl_sync()
            for i in T.Parallel(H):
                xs[i] = HN[i]
            T.sync_threads()
            for r, l in T.Parallel(BN, 32):
                acc[r, l] = 0.0
            T.sync_threads()
            for step in T.serial(S_RT):
                for r, l in T.Parallel(BN, 32):
                    k0 = step * 64 + l * 2
                    acc[r, l] += (xs[k0] * T.Cast("float32", gsh[r, k0])
                                  + xs[k0 + 1] * T.Cast("float32", gsh[r, k0 + 1]))
            T.sync_threads()
            T.reduce_sum(acc, rowv, dim=1)
            T.sync_threads()
            for r in T.Parallel(BN):
                LG[blk * BN + r] = rowv[r]
            if pdl:
                T.pdl_trigger()

    return main, ["GW", "HN", "LG"]


def k_topk(pdl=True):
    """sqrtsoftplus, top-6, renormalise. One block: the selection is inherently
    serial over 384 candidates, and nothing here can be prefetched."""

    @T.prim_func
    def main(LG: T.Tensor((NEXP,), "float32"),
             GB: T.Tensor((NEXP,), "float32"),
             EXPID: T.Tensor((TOPK,), "int32"),
             WGT: T.Tensor((NGU,), "float32")):
        with T.Kernel(1, threads=THREADS) as blk:
            lval = T.alloc_shared((NEXP,), "float32")
            selv = T.alloc_shared((NEXP,), "float32")
            lpart = T.alloc_shared((512,), "float32")
            smt = T.alloc_shared((TOPK,), "float32")
            sd = T.alloc_shared((1,), "float32")
            bv = T.alloc_shared((1,), "float32")
            bi = T.alloc_shared((1,), "int32")
            if pdl:
                T.pdl_sync()
            for j in T.Parallel(NEXP):
                lval[j] = T.sqrt(T.max(LG[j], 0.0)
                                 + T.log(1.0 + T.exp(-T.abs(LG[j]))))
            T.sync_threads()
            # ffn.gate.bias steers the choice only; the weight stays the raw score
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
                    bi[0] = NEXP
                T.sync_threads()
                for j in T.Parallel(NEXP):
                    if selv[j] == bv[0]:
                        T.atomic_min(bi[0], T.Cast("int32", j))
                T.sync_threads()
                for i in T.Parallel(1):
                    EXPID[rr] = bi[0]
                    smt[rr] = lval[bi[0]]      # bv is biased; the weight is not
                    sd[0] += lval[bi[0]]
                    lval[bi[0]] = -3.0e38
                    selv[bi[0]] = -3.0e38
                T.sync_threads()
            for i in T.Parallel(1):
                for rr in T.serial(TOPK):
                    WGT[rr] = RSCALE * smt[rr] / sd[0]
                WGT[NGU - 1] = 1.0
            if pdl:
                T.pdl_trigger()

    return main, ["LG", "GB", "EXPID", "WGT"]


# ------------------------------------------------------------------ MoE
def k_moe_gu(pdl=True):
    """gate_up for all 7 slots (6 routed + shared) as one tile sweep.

    The shared slot is **fp8 e4m3**, not a 7th fp4 expert: the checkpoint stores
    `ffn.shared_experts.w1/w3` as F8_E4M3 with 32x32 ue8m0 scales, twice the bytes
    of a routed expert (11.8 MB vs 5.9 MB per full matrix). Reading it out of the
    fp4 table would decode the wrong format and under-count the stage by 2x.

    Which path a block takes is uniform across the block (it follows `tq`), so the
    two weight formats can share one launch.
    """
    C_GU = 10
    S_GU = (H // 8) // 32
    S_GU8 = (H // 4) // 32
    NROW = E_GU // BN
    NT_GU = NGU * NROW

    @T.prim_func
    def main(WGU: T.Tensor((NEXP + NSH, E_GU, H // 8), "uint32"),
             SGU: T.Tensor((NEXP + NSH, E_GU, H // GRAN), "uint8"),
             SHW: T.Tensor((E_GU, H // 4, 4), FP8),
             SHS: T.Tensor((E_GU, H // GRAN), "uint8"),
             HF: T.Tensor((H,), "float32"),
             EXPID: T.Tensor((TOPK,), "int32"),
             GU: T.Tensor((NGU, E_GU), "float32"),
             LUT: T.Tensor((16,), "float32")):
        with T.Kernel(NT_GU, threads=THREADS) as tq:
            xs = T.alloc_shared((H,), "float32")
            scl = T.alloc_shared((BN, H // GRAN), "float32")
            acc = T.alloc_fragment((BN, 32), "float32")
            rowv = T.alloc_shared((BN,), "float32")
            lut = T.alloc_shared((16,), "float32")
            wgu = T.alloc_shared((BN, C_GU * 32), "uint32")
            s_ = tq // NROW
            ti = tq % NROW
            for i in T.Parallel(16):
                lut[i] = LUT[i]
            if pdl:
                T.pdl_sync()
            for i in T.Parallel(H):
                xs[i] = HF[i]
            T.sync_threads()
            for r, l in T.Parallel(BN, 32):
                acc[r, l] = 0.0
            T.sync_threads()
            eid = T.if_then_else(s_ < TOPK, EXPID[T.min(s_, TOPK - 1)], NEXP)
            if s_ < TOPK:
                for r, g in T.Parallel(BN, H // GRAN):
                    scl[r, g] = T.exp2(T.Cast("float32",
                                              SGU[eid, ti * BN + r, g]) - 127.0)
                T.sync_threads()
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
            else:
                # shared expert: fp8, four values per lane per step, one ue8m0
                # scale per 32 values (so every 8th `wi` opens a new group)
                for r, g in T.Parallel(BN, H // GRAN):
                    scl[r, g] = T.exp2(T.Cast("float32",
                                              SHS[ti * BN + r, g]) - 127.0)
                T.sync_threads()
                for step in T.serial(S_GU8):
                    for r, l in T.Parallel(BN, 32):
                        wi = step * 32 + l
                        acc[r, l] += (
                            xs[wi * 4 + 0] * T.Cast("float32", SHW[ti * BN + r, wi, 0])
                            + xs[wi * 4 + 1] * T.Cast("float32", SHW[ti * BN + r, wi, 1])
                            + xs[wi * 4 + 2] * T.Cast("float32", SHW[ti * BN + r, wi, 2])
                            + xs[wi * 4 + 3] * T.Cast("float32", SHW[ti * BN + r, wi, 3])
                        ) * scl[r, wi // 8]
                T.sync_threads()
            T.reduce_sum(acc, rowv, dim=1)
            T.sync_threads()
            for r in T.Parallel(BN):
                GU[s_, ti * BN + r] = rowv[r]
            if pdl:
                T.pdl_trigger()

    return main, ["WGU", "SGU", "SHGW", "SHGS", "HF", "EXPID", "GU", "LUT"]


def k_resid_add(pdl=True):
    """OUT += X, at the head of seg c.

    At TP8 wo_b emits a partial and the all_reduce sums it. Adding X inside wo_b
    would sum X once per rank; adding it here adds it exactly once.
    """

    @T.prim_func
    def main(X: T.Tensor((H,), "float32"), OUT: T.Tensor((H,), "float32")):
        with T.Kernel(H // THREADS, threads=THREADS) as blk:
            if pdl:
                T.pdl_sync()
            for t0 in T.Parallel(THREADS):
                OUT[blk * THREADS + t0] = OUT[blk * THREADS + t0] + X[blk * THREADS + t0]
            if pdl:
                T.pdl_trigger()

    return main, ["X", "OUT"]


def k_swiglu(pdl=True):
    """Clamped SwiGLU, one block per slot, routing weight folded in."""

    @T.prim_func
    def main(GU: T.Tensor((NGU, E_GU), "float32"),
             ME: T.Tensor((NGU, E_K), "float32"),
             WGT: T.Tensor((NGU,), "float32")):
        with T.Kernel(NGU, threads=THREADS) as blk:
            if pdl:
                T.pdl_sync()
            for i in T.Parallel(E_K):
                g = T.Cast("float32", T.min(T.Cast("bfloat16", GU[blk, i]),
                                            T.Cast("bfloat16", SWIGLU_LIMIT)))
                u = T.Cast("float32", T.max(
                    T.min(T.Cast("bfloat16", GU[blk, E_K + i]),
                          T.Cast("bfloat16", SWIGLU_LIMIT)),
                    T.Cast("bfloat16", -SWIGLU_LIMIT)))
                ME[blk, i] = g / (1.0 + T.exp(-g)) * u * WGT[blk]
            if pdl:
                T.pdl_trigger()

    return main, ["GU", "ME", "WGT"]


def k_moe_dn(pdl=True, dst="OUT"):
    """down for all 7 slots, concatenated along K into one accumulator.

    Slots 0..5 are fp4 (one ue8m0 per 32 values, two values per byte); slot 6 is
    the shared expert in **fp8**, whose tile is small enough (BN x 288 = 2.3 KB) to
    prefetch whole before the sync, so it still rides for free.
    """
    W_DN = E_K // 8
    S_DN = (W_DN + 31) // 32
    NROW = H // BN

    @T.prim_func
    def main(WDN: T.Tensor((NEXP + NSH, H, E_K // 8), "uint32"),
             SDN: T.Tensor((NEXP + NSH, H, E_K // GRAN), "uint8"),
             SHW: T.Tensor((H, E_K), FP8),
             SHS: T.Tensor((H, E_K // GRAN), "uint8"),
             ME: T.Tensor((NGU, E_K), "float32"),
             EXPID: T.Tensor((TOPK,), "int32"),
             LUT: T.Tensor((16,), "float32"),
             OUT: T.Tensor((H,), "float32"),
             MO: T.Tensor((H,), "float32")):
        with T.Kernel(NROW, threads=THREADS) as tq:
            xs = T.alloc_shared((NGU * E_K,), "float32")
            scl = T.alloc_shared((BN, NGU * (E_K // GRAN)), "float32")
            acc = T.alloc_fragment((BN, 32), "float32")
            rowv = T.alloc_shared((BN,), "float32")
            lut = T.alloc_shared((16,), "float32")
            wsh = T.alloc_shared((NGU, BN, W_DN), "uint32")
            wsh8 = T.alloc_shared((BN, E_K), FP8)
            deid = T.alloc_shared((NGU,), "int32")
            for i in T.Parallel(16):
                lut[i] = LUT[i]
            # ---- the shared slot is a fixed row, so it can land before the sync
            for r, i8 in T.Parallel(BN, E_K):
                wsh8[r, i8] = SHW[tq * BN + r, i8]
            if pdl:
                T.pdl_sync()
            for s2 in T.Parallel(NGU):
                deid[s2] = T.if_then_else(s2 < TOPK, EXPID[T.min(s2, TOPK - 1)],
                                          NEXP)
            T.sync_threads()
            for s2, i in T.Parallel(NGU, E_K):
                xs[s2 * E_K + i] = ME[s2, i]
            for s2, r, g in T.Parallel(NGU, BN, E_K // GRAN):
                scl[r, s2 * (E_K // GRAN) + g] = T.if_then_else(
                    s2 < TOPK,
                    T.exp2(T.Cast("float32", SDN[deid[s2], tq * BN + r, g]) - 127.0),
                    T.exp2(T.Cast("float32", SHS[tq * BN + r, g]) - 127.0))
            T.sync_threads()
            for r, l in T.Parallel(BN, 32):
                acc[r, l] = 0.0
            # ---- routed experts only; the shared slot is already in wsh8
            for es, r, wi in T.Parallel(NGU - 1, BN, W_DN):
                wsh[es, r, wi] = WDN[deid[es], tq * BN + r, wi]
            T.sync_threads()
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
                    # fp8: 288 = 9 x 32, so every lane covers exactly one value
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
            # At TP8 the MoE output is a *partial* until the all_reduce lands, so it
            # goes to its own buffer: accumulating into OUT here would make the
            # reduce sum the attention residual a second time.
            if dst == "OUT":
                for r in T.Parallel(BN):
                    OUT[tq * BN + r] = OUT[tq * BN + r] + rowv[r]
            else:
                for r in T.Parallel(BN):
                    MO[tq * BN + r] = rowv[r]
            if pdl:
                T.pdl_trigger()

    return main, ["WDN", "SDN", "SHDW", "SHDS", "ME", "EXPID", "LUT", "OUT", "MO"]


# ------------------------------------------------------------------ the chain
def build_chain(pdl=True):
    """Every kernel in dependency order, each with the argument names it takes.

    Builders return (prim_func, arg-names); compile here so `pdl` stays a plain
    Python flag that changes the emitted source rather than a kernel argument.
    """
    specs = [
        k_norm_partial("X", pdl),                          # rmsnorm1 partials
        k_norm_apply(H, "X", "GN", "XF", False, pdl),       # rmsnorm1 apply
        k_gemv("qa", QLR, H, 0, pdl),                      # wq_a
        k_norm_partial("QL", pdl),                         # rmsnormQL partials
        k_norm_apply(QLR, "QL", "QNORM", "QLF", False, pdl),  # rmsnormQL apply
        k_gemv("qb", QN, QLR, 0, pdl),                     # wq_b
        k_mla_split(pdl),                                  # MLA split
        k_mla_combine(pdl),                                # MLA combine
        k_gemv("oa", WOA, QN, 0, pdl),                     # wo_a
        k_gemv("ob", H, WOA, 0, pdl),                      # wo_b  (+ residual)
        k_norm_partial("OUT", pdl),                        # rmsnorm2 partials
        k_norm_apply(H, "OUT", "GN2", "HF", True, pdl),     # rmsnorm2 (+ HN)
        k_router(pdl),                                     # router logits
        k_topk(pdl),                                       # top-6 + weights
        k_moe_gu(pdl),                                     # gate_up x7
        k_swiglu(pdl),                                     # swiglu x7
        k_moe_dn(pdl),                                     # down x7
    ]
    return [(tilelang.compile(_named(main, f"pdl_{i}")), n)
            for i, (main, n) in enumerate(specs)]


def _named(main, name):
    """Give the prim_func a unique global symbol.

    Every builder's inner function is called `main`, and tilelang.compile caches by
    kernel name -- without this all 17 stages collide on "main" and the first
    compiled kernel is handed back for every stage.
    """
    return main.with_attr("global_symbol", name)


def call_chain(chain, args):
    for main, names in chain:
        main(*[args[n] for n in names])


# ------------------------------------------------------------------ driver
if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "check"
    print(f"PDL chain: {len(build_chain(True))} kernels, THREADS={THREADS}, BN={BN}")

    if what == "check":
        chain = build_chain(True)
        t, args, ref_in = build()
        call_chain(chain, args)
        torch.cuda.synchronize()
        rql, rq, raq, rout, rtopi = ref_layer(t, *ref_in)
        e_ql = (args["QL"] - rql).abs().max().item() / rql.abs().max().item()
        e_aq = (args["AQ"] - raq).abs().max().item() / raq.abs().max().item()
        e_q = (args["OUT"] - rout).abs().max().item() / rout.abs().max().item()
        print(f"  q_lora           rel={e_ql:.3e}")
        print(f"  attn out (quant) rel={e_aq:.3e}")
        print(f"  layer out        rel={e_q:.3e}  "
              f"{'MATCH' if e_q < 5e-3 else 'MISMATCH'}")
        ktopi = torch.sort(args["EXPID"].long())[0].tolist()
        rtopi = torch.sort(rtopi.long())[0].tolist()
        same = sum(a == b for a, b in zip(ktopi, rtopi))
        print(f"  routed experts   {same}/{TOPK} identical "
              f"(kernel {ktopi} ref {rtopi})")
        sys.exit(0)

    if what == "prof":
        chain = build_chain(True)
        _, args, _ = build()
        for _ in range(10):
            call_chain(chain, args)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            call_chain(chain, args)
        torch.cuda.synchronize()
        from torch.profiler import profile, ProfilerActivity
        N = 200
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(N):
                g.replay()
            torch.cuda.synchronize()
        agg = {}
        for e in prof.key_averages():
            if e.device_time_total > 0 and ("pdl_" in e.key or "kernel" in e.key):
                agg[e.key] = e.device_time_total / N
        names = ["norm1 partial", "norm1 apply", "wq_a", "normQL partial",
                 "normQL apply", "wq_b", "mla split", "mla combine", "wo_a",
                 "wo_b", "norm2 partial", "norm2 apply", "router", "topk",
                 "moe gate_up", "swiglu", "moe down"]
        tot = 0.0
        for i, nm in enumerate(names):
            hit = [v for k, v in agg.items() if f"pdl_{i}_" in k]
            us = sum(hit)
            tot += us
            print(f"    {nm:16s} {us:7.2f} us")
        print(f"    {'sum':16s} {tot:7.2f} us")
        sys.exit(0)

    if what == "bench":
        # One layer, replayed: at B=1 the whole question is per-layer latency.
        for label, pdl in (("no-PDL", False), ("PDL", True)):
            chain = build_chain(pdl)
            _, args, _ = build()
            for _ in range(10):
                call_chain(chain, args)
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                call_chain(chain, args)
            torch.cuda.synchronize()
            best = float("inf")
            for _ in range(12):
                t0 = time.perf_counter()
                for _ in range(50):
                    g.replay()
                torch.cuda.synchronize()
                best = min(best, (time.perf_counter() - t0) / 50 * 1e6)
            print(f"  {label:8s} layer {best:7.2f} us  "
                  f"-> {1e6/(best*40):6.1f} tok/s (x40 layers)")
        sys.exit(0)
