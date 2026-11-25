#!/usr/bin/env python3
"""Pipelined (multi-stage) MXFP4 blockscaled GEMM on Blackwell B300.

Same verified recipe as `mxfp4_blockscaled_b300.py`, plus a num_stages ring
buffer so TMA loads overlap with UTCCP + tcgen05.mma.

Warp roles (128 threads, one CTA per 128x128 output tile):
  warp 0  : TMA producer (also performs the fp4 packed -> unpacked expansion)
  warp 1  : UTCCP of scale factors + tcgen05.mma.block_scale issue
  warp 2  : SMEM scale-factor warp transpose + async-proxy fence
  all     : epilogue (TMEM -> fragment -> global)

Usage: python mxfp4_blockscaled_pipelined.py [stages]
"""

import sys
import time

import torch
import tilelang
import tilelang.language as T

from mxfp4_blockscaled_b300 import (BM, BN, BK, GRAN, FP4, FP4_SMEM,
                                    quantize_ref)


@tilelang.jit(out_idx=[-1])
def blockscaled_gemm_pipelined(M: int, N: int, K: int, ns: int = 2,
                              dtype: str = "float32"):
    assert M % BM == 0 and N % BN == 0 and K % BK == 0
    assert ns >= 1
    ki = K // BK
    sfg = K // GRAN // 4

    @T.prim_func
    def main(
        A: T.Tensor((M, K), FP4),
        B: T.Tensor((K, N), FP4),
        SFA: T.Tensor((sfg * M,), "uint32"),
        SFB: T.Tensor((sfg * N,), "uint32"),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, BN), T.ceildiv(M, BM), threads=128) as (bx, by):
            As = T.alloc_shared((ns, BM, BK), FP4_SMEM)
            Bs = T.alloc_shared((ns, BK, BN), FP4_SMEM)
            SAs = T.alloc_shared((ns, BM), "uint32")
            SBs = T.alloc_shared((ns, BN), "uint32")

            Ct = T.alloc_tmem((BM, BN), "float32")
            Cf = T.alloc_fragment((BM, BN), "float32")
            SFAt = T.alloc_tmem((BM, BM // 128 * 4), "uint32")
            SFBt = T.alloc_tmem((BM, BN // 128 * 4), "uint32")

            loaded = T.alloc_barrier([32] * ns)
            with_sf = T.alloc_barrier([32] * ns)
            consumed = T.alloc_barrier([1] * ns)
            tmem_full = T.alloc_barrier([1])

            T.annotate_layout({Ct: T.make_blockscaled_gemm_layout(Ct, A)})

            tx = T.get_thread_binding()
            for k in T.serial(ki):
                s = k % ns
                ph = (k // ns) & 1

                if tx < 32:
                    # Reuse stage s only after the MMA that consumed it retired.
                    T.mbarrier_wait_parity(consumed[s], ph ^ 1)
                    T.tma_copy(
                        A[by * BM:(by + 1) * BM, k * BK:(k + 1) * BK],
                        As[s, :, :], barrier=loaded[s])
                    T.tma_copy(
                        B[k * BK:(k + 1) * BK, bx * BN:(bx + 1) * BN],
                        Bs[s, :, :], barrier=loaded[s])
                    T.tma_copy(SFA[k * M + by * BM:k * M + (by + 1) * BM],
                               SAs[s, :], barrier=loaded[s])
                    T.tma_copy(SFB[k * N + bx * BN:k * N + (bx + 1) * BN],
                               SBs[s, :], barrier=loaded[s])
                    T.mbarrier_arrive(loaded[s])

                elif tx < 64:
                    T.mbarrier_wait_parity(loaded[s], ph)
                    T.mbarrier_wait_parity(with_sf[s], ph)
                    T.tcgen05_cp_warpx4(SAs[s, :], SFAt)
                    T.tcgen05_cp_warpx4(SBs[s, :], SFBt)
                    T.tcgen05_gemm_blockscaled(
                        As[s, :, :], Bs[s, :, :], Ct, SFAt, SFBt,
                        transpose_B=False,
                        clear_accum=(k == 0),
                        mbar=consumed[s],
                        k_start=k * BK,
                        sf_a_granularity_k=GRAN,
                        sf_b_granularity_k=GRAN,
                    )

                elif tx < 96:
                    T.mbarrier_wait_parity(loaded[s], ph)
                    T.tcgen05_sf_warp_transpose(SAs[s, :])
                    T.tcgen05_sf_warp_transpose(SBs[s, :])
                    T.fence_proxy_async()
                    T.mbarrier_arrive(with_sf[s])

            T.tcgen05_mma_arrive(tmem_full)
            T.mbarrier_wait_parity(tmem_full, 0)
            T.sync_threads()

            T.tcgen05_before_thread_sync()
            T.copy(Ct, Cf)
            T.tcgen05_after_thread_sync()
            T.copy(Cf, C[by * BM:(by + 1) * BM, bx * BN:(bx + 1) * BN])

    return main


def run(M=4096, N=4096, K=4096, ns=2, iters=20):
    A, B, SFA, SFB, ref = quantize_ref(M, N, K, seed=1, ref_dtype="float32")
    kern = blockscaled_gemm_pipelined(M, N, K, ns)
    out = kern(A, B, SFA, SFB)
    err = (out.double() - ref).abs().max().item()

    for _ in range(3):
        kern(A, B, SFA, SFB)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        kern(A, B, SFA, SFB)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / iters

    tf = 2 * M * N * K / dt / 1e12
    print(f"stages={ns}  {M}x{N}x{K}  err={err:.5f}  "
          f"{dt * 1e3:.3f} ms/op  {tf:.1f} TFLOP/s")
    return err, tf


if __name__ == "__main__":
    stages = [int(x) for x in sys.argv[1:]] or [2]
    for s in stages:
        run(ns=s)
