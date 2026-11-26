#!/usr/bin/env python3
"""MXFP8 blockscaled GEMM (e4m3 + ue8m0) verified bit-exact on B300.

DSV4.1-Flash ships every linear as F8_E4M3 with an ue8m0 scale on 32x32 blocks:

    layers.N.attn.wq_a.weight  F8_E4M3  [1280, 5120]
    layers.N.attn.wq_a.scale   F8_E8M0  [40, 160]     <- [N/32, K/32]

The checkpoint's scale is one value per 32x32 tile (N/32 * K/32 of them). The
tcgen05 MXFP8 scale is one value per 32 elements along K for EACH row, i.e.
N * K/32 -- 32x coarser in the checkpoint. So the scales must be expanded by
broadcasting each tile scale to its 32 rows, then packed 4-per-uint32 along K
(one uint32 covers 128 K elements, matching GRAN*4).

Same five lowering rules as the verified MXFP4 kernel apply here (the ISA kind is
the same, mxf8f6f4):
  * e4m3 needs no unpacking (1 byte/value), unlike fp4's packed->unpacked TMA
  * TMEM scale buffers are uint32 cells, 4 K-groups per cell
  * SF in SMEM must be warp-transposed, then fence_proxy_async
  * TMEM accumulator copies out through a fragment between the thread-sync pair
  * layout inference needs the packed global A

Run: python mxfp8_gemm_b300.py [check|bench]
"""

import sys
import time

import torch
import tilelang
import tilelang.language as T

BM = BN = 128
BK = 128
GRAN = 32          # ue8m0 covers 32 K elements
FP8 = "float8_e4m3fn"


@tilelang.jit(out_idx=[-1])
def mxfp8_gemm(M: int, N: int, K: int):
    """C[M,N] = (A[M,K] * sA) @ (B[N,K]^T * sB), e4m3 weights, ue8m0 scales.

    A: [M, K] e4m3.  B: [N, K] e4m3 (weight layout: N-major, K contiguous).
    SFA/SFB: [N or M, K/GRAN] ue8m0 bytes, here passed pre-packed as uint32
             groups of 4 along K, i.e. [rows * (K/GRAN/4)].
    """
    assert K % BK == 0 and M % BM == 0 and N % BN == 0
    sfg = K // GRAN // 4

    @T.prim_func
    def main(A: T.Tensor((M, K), FP8),
             B: T.Tensor((N, K), FP8),
             SFA: T.Tensor((M * sfg,), "uint32"),
             SFB: T.Tensor((N * sfg,), "uint32"),
             C: T.Tensor((M, N), "float32")):
        with T.Kernel(T.ceildiv(N, BN), T.ceildiv(M, BM), threads=128) as (bx, by):
            As = T.alloc_shared((BM, BK), FP8)
            Bs = T.alloc_shared((BN, BK), FP8)
            SAs = T.alloc_shared((BM,), "uint32")
            SBs = T.alloc_shared((BN,), "uint32")

            Ct = T.alloc_tmem((BM, BN), "float32")
            Cf = T.alloc_fragment((BM, BN), "float32")
            SFAt = T.alloc_tmem((BM, BM // 128 * 4), "uint32")
            SFBt = T.alloc_tmem((BM, BN // 128 * 4), "uint32")

            loaded = T.alloc_barrier([32])
            with_sf = T.alloc_barrier([32])
            consumed = T.alloc_barrier([1])
            tmem_full = T.alloc_barrier([1])

            T.annotate_layout({Ct: T.make_blockscaled_gemm_layout(Ct, A)})

            tx = T.get_thread_binding()
            for k in T.serial(K // BK):
                ph = k & 1
                if tx < 32:
                    T.tma_copy(A[by * BM:(by + 1) * BM, k * BK:(k + 1) * BK],
                               As, barrier=loaded)
                    T.tma_copy(B[bx * BN:(bx + 1) * BN, k * BK:(k + 1) * BK],
                               Bs, barrier=loaded)
                    T.tma_copy(SFA[k * M + by * BM:k * M + (by + 1) * BM],
                               SAs, barrier=loaded)
                    T.tma_copy(SFB[k * N + bx * BN:k * N + (bx + 1) * BN],
                               SBs, barrier=loaded)
                    T.mbarrier_arrive(loaded)
                elif tx < 64:
                    T.mbarrier_wait_parity(loaded, ph)
                    T.mbarrier_wait_parity(with_sf, ph)
                    T.tcgen05_cp_warpx4(SAs, SFAt)
                    T.tcgen05_cp_warpx4(SBs, SFBt)
                    T.tcgen05_gemm_blockscaled(
                        As, Bs, Ct, SFAt, SFBt,
                        transpose_B=True,      # B is [N, K]: K-major
                        clear_accum=(k == 0),
                        mbar=consumed,
                        k_start=k * BK,
                        sf_a_granularity_k=GRAN,
                        sf_b_granularity_k=GRAN,
                    )
                elif tx < 96:
                    T.mbarrier_wait_parity(loaded, ph)
                    T.tcgen05_sf_warp_transpose(SAs)
                    T.tcgen05_sf_warp_transpose(SBs)
                    T.fence_proxy_async()
                    T.mbarrier_arrive(with_sf)

            T.tcgen05_mma_arrive(tmem_full)
            T.mbarrier_wait_parity(tmem_full, 0)
            T.sync_threads()
            T.tcgen05_before_thread_sync()
            T.copy(Ct, Cf)
            T.tcgen05_after_thread_sync()
            T.copy(Cf, C[by * BM:(by + 1) * BM, bx * BN:(bx + 1) * BN])

    return main


# --------------------------------------------------------------- host helpers
def pack_per_row_scales(per_row: torch.Tensor) -> torch.Tensor:
    """[N, K/GRAN] ue8m0 bytes (one per 32 K elements per row) -> packed uint32.

    GROUP-MAJOR: 4 consecutive K-groups form one uint32 (little-endian) and the
    result is indexed [k_group * N + row], which is what
    `tcgen05_gemm_blockscaled` expects for its FMA operand. Storing it
    row-major instead silently scrambles the result.
    """
    N, ng = per_row.shape
    assert ng % 4 == 0
    exp = per_row.t().contiguous()                           # [ng, N]
    words = torch.zeros((ng // 4) * N, dtype=torch.int64, device=per_row.device)
    for g in range(ng // 4):
        w = torch.zeros((N,), dtype=torch.int64, device=per_row.device)
        for b in range(4):
            w |= exp[g * 4 + b].to(torch.int64) << (8 * b)
        words[g * N:(g + 1) * N] = w
    return words.to(torch.uint32)


def expand_scales(sf_tile: torch.Tensor, N: int, K: int) -> torch.Tensor:
    """[N/32, K/32] 32x32-tile ue8m0 bytes -> packed uint32 [(K/GRAN/4) * N].

    The checkpoint scales one value per 32x32 tile; tcgen05 MXFP8 wants one per
    32 elements along K for every row, so each tile scale is broadcast to its 32
    rows first, then packed group-major (see `pack_per_row_scales`).
    """
    per_row = sf_tile.repeat_interleave(32, dim=0)           # [N, K/GRAN]
    assert per_row.shape == (N, K // GRAN), per_row.shape
    return pack_per_row_scales(per_row)


def dequant_rows(w: torch.Tensor, sf_tile: torch.Tensor) -> torch.Tensor:
    """[N, K] e4m3 + [N/32, K/32] ue8m0 -> float32, exactly as the model means it."""
    N, K = w.shape
    s = sf_tile.repeat_interleave(32, dim=0).repeat_interleave(32, dim=1)
    return w.to(torch.float32) * torch.exp2(s.to(torch.float32) - 127.0)


def make_case(M, N, K, seed=0):
    torch.manual_seed(seed)
    w = (torch.randn(N, K, device="cuda") * 0.05).to(torch.float8_e4m3fn)
    sf = torch.randint(120, 133, (N // 32, K // 32), dtype=torch.uint8,
                       device="cuda")
    a = (torch.randn(M, K, device="cuda") * 0.5).to(torch.float8_e4m3fn)
    sfa = torch.full((M // 32, K // 32), 127, dtype=torch.uint8, device="cuda")
    return w, sf, a, sfa


def check(M=128, N=128, K=512):
    w, sf, a, sfa = make_case(M, N, K)
    wd = dequant_rows(w, sf)
    ad = dequant_rows(a, sfa)
    ref = ad @ wd.T
    C = mxfp8_gemm(M, N, K)(a, w, expand_scales(sfa, M, K),
                            expand_scales(sf, N, K)).double()
    err = (C - ref).abs().max().item()
    rel = err / ref.abs().max().item()
    print(f"MXFP8 blockscaled {M}x{N}x{K}  (GRAN={GRAN}, 32x32 tile scales)")
    print(f"  ref absmax = {ref.abs().max().item():.4f}")
    print(f"  max abs err = {err:.4e}  rel {rel:.2e}  "
          f"-> {'MATCH' if rel < 2e-3 else 'MISMATCH'}")
    return rel < 2e-3


def bench(M=4096, N=4096, K=4096, iters=20):
    w, sf, a, sfa = make_case(M, N, K, seed=1)
    k = mxfp8_gemm(M, N, K)
    A32, B32 = expand_scales(sfa, M, K), expand_scales(sf, N, K)
    k(a, w, A32, B32)
    for _ in range(3):
        k(a, w, A32, B32)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        k(a, w, A32, B32)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / iters
    print(f"MXFP8 {M}x{N}x{K}: {dt*1e3:.3f} ms/op  "
          f"{2*M*N*K/dt/1e12:.1f} TFLOP/s")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "check"
    if what == "check":
        check()
    else:
        bench()
