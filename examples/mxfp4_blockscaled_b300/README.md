# MXFP4 block-scaled GEMM on Blackwell (B300 / SM103a)

Standalone TileLang implementations of `tcgen05.mma.kind::mxf8f6f4.block_scale`
for MXFP4 operands (`float4_e2m1fn` payloads with one `e8m0` scale per 32 K
elements), verified bit-exact on a B300.

This directory is a **reference kernel study**, not part of the TileRT runtime.
TileRT itself ships precompiled CUDA backends (no kernel source in the tree), so
these files exist to document the exact instruction sequence and the pitfalls
required to make block-scaled FP4 MMA produce correct numbers on SM103a.

## Files

| File | Purpose |
| --- | --- |
| `mxfp4_blockscaled_b300.py` | Single-stage kernel, bit-exact correctness harness, throughput mode. |
| `mxfp4_blockscaled_pipelined.py` | Multi-stage (`num_stages`) ring-buffer variant. |

```bash
python mxfp4_blockscaled_b300.py check    # bit-exact correctness (128x128x512)
python mxfp4_blockscaled_b300.py bench    # 4096^3 throughput
python mxfp4_blockscaled_pipelined.py 2 3 # pipelined variants, 2 and 3 stages
```

## Measured on B300 (SM 10.3a)

Correctness uses random 4-bit payloads with **non-uniform** `e8m0` scales
varying per 32-element K group across 2^-3..2^1, compared against a torch
reference — so it cannot pass by accident on degenerate inputs.

```
MXFP4 blockscaled 128x128x512  (GRAN=32)
  ref absmax  = 1191.7
  max abs err = 0.0000      -> BIT-EXACT MATCH
```

Throughput, 4096x4096x4096, 128x128x128 CTA tile, error stays 0.00000:

| Variant | ms/op | TFLOP/s |
| --- | --- | --- |
| single-stage (warp-specialized) | 0.213 | 646 |
| pipelined, 1 stage | 0.604 | 228 |
| pipelined, 2 stages | 0.246 | 560 |
| pipelined, 3 stages | 0.244 | 563 |
| pipelined, 4 stages | 0.247 | 557 |

Two honest notes:

- Multi-stage pipelining does **not** beat the single-stage warp-specialized
  version here. Going from 1 to 2 stages recovers most of the loss (the
  single-buffered variant stalls on the `consumed` barrier every iteration), but
  beyond 2 stages it plateaus and slightly regresses from extra SMEM pressure.
- 646 TFLOP/s is far below the FP4 peak of the part. The `128x128` output tile
  with three warps and one CTA per tile is the limiter; closing the gap needs
  larger tiles, 2-CTA UMMA, and a persistent grid — not more stages.

## The five things that must be right

Each of these was a silent-wrong-answer bug: the kernel compiled, ran, and
produced plausible-magnitude garbage.

1. **Operands must be in unpacked shared memory.**
   A/B shared buffers are `float4_e2m1_unpacked`; only the global tensors are
   packed (`float4_e2m1fn`, two values per byte). Packed SMEM compiles and runs
   but the MMA reads wrong data (measured error 1419.6 vs a 994 absmax
   reference). The expansion happens inside the TMA load, and **only TMA can do
   it** — a SIMT `T.copy` from packed global to unpacked shared is rejected at
   compile time with an explicit message.

2. **Scale factors in TMEM are `uint32` cells, not `uint8`.**
   ```python
   SFA_tmem = T.alloc_tmem([block_M, block_M // 128 * 4], "uint32")
   SFB_tmem = T.alloc_tmem([block_M, block_N // 128 * 4], "uint32")
   ```
   One `uint32` packs four consecutive K-groups of `e8m0` bytes (group-major),
   matching what `tcgen05.cp.32x128b.warpx4` expects.

3. **Scale factors must be warp-transposed in SMEM before UTCCP.**
   ```python
   T.tcgen05_sf_warp_transpose(SFA_shared)   # 4x32 words -> 32x4 words
   T.fence_proxy_async()                     # SMEM written via the async proxy
   ```
   Skipping the transpose yields right-magnitude, wrong-value results. The fence
   is required because the MMA warp reads that SMEM through the async proxy.

4. **The epilogue cannot copy TMEM straight to global.**
   Route through a fragment and bracket with the thread-sync pair:
   ```python
   T.tcgen05_before_thread_sync()
   T.copy(C_tmem, C_frag)
   T.tcgen05_after_thread_sync()
   T.copy(C_frag, C)
   ```
   A direct TMEM->global copy returns uninitialized tensor memory. This bug is
   not FP4-specific: it reproduced identically on a plain bf16 `tcgen05_gemm`
   baseline, which is how it was isolated.

5. **`make_blockscaled_gemm_layout` must be given the packed global operand.**
   ```python
   T.annotate_layout({C_tmem: T.make_blockscaled_gemm_layout(C_tmem, A)})
   ```
   Passing the unpacked SMEM buffer fails with
   `Unsupported TCGEN5MMA configuration: ... A dtype=custom[float4_e2m1_unpacked]`
   because the MMA shape/dtype config cannot be inferred from it.

## Debugging note that saved time

The FP4-specific bugs were isolated by swapping the operand dtype to `float8_e4m3fn`
while keeping the scale-factor path, warp roles, and epilogue byte-for-byte
identical. That variant matched the torch reference exactly under all four
A/B major-layout combinations, which proved the synchronization and SF plumbing
were correct and confined the fault to the FP4 packed/unpacked data path.

## Environment

- TileLang 0.1.12, `sm_103a`
- torch with CUDA 13, one B300 (148 SMs)
