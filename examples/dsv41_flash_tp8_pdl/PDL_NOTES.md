# DSV4.1-Flash B=1 TP8 kernels (TileLang, B300 / SM103a)

Hand-written decode kernels for `deepseek-v4.1-flash-1017`, specialised for
**B=1** and **TP8**. Everything here is verified against an exact torch
reference; the fp8/fp4 payloads and ue8m0 scale bytes are compared bitwise, not
just by tolerance.

## Why B=1 needs a different kernel set

At B=1 every linear is a GEMV. M=1 occupies 1 of 128 MMA lanes, so the tensor
cores are not the tool; the problem is purely **HBM bandwidth**, and each weight
byte is read exactly once. That single fact drives all the design choices below:

- weights stream global -> registers, with **no shared-memory round trip** for
  the body (a staging buffer would be a pure copy);
- one scalar accumulator per `(row, lane)` in a **fragment**, never a shared
  buffer indexed by a serial loop variable;
- **4-byte words per lane per step**, so a warp covers 128 contiguous bytes.
  The first version used 1 byte per lane and reached only 407 GB/s; the 4-byte
  version reached 1491 GB/s on the same shape.

## Files

| file | contents | status |
| --- | --- | --- |
| `mxfp8_gemv_b1.py` | attention projections `wq_a wq_b wo_a wo_b` | verified, 950 GB/s aggregate |
| `mxfp4_gemv_b1.py` | MoE `gate_up` / `down`, full-expert and TP8-shard shapes | verified, 1142 GB/s peak |
| `fused_ew_b300.py` | `rmsnorm+quant`, `swiglu_clamp+quant`, `rope_partial` | **bit-exact** |
| `dsv41_mla_flash.py` | MLA decode, split-KV + online softmax | verified, 31.6 us |
| `mxfp8_gemm_b300.py` | MXFP8 `tcgen05` blockscaled GEMM (prefill path) | verified, rel 2.2e-7 |
| `hetero_megakernel.py` | SIMT reduce -> `sync_grid` -> `tcgen05` MMA in one kernel | verified, barrier 0.02 us |
| `dsv41_layer_megakernel.py` | whole layer: dense + MLA + router + 7-slot MoE, one cooperative launch | verified, rel 1.4e-7 |
| `dsv41_e2e_tp8.py` | the same layer cut at the 3 TP8 collectives, 8 ranks, 40 layers | verified PASS, timing pending |

## Measured, TP8 per rank, B=1

Attention projections (`mxfp8_gemv_b1.py`):

```
wq_a     160x5120   0.82 MB   2.72 us    301 GB/s
wq_b    4096x1280   5.24 MB   3.52 us   1491 GB/s
wo_a    1024x4096   4.19 MB   4.97 us    844 GB/s
wo_b    5120x1024   5.24 MB   5.10 us   1027 GB/s
sum                          16.31 us    950 GB/s
```

MoE (`mxfp4_gemv_b1.py`), per rank at TP8 (intermediate 2304 -> 288):

```
tp8_gate_up   576x5120   2.95 MB   4.12 us    358 GB/s
tp8_down     5120x 288   0.74 MB   2.43 us    304 GB/s
```

Elementwise + quant, fused (`fused_ew_b300.py`), K=5120:

```
rmsnorm+quant   3.29 us      swiglu+quant   5.51 us
```

## Projected layer budget (TP8, no EP, B=1)

```
rmsnorm+quant             3.3
wq_a + wq_b               6.2
MLA decode               31.6
wo_a + wo_b              10.1
all-reduce (hidden)      ~2.5
rmsnorm+quant             3.3
MoE 6 experts (2 GEMV)   39.3   <- 6 launches x ~2-4 us, launch bound
swiglu+quant x6         ~12.0
all-reduce               ~2.5
------------------------------
per layer              ~111 us   ->  ~4.4 ms/token  ->  ~230 tok/s
```

Treat that as a projection from per-kernel measurements, not a measured
end-to-end number.

## Where the remaining time is

1. **MoE expert launches (39 us of 111).** Each TP8 shard is only 0.7-3 MB, so
   the six per-expert `gate_up`/`down` pairs run at 300-360 GB/s — pure launch
   and parallelism overhead. Fusing all six routed experts into one launch with
   `grid = 6 * (N/BN)` and an `expert_ids[6]` indirection should collapse this
   to ~10 us. **This is the single highest-value change** (~230 -> ~330 tok/s).
2. **MLA decode at 31.6 us** is next; it is already the best of the set relative
   to its own roofline.
3. `swiglu+quant` at 5.5 us for 5120 elements is one CTA on a 148-SM part.

## Correctness notes worth keeping

- `rms_norm_eps` is **1e-20** in `config.json`. A 1e-6 default is wrong here.
- ue8m0 rounding is **ceil to a power of two** (`exp + (mant != 0)`), from
  sglang's `fp8_utils.cuh::n()`. Round-to-nearest silently saturates the largest
  element in roughly half the groups.
- `swiglu_limit` = 10.0 clamps **in bf16 before widening**: `gate = min(gate,
  limit)`, `up = clamp(up, -limit, limit)`, then `silu(gate) * up`.
- e2m1 decode: mask the sign off before shifting (`nib & 7`), and exponent 0 is
  the subnormal `0.5 * m`, not `2^(e-1) * (1 + m/2)`. Both bugs were live here;
  a shared 16-entry LUT avoids them and was **4.6x faster** than the arithmetic
  form (47.7 -> 10.3 us on `gate_up`).
- `T.tma_copy` cannot collect the uint8 scale tile — its shape is not a
  TMA-capable `global<->shared` pattern. Scales use plain vectorised copies;
  they are ~1/128 of the bytes.
- Inside a `prim_func`, `range()` is rebound by the AST rewriter and a list
  comprehension over it throws `'ForFrame' object is not iterable`. Write the
  unrolled statements out.

## Running

```bash
P=/opt/dlami/nvme/saboxu/venvs/sglang/bin/python
$P mxfp8_gemv_b1.py check    # attention projections
$P mxfp4_gemv_b1.py check    # MoE
$P fused_ew_b300.py          # elementwise, bit-exact
```

Timings use CUDA graphs. Eager timing measures Python launch overhead and reads
a flat ~13 us for *every* shape, which is misleading at this scale.

## Single-layer megakernel (`dsv41_layer_megakernel.py`)

The whole decoder layer -- dense backbone + MoE, attention excluded -- fused into
ONE cooperative launch and chained with `T.sync_grid()`, so the layer count drops
from ~20 launches to one. Verified stage-by-stage against a torch replica:
`q_lora` rel 1.0e-7, layer output rel 2.4e-7 on the dequantised fp8 payload.

```bash
$P dsv41_layer_megakernel.py check     # numerics vs torch
$P dsv41_layer_megakernel.py bench     # CUDA-graph timing + per-stage breakdown
MEGAK_G=148 $P dsv41_layer_megakernel.py bench   # grid sweep
```

| stage | us | cycles | notes |
|---|---|---|---|
| rmsnorm1 | 3.97 | 8.1 k | split over 20 blocks |
| wq_a | 6.54 | 13.3 k | full-width 1280 rows, ~1000 GB/s |
| rmsnormQL | 3.93 | 8.0 k | split over 5 blocks |
| wq_b | 4.69 | 9.5 k | 1049 GB/s |
| mla split | 10.39 | 21.1 k | 80 split CTAs, fp8 window + fp4 compressed |
| mla combine | 11.10 | 22.6 k | 1 CTA/head, coalesced `PO` |
| wo_a | 5.23 | 10.6 k | 723 GB/s (tile fill 43%) |
| wo_b | 5.42 | 11.0 k | 999 GB/s |
| rmsnorm2 | 4.33 | 8.8 k | also writes the f32 `HN` the router scores |
| router logits | 4.74 | 9.6 k | `384x5120` bf16, tile fill only 16% |
| router topk | 4.60 | 9.3 k | sqrtsoftplus + top-6 + renorm, 1 block |
| moe gate_up | 17.64 | 35.8 k | 7 slots; 10-step staging in smem |
| swiglu | 2.68 | 5.5 k | clamp in bf16 before widening |
| moe down | 11.73 | 23.8 k | 7 slots; weights staged per tile |
| **total** | **96.98** | **197 k** | **1 launch** |

`us` is `cycles / 2.032 GHz`, never `cycles / wall`. Reporting the latter silently
absorbs cooperative-launch queueing and makes `sum(stages)` equal the wall time by
construction -- which it did, until this was fixed.

`x40 layers` (kernel-internal only) = 3.88 ms/token -> **258 tok/s**; the wall-time
figure is contaminated on a shared box (see the timing note below).

### Two latency fixes, and the register budget

Both MoE stages were pure **latency**, not bandwidth: `down` streams 5.16 MB in
13.16 us = 4.9 % of peak, and the model that fits is 14 dependent HBM round trips
per tile (`7 experts x 2 steps`, all accumulating into one register pair, so each
`acc +=` is a read-modify-write and nothing overlaps):

```
14 x ~700cy = 4.8 us/wave  x 2.16 waves = 10.4 us   (measured 13.16)
```

Staging the tile's weights in shared memory breaks the chain: 29.3 -> 23.8 kcyc
for `down`, 43.5 -> 35.7 kcyc for `gate_up` (10-step chunks, `C_GU=10`).

Per-thread registers were the obvious first attempt and **failed loudly**: 14
registers dropped occupancy from 2 blocks/SM to 1 and the cooperative launch of
G=296 died with `CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE`. The whole layer has
about 13 registers of headroom left -- any further change has to spend them
deliberately.

Measurement note: the timestamps are overwritten by every launch, so a single read
is a single sample (+-8 % per stage on a shared GPU, wider than the effects being
measured). The bench now replays 12 times and keeps the per-slot minimum.

`x40 layers` = 2.55 ms/token -> **392 tok/s** excluding MLA, **262 tok/s** with the
separately-measured 31.6 us MLA decode added in. Adjusting `wq_a` for a real TP8
shard (1/8 of the rows, see below) gives ~59.5 us/layer, ~274 tok/s.

### What the fusion actually bought, and what it cost

- **gate_up fusion is the big win.** Six 2.95 MB GEMMs that ran at 358 GB/s each
  when launched separately now sweep as one stage at **1424 GB/s**. The MoE was
  the whole reason to fuse, and it pays.
- **Fusion is not free.** A single threads-per-block for every stage forces BN=8
  everywhere, where the standalone kernels could each pick their own shape
  (BN=32 for fp8). `wq_b`/`wo_b` lose a little, `moe down` loses a lot.
- **The grid is capped at 296 blocks** (2 per SM). Fully unrolled K sweeps cost
  enough registers that 444 blocks fails with
  `CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE`. So every stage runs persistent tile
  loops, and `wo_b`/`down` (640 tiles) need 3 waves with the last only 16% full.
  Occupancy, not bandwidth, is the ceiling: the layer streams 37.6 MB in 63.6 us
  = 591 GB/s aggregate, well under the ~8 TB/s the hardware can do.
- **Barriers are free.** 10 × `T.sync_grid()` is inside the noise (0.02 us each).

### Three measurement-driven rewrites

1. **moe down 21.0 -> 13.1 us.** Reducing the six experts one at a time, each
   with its own AllReduce and syncthreads, ran at 211 GB/s. Concatenating the
   experts along K into a single accumulator turns six bursts of 1.125 warp loads
   into 7 back-to-back loads. It was latency, not bandwidth, the whole time.
2. **rmsnorms 17.5 -> 14.1 us** by splitting each into a partials stage over
   K/256 blocks plus a parallel apply. Less than expected: the double AllReduce
   eats most of the gain, and for QLR=1280 it is a wash.
3. **SwiGLU inlined into the down stage was WORSE** (down+swiglu 20.0 us vs
   13.1 + 2.4). Every down block would re-read all 6x576 gate_up values -- 8.8 MB
   of redundant traffic against only 4.4 MB of down weights. Reverted.

### Traps hit while building this

- `xf = xs * exp2(-(e-127))` is *dividing* by the scale, not dequantising. It must
  be `cast_fp8(xs / sc) * sc`. Caught by a q_lora rel error of 144.
- fp4 and fp8 stages of the same K need different step counts (8 vs 4 values per
  32-bit word). Using the fp8 count for `gate_up` read 2x past the end of `xs`.
- Taking the group absmax of `x*w` and multiplying by `rsqrt` afterwards differs
  from the reference's `(x*rsqrt)*w` by one rounding, which flips the ue8m0 ceil
  exponent on some groups: 1.9e-2 error from a single association change.
- `wg // 36` (non-power-of-2) on a vectorised lane index builds a `Ramp` with a
  non-scalar base and aborts in TVM's simplifier. Walk experts in a serial loop
  and keep the lane index division-free.
- `T.if_then_else(c, PART[t0], 0)` still emits the load, so clamp the index
  (`PART[T.min(t0, NB_MAX-1)]`) or it reads past the buffer.

### Attention IS in here now

MLA decode was merged into the megakernel: phase 1 (split-KV over NT=80 blocks)
and phase 2 (combine, sink logit, inverse rope, dynamic fp8 quant) run as two
stages of the same cooperative launch. Numerics: `attn out` is bit-exact
(`rel=0.000e+00`), whole layer `rel=1.4e-07`.

`PO` -- the per-split partial output -- is laid out `[HEADS, NT, HEAD_DIM]`, not
`[HEADS, HEAD_DIM, NT]`. With `[H, D, NT]` the combine stage read stride-320B
columns; that was 32.6 us standalone-shaped but 12.5 us in the megakernel's
70 KB-of-smem world where L1 has shrunk. Flipping the two axes makes every warp
load contiguous along `d`: combine 32.6 -> 11.0 us.

## TP8: how the MoE is sharded, and why an all-reduce per layer is enough

This is **TP, not EP**: every rank holds all 384 experts, sliced along the
intermediate dimension. No token ever crosses a rank boundary -- there is no
all-to-all dispatch and no combine.

| tensor | full shape | per rank at TP8 | split axis | who sums |
|---|---|---|---|---|
| `gate_up` | `[2*2304, 5120]` | `[576, 5120]` | **N** (rows) | nobody: each rank owns different output rows |
| `down` | `[5120, 2304]` | `[5120, 288]` | **K** (columns) | **all_reduce** of 5120 fp32 |
| `gate` (router) | `[384, 5120]` | `[384, 5120]` | replicated | nobody |
| shared expert | same as routed | same as routed | same as routed | same as routed |

`2304 / 8 = 288`, and `gate_up` concatenates gate and up, hence `2*288 = 576`.

**Why the router needs no communication.** After the attention all-reduce every
rank holds the same hidden vector, so every rank independently computes all 384
logits, the same `sqrtsoftplus` scores, and therefore the same top-6 -- and it
already has those six experts' weights locally. EP has to broadcast expert ids
precisely because a rank only holds a subset; TP does not. That is the structural
reason TP beats EP at B=1 here.

Cost of replication: ~1.70 GB per layer of fp4 expert weights (`gate_up` 1.13 GB
+ `down` 0.57 GB) before scale bytes, i.e. ~68 GB over 40 layers, divided by 8
along the intermediate axis.

The shared expert is `n_shared_experts=1` with the same 2304 intermediate, so it
shards identically and rides the same code path as a 7th slot: weight fixed at
1.0, excluded from `routed_scaling_factor`.

## End to end over 8 GPUs (`dsv41_e2e_tp8.py`)

A cooperative launch cannot contain an NCCL call, so the layer is cut at exactly
the three points the TP algebra forces a sync:

```
seg a   rmsnorm1 + wq_a(160 rows)           -> all_gather(q_lora)      5 KB
seg b   rmsnormQL + wq_b + MLA + wo_a/wo_b  -> all_reduce(attn out)    20 KB
seg c   rmsnorm2 + router + MoE             -> all_reduce(moe out)     20 KB
```

```bash
torchrun --nproc_per_node=8 dsv41_e2e_tp8.py check
torchrun --nproc_per_node=8 dsv41_e2e_tp8.py bench
```

Verified (all three exact, `RESULT PASS`, exit 0):

```
1. wq_a shard 160 rows vs full      : 0.000e+00   (rows outside untouched: 0.0)
2. seg a+b vs one rank, gathered    : 0.000e+00
3. full layer vs torch reference    : 1.404e-07   (same as single GPU)
   routed experts  [52, 94, 120, 280, 343, 346] == ref
```

Two subtleties the check had to be built around, both worth knowing:

- **`wq_a` sharding is bit-exact.** A rank's 160 rows of the sharded product equal
  the same 160 rows of the full 1280-row product, to the bit (0.0).
- **Every synthetic shard is identical**, so a K-parallel linear's 8 partials are
  all equal and the reduce returns 8x one rank's value. That pushes the downstream
  rmsnorm/router to a different operating point (rmsnorm is scale invariant,
  `X + 8*attn` is not), and a 1-ulp reduce difference is enough to flip a ue8m0
  ceiling -- which showed up as a 1.1e-2 "error" that was entirely an artefact of
  the test, not the kernel. Single-GPU split-vs-fused is **bit-exact (0.0)**. The
  check therefore zeroes the K-parallel weights on ranks>0, making the reduction
  return exactly rank 0's shard.

### Timing: not yet usable

The e2e numbers on this box are dominated by contention, not by the kernels. With
another tenant saturating all 8 GPUs, successive runs of the same binary gave
`collectives only = 218 us` and then `19.5 ms` per layer for the same three
collectives -- NCCL's busy-poll gets starved. For reference the kernel-only loop
measured 180-270 us/layer, which is the 3 cooperative launches plus the ~100 us of
kernel-internal work the cycle counters report. **A clean-box rerun is required
before quoting a tok/s figure.**

## Cross-GPU: what TileLang gives you, and what it does not

Short version: the language has **no cross-GPU primitives**, but it is also not
limited to NCCL -- a kernel can reach peer memory directly, which makes a
hand-written in-kernel all-reduce possible. Measured, not assumed.

### The language surface is single-GPU only

| API | scope | crosses GPUs? |
|---|---|---|
| `T.sync_grid()` | every block of one launch | no |
| `cluster_arrive/wait/sync`, `mapa` | SMs inside one thread-block cluster (DSMEM) | no, same GPU |
| `barrier_arrive/wait` (mbarrier) | one block | no |

TVM does ship an NVSHMEM runtime (`3rdparty/tvm/src/runtime/contrib/nvshmem/`:
`init.cc`, `memory_allocator.cc`, `kv_transfer.cu`), but **it is not compiled into
this build** and `libnvshmem` is not installed:

```
tvm.get_global_func('runtime.disco.nvshmem.init_nvshmem', allow_missing=True) -> None
(all nvshmem entry points missing)
libnvshmem.so.3 -> not found
```

Even if it were, it lives at the **disco runtime** layer (host-side symmetric
allocation, uid exchange), not as a device-side callable API inside a kernel.

### Device-side P2P does work

Verified on this box: a TileLang kernel launched on GPU0 writing GPU1's memory.

```python
HELPER = r'''
__device__ __forceinline__ void tl_pst(long long addr, int i, float v) {
  ((float*)addr)[i] = v;
}
'''

@T.prim_func
def main(SRC: T.Tensor((N,), "float32"), PADDR: T.int64,
         PEER: T.Tensor((N,), "float32")):
    with T.Kernel(1, threads=128) as blk:
        T.import_source(HELPER)
        for i in T.Parallel(N):
            T.call_extern("void", "tl_pst", PADDR, i, SRC[i] * 2.0)
```

```
enablePeerAccess 0->1: rc=0
GPU1 memory after a kernel launched on GPU0: [0.0, 2.0, 4.0, 6.0, 8.0, 10.0]
P2P DEVICE-SIDE WRITE: WORKS
```

**The one trap**: you cannot pass the peer tensor as a tensor argument. TileLang's
packed ABI validates the device id and refuses:

```
RuntimeError: kernel main input PEER device_id violates packed ABI constraint;
expected: 1, got: 0
```

Pass the peer **address as an `int64` scalar** instead, and do the access from a
`T.import_source` helper via `T.call_extern`. That is a normal TileLang path, not
a hack.

### What that unlocks, and what it costs

```
write this rank's partial to the 7 peer addresses -> spin on peer flags -> read back
```

The spin is **safe here specifically because the launch is cooperative**: all 296
blocks are resident, so no block can be starved waiting for another. That is the
precondition a hand-rolled cross-GPU barrier needs, and this kernel already has it.

The cost is that the cross-GPU barrier has to be built by hand (`volatile` +
`__threadfence_system()` + atomic release/acquire). That is the most deadlock-prone
code in this whole design, and NCCL's version is battle-tested with topology
awareness and NVLS/shuffle on top.

If this is pursued, the three-segment split becomes unnecessary: it exists **only**
because a cooperative launch cannot contain an NCCL call. With in-kernel P2P the
layer goes back to one launch and all three collective latencies disappear -- which
at B=1, where the payload is 5-20 KB and everything is latency, is the whole cost.

### Hardware all-reduce is reachable without NVSHMEM

`libnvshmem` is absent, but the CUDA **driver's** multicast API is present -- the
layer NVSHMEM itself sits on:

```
FOUND   cuMulticastCreate
FOUND   cuMulticastAddDevice
FOUND   cuMulticastBindMem
FOUND   cuMulticastGetGranularity
```

So Blackwell's `multimem.red.*` (true hardware aggregation) does not require the
NVSHMEM library: create the multicast object through the driver API, then use
inline PTX. That beats "7 writes + 1 read" on NVLink traffic by 7x. Topology
supports it -- full all-to-all, NV18 (18 links) between every pair.

---

## The real model, per layer (measured against `models/DeepSeek-V4.1-Flash-1017`)

Everything above was built against a *single* layer archetype and then replayed 40
times. The real checkpoint's 40 layers are not identical, and the difference is
structural, not just weights. Read straight from the checkpoint's `config.json`:

```
num_hidden_layers = 40
compress_ratios   = [0, 0, 2 x 18, 1 x 20]        # 43 entries, first 40 used
engram_layer_ids    = [1, 14]
kv_source_layer_ids = [2, 8, 14, 20]
index_source_layer_ids = [2, 8, 14, 20, 24, 28, 32, 36]
sliding_window = 128,  index_topk = 512,  hc_mult = 4
```

Expanding `compress_ratios` through the variant rule in
`mirage/demo/deepseek_v4_1_csa2/weights.py:63`:

| variant | layers | count | what it does |
|---|---|---|---|
| `swa` | L0, L1 | 2 | sliding window only (128 keys), no compressed KV, no indexer |
| `full_r2` | L2, L8, L14 | 3 | + compressor (`wkv`, `wgate`, pooling) writing ratio-2 latents, + indexer |
| `full_r1` | L20 | 1 | + compressor (no `wgate`: ratio 1 is a plain projection), + indexer |
| `reindex` | L24, 28, 32, 36 | 4 | + indexer only (reuses its source's compressor output) |
| `reuse` | the other 30 | 30 | reads the compressed KV and top-k its source published |
| + Engram | L1, L14 | 2 | n-gram lookup into the residual, `wkv` [25600, 6144] |

The 18-kernel chain in `dsv41_pdl_layer.py` matches the **`reuse`** archetype: it
reads `128 window + 512 compressed = 640` keys, which is exactly
`sliding_window + index_topk`. So it represents 30 of 40 layers, and by
construction the *cheapest* of the three non-trivial ones.

### Six places the kernels disagreed with the checkpoint

1. **The shared expert is FP8, not FP4.** `ffn.shared_experts.w1.weight` is
   `F8_E4M3 [2304, 5120]` (11.8 MB) while a routed expert's is `I8 [2304, 2560]`
   (5.9 MB packed MXFP4). The kernel read the shared expert as a 7th fp4 slot at
   row `NEXP`, which is half the bytes and a different format. At TP8 the shared
   gate_up is 2.95 MB, not 1.47 MB.
2. **`wq_a` is replicated, so there is no q_lora all_gather.**
   `inference/model.py:643` declares `wq_a` a plain `Linear`; only `wq_b` is
   `ColumnParallelLinear`. Every rank therefore multiplies the full 1280 x 5120
   (6.55 MB), not a 160-row shard (0.82 MB). The e2e runner's all_gather of 1280
   floats is not part of this model's TP8 step.
3. **The router bias selects experts.** `(scores + bias).topk()` at
   `inference/model.py:824` -- the bias steers *selection only*, while the weights
   come from the unbiased scores. The kernel ignored `ffn.gate.bias` entirely, so
   its expert set could differ from the model's on near-ties.
4. **The router runs in fp32.** `linear(x.float(), self.weight.float())`: the gate
   weight is stored bf16 but upcast before the GEMV, so that stage moves 7.86 MB
   per layer, not 3.93 MB.
5. **mHC is missing.** `hc_mult = 4` residual streams. `mix_hc = (2 + hc_mult) *
   hc_mult = 24` and `hc_dim = 4 x 5120 = 20480`, which is exactly the checkpoint's
   `hc_attn_fn [24, 20480]`. Two coefficient GEMVs per layer (attn + ffn), each
   1.97 MB f32, plus a 20-iteration Sinkhorn on the 4x4 comb matrix.
6. **The ratio-2 compressor only emits every other step.** `Compressor.forward`
   returns `None` while a group is still filling, so a `full_r2` layer does that
   work on half of the decode steps. `kv_state` / `score_state` carry the partial
   group across steps.

### TP8 sharding as the model actually does it

| tensor | checkpoint | this rank (TP8) | split |
|---|---|---|---|
| `wq_a` | `[1280, 5120]` fp8 | `[1280, 5120]` | **replicated** |
| `wq_b` | `[32768, 1280]` fp8 | `[4096, 1280]` | column, 8 of 64 heads |
| `wkv` | `[512, 5120]` fp8 | `[512, 5120]` | **replicated** (1 KV head) |
| `wo_a` | `[8192, 4096]` fp8 | `[1024, 4096]` | column, 1 of 8 `o_groups` |
| `wo_b` | `[5120, 8192]` fp8 | `[5120, 1024]` | row, partial sums |
| `gate` | `[384, 5120]` bf16 | same | replicated |
| `experts.w1/w3` | `[2304, 2560]` fp4 | `[288, 2560]` | intermediate 2304/8 |
| `experts.w2` | `[5120, 1152]` fp4 | `[5120, 144]` | intermediate 2304/8 |
| `shared.w1/w3` | `[2304, 5120]` **fp8** | `[288, 5120]` | intermediate 2304/8 |
| `shared.w2` | `[5120, 2304]` **fp8** | `[5120, 288]` | intermediate 2304/8 |

Collectives per layer are **two** all_reduces (`wo_b`, MoE `w2`), plus one more on
each of the 8 index-source layers for the indexer score. No all_gather per layer.
That is *fewer* chain breaks than the three-collective pipeline assumed earlier,
which is good news for PDL -- but more than two on 8 of the 40 layers.

Measured per-rank footprint: `reuse` 895 MiB, `full_r2` 911 MiB, `swa`+Engram
1045 MiB (excluding the 98 GB Engram embedding table, which is vocab-sharded to
~12.3 GB/rank/layer). 40 layers ~= 37 GiB/rank, ~62 GiB with Engram.

### Correction landed: the router bias (`ffn.gate.bias`)

The first delta is fixed and re-verified in both implementations. The router now
computes the score twice over: `selv[j] = score[j] + GB[j]` drives the top-6 scan
(`T.reduce_max` + the `atomic_min` tie-break), while `lval[j]` -- the unbiased
score -- supplies the routing weights, matching `inference/model.py:823-825`.
Ghosting one array and masking only the other would silently give an expert the
weight of a different one, so both are masked after each pick.

Both paths still match the reference bit for bit:

```
single launch   layer out rel=1.702e-07  MATCH   experts 6/6
PDL chain       layer out rel=1.702e-07  MATCH   experts 6/6
```

Re-measured on B300, three runs each (the first run of any process is ~3 us slow
while the L2 warms, so the outlier is dropped):

| | no PDL | PDL | delta |
|---|---|---|---|
| one layer, 1 GPU, no collective, L2 hot | 85.2 us | **82.1 us** | -3.1 us (3.6%) |

**These are not the model's tok/s.** Do not multiply by 40: a single layer on one
GPU with no collectives and a warm L2 is the most favourable configuration there
is. The real 40-layer TP8 step is in `FINDINGS.md` and is **202.30 us/layer,
123.6 tok/s** under CUDA graph -- 2.5x slower than `82.1 x 40` because each layer
pays 85 us of all_reduce latency and the weights no longer fit in L2. An earlier
revision of this file extrapolated that single-layer number to "304.5 tok/s";
that was wrong.

Note the chain below is the **`reuse` archetype only** -- 30 of the 40 layers.

Still open from the six deltas above: the fp8 shared expert (needs the kernel to
stop reading slot `NEXP` of the fp4 table), `wq_a`'s replication (the e2e runner
still all-gathers a q_lora shard that this model never produces), mHC, and the
five attention variants.

---

## Read this first: `FINDINGS.md`

`FINDINGS.md` in this directory is the authoritative record: the real per-layer
structure of the checkpoint, the six places these kernels disagree with it, the
measured performance decomposition, the all_reduce latency table, and a
"do not repeat these" list of the eleven traps hit while building this (single-layer
numbers extrapolated to model tok/s, buffered `grep` masquerading as a hang, NCCL
capture needing a side-stream warmup, safetensors header offsets, TileLang caching
by `global_symbol`, and others). The numbers below predate several of those fixes;
treat `FINDINGS.md` as the current state.
