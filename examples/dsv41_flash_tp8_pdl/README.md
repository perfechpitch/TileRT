# DeepSeek-V4.1-Flash, TP8 decode step as a PDL chain

A batch-1 decode step of DeepSeek-V4.1-Flash-1017 on 8×B300 (SM103a), written in
TileLang, with the layer built as a chain of specialized kernels joined by
`T.pdl_sync()` / `T.pdl_trigger()` rather than one cooperative launch.

`FINDINGS.md` is the authoritative record: the checkpoint's real per-layer
structure, the measured decomposition, the all_reduce latency table, and the list
of traps hit while building this. Read it before changing anything.

## Headline numbers

40 layers, TP8, CUDA graph: **8092 µs/step = 123.6 tok/s**, which decomposes as
**117 µs/layer of compute + 85 µs/layer of all_reduce latency**.

Do not quote the single-layer figure as a model throughput. One layer on one GPU
with no collectives and a warm L2 is 82.1 µs; multiplying by 40 gives 304 tok/s and
is wrong, because in the full model each layer pays two 31 µs all_reduces and no
longer fits its 931 MB of weights in L2.

## The layers are not 40 copies of one layer

The checkpoint's `compress_ratios` gives five attention variants:

| variant | layers | count |
|---|---|---|
| `swa` | L0, L1 | 2 |
| `full_r2` | L2, L8, L14 | 3 |
| `full_r1` | L20 | 1 |
| `reindex` | L24, 28, 32, 36 | 4 |
| `reuse` | the rest | 30 |

plus Engram on L1 and L14, and mHC (`hc_mult = 4`) on every layer. The chain here
implements the **`reuse`** archetype only -- the other ten layers need stages that
do not exist yet. See `FINDINGS.md` §2 and §7.

The MoE half *is* identical across all 40 layers (attention compression is the
only per-layer variable), so the MoE numbers generalize.

## Files

| file | what |
|---|---|
| `FINDINGS.md` | **read first**: real layer structure, perf decomposition, latency table, traps |
| `PDL_NOTES.md` | the long-form build log: single launch vs PDL, NVSHMEM/multimem experiments, stage-by-stage reasoning |
| `dsv41_pdl_layer.py` | one layer as 17 PDL-linked kernels, plus the torch reference and the bench |
| `dsv41_pdl_e2e_tp8.py` | the 8-GPU 40-layer runner: two PDL clusters per layer, two all_reduces |
| `dsv41_layer_megakernel.py` | the same layer as one cooperative launch, kept as the comparison point |
| `dsv41_real_weights.py` | TP8 reader for the real checkpoint (48 sharded safetensors) |
| `dsv41_mla_*.py` | the MLA decode kernels and their reference |
| `dsv41_graph_probe2.py` | isolates a CUDA-graph capture hang step by step (plain op / PDL kernel / collectives) |
| `mxfp4_gemv_b1.py`, `mxfp8_gemv_b1.py`, `mxfp8_gemm_b300.py`, `fused_ew_b300.py`, `tp8_linear_b1.py` | the low-level GEMV / quantise helpers the layer calls |

## Running

```bash
# correctness: one layer, kernel vs torch reference
python dsv41_pdl_layer.py check

# one layer, PDL vs no-PDL timing
python dsv41_pdl_layer.py bench

# 40 layers, TP8
torchrun --nproc_per_node=8 dsv41_pdl_e2e_tp8.py check     # numerics
torchrun --nproc_per_node=8 dsv41_pdl_e2e_tp8.py bench     # eager + graph timing
torchrun --nproc_per_node=8 dsv41_pdl_e2e_tp8.py eager     # skip capture

# the split between compute and communication
DSV41_NOCOMM=1 torchrun --nproc_per_node=8 dsv41_pdl_e2e_tp8.py bench

# the real checkpoint's shapes and TP8 sharding
DSV41_CKPT=/path/to/DeepSeek-V4.1-Flash-1017 python dsv41_real_weights.py probe
```

`DSV41_LAYERS=4` shrinks the e2e run for a smoke test.

Long runs should go to a file, not through a pipe: `| grep` buffers, which has
been mistaken for a hang more than once (see `FINDINGS.md` §6.4).

## Known state

Correctness: the layer matches the torch reference to `rel=2.428e-07` with the
routed experts identical, in both the single-launch and PDL forms, and across all
8 TP8 ranks.

Open, in the order `FINDINGS.md` §7 argues for:

1. **Communication is the largest single cost** (85 of 202 µs/layer). A 20 KB
   `all_reduce` takes 31 µs on this machine -- pure latency, 0.3 GB/s effective --
   and PDL and CUDA graph cannot touch it. Replacing the two collects with a
   `multimem.red` (GIN) kernel is the change with the most headroom.
2. The five attention variants, mHC, and Engram.
