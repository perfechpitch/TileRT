#!/usr/bin/env python3
"""40-layer DeepSeek-V4.1-Flash decode step, TP8, as a PDL chain.

    torchrun --nproc_per_node=8 dsv41_pdl_e2e_tp8.py check
    torchrun --nproc_per_node=8 dsv41_pdl_e2e_tp8.py bench

Per layer, 18 kernels in TWO PDL clusters joined by the two collectives the
tensor-parallel algebra of *this* model actually requires:

    seg ab (10) rmsnorm1 p/a -> wq_a(1280 rows, replicated) -> rmsnormQL p/a
                -> wq_b -> MLA split -> MLA combine -> wo_a -> wo_b
                -- all_reduce(attn partial, 5120) --
    seg c  (8)  resid add -> rmsnorm2 p/a -> router -> top-6 -> gate_up x7
                -> swiglu x7 -> down x7
                -- all_reduce(moe partial, 5120) -- OUT += MO

There is **no q_lora all_gather**: `inference/model.py:643` declares `wq_a` a plain
`Linear`, so every rank multiplies the whole 1280 x 5120 and holds all of q_lora.
Only `wq_b` is column-parallel. That collapses what used to be two clusters (a and
b) into one, and drops a collective per layer -- eight collectives across the 40
layers, not sixteen. It also makes that whole stretch one uninterrupted PDL chain
of ten kernels.

Note this holds for the `reuse`/`swa` archetypes; `full_r*`/`reindex` add an
indexer all_reduce on their 8 index-source layers.

PDL cannot cross a collective. `griddepcontrol.launch_dependents` only releases the
next grid in the same stream, and an NCCL kernel never emits it, so each boundary
is a full stream dependency. That much is a property of pure TP8, not of this
implementation.

Residual placement (the part that is easy to get wrong at TP8)
--------------------------------------------------------------
wo_b emits a *partial* and the all_reduce sums it, so X must not be folded into
wo_b's store: X is identical on every rank and would be summed TP times. It is
added once, by `resid_add`, at the head of seg c. For the same reason the MoE
result lands in MO, is reduced, and only then added to OUT -- folding it into OUT
before the reduce would sum the already-reduced attention output a second time.

Weights are per layer, as in the real model: 40 distinct sets, ~931 MB each at TP8
(384 experts held whole on every rank, intermediate sharded to 288), ~37 GB/GPU.

Modes
-----
check  one layer, ranks > 0 have their K-parallel weights zeroed so the reductions
       evaluate rank 0's shard exactly; rank 0 then compares the distributed result
       against the torch replica. This validates the collective plumbing (no
       residual double-counting) rather than full-model math.
bench  40 layers, all weights live, captured into one CUDA graph.
"""

import os
import sys
import time

import torch
import torch.distributed as dist
import tilelang

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "..", "dsv41_mla"))

from dsv41_layer_megakernel import (  # noqa: E402
    H, QLR, QN, WOA, E_GU, E_K, NEXP, TOPK, NSH, NGU, GRAN, NB_MAX,
    NWIN, NCMP, NT, HEADS, HEAD_DIM, build, ref_layer,
)
from dsv41_pdl_layer import (  # noqa: E402
    THREADS, BN, k_norm_partial, k_norm_apply, k_gemv, k_mla_split,
    k_mla_combine, k_router, k_topk, k_moe_gu, k_swiglu, k_moe_dn, k_resid_add,
    _named,
)

WQ_ROWS = QLR // 8          # 160 q_lora rows per rank
NLAYERS = int(os.environ.get("DSV41_LAYERS", "40"))
# DSV41_NOCOMM=1 drops both all_reduces, to split compute from communication.
# The result is wrong on purpose; it is a timing probe, not a numerics mode.
NOCOMM = os.environ.get("DSV41_NOCOMM", "0") != "0"
CODE = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
        0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]


# ------------------------------------------------------------------ kernels
def compile_kernels(off, pdl=True):
    """The 18 kernels of one layer, compiled once and reused for all 40.

    `off` is this rank's row offset into q_lora. It is baked into the wq_a kernel
    (and its compiled symbol) rather than passed as an argument, because it is
    fixed for the whole run.
    """
    specs = [
        ("ab", k_norm_partial("X", pdl)),                       # rmsnorm1 partials
        ("ab", k_norm_apply(H, "X", "GN", "XF", False, pdl)),    # rmsnorm1 apply
        # wq_a is replicated in this model: all 1280 q_lora rows on every rank,
        # so nothing has to be gathered before rmsnormQL
        ("ab", k_gemv("qa", QLR, H, 0, pdl)),
        ("ab", k_norm_partial("QL", pdl)),                      # rmsnormQL partials
        ("ab", k_norm_apply(QLR, "QL", "QNORM", "QLF", False, pdl)),
        ("ab", k_gemv("qb", QN, QLR, 0, pdl)),                  # wq_b
        ("ab", k_mla_split(pdl)),
        ("ab", k_mla_combine(pdl)),
        ("ab", k_gemv("oa", WOA, QN, 0, pdl)),                  # wo_a
        ("ab", k_gemv("ob", H, WOA, 0, pdl, resid=False)),      # wo_b: partial only
        ("c", k_resid_add(pdl)),                                # OUT += X
        ("c", k_norm_partial("OUT", pdl)),                      # rmsnorm2 partials
        ("c", k_norm_apply(H, "OUT", "GN2", "HF", True, pdl)),   # rmsnorm2 (+ HN)
        ("c", k_router(pdl)),
        ("c", k_topk(pdl)),
        ("c", k_moe_gu(pdl)),
        ("c", k_swiglu(pdl)),
        ("c", k_moe_dn(pdl, dst="MO")),                         # down, to its own buffer
    ]
    out = []
    for i, (seg, (main, names)) in enumerate(specs):
        out.append((seg, tilelang.compile(_named(main, f"e2e_{off}_{i}")), names))
    return out


# ------------------------------------------------------------------ weights
def _u8(shape, dev, lo=None, hi=None):
    """Fast arbitrary bytes. Values are irrelevant to timing; what matters is that
    every layer's tensors are distinct so nothing is reused across layers."""
    t = torch.empty(shape, dtype=torch.uint8, device=dev)
    if lo is None:
        t.random_()
    else:
        t.random_(lo, hi)
    return t


def layer_weights(seed, dev, rank):
    """One layer's TP8 weights, generated on-device.

    wq_a is **replicated**: each rank multiplies all 1280 q_lora rows, matching
    `inference/model.py:643` (a plain `Linear`, unlike the column-parallel wq_b).
    """
    torch.manual_seed(seed)
    d = {}

    def fp8(N, K):
        return (torch.randn(N, K, device=dev) * 0.05).to(torch.float8_e4m3fn)

    d["WQA"] = fp8(QLR, H).view(QLR, H // 4, 4)
    d["SQA"] = _u8((QLR, H // GRAN), dev, 120, 133)
    d["WQB"] = fp8(QN, QLR).view(QN, QLR // 4, 4)
    d["SQB"] = _u8((QN, QLR // GRAN), dev, 120, 133)
    d["WOAW"] = fp8(WOA, QN).view(WOA, QN // 4, 4)
    d["SOA"] = _u8((WOA, QN // GRAN), dev, 120, 133)
    d["WOBW"] = fp8(H, WOA).view(H, WOA // 4, 4)
    d["SOB"] = _u8((H, WOA // GRAN), dev, 120, 133)
    d["GW"] = (torch.randn(NEXP, H, device=dev) * 0.02).to(torch.bfloat16)

    # Every rank holds all 384 experts (pure TP8, no EP), sharded along the
    # intermediate axis: 2304/8 = 288 per side.
    # (H // 8) *uint32* words per row, so 4 bytes each: the byte shape is 4x the
    # word shape, which is what makes the int32 <-> uint32 view a no-op.
    d["WGU"] = _u8((NEXP + NSH, E_GU, (H // 8) * 4), dev).view(torch.int32).view(
        torch.uint32).view(NEXP + NSH, E_GU, H // 8)
    d["SGU"] = _u8((NEXP + NSH, E_GU, H // GRAN), dev, 120, 133)
    d["WDN"] = _u8((NEXP + NSH, H, (E_K // 8) * 4), dev).view(torch.int32).view(
        torch.uint32).view(NEXP + NSH, H, E_K // 8)
    d["SDN"] = _u8((NEXP + NSH, H, E_K // GRAN), dev, 120, 133)

    # the shared expert is fp8 e4m3 with 32x32 ue8m0 scales, not a 7th fp4 slot
    d["SHGW"] = fp8(E_GU, H).view(E_GU, H // 4, 4)
    d["SHGS"] = _u8((E_GU, H // GRAN), dev, 120, 133)
    d["SHDW"] = fp8(H, E_K)
    d["SHDS"] = _u8((H, E_K // GRAN), dev, 120, 133)
    d["GB"] = (torch.randn(NEXP, device=dev) * 0.5).float()

    from dsv41_mla_flash import make_inputs as mla_inputs
    _q, win, cmp_, sink, freq, pos = mla_inputs(NWIN, NCMP, seed=seed)
    d["WIN"], d["CMP"], d["SINK"], d["FREQ"] = win, cmp_, sink, freq
    d["pos"] = pos

    d["GN"] = (torch.randn(H, device=dev) * 0.1 + 1.0)
    d["QNORM"] = (torch.randn(QLR, device=dev) * 0.1 + 1.0)
    d["GN2"] = (torch.randn(H, device=dev) * 0.1 + 1.0)
    return d


def shared_buffers(dev):
    z = lambda *s: torch.zeros(*s, dtype=torch.float32, device=dev)
    return dict(
        A=z(H), B=z(H),
        QS=z(QLR), QL=z(QLR), Q=z(QN),
        PO=z(HEADS, NT, HEAD_DIM), MXA=z(NT, HEADS), SMA=z(NT, HEADS),
        AQ=z(QN), HN=z(H), LG=z(NEXP),
        GU=z(NGU, E_GU), ME=z(NGU, E_K), MO=z(H),
        XF=z(H), QLF=z(QLR), HF=z(H), WOAO=z(WOA),
        PART=z(NB_MAX),
        LUT=torch.tensor(CODE, dtype=torch.float32, device=dev),
        EXPID=torch.zeros(TOPK, dtype=torch.int32, device=dev),
        WGT=torch.zeros(NGU, device=dev),
    )


# ------------------------------------------------------------------ the step
def run_layer(kern, L, X, OUT, rank):
    """One layer: two PDL clusters, two collectives.

    `X` is this layer's input hidden state and `OUT` the buffer it writes; they are
    distinct so the ping-pong across 40 layers needs no copy.
    """
    a = dict(L, X=X, OUT=OUT)
    for seg, k, names in kern:
        if seg != "ab":
            break
        k(*[a[n] for n in names])

    if not NOCOMM:
        dist.all_reduce(OUT)

    c = dict(L, X=X, OUT=OUT)
    for seg, k, names in kern:
        if seg != "c":
            continue
        k(*[c[n] for n in names])

    if not NOCOMM:
        dist.all_reduce(L["MO"])
    OUT.add_(L["MO"])


def run_step(kern, layers, buf, rank):
    """40 layers, hidden state ping-ponged between two buffers."""
    x, y = buf["A"], buf["B"]
    for i, L in enumerate(layers):
        run_layer(kern, L, x, y, rank)
        x, y = y, x


# ------------------------------------------------------------------ modes
def main():
    rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ.get("WORLD_SIZE", "8"))
    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)
    dist.init_process_group("nccl")

    mode = sys.argv[1] if len(sys.argv) > 1 else "check"
    nlay = 1 if mode == "check" else NLAYERS

    off = rank * WQ_ROWS
    kern = compile_kernels(off, True)
    buf = shared_buffers(dev)
    dist.barrier()

    if mode == "check":
        # Reuse the validated single-layer harness so the reference and the kernel
        # see identical weights.
        t, args, ref_in = build()
        if rank > 0:
            # Zero the K-parallel weights on every rank but 0, so the all_reduces
            # evaluate exactly rank 0's shard. Fresh tensors, not .zero_(): `args`
            # holds views, and mutating in place would corrupt rank 0's reference.
            # The shared expert is sharded along the same intermediate axis
            # (2304/8 = 288 per rank), so its fp8 matrices belong here too --
            # leaving them out sums the shared expert TP times.
            for k in ("WOBW", "SOB", "WDN", "SDN", "SHGW", "SHDW"):
                args[k] = torch.zeros_like(args[k])
        L = dict(args)
        L["QS"] = torch.zeros(QLR, device=dev)
        x = L["X"].clone()
        out = torch.zeros(H, device=dev)
        run_layer(kern, L, x, out, rank)
        torch.cuda.synchronize()

        if rank == 0:
            rql, rq, raq, rout, rtopi = ref_layer(t, *ref_in)
            e = (out - rout).abs().max().item() / rout.abs().max().item()
            print(f"  e2e layer out   rel={e:.3e}  "
                  f"{'MATCH' if e < 5e-3 else 'MISMATCH'}")
            kt = torch.sort(L["EXPID"].long())[0].tolist()
            rt = torch.sort(rtopi.long())[0].tolist()
            print(f"  routed experts  {sum(a == b for a, b in zip(kt, rt))}/{TOPK} "
                  f"(kernel {kt} ref {rt})")
        dist.barrier()
        dist.destroy_process_group()
        sys.exit(0)

    # ---- bench: 40 distinct layers, one graph
    layers = []
    for i in range(nlay):
        # merge the shared scratch in here, so every per-layer dict `run_layer`
        # receives is self-contained (weights + buffers)
        layers.append(dict(layer_weights(1000 + i, dev, rank), **buf))
    dist.barrier()
    if rank == 0:
        got = torch.cuda.memory_allocated() / 2**30
        print(f"  weights: {nlay} layers, {got:.1f} GiB/GPU")

    for _ in range(3):                      # collective warmup before capture
        run_step(kern, layers, buf, rank)
    torch.cuda.synchronize()
    dist.barrier()

    # ---- eager timing first: it is the honest launch pattern for a PDL chain
    # (every stage is its own launch, which is what the chain is), and it does not
    # depend on the capture path working.
    best = float("inf")
    for _ in range(6):
        dist.barrier()
        t0 = time.perf_counter()
        for _ in range(10):
            run_step(kern, layers, buf, rank)
        torch.cuda.synchronize()
        best = min(best, (time.perf_counter() - t0) / 10 * 1e6)
    eager_us = best
    tt = torch.tensor([eager_us], device=dev)
    dist.all_reduce(tt, op=dist.ReduceOp.MAX)
    eager_us = tt.item()
    if rank == 0:
        print(f"  eager  step ({nlay} layers, TP8): {eager_us:8.2f} us")
        print(f"  eager  per layer:                 {eager_us/nlay:8.2f} us")
        print(f"  eager  -> {1e6/eager_us:6.1f} tok/s")

    if mode == "eager":
        # stop here: the eager number is the one that does not depend on the
        # capture path, so it is useful even while capture is being fixed
        dist.barrier()
        dist.destroy_process_group()
        sys.exit(0)

    # ---- graph timing.
    # NCCL capture needs the communicator warmed on a *side* stream inside the
    # capture region; warming on the default stream leaves the graph's NCCL
    # handles unregistered and the capture hangs at the first collective with the
    # GPU idle, which is exactly what happened here.
    if rank == 0:
        print("  [graph] warming on a side stream", flush=True)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            run_step(kern, layers, buf, rank)
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    dist.barrier()

    if rank == 0:
        print("  [graph] entering capture", flush=True)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        if rank == 0:
            print("  [graph] recording", flush=True)
        run_step(kern, layers, buf, rank)
    if rank == 0:
        print("  [graph] captured", flush=True)
    torch.cuda.synchronize()
    dist.barrier()

    best = float("inf")
    for _ in range(8):
        dist.barrier()
        t0 = time.perf_counter()
        for _ in range(20):
            g.replay()
        torch.cuda.synchronize()
        best = min(best, (time.perf_counter() - t0) / 20 * 1e6)

    tt = torch.tensor([best], device=dev)
    dist.all_reduce(tt, op=dist.ReduceOp.MAX)
    step_us = tt.item()
    per_layer = step_us / nlay
    if rank == 0:
        print(f"  graph  step ({nlay} layers, TP8): {step_us:8.2f} us")
        print(f"  graph  per layer:                 {per_layer:8.2f} us")
        print(f"  graph  -> {1e6/step_us:6.1f} tok/s")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
