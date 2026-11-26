#!/usr/bin/env python3
"""Isolate the NCCL + CUDA-graph capture hang seen in dsv41_pdl_e2e_tp8.py.

Three captures, each a superset of the last, so the first one that hangs names the
culprit:

  1  plain torch op only            -- rules out the TileLang kernels
  2  one TileLang PDL kernel        -- rules out the PDL launch attribute
  3  + one all_reduce               -- the collective inside the capture
  4  + a second all_reduce          -- two collectives, as a real layer has

Each phase prints before it blocks, so a hang is attributable from the log alone.

  torchrun --nproc_per_node=8 dsv41_graph_probe2.py
"""
import os
import sys
import time

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def log(rank, msg):
    if rank == 0:
        print(msg, flush=True)


def phase(name, rank, fn, world):
    log(rank, f"[{name}] starting")
    g = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    dist.barrier()
    with torch.cuda.graph(g):
        fn()
    torch.cuda.synchronize()
    dist.barrier()
    t0 = time.perf_counter()
    for _ in range(10):
        g.replay()
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / 10 * 1e6
    if rank == 0:
        print(f"[{name}] OK  {dt:.1f} us/replay", flush=True)
    dist.barrier()


def main():
    rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ.get("WORLD_SIZE", "8"))
    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)
    dist.init_process_group("nccl")

    N = 5120
    x = torch.randn(N, device=dev)
    y = torch.zeros(N, device=dev)

    # ---- 1: a plain elementwise op
    phase("1 plain add", rank, lambda: y.add_(x, alpha=1.0), world)

    # ---- 2: one real TileLang kernel from the chain, PDL on
    from dsv41_pdl_layer import k_resid_add, _named
    import tilelang
    # k_resid_add is the simplest stage with a real PDL sync/trigger pair
    main_fn, names = k_resid_add(True)
    k = tilelang.compile(_named(main_fn, "probe_pdl"))
    buf = dict(X=x, OUT=y)
    phase("2 tilelang pdl", rank,
          lambda: k(*[buf[n] for n in names]), world)

    # ---- 3: one all_reduce
    phase("3 one all_reduce", rank, lambda: dist.all_reduce(y), world)

    # ---- 4: two all_reduces, interleaved with a kernel (a real layer's shape)
    z = torch.zeros(N, device=dev)

    def two():
        dist.all_reduce(y)
        z.add_(y)
        dist.all_reduce(z)

    phase("4 two all_reduce", rank, two, world)

    if rank == 0:
        print("all phases captured", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
