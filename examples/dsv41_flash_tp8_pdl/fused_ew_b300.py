#!/usr/bin/env python3
"""DSV4.1-Flash fused elementwise kernels for B=1, TP8, B300.

Three kernels, each verified against an exact torch reference:

1. `rmsnorm_quant_mxfp8`  RMSNorm -> per-32-group ue8m0 -> fp8.
   Feeds every linear, which is why the quant is fused into the norm rather than
   run as its own pass. eps must be 1e-20 (config.json `rms_norm_eps`); a default
   like 1e-6 is wrong here.

2. `swiglu_clamp_quant_mxfp8`  silu(gate)*up with the clamped branches, -> fp8.
   Exact semantics copied from sglang's
   `kernels/jit/csrc/deepseek_v4/silu_and_mul_masked_post_quant.cuh`:

       gate = min(gate, limit)          # limit = swiglu_limit = 10.0
       up   = clamp(up, -limit, limit)
       out  = silu(gate) * up

   and the clamp happens in **bf16**, before widening to fp32 ("must clamp in
   bf16" in that source), so we clamp then convert.

3. `rope_partial`  rotary on the trailing 64 dims (head_dim 512 = 448 NoPE + 64
   RoPE), GPT-NeoX half-split convention.

The ue8m0 rounding is **round the exponent up**, not round-to-nearest:

    raw = max(amax, 1e-10) / 448
    byte = exponent(raw) + (mantissa(raw) != 0)      # i.e. ceil to a power of 2
    scale = 2^(byte - 127)

so the scaled value can never exceed 448 and overflow the e4m3 range. Rounding to
nearest instead silently saturates the largest element of roughly half the groups.

Run: python fused_ew_b300.py [check|bench]
"""

import sys
import time

import torch
import tilelang
import tilelang.language as T

FP8 = "float8_e4m3fn"
FP8_MAX = 448.0
GRAN = 32                      # one ue8m0 scale per 32 elements
AMAX_FLOOR = 1e-10


# ------------------------------------------------------------------ ue8m0 bits
def _ceil_exp_byte(raw_f32_bits):
    """float32 bits -> ue8m0 byte, rounding the exponent UP (see module docstring)."""
    expo = (raw_f32_bits >> 23) & 0xFF
    mant = raw_f32_bits & 0x7FFFFF
    return expo + T.if_then_else(mant != 0, 1, 0)


@tilelang.jit
def rmsnorm_quant_mxfp8(K: int, eps: float, threads: int = 256):
    """y = fp8(rmsnorm(x) * w) with a ue8m0 scale per 32 elements of y.

    X: [K] fp32 input row.  Wt: [K] fp32 norm weight.
    Y: [K] e4m3.            S: [K/32] uint8 ue8m0 exponent bytes.
    """
    G = K // GRAN
    blocks = (K + threads - 1) // threads

    @T.prim_func
    def main(X: T.Tensor((K,), "float32"),
             Wt: T.Tensor((K,), "float32"),
             Y: T.Tensor((K,), FP8),
             S: T.Tensor((G,), "uint8")):
        with T.Kernel(1, threads=threads):
            xs = T.alloc_shared((K,), "float32")
            ys = T.alloc_shared((K,), "float32")
            part = T.alloc_shared((threads,), "float32")
            ss = T.alloc_shared((1,), "float32")
            amax = T.alloc_shared((G,), "float32")
            ebyte = T.alloc_shared((G,), "uint32")

            for i in T.Parallel(K):
                xs[i] = X[i]
            T.sync_threads()

            # ---- sum of squares, one partial per thread, then a tree reduce
            for t in T.Parallel(threads):
                part[t] = 0.0
            T.sync_threads()
            for b in T.serial(blocks):
                for t in T.Parallel(threads):
                    idx = b * threads + t
                    if idx < K:
                        part[t] += xs[idx] * xs[idx]
            T.sync_threads()
            T.reduce_sum(part, ss, dim=0)
            T.sync_threads()

            # ---- y = x * rsqrt(mean(x^2) + eps) * w
            for i in T.Parallel(K):
                ys[i] = (xs[i] * T.ieee_frsqrt(ss[0] / T.Cast("float32", K) + eps)
                         * Wt[i])
            T.sync_threads()

            # ---- per-group absmax, one group per lane
            for g in T.Parallel(G):
                amax[g] = 0.0
            T.sync_threads()
            for j in T.serial(GRAN):
                for g in T.Parallel(G):
                    amax[g] = T.max(amax[g], T.abs(ys[g * GRAN + j]))
            T.sync_threads()

            for g in T.Parallel(G):
                ebyte[g] = _ceil_exp_byte(
                    T.reinterpret(T.max(amax[g], AMAX_FLOOR) / FP8_MAX, "uint32"))
            T.sync_threads()

            for g in T.Parallel(G):
                S[g] = T.Cast("uint8", ebyte[g])
            for i in T.Parallel(K):
                Y[i] = T.Cast(FP8, ys[i] * T.exp2(-(T.Cast("float32", ebyte[div(i, GRAN)])
                                                   - 127.0)))

    return main


def div(a, b):
    """Euclidean divide helper usable inside T.Parallel index expressions."""
    return T.floordiv(a, b)


@tilelang.jit
def swiglu_clamp_quant_mxfp8(K: int, limit: float, threads: int = 256):
    """out = fp8(silu(min(gate,limit)) * clamp(up,-limit,limit)), ue8m0 per 32.

    Gate and up are bf16 (the MoE intermediate dtype); the clamp is applied
    before widening, matching the reference.
    """
    G = K // GRAN

    @T.prim_func
    def main(Gate: T.Tensor((K,), "bfloat16"),
             Up: T.Tensor((K,), "bfloat16"),
             Y: T.Tensor((K,), FP8),
             S: T.Tensor((G,), "uint8")):
        with T.Kernel(1, threads=threads):
            ys = T.alloc_shared((K,), "float32")
            amax = T.alloc_shared((G,), "float32")
            ebyte = T.alloc_shared((G,), "uint32")

            for i in T.Parallel(K):
                g = T.Cast("float32", T.min(Gate[i], T.Cast("bfloat16", limit)))
                u = T.Cast("float32", T.max(T.min(Up[i], T.Cast("bfloat16", limit)),
                                            T.Cast("bfloat16", -limit)))
                # silu(g) = g * sigmoid(g), computed in fp32
                ys[i] = g / (1.0 + T.exp(-g)) * u
            T.sync_threads()

            for g in T.Parallel(G):
                amax[g] = 0.0
            T.sync_threads()
            for j in T.serial(GRAN):
                for g in T.Parallel(G):
                    amax[g] = T.max(amax[g], T.abs(ys[g * GRAN + j]))
            T.sync_threads()

            for g in T.Parallel(G):
                ebyte[g] = _ceil_exp_byte(
                    T.reinterpret(T.max(amax[g], AMAX_FLOOR) / FP8_MAX, "uint32"))
            T.sync_threads()

            for g in T.Parallel(G):
                S[g] = T.Cast("uint8", ebyte[g])
            for i in T.Parallel(K):
                Y[i] = T.Cast(FP8, ys[i] * T.exp2(-(T.Cast("float32", ebyte[div(i, GRAN)])
                                                   - 127.0)))

    return main


@tilelang.jit
def rope_partial(D: int, NOPE: int, threads: int = 256):
    """In-place rotary on the trailing (D - NOPE) dims, GPT-NeoX half-split.

    X: [D] fp32, modified in place.  CS: [half, 2] fp32 (cos, sin) for this step.
    """
    R = D - NOPE
    HALF = R // 2

    @T.prim_func
    def main(X: T.Tensor((D,), "float32"), CS: T.Tensor((HALF, 2), "float32")):
        with T.Kernel(1, threads=threads):
            xs = T.alloc_shared((R,), "float32")
            for i in T.Parallel(R):
                xs[i] = X[NOPE + i]
            T.sync_threads()
            for k in T.Parallel(HALF):
                a = xs[k]
                b = xs[HALF + k]
                X[NOPE + k] = xs[k] * CS[k, 0] - xs[HALF + k] * CS[k, 1]
                X[NOPE + HALF + k] = a * CS[k, 1] + b * CS[k, 0]

    return main


# ------------------------------------------------------------------- reference
def ref_rmsnorm_quant(x, w, eps, gran=GRAN):
    y = x * torch.rsqrt(x.pow(2).mean() + eps) * w
    G = y.numel() // gran
    yg = y.view(G, gran)
    amax = yg.abs().amax(dim=1).clamp_min(AMAX_FLOOR)
    raw = amax / FP8_MAX
    bits = raw.view(torch.int32)
    expo = ((bits >> 23) & 0xFF).to(torch.int64)
    mant = (bits & 0x7FFFFF) != 0
    byte = (expo + mant.to(torch.int64))
    scale = torch.exp2(byte.to(torch.float32) - 127.0)
    out = (yg / scale[:, None]).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return out.view(-1), byte.to(torch.uint8)


def ref_swiglu_quant(gate, up, limit, gran=GRAN):
    g = torch.minimum(gate, torch.tensor(limit, dtype=torch.bfloat16))
    u = torch.maximum(torch.minimum(up, torch.tensor(limit, dtype=torch.bfloat16)),
                      torch.tensor(-limit, dtype=torch.bfloat16))
    y = torch.nn.functional.silu(g.float()) * u.float()
    G = y.numel() // gran
    yg = y.view(G, gran)
    amax = yg.abs().amax(dim=1).clamp_min(AMAX_FLOOR)
    bits = (amax / FP8_MAX).view(torch.int32)
    byte = (((bits >> 23) & 0xFF).to(torch.int64)
            + ((bits & 0x7FFFFF) != 0).to(torch.int64))
    scale = torch.exp2(byte.to(torch.float32) - 127.0)
    out = (yg / scale[:, None]).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return out.view(-1), byte.to(torch.uint8)


# ------------------------------------------------------------------------ test
def check(K=5120, eps=1e-20, limit=10.0):
    torch.manual_seed(0)
    x = torch.randn(K, device="cuda") * 2.0
    w = torch.randn(K, device="cuda") * 0.1 + 1.0
    y = torch.empty(K, dtype=torch.float8_e4m3fn, device="cuda")
    s = torch.empty(K // GRAN, dtype=torch.uint8, device="cuda")
    rmsnorm_quant_mxfp8(K, eps)(x, w, y, s)
    torch.cuda.synchronize()
    ry, rs = ref_rmsnorm_quant(x, w, eps)
    ey = (y.float() - ry.float()).abs().max().item()
    es = (s != rs).sum().item()
    print(f"rmsnorm_quant  K={K} eps={eps}")
    print(f"  fp8 payload exact: {bool(torch.equal(y.view(torch.uint8), ry.view(torch.uint8)))}"
          f"   max|dy|={ey:.3e}   scale bytes differing: {es}/{K//GRAN}")

    gate = (torch.randn(K, device="cuda") * 3).to(torch.bfloat16)
    up = (torch.randn(K, device="cuda") * 8).to(torch.bfloat16)
    y2 = torch.empty(K, dtype=torch.float8_e4m3fn, device="cuda")
    s2 = torch.empty(K // GRAN, dtype=torch.uint8, device="cuda")
    swiglu_clamp_quant_mxfp8(K, limit)(gate, up, y2, s2)
    torch.cuda.synchronize()
    ry2, rs2 = ref_swiglu_quant(gate, up, limit)
    e2 = (y2.float() - ry2.float()).abs().max().item()
    e2s = (s2 != rs2).sum().item()
    print(f"swiglu_quant   K={K} limit={limit}")
    print(f"  fp8 payload exact: {bool(torch.equal(y2.view(torch.uint8), ry2.view(torch.uint8)))}"
          f"   max|dy|={e2:.3e}   scale bytes differing: {e2s}/{K//GRAN}")

    # rope
    D, NOPE, R = 512, 448, 64
    pos = 37.0
    inv = 10000.0 ** (-torch.arange(0, R // 2, device="cuda", dtype=torch.float32)
                      * 2.0 / R)
    ang = pos * inv
    cs = torch.stack([ang.cos(), ang.sin()], dim=-1).contiguous()
    xr = torch.randn(D, device="cuda")
    got = xr.clone()
    rope_partial(D, NOPE)(got, cs)
    torch.cuda.synchronize()
    a, b = xr[NOPE:NOPE + R // 2], xr[NOPE + R // 2:]
    want = torch.cat([a * cs[:, 0] - b * cs[:, 1], a * cs[:, 1] + b * cs[:, 0]])
    e3 = (got[NOPE:] - want).abs().max().item()
    e3n = (got[:NOPE] - xr[:NOPE]).abs().max().item()
    print(f"rope_partial   D={D} nope={NOPE}")
    print(f"  max|d|={e3:.3e}   NoPE untouched: {e3n == 0.0}")
    return ey < 0.06 and es == 0 and e2 < 0.06 and e2s == 0 and e3 < 1e-6


def graph_time(fn, iters=200, warmup=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(iters):
            fn()
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(10):
        t0 = time.perf_counter()
        g.replay()
        torch.cuda.synchronize()
        best = min(best, (time.perf_counter() - t0) / iters * 1e6)
    return best


def bench(K=5120):
    x = torch.randn(K, device="cuda")
    w = torch.randn(K, device="cuda") * 0.1 + 1.0
    y = torch.empty(K, dtype=torch.float8_e4m3fn, device="cuda")
    s = torch.empty(K // GRAN, dtype=torch.uint8, device="cuda")
    gate = torch.randn(K, device="cuda", dtype=torch.bfloat16)
    up = torch.randn(K, device="cuda", dtype=torch.bfloat16)
    kn = rmsnorm_quant_mxfp8(K, 1e-20)
    ks = swiglu_clamp_quant_mxfp8(K, 10.0)
    print(f"  rmsnorm+quant K={K}  {graph_time(lambda: kn(x, w, y, s)):6.2f} us")
    print(f"  swiglu+quant  K={K}  {graph_time(lambda: ks(gate, up, y, s)):6.2f} us")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "check"
    if what == "bench":
        bench()
    else:
        ok = check()
        print("  -> PASS" if ok else "  -> FAIL")
