#!/usr/bin/env python3
"""TP8 weight loader for the real DeepSeek-V4.1-Flash-1017 checkpoint.

Reads the sharded safetensors directly and applies the *reference model's* TP8
split, which is not the one the earlier bench assumed. From
`inference/model.py` (the model's own parallel code):

    wq_a    Linear                  -> REPLICATED, all 1280 q_lora rows on every rank
    wq_b    ColumnParallelLinear    -> 32768 rows (64 heads) / 8 = 4096 (8 heads)
    wkv     Linear                  -> REPLICATED (num_key_value_heads == 1, latent is shared)
    wo_a    ColumnParallelLinear    -> 8192 (= 8 o_groups x 1024) / 8 = 1024 (one group)
    wo_b    RowParallelLinear       -> reduction dim 8192 / 8 = 1024 columns, partial sums
    gate    (replicated)            -> 384 x 5120 bf16, plus a bias
    experts w1/w3                   -> intermediate 2304 / 8 = 288 rows
            w2                      -> intermediate 2304 / 8 = 288 columns
    shared  w1/w3/w2                -> same split, but **fp8 e4m3, not fp4**

Two of these correct earlier assumptions and change the per-layer cost:

  * `wq_a` is replicated, so there is NO all_gather of q_lora and each rank reads
    the full 6.55 MB instead of a 0.82 MB shard. The three-collective TP8 pipeline
    the megakernel bench used (all_gather + 2 all_reduce) is wrong; this model has
    two all_reduce per layer (wo_b, MoE down) plus one per index-source layer.
  * the shared expert is stored fp8 while the routed experts are MXFP4, so the
    shared slot moves twice the bytes of a routed slot. Treating it as a 7th fp4
    expert -- which the earlier kernel did -- under-counts it by 2x.

Expert storage: routed experts are `I8` with shape [2304, 2560] and an
`F8_E8M0` scale of [2304, 160], i.e. MXFP4 packed two values per byte with one
ue8m0 scale per 32 values. `w2` is [5120, 1152] with scale [5120, 72].

Run: python dsv41_real_weights.py [probe|check]
"""

import json
import os
import struct
import sys

import torch

CKPT = os.environ.get(
    "DSV41_CKPT", "/opt/dlami/nvme/saboxu/models/DeepSeek-V4.1-Flash-1017")
INDEX = os.path.join(CKPT, "model.safetensors.index.json")


# ------------------------------------------------------------------ config
def load_config(path=os.path.join(CKPT, "config.json")):
    c = json.load(open(path))["text_config"]
    n = c["num_hidden_layers"]
    out = dict(
        n_layers=n,
        hidden=c["hidden_size"],
        n_heads=c["num_attention_heads"],
        head_dim=c["head_dim"],
        q_lora_rank=c["q_lora_rank"],
        o_lora_rank=c["o_lora_rank"],
        o_groups=c["o_groups"],
        moe_inter=c["moe_intermediate_size"],
        n_routed=c["n_routed_experts"],
        n_shared=c["n_shared_experts"],
        topk=c["num_experts_per_tok"],
        nope_dim=c["head_dim"] - c["qk_rope_head_dim"],
        rope_dim=c["qk_rope_head_dim"],
        window=c["sliding_window"],
        index_topk=c["index_topk"],
        index_n_heads=c["index_n_heads"],
        index_head_dim=c["index_head_dim"],
        hc_mult=c["hc_mult"],
        compress_ratios=c["compress_ratios"][:n],
        kv_source=tuple(c["kv_source_layer_ids"]),
        index_source=tuple(c["index_source_layer_ids"]),
        engram_layers=tuple(c["engram_layer_ids"]),
        candidate_source=c["candidate_source_layer_id"],
        candidate_topk_blocks=c["candidate_topk_blocks"],
        candidate_block_size=c["candidate_block_size"],
        eps=c["rms_norm_eps"],
        rscale=c["routed_scaling_factor"],
        swiglu_limit=c["swiglu_limit"],
    )
    out["variants"] = [variant_of(out, L) for L in range(n)]
    return out


def variant_of(cfg, L):
    """The layer's attention variant, as `mirage/.../csa2/weights.py:63` defines it."""
    r = cfg["compress_ratios"][L]
    if r == 0:
        return "swa"
    if L in cfg["kv_source"]:
        return f"full_r{r}"
    return "reindex" if L in cfg["index_source"] else "reuse"


# ------------------------------------------------------------------ shards
class Shards:
    """Header-only reader: every tensor is fetched by name, one shard open at a time.

    96085 tensors across 48 files; opening the shard on demand keeps the reader from
    holding 475 GB of handles and lets a single layer load without touching the rest.
    """

    def __init__(self, index_path=INDEX):
        idx = json.load(open(index_path))
        self.where = idx["weight_map"]
        self._open = {}
        self._hdr = {}
        self._base = {}

    def _shard(self, path):
        if path not in self._open:
            f = open(os.path.join(CKPT, path), "rb")
            # The header length is the first 8 bytes. Re-serialising the parsed
            # dict to measure it is wrong: json.dumps does not reproduce the
            # file's key order or spacing, so the data offsets would land a few
            # thousand bytes off and every tensor would come back truncated.
            n = struct.unpack("<Q", f.read(8))[0]
            self._open[path] = f
            self._hdr[path] = json.loads(f.read(n))
            self._base[path] = 8 + n
        return self._open[path], self._hdr[path], self._base[path]

    def raw(self, name):
        """The tensor's raw bytes plus its dtype/shape, without interpretation."""
        path = self.where[name]
        f, hdr, base = self._shard(path)
        meta = hdr[name]
        begin, end = meta["data_offsets"]
        f.seek(base + begin)
        return f.read(end - begin), meta["dtype"], meta["shape"]

    def shape(self, name):
        return self.hdr(name)[1]

    def dtype(self, name):
        return self.hdr(name)[0]

    def hdr(self, name):
        path = self.where[name]
        self._shard(path)
        return self._hdr[path][name]["dtype"], self._hdr[path][name]["shape"]

    def has(self, name):
        return name in self.where

    def u8(self, name):
        b, dt, shp = self.raw(name)
        return torch.frombuffer(bytearray(b), dtype=torch.uint8).view(*shp)


DTYPE_MAP = {
    "BF16": torch.bfloat16,
    "F32": torch.float32,
    "F16": torch.float16,
    "F8_E4M3": torch.float8_e4m3fn,
    "F8_E8M0": torch.uint8,       # ue8m0 stored as raw exponent bytes
    "I8": torch.uint8,            # packed fp4 nibbles
    "I32": torch.int32,
    "I64": torch.int64,
}


def load_tensor(sh, name, device="cuda", rows=None, cols=None):
    """Read one tensor, optionally keeping only a slice, as a CUDA tensor.

    `rows` / `cols` slice the first / second axis *while reading*, so a rank never
    materialises a shard it does not own: wq_b at TP8 is 32768 x 1280 fp8 = 42 MB
    full width, of which this rank wants 4096 rows.
    """
    b, dt, shp = sh.raw(name)
    t = torch.frombuffer(bytearray(b), dtype=DTYPE_MAP[dt]).view(*shp)
    if rows is not None:
        lo, hi = rows
        t = t[lo:hi]
    if cols is not None:
        lo, hi = cols
        t = t[:, lo:hi]
    return t.contiguous().to(device, non_blocking=False)


# ------------------------------------------------------------------ one layer
def layer_tensors(sh, cfg, L, rank, tp=8, device="cuda", experts=None,
                  include_engram=False):
    """This rank's TP8 view of layer L's weights.

    `experts` restricts the routed experts loaded (None = all 384); the checkpoint
    holds every expert on every rank in pure TP8, sharded only along the
    intermediate axis.
    """
    P = f"layers.{L}."
    m = cfg["moe_inter"] // tp          # 288: intermediate per rank
    d = {}
    r = lambda name: load_tensor(sh, P + name, device)

    # ---- attention: replicated pieces first
    d["wq_a"] = r("attn.wq_a.weight")                    # [1280, 5120] REPLICATED
    d["q_norm"] = r("attn.q_norm.weight")                # [1280]
    d["wkv"] = r("attn.wkv.weight")                      # [512, 5120] REPLICATED
    d["kv_norm"] = r("attn.kv_norm.weight")              # [512]
    d["attn_sink"] = r("attn.attn_sink")                 # [64] -> 8 per rank
    d["attn_norm"] = r("attn_norm.weight")
    d["ffn_norm"] = r("ffn_norm.weight")

    # ---- attention: column/row parallel pieces
    nh = cfg["n_heads"] // tp                            # 8 heads
    d["wq_b"] = load_tensor(sh, P + "attn.wq_b.weight", device,
                            rows=(rank * nh * cfg["head_dim"],
                                  (rank + 1) * nh * cfg["head_dim"]))
    d["wo_a"] = load_tensor(sh, P + "attn.wo_a.weight", device,
                            rows=(rank * cfg["o_lora_rank"],
                                  (rank + 1) * cfg["o_lora_rank"]))
    d["wo_b"] = load_tensor(sh, P + "attn.wo_b.weight", device,
                            cols=(rank * cfg["o_lora_rank"],
                                  (rank + 1) * cfg["o_lora_rank"]))
    d["attn_sink"] = d["attn_sink"][rank * nh:(rank + 1) * nh].contiguous()

    # ---- router (replicated: every rank routes identically)
    d["gate"] = r("ffn.gate.weight")                      # [384, 5120] bf16
    d["gate_bias"] = r("ffn.gate.bias")                   # [384] f32, selection only
    if sh.has(P + "ffn.gate.bias_vl"):
        d["gate_bias_vl"] = r("ffn.gate.bias_vl")

    # ---- shared expert: FP8, intermediate sharded to 288
    d["sh_w1"] = load_tensor(sh, P + "ffn.shared_experts.w1.weight", device,
                             rows=(rank * m, (rank + 1) * m))
    d["sh_w3"] = load_tensor(sh, P + "ffn.shared_experts.w3.weight", device,
                             rows=(rank * m, (rank + 1) * m))
    d["sh_w2"] = load_tensor(sh, P + "ffn.shared_experts.w2.weight", device,
                             cols=(rank * m, (rank + 1) * m))

    # ---- routed experts: MXFP4 packed nibbles, intermediate sharded
    sel = range(cfg["n_routed"]) if experts is None else experts
    gu_w, gu_s, dn_w, dn_s = [], [], [], []
    for e in sel:
        w1 = load_tensor(sh, P + f"ffn.experts.{e}.w1.weight", device,
                         rows=(rank * m, (rank + 1) * m))
        w3 = load_tensor(sh, P + f"ffn.experts.{e}.w3.weight", device,
                         rows=(rank * m, (rank + 1) * m))
        s1 = load_tensor(sh, P + f"ffn.experts.{e}.w1.scale", device,
                         rows=(rank * m, (rank + 1) * m))
        s3 = load_tensor(sh, P + f"ffn.experts.{e}.w3.scale", device,
                         rows=(rank * m, (rank + 1) * m))
        w2 = load_tensor(sh, P + f"ffn.experts.{e}.w2.weight", device,
                         cols=(rank * m // 2, (rank + 1) * m // 2))
        s2 = load_tensor(sh, P + f"ffn.experts.{e}.w2.scale", device,
                         cols=(rank * m // 32, (rank + 1) * m // 32))
        gu_w.append(torch.cat([w1, w3], 0))
        gu_s.append(torch.cat([s1, s3], 0))
        dn_w.append(w2)
        dn_s.append(s2)
    d["gu_w"] = torch.stack(gu_w)
    d["gu_s"] = torch.stack(gu_s)
    d["dn_w"] = torch.stack(dn_w)
    d["dn_s"] = torch.stack(dn_s)

    # ---- mHC (hyper-connections): always present, hc_mult=4 streams
    for nm in ("hc_attn_base", "hc_attn_fn", "hc_attn_scale",
               "hc_ffn_base", "hc_ffn_fn", "hc_ffn_scale"):
        if sh.has(P + nm):
            d[nm] = r(nm)

    # ---- compressor / indexer: only on the source layers
    for nm in ("attn.compressor.wkv.weight", "attn.compressor.norm.weight",
               "attn.compressor.wgate.weight",
               "attn.indexer.wk.weight", "attn.indexer.k_norm.weight",
               "attn.indexer.wq_b.weight", "attn.indexer.weights_proj.weight"):
        if sh.has(P + nm):
            d[nm] = r(nm)

    if include_engram and sh.has(P + "engram.wkv.weight"):
        for nm in ("engram.wkv.weight", "engram.k_weight", "engram.q_weight"):
            d[nm] = r(nm)
        d["has_engram"] = True
    return d


def report(cfg, L=3):
    """Shapes as this rank will see them, next to the checkpoint's own."""
    sh = Shards()
    print(f"layer {L} ({cfg['variants'][L]}), TP8 rank 0 view")
    print(f"{'tensor':28s} {'ckpt':>18s} {'rank0':>18s}")
    spec = [
        ("attn.wq_a.weight", None, None),
        ("attn.wq_b.weight", (0, 4096), None),
        ("attn.wkv.weight", None, None),
        ("attn.wo_a.weight", (0, 1024), None),
        ("attn.wo_b.weight", None, (0, 1024)),
        ("ffn.gate.weight", None, None),
        ("ffn.shared_experts.w1.weight", (0, 288), None),
        ("ffn.shared_experts.w2.weight", None, (0, 288)),
        ("ffn.experts.0.w1.weight", (0, 288), None),
        ("ffn.experts.0.w2.weight", None, (0, 144)),
        ("ffn.experts.0.w1.scale", (0, 288), None),
        ("ffn.experts.0.w2.scale", None, (0, 9)),
    ]
    for nm, rw, cl in spec:
        name = f"layers.{L}.{nm}"
        dt, shp = sh.hdr(name)
        t = load_tensor(sh, name, "cpu", rows=rw, cols=cl)
        print(f"  {nm:26s} {dt:8s} {str(shp):>16s} {str(tuple(t.shape)):>16s} {t.dtype}")
    print()
    print("variants:", {v: cfg["variants"].count(v) for v in set(cfg["variants"])})
    print("engram layers:", cfg["engram_layers"])
    print("collectives per layer: wo_b all_reduce + MoE down all_reduce"
          " (+ indexer all_reduce on index-source layers)")
    print("NO q_lora all_gather: wq_a is replicated")


if __name__ == "__main__":
    cfg = load_config()
    what = sys.argv[1] if len(sys.argv) > 1 else "probe"
    if what == "probe":
        report(cfg, int(sys.argv[2]) if len(sys.argv) > 2 else 3)
    elif what == "check":
        # per-layer byte footprint, to confirm 40 layers fit
        sh = Shards()
        tot = 0
        for L in (3, 2, 1):
            d = layer_tensors(sh, cfg, L, 0, device="cpu", include_engram=True)
            b = sum(v.numel() * v.element_size() for v in d.values()
                    if isinstance(v, torch.Tensor))
            print(f"  L{L} ({cfg['variants'][L]:9s}): {b/2**20:8.1f} MiB/rank")
            tot += b
        print(f"  40 layers, all reuse-sized: {tot*40/3/2**30:.1f} GiB/rank")
