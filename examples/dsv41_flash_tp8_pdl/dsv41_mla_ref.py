#!/usr/bin/env python3
"""DSV4.1-Flash decode-attention reference (dual KV), from the shipped encodings.

Geometry taken from `sgl_nvidia_mega/deepseek_v41`:
  HF_CONFIG        hidden 5120, head_dim 512, q_lora_rank 1280, sliding_window 128
  sm103/attention/kv_layout.py + core_lean.py
      window row     528 B = 512 E4M3 values + 16 UE8M0 scales (1 per 32 values)
      compressed row 288 B = 512 E2M1 values (256 B) + 32 E4M3 scales (1 per 16)
  core_lean        TOP_K 512 compressed keys, HEADS 8 (TP8-local), HEAD_DIM 512

Note the two scale formats are NOT interchangeable:
  window      UE8M0,   granularity 32   -> matches block-scaled MMA SF layout
  compressed  E4M3,    granularity 16   -> NVFP4 style, needs on-the-fly dequant

This module is the correctness oracle. It is deliberately written in plain torch
so the TileLang kernel can be checked digit-for-digit against it.
"""

from __future__ import annotations

import torch

HEAD_DIM = 512
NOPE_DIM = 448          # first 448 dims, no positional encoding
ROPE_DIM = 64           # last 64 dims carry RoPE
WINDOW = 128            # sliding-window keys
TOP_K = 512             # compressed / sparse keys
HEADS = 8               # TP8-local heads

WIN_DATA_B = 512
WIN_SCALE_B = 16
WIN_ROW_B = WIN_DATA_B + WIN_SCALE_B        # 528
WIN_GRAN = 32

CMP_DATA_B = 256
CMP_SCALE_B = 32
CMP_ROW_B = CMP_DATA_B + CMP_SCALE_B        # 288
CMP_GRAN = 16

E2M1 = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])


# ------------------------------------------------------------------ quantizers
def _to_e2m1(x: torch.Tensor) -> torch.Tensor:
    """Nearest e2m1 code index (0..15) for each element."""
    tbl = E2M1.to(x.device)
    d = (x.unsqueeze(-1) - tbl).abs()
    return d.argmin(-1)


def _to_e4m3_scale(x: torch.Tensor) -> torch.Tensor:
    """Round a positive scale to the e4m3 grid (as an fp32 value)."""
    return x.to(torch.float8_e4m3fn).float()


def quant_window_row(k: torch.Tensor) -> torch.Tensor:
    """k: [HEAD_DIM] fp32 -> 528-byte row (uint8), UE8M0 scale per 32 values."""
    assert k.shape[-1] == HEAD_DIM
    blocks = k.view(WIN_GRAN, -1)  # this is just for the scale sweep below
    amax = k.abs().view(-1, WIN_GRAN).amax(-1)
    # UE8M0: power-of-two scale so that amax / scale <= 448 (max e4m3)
    exp = torch.ceil(torch.log2(torch.clamp(amax, min=1e-30) / 448.0))
    ue8 = torch.clamp(exp + 127, 0, 255).to(torch.uint8)
    scale = torch.pow(2.0, (ue8.float() - 127))
    val = (k.view(-1, WIN_GRAN) / scale[:, None]).reshape(HEAD_DIM)
    data = val.to(torch.float8_e4m3fn).view(torch.uint8)
    return torch.cat([data, ue8]).contiguous()


def dequant_window_row(row: torch.Tensor) -> torch.Tensor:
    """528-byte row -> [HEAD_DIM] fp32."""
    data = row[:WIN_DATA_B].view(torch.float8_e4m3fn).float()
    ue8 = row[WIN_DATA_B:WIN_DATA_B + WIN_SCALE_B].float()
    scale = torch.pow(2.0, ue8 - 127).repeat_interleave(WIN_GRAN)
    return data * scale


def quant_compressed_row(k: torch.Tensor) -> torch.Tensor:
    """k: [HEAD_DIM] fp32 -> 288-byte row, E4M3 scale per 16 values (NVFP4 style)."""
    assert k.shape[-1] == HEAD_DIM
    amax = k.abs().view(-1, CMP_GRAN).amax(-1)
    scale = _to_e4m3_scale(torch.clamp(amax / 6.0, min=1e-30))
    q = (k.view(-1, CMP_GRAN) / scale[:, None]).reshape(HEAD_DIM)
    idx = _to_e2m1(q).to(torch.uint8)
    packed = (idx[0::2] | (idx[1::2] << 4)).to(torch.uint8)
    return torch.cat([packed, scale.to(torch.float8_e4m3fn).view(torch.uint8)])


def dequant_compressed_row(row: torch.Tensor) -> torch.Tensor:
    """288-byte row -> [HEAD_DIM] fp32."""
    p = row[:CMP_DATA_B]
    idx = torch.empty(CMP_DATA_B * 2, dtype=torch.long, device=row.device)
    idx[0::2] = (p & 0x0F).long()
    idx[1::2] = (p >> 4).long()
    vals = E2M1.to(row.device)[idx]
    scale = row[CMP_DATA_B:CMP_DATA_B + CMP_SCALE_B].view(
        torch.float8_e4m3fn).float()
    return vals * scale.repeat_interleave(CMP_GRAN)


# --------------------------------------------------------------------- forward
def rope(q: torch.Tensor, positions: torch.Tensor, theta: float = 50000.0):
    """Rotate the trailing ROPE_DIM dims in place-ish (returns a new tensor)."""
    d = ROPE_DIM
    half = d // 2
    inv = theta ** (-torch.arange(0, half, device=q.device, dtype=torch.float32)
                    * 2.0 / d)
    ang = positions.float().view(-1, 1) * inv.view(1, -1)     # [rows, half]
    cos, sin = ang.cos(), ang.sin()
    x = q[..., NOPE_DIM:]
    x1, x2 = x[..., :half], x[..., half:]
    rot = torch.cat([x1 * cos.unsqueeze(1) - x2 * sin.unsqueeze(1),
                     x1 * sin.unsqueeze(1) + x2 * cos.unsqueeze(1)], dim=-1)
    return torch.cat([q[..., :NOPE_DIM], rot], dim=-1)


def inverse_rope(o, position: int, theta: float = 10000.0):
    """Inverse-rotate the trailing ROPE_DIM dims at one position (decode)."""
    d = ROPE_DIM
    half = d // 2
    inv = theta ** (-torch.arange(0, half, device=o.device, dtype=torch.float32)
                    * 2.0 / d)
    ang = float(position) * inv
    cos, sin = ang.cos(), ang.sin()
    x = o[..., NOPE_DIM:]
    x1, x2 = x[..., :half], x[..., half:]
    rot = torch.cat([x1 * cos + x2 * sin,
                     -x1 * sin + x2 * cos], dim=-1)
    return torch.cat([o[..., :NOPE_DIM], rot], dim=-1)


def decode_attention_ref(q, win_rows, cmp_rows, out_dtype=torch.float32,
                         scale=None, sink=None, position=None, theta=10000.0):
    """Reference DSV4.1 decode attention.

    q:        [rows, HEADS, HEAD_DIM] fp32, RoPE already applied
    win_rows: [rows, NW, WIN_ROW_B] uint8   (already gathered per row)
    cmp_rows: [rows, NC, CMP_ROW_B] uint8 or None
    scale:    softmax scale; None -> 1/sqrt(HEAD_DIM) (the model's value)
    sink:     [heads] per-head attention-sink logit, or None
    position: if given, apply the inverse RoPE of this position to the output
    returns:  [rows, HEADS, HEAD_DIM] fp32
    """
    rows, heads, _ = q.shape
    dev = q.device
    if scale is None:
        scale = HEAD_DIM ** -0.5
    K_win = torch.stack([torch.stack([dequant_window_row(r) for r in row])
                         for row in win_rows])           # [rows, NW, D]
    keys = [K_win]
    if cmp_rows is not None:
        K_cmp = torch.stack([torch.stack([dequant_compressed_row(r) for r in row])
                             for row in cmp_rows])
        keys.append(K_cmp)
    kv = torch.cat(keys, dim=1)                          # [rows, N, D]

    # scores = q @ kv^T, K and V are the same rows in DSV4.1 (head_dim shared)
    scores = torch.einsum("rhd,rnd->rhn", q, kv) * scale

    if sink is None:
        p = torch.softmax(scores, dim=-1)
        out = torch.einsum("rhn,rnd->rhd", p, kv)
    else:
        # sink is an extra logit in the denominator only, folded into the max
        sk = sink.view(1, heads, 1)
        m = torch.maximum(scores.amax(-1, keepdim=True), sk)
        e = torch.exp(scores - m)
        den = e.sum(-1, keepdim=True) + torch.exp(sk - m)
        out = torch.einsum("rhn,rnd->rhd", e / den, kv)

    if position is not None:
        out = inverse_rope(out, position, theta)
    return out.to(out_dtype)


if __name__ == "__main__":
    torch.manual_seed(0)
    dev = "cuda"
    D = HEAD_DIM

    k = torch.randn(D, device=dev)
    w = quant_window_row(k)
    dq = dequant_window_row(w)
    print(f"window row: {w.numel()} B (expect {WIN_ROW_B})  "
          f"roundtrip rel err {((dq - k).norm() / k.norm()).item():.4f}")

    c = torch.randn(D, device=dev)
    cr = quant_compressed_row(c)
    dqc = dequant_compressed_row(cr)
    print(f"compressed row: {cr.numel()} B (expect {CMP_ROW_B})  "
          f"roundtrip rel err {((dqc - c).norm() / c.norm()).item():.4f}")

    rows, NW, NC = 1, WINDOW, 64
    q = torch.randn(rows, HEADS, D, device=dev)
    q = rope(q, torch.tensor([7], device=dev))
    win = torch.stack([torch.stack([quant_window_row(torch.randn(D, device=dev))
                                    for _ in range(NW)]) for _ in range(rows)])
    cmp_ = torch.stack([torch.stack([quant_compressed_row(torch.randn(D, device=dev))
                                     for _ in range(NC)]) for _ in range(rows)])
    out = decode_attention_ref(q, win, cmp_)
    print(f"decode out: {tuple(out.shape)}  absmax {out.abs().max().item():.3f}")
