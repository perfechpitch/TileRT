# DeepSeek-V4.1-Flash TP8 PDL 链 —— Findings

日期：2026-11-26
代码：`bench/dsv41_tp8/`（`dsv41_pdl_layer.py` 单层、`dsv41_pdl_e2e_tp8.py` TP8 40 层、
`dsv41_real_weights.py` 真实权重加载器）
硬件：B300 SM103a × 8

---

## 0. 一句话结论

真实 40 层、TP8、CUDA graph：**8092 µs/step = 123.6 tok/s**。
拆开是 **117 µs/层纯计算 + 85 µs/层通信**。

之前报的 304.5 tok/s（82.1 µs×40）是**单层、单卡、零通信、L2 全热**的数字，×40 外推是错的。
真实瓶颈不是算力也不是 PDL，是 **20 KB 的 all_reduce 每次 31 µs**。

---

## 1. 结论速览

| 项 | 状态 |
|---|---|
| 真实 checkpoint TP8 加载器 | ✅ 48 shard 直读，按模型自己的分片规则，40 层 35.0 GiB/GPU 实测 |
| 40 层结构查清 | ✅ 5 种 attention 变体 + Engram(L1,L14) + mHC(hc_mult=4) |
| router 修正（gate bias） | ✅ 单 launch / PDL 链均 rel=1.702e-07，专家 6/6 |
| 共享专家 FP4→FP8 | ✅ 单 launch / PDL 链均 rel=2.428e-07 |
| 去掉 q_lora all_gather | ✅ 18 kernel → 2 个 PDL cluster |
| TP8 40 层 e2e（graph） | ✅ 跑通，8092 µs，123.6 tok/s |
| **5 种 attention 变体** | ❌ 只有 `reuse` 一种（30/40 层） |
| **mHC** | ❌ 完全未建模 |
| **Engram** | ❌ 未建模（权重已能加载） |
| **通信替代（GIN/multimem）** | ❌ 未做，当前是最大单项 |

---

## 2. 真实模型每层结构（**不能复用**）

从 `models/DeepSeek-V4.1-Flash-1017/config.json` 的 `text_config` 直读：

```
num_hidden_layers      = 40
compress_ratios        = [0, 0, 2 x18, 1 x20]     # 数组 43 项，只用前 40
engram_layer_ids       = [1, 14]
kv_source_layer_ids    = [2, 8, 14, 20]
index_source_layer_ids = [2, 8, 14, 20, 24, 28, 32, 36]
candidate_source_layer_id = 20
sliding_window = 128   index_topk = 512   hc_mult = 4
candidate_topk_blocks = 2048   candidate_block_size = 8
```

按 `mirage/demo/deepseek_v4_1_csa2/weights.py:63` 的规则展开：

| 变体 | 层 | 数量 | 内容 |
|---|---|---|---|
| `swa` | L0, L1 | 2 | 只有滑窗 128 key，无压缩 KV、无 indexer |
| `full_r2` | L2, L8, L14 | 3 | + compressor(`wkv`/`wgate`/池化) + indexer |
| `full_r1` | L20 | 1 | + compressor(无 `wgate`，ratio 1 是纯投影) + indexer |
| `reindex` | L24, 28, 32, 36 | 4 | + indexer（复用源的 compressor 输出） |
| `reuse` | 其余 | 30 | 读源层发布好的压缩 KV 与 top-k |
| +Engram | L1, L14 | 2 | n-gram 查表写进残差 |

现有 18-kernel 链读 `128 window + 512 compressed = 640` key，正好是
`sliding_window + index_topk`，即 **`reuse` 变体**——40 层里最便宜的 30 层。

**MoE 部分是 40 层同构的**（`compress_ratios` 只管 attention，MLP 无 per-layer 字段），
所以 MoE 的结论可以推全体；attention 不行。

---

## 3. kernel 与真实模型的 6 处差异

| # | 差异 | 量级影响 | 状态 |
|---|---|---|---|
| 1 | 共享专家是 **FP8** (`F8_E4M3 [2304,5120]` 11.8 MB)，不是 fp4 的第 7 个专家 (5.9 MB) | gate_up 12.11→12.79 µs | ✅ 已修 |
| 2 | **`wq_a` 是复制的**，无 q_lora all_gather；只有 `wq_b` 是 ColumnParallel | 每层少一次 collective | ✅ 已修 |
| 3 | `ffn.gate.bias` **参与选专家**：`(score+bias).topk`，但权重取无 bias 的 score | 近 tie 时专家集不同 | ✅ 已修 |
| 4 | router 跑 **fp32**（bf16 权重先升 fp32），每层 7.86 MB 而非 3.93 MB | router 3.59 µs | ✅ 已修 |
| 5 | **mHC 缺失**：`hc_mult=4`，`mix_hc=24`、`hc_dim=20480`（对上 `hc_attn_fn [24,20480]`） | 每层 2 个 1.97 MB f32 系数 GEMV + 20 次 Sinkhorn | ❌ 未做 |
| 6 | **压缩率 2 的 compressor 隔层才产出**（组没填满返回 None，状态跨 step 携带） | `full_r2` 层一半 step 才做 | ❌ 未做 |

TP8 真实分片（`inference/model.py`）：

| tensor | checkpoint | 本 rank | 切法 |
|---|---|---|---|
| `wq_a` | `[1280,5120]` fp8 | `[1280,5120]` | **复制** |
| `wq_b` | `[32768,1280]` fp8 | `[4096,1280]` | column，64 头取 8 |
| `wkv` | `[512,5120]` fp8 | `[512,5120]` | **复制**（1 个 KV 头） |
| `wo_a` | `[8192,4096]` fp8 | `[1024,4096]` | column，8 个 o_group 取 1 |
| `wo_b` | `[5120,8192]` fp8 | `[5120,1024]` | row，部分和 |
| `experts.w1/w3` | `[2304,2560]` fp4 | `[288,2560]` | 中间维 2304/8 |
| `shared.w1/w3` | `[2304,5120]` **fp8** | `[288,5120]` | 中间维 2304/8 |

**每层 collective：2 次 all_reduce**（`wo_b`、MoE `w2`），另有 8 个 index-source 层多一次
indexer 的 all_reduce；Engram 层（L1/L14）还有一次词表 all_reduce。
**没有** 每层 all_gather。

实测 footprint/rank：`reuse` 895.1 MiB、`full_r2` 910.6 MiB、`swa`+Engram 1045.2 MiB
（不含 Engram 表，`engram.embed` 单个 94.5 GiB、按词表切到 8 卡约 11.8 GiB/rank/层）。

---

## 4. 性能分解（同一 harness 的 A/B，可信）

全部来自 `dsv41_pdl_e2e_tp8.py bench`，TP8、8 卡、40 层、35.0 GiB/GPU：

| 配置 | step | 每层 | tok/s |
|---|---|---|---|
| eager，带通信 | 10359.27 µs | 258.98 | 96.5 |
| **graph，带通信** | **8092.06 µs** | **202.30** | **123.6** |
| eager，`DSV41_NOCOMM=1` | 6277.99 µs | 156.95 | 159.3 |
| graph，`DSV41_NOCOMM=1` | 4686.05 µs | 117.15 | 213.4 |

由同一 harness 相减得到的两项开销：

- **通信：202.30 − 117.15 = 85.15 µs/层**（42%）
- **eager→graph：258.98 − 202.30 = 56.7 µs/层**（22%），即暴露出来的 CPU launch
  开销，**3.15 µs/次 × 18 个 kernel**。720 次 launch 在单线程 Python 里发不完，
  graph 把它们收成一次 replay。

> 注意这 3.15 µs 是**暴露**的部分。eager 下 launch 是异步的，CPU 的发射时间能和
> GPU 计算重叠，所以单次 launch 的真实 CPU 成本（更高）大部分被掩盖了。
> 我早先说的"9 µs/kernel"是拿单卡 82 µs 和 40 层 eager 244 µs 相减得出的，
> 那是跨配置相减（§6.3），**不成立**，已作废。

单 launch 版本（`dsv41_layer_megakernel.py`）对照：96.98 µs/层。
PDL 链单层、零通信：**82.1 µs**（vs 无 PDL 85.2 µs，三次运行 85.74/82.07、84.56/82.11、85.47/82.11）。

> ⚠️ 82.1 是**单 GPU、单层、无 collective、L2 全热**的数字，**不能**和上表的
> 117.15（8 卡、40 层）相减。见 §6.3。

分段（PDL 链，单层，µs）：norm1 1.25+2.81、wq_a 6.72、normQL 1.73+2.78、wq_b 4.54、
mla split 11.90、mla combine 11.55、wo_a 5.45、wo_b 4.34、norm2 1.53+2.63、
router 3.59、topk 4.91、gate_up 12.79、swiglu 1.86、down 11.09。

---

## 5. 通信延迟实测（8 卡，20 KB 是关键点）

`torch.distributed all_reduce`，B300 × 8：

| bytes | eager µs | graph µs |
|---|---|---|
| 5 120 | 27.01 | 26.28 |
| **20 480** | **31.45** | **28.76** |
| 81 920 | 31.66 | 30.81 |
| 1 048 576 | 31.91 | 32.72 |
| 8 388 608 | 84.19 | 81.39 |

结论：

1. **纯延迟受限**：5 KB 到 1 MB 全在 27–33 µs，带宽完全没用上。20 KB 的有效带宽约 0.3 GB/s。
2. **graph 对通信几乎无改善**（−2.7 µs）。NCCL kernel 本来就在 device 侧排队，
   PDL/graph 都碰不到这段。
3. 每层 2 次 → ~62 µs 理论，实测吃到 85 µs（多出的部分是它把 PDL 链截成三段、
   流水重新暖机的代价）。

**这决定了优化顺序：通信替代（GIN `multimem.red` / 对称内存）的收益远大于继续调 PDL。**
按 85 → 10 µs 估算，整模型 202 → ~127 µs/层，即 123.6 → ~197 tok/s。

---

## 6. 不要重蹈覆辙（方法论，逐条都有实测代价）

### 6.1 单层数字不能当模型 tok/s
**代价**：报了 "304.5 tok/s"，实际 123.6。
**规则**：报 tok/s 必须同时写明 `{层数, 卡数, graph/eager, 通信开关, L2 冷热}`。
单层 ×40 只在外推理想值，不能当结论。README 里旧的那行已改。

### 6.2 必须有 A/B，不要只报一个数
**做法**：`DSV41_NOCOMM=1` 开关（`dsv41_pdl_e2e_tp8.py`）+ `eager` / `bench` 两种模式。
有了 A/B 才能把 202 µs 拆成 117+85，否则会误以为是算力问题去优化 kernel。

### 6.3 单卡基线不能和 TP8 基线相减
**陷阱**：单卡跑 `wq_b` 是全宽 32768 行、`wo_a` 全宽 8192；TP8 每 rank 只有 4096/1024 行。
两者"每层计算量"根本不同，82.1 − 117.15 这种减法没有意义。
**规则**：跨 harness 只比趋势，绝对分解只用同一 harness 内部 A/B。
注意两个 harness 现在都用全宽 `wq_a`（`QLR`），这一项是对齐的；`wq_b`/`wo_a` 不是。

### 6.4 管道 grep 会缓冲，会伪造成"卡死"
**代价**：好几次判定 "hang"，实际是 `| grep` 的全缓冲把输出吞了，看不到进度。
**规则**：长任务一律 `setsid nohup ... > /tmp/x.log 2>&1 < /dev/null &`，
然后 `grep -a` 读文件。别在长跑命令后面接管道。

### 6.5 CUDA graph 捕获 NCCL 前必须在 side stream 预热
**代价**：4 层和 40 层两次"死锁"，GPU 利用率 3%、显存已占，看起来像 capture 挂了。
**根因**：default stream 上预热过的 communicator 在 capture 区内不会被注册，
第一个 collective 处挂住。用独立探针（`dsv41_graph_probe2.py`，逐级加 PDL kernel /
1 个 / 2 个 all_reduce）证明 **capture、PDL、NCCL 单独都没问题**，才定位到预热。
**做法**：
```python
side = torch.cuda.Stream()
side.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(side):
    for _ in range(3): run_step(...)
torch.cuda.current_stream().wait_stream(side)
```
**规则**：先做最小可复现探针逐级加码，再改主代码；不要一次改多处。

### 6.6 safetensors header 偏移不能重新序列化
**代价**：`RuntimeError: shape '[2304, 2560]' is invalid for input of size 5877248`。
**根因**：用 `len(json.dumps(解析后的 header))` 算 header 长度，key 顺序/空格与文件里不同，
数据区偏移差几千字节，每个 tensor 都被截断。
**规则**：`n = struct.unpack("<Q", f.read(8))[0]`，然后 `base = 8 + n`，原样使用。

### 6.7 `uint32` 视图要按字节数分配
**代价**：`shape '[385, 576, 640]' is invalid for input of size 35481600`。
**根因**：`uint8 → view(int32) → view(uint32)` 每步字节数不变，所以分配字节数必须是
字数的 **4 倍**：`_u8((N, M, (K//8)*4)).view(torch.int32).view(torch.uint32).view(N, M, K//8)`。

### 6.8 TileLang：变量不能定义在 runtime `if` 分支里给兄弟分支用
**代价**：`if s_ < TOPK: eid = ...` 之后 `else:` 分支看不到 `eid`。
**规则**：用 `T.if_then_else` 在分支外提升（`min()` 兜住两端防止越界）。

### 6.9 TileLang：`tilelang.compile` 按 `global_symbol` 缓存
**代价**：17 个 kernel 全叫 `main`，编译返回第一个（4 参数），报
`Kernel expected 4 inputs, but 2 are provided`。
**规则**：每个 prim_func 用 `.with_attr("global_symbol", f"pdl_{i}")` 取唯一符号。

### 6.10 TP8 check 时，K 维分片的权重在 rank>0 上要清零
**代价**：`rel=2.583e-01 MISMATCH`。
**根因**：check 让 all_reduce 只等于 rank 0 的分片，把 rank>0 的
`WOBW/SOB/WDN/SDN` 清零；共享专家换成 FP8 独立张量后没加进清零列表，
共享专家被累加 8 次。
**规则**：新增任何走 all_reduce 的 K 维分片权重，必须同步加进清零列表
（现在是 `WOBW, SOB, WDN, SDN, SHGW, SHDW`）。

### 6.11 共享专家的字节量不能当普通专家
fp8 共享专家是 routed fp4 专家的 **2 倍字节**（11.8 vs 5.9 MB/矩阵）。
把它当"第 7 个 fp4 专家"会让 gate_up 少算一半流量。

---

## 7. 未完成 / 下一步

1. **通信替代**（最大单项，85 µs/层）。用之前验证过的 `multimem.red.*` 走 driver API
   建 multicast object + 内联 PTX，替换两个 `all_reduce`。目标 85 → ~10 µs。
   *注：`griddepcontrol.launch_dependents` 只对同 stream 生效，NCCL kernel 不发，
   所以只要还有 collective，PDL 链就必然被截断——换成非 NCCL 的聚合 kernel 后，
   这一段才有可能接回同一条 PDL 链。*
2. **5 种 attention 变体**（`swa`/`full_r2`/`full_r1`/`reindex`）。`swa` 只读 128 key，
   当前按 640 算会**高估**约 35 µs/层；`full_r*`/`reindex` 需要新 stage
   （`compress_kv` 写 / indexer 打分 512 of 2048 候选块），会**低估**。
3. **mHC**（`hc_mult=4`）：每层 2 个系数 GEMV + 20 次 Sinkhorn，纯缺项，无抵消对象。
4. **Engram**（L1/L14）：权重已能加载（`include_engram=True`），kernel 未写；
   自带一次词表 all_reduce。
5. 用 `dsv41_real_weights.py` 替换 e2e 里的随机权重，做一次真实权重的端到端数值校验。
