# GPU / 依赖基准与存储规划 (PR-01)

**测量日期 (UTC)**：2026-11-20（本地 2026-11-21 01:5x，Asia/Shanghai）  
**主机**：`ccpc`，WSL2（`Linux-6.6.87.2-microsoft-standard-WSL2-x86_64`）  
**产物**：`artifacts/riichiformer/bench/gpu_baseline.json`（gitignored），SHA-256 `db844fb30c5b365b94abe49928abf4ec6ebf0133b1ad5c740b1640963983cc37`  
**配置**：`configs/riichiformer/benchmark.yaml`  
**复现**：`.venv/bin/python scripts/rf_bench.py --config configs/riichiformer/benchmark.yaml --out artifacts/riichiformer/bench/gpu_baseline.json`

> 这是**通用 causal transformer block 的算力/显存容量测量**，不是 V2 模型，也不是对局强度或数据结论。
> 所有数值都是真实运行结果；OOM 会记录错误与显存，不会写成 0 ms。

---

## 1. 设备与运行时（实测）

| 项 | 值 | 来源 |
|---|---|---|
| GPU | NVIDIA GeForce RTX 5070 Ti | torch + nvidia-smi |
| UUID | `GPU-349f95f5-b252-bdea-1895-5fb0cbdceea7` | 两者一致 |
| Compute capability | `12.0`（sm_120，Blackwell） | torch props |
| SM 数 | 70 | torch props |
| VRAM（torch） | 17,066,360,832 B = 16,276 MiB | `torch.cuda.get_device_properties` |
| VRAM（nvidia-smi total） | 16,303 MiB | `/usr/lib/wsl/lib/nvidia-smi` |
| Driver | 621.01 | nvidia-smi |
| PyTorch / CUDA / cuDNN | `2.11.0+cu128` / `12.8` / `91900` (9.19.0.56) | torch |
| BF16 | **支持**（`is_bf16_supported=True`） | torch |
| BF16 reduced-precision reduction | `True`（默认） | torch |
| Python | 3.12.3（`.venv/bin/python`） | — |
| NumPy / PyYAML | 2.5.2 / 6.0.3 | — |
| CPU / RAM | 14 threads / 15.6 GiB | `/proc/cpuinfo`、`/proc/meminfo` |
| WSL | 是；`/dev/shm` 7.9 GiB（测量时 7.8 GiB free） | `df /dev/shm` |
| multiprocessing start method | `fork` | torch |
| git HEAD | `aceb281602fb2703a1ed2067a006992cd9ddb680` | `git rev-parse` |

**共享 GPU 的事实（必须保留）**：测量时 `nvidia-smi` 报告 `memory_used≈4.8–5.0 GiB`、利用率一度 12%，
但 `torch.cuda.mem_get_info()` 报告 free=15,009 MiB / used=1,267 MiB。两者不一致是 WSL 的
GPU 记账差异（torch 看不到部分 WSL 侧占用）。因此：

- 本次数字是在**非独占 GPU**上取得的，可能偏保守/有噪声；
- 正式容量规划（PR-17）必须在持锁、安静 GPU 上重跑；
- 不得把本次结果当作“5070 Ti 独占性能”对外表述。

---

## 2. 基准结果（BF16 autocast，warmup=10，measure=30）

模型：2 层 pre-norm transformer，`n_heads=4`，`ffn=4×d`，`F.scaled_dot_product_attention`
（causal），AdamW(lr=1e-4, wd=0.01)。`params` 是该合成模型参数量，不是 V2 规模。

| shape (B×L×d) | params | forward ms | train ms | step ms | train tok/s | step tok/s | peak alloc MiB | peak reserved MiB | step p95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8×512×256 | 1,640,960 | 1.107 | 3.349 | 3.357 | 1,223,028 | 1,220,090 | 140.31 | 174.0 | 5.217 |
| 4×1024×384 | 3,690,240 | 1.626 | 5.512 | 5.509 | 743,116 | 743,486 | 219.39 | 264.0 | 8.29 |
| 2×2048×384 | 3,690,240 | 1.631 | 5.513 | 6.32 | 743,018 | 648,141 | 219.39 | 264.0 | 10.036 |
| 1×4096×384 | 3,690,240 | 1.983 | 5.995 | 6.237 | 683,211 | 656,718 | 219.39 | 264.0 | 6.824 |

判读：

- **4/4 shape 的 forward / train / step 全部 PASS，无 OOM、无 NaN**；
- 峰值 allocated ≤ 219.4 MiB、reserved ≤ 264.0 MiB —— 但这只是 **2 层、无词表 embedding、
  无 per-game batch、无 KV cache** 的合成模型，**不能**线性外推到 Main-H/Flat-F；
- `train` 与 `step` 的 ms 接近，说明 AdamW 在 warmup 后已分配好状态，测量段没有额外分配尖峰；
- B=1×4096 的 step p95 反而低于 2×2048，属于共享 GPU 上的抖动，不应当作规律；
- 吞吐数字仅用于“同一机器上相对比较”，不能换算成“每天多少 labeled decisions”。

---

## 3. 存储预算（只读统计，未删任何 zip）

| 项 | 数值 | 备注 |
|---|---:|---|
| 年度 zip | 18 个 / 12,777,084,723 B（11.90 GiB） | 2009–2026；保留原文件 |
| 已解压 `.mjson` | 12,282,818,151 B（11.44 GiB） | `datasets/extracted`；约 250 万文件 |
| `datasets/` 合计 | 23.34 GiB（25,059,903,789 B） | 含 zip + extracted + 其他 |
| `checkpoints/` | 0.89 GiB（954,119,705 B，8 文件） | 含 6 个权重 + 2 个 index 缓存 |
| `artifacts/riichiformer/` | 38,306 B（preflight.json） | 新产物目录 |
| 磁盘可用 / 总量 | 772.3 GiB / 1,006.5 GiB | 挂载点 `/`（同一卷） |

风险：AGENTS 记录的预期是“12.78 GB zips → 40–60 GB extracted”，而本机只有 11.44 GiB extracted；
说明解压可能不完整（或数据版本不同）。**PR-02 必须以本地扫描为准，不得用文档数字替代**；
在审计清楚前不要删除 zip、不要假设 2009–2026 全部可用。

---

## 4. WSL / Windows dataloader 注意事项（实测 + 约定）

1. 当前 `torch.multiprocessing` 启动方式是 **`fork`**（Linux/WSL），不是 Windows 的 `spawn`；
   迁移到原生 Windows 时必须显式 `spawn` + `if __name__ == "__main__":` 入口保护，否则 worker 会重复执行主模块。
2. **worker 里不要初始化 CUDA**。`fork` 出来的 worker 若在父进程已建 CUDA context 后触碰 CUDA，
   会继承上下文并可能挂死/报错；数据加载只在 CPU，collate 后再 `.to(device, non_blocking=True)`。
3. `/dev/shm` 只有 7.9 GiB；`num_workers` 提高 + `prefetch_factor` 过大时容易触发
   `DataLoader worker (pid ...) is killed by signal: Bus error` 或共享内存不足。
   首版建议 `num_workers=0..2`、`persistent_workers=True`、`prefetch_factor=2`，实测后再加。
4. 不要把 `pin_memory` 和 `cudaMallocAsync` 当默认；本机没有设置 allocator 环境变量，保持默认。
5. **`torch.compile` 不是硬依赖**：本次 benchmark 未启用；若后续启用，必须单独记录首次编译时间
   与 peak VRAM，不能混入 steady-state 数字。
6. WSL 下用绝对路径（本项目 config 已全绝对路径）；不要在 worker 内 `os.chdir`。

---

## 5. 依赖固定建议

```text
python==3.12.3
torch==2.11.0+cu128        # 官方 wheel；CUDA runtime 12.8
numpy==2.5.2
PyYAML==6.0.3              # 仅 rf_* YAML 配置需要
pytest==9.2.0              # 仅开发/测试
```

`torch.compile`、FlashAttention、Triton 均**不**是当前依赖；任何启用都要单独 PR + 单独测
编译时间/显存/正确性。

---

## 6. 本 PR 的边界与下一关

- 本文件只证明“在这台机器上，4 个候选 shape 的 forward/train/step 能跑通并记录真实显存”。
- **G3 仍未通过**；真正的 MVP-S 训练显存/吞吐由 PR-13/PR-14/PR-17 重新测量。
- PR-01 PASS 的唯一含义是：GPU 环境可测、benchmark 工具可复现、失败会显式记录、磁盘预算已知。
