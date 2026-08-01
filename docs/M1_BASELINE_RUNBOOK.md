# M1 基线运行手册

## 1. 本阶段目标

在目标 RTX 5070 上建立第一份可用于后续对照的 BF16 基线。M1 不评价 Prefix Cache、FP8 KV 或 LoRA，只回答输入长度、并发度和基础容量的关系。

完成后应得到：

- 一个通过 G0 的环境快照；
- 一个通过 E00 的 smoke 结果；
- 12 个 profile、每个 3 次重复的原始 vLLM JSON；
- 每次重复对应的 GPU telemetry CSV；
- manifest 驱动生成的 `runs.csv` 和 `summary.md`；
- 对异常轮次和失败配置的书面记录。

## 2. 实验前准备

1. 使用 Windows 侧最新 NVIDIA 驱动，WSL2 内不要安装 Linux 显卡驱动。
2. 仓库应位于 WSL2 的 Linux 文件系统，不放在 `/mnt/c` 下。
3. 在第一轮实验前提交 Git commit，确保环境 manifest 可以记录 commit。
4. 关闭游戏、浏览器硬件加速任务和其他 GPU 推理程序。
5. 记录 Windows 显示器连接状态；整个矩阵期间不要改变显示配置。

初始化：

```bash
bash scripts/bootstrap_wsl.sh
source .venv/bin/activate
make test
make doctor
```

只有 `linux`、`wsl2`、`nvidia-smi`、`target-gpu`、`gpu-memory`、`vllm` 和 `torch-cuda` 全部 PASS 才通过 G0。

若 WSL2 可访问 PyPI 但无法访问 `huggingface.co`，不更改 vLLM 或 CUDA 环境。改用 Qwen 官方 ModelScope 源下载本地快照：

```bash
make download-model-modelscope
```

完成后必须存在 `~/models/Qwen2.5-3B-Instruct/SHA256SUMS`。保留下载来源、该文件及其 SHA-256，并在整个 M1 矩阵中使用同一快照。

## 3. 启动受控基线

渲染并检查命令：

```bash
make render-baseline
```

启动参数必须包含：

```text
revision=a1d308dfcc03e09da285d49d912439a655a571e8
dtype=bfloat16
max_model_len=8192
gpu_memory_utilization=0.82
max_num_seqs=8
max_num_batched_tokens=8192
kv_cache_dtype=bfloat16
prefix_caching=false
```

使用证据采集入口启动：

```bash
make serve-baseline
```

使用 ModelScope 本地快照时，对应命令为：

```bash
make render-baseline-local
make serve-baseline-local
```

本地快照启动会省略 Hub `--revision`，但 server manifest 仍保留规范模型 ID、规范 revision、本地绝对路径与 `SHA256SUMS` 指纹。对外 `served_model_name` 不变，因此 benchmark 命令不变。

RTX 5070 在 WSL2 上使用 vLLM V2 Model Runner 时需要 UVA。两个受控 serve profile 都显式设置 `wsl2_enable_pin_memory=true`，运行命令对应 `VLLM_WSL2_ENABLE_PIN_MEMORY=1`。该值必须出现在 server manifest 的 `environment_overrides` 中；正式矩阵不得在轮次间改变。

vLLM 0.25.1 所带 FlashInfer 采样器无法正确识别 RTX 5070 的 `sm120`，会在模型加载和 CUDA Graph 捕获完成后错误报告 `FlashInfer requires GPUs with sm75 or higher`。两个 profile 固定设置 `use_flashinfer_sampler=false`，对应 `VLLM_USE_FLASHINFER_SAMPLER=0`，使 top-k/top-p 采样使用 PyTorch 原生实现。它是平台兼容条件，不是实验优化变量，正式矩阵中不得改变。

它会把环境、配置哈希和命令写入 `artifacts/env`，完整启动日志写入 `artifacts/server`。若 OOM，先保留失败 manifest 和日志，再按照可行性文档中的顺序降低资源参数；修改后必须创建新的 server profile，不能覆盖 baseline。

## 4. E00 Smoke

另开 WSL2 终端：

```bash
source .venv/bin/activate
make bench-smoke
```

通过条件：

- 8 个请求全部成功；
- 产生一个 vLLM result JSON；
- manifest 中该轮 `returncode=0` 且只关联一个 result JSON；
- telemetry CSV 至少包含一个有效 GPU 样本；
- 没有服务重启、OOM 或 CUDA kernel error。

任一条件不满足时停止，不运行正式矩阵。

## 5. 单 Profile 稳定性

```bash
qsl run-matrix configs/matrix/baseline.toml --only e01_baseline_short_c1
```

它会执行三次，每次 100 个请求，轮次间冷却 30 秒。检查三轮是否使用相同的 benchmark/server SHA-256，错误率是否低于 1%，温度和时钟是否出现明显漂移。

然后生成临时报告：

```bash
make summarize
```

只有该 profile 状态为 `PASS`，才进入完整矩阵。

## 6. 完整矩阵

先确认矩阵确实展开为 12 组：

```bash
make render-baseline-matrix | less
```

执行：

```bash
make bench-baseline-matrix
make summarize
```

矩阵包含：

| Workload | Input | Output | Concurrency |
|---|---:|---:|---|
| short | 128 | 128 | 1, 4, 8, 16 |
| medium | 512 | 256 | 1, 4, 8, 16 |
| long | 2048 | 256 | 1, 4, 8, 16 |

共 36 个正式 benchmark run。每个 run 都显式使用 `temperature=0`、相同随机种子和固定 SLO。

## 7. M1 退出条件

- `reports/baseline/summary.md` 中 12 个 profile 均有三次同配置重复；
- 没有静默删除失败轮次；
- 能指出 TTFT、TPOT 和 goodput 随并发变化的拐点；
- 能区分服务吞吐提升与 SLO goodput 提升；
- 能解释峰值显存、GPU 利用率、温度和时钟对异常结果的影响。
- 将最终 manifest、原始 JSON、telemetry CSV、server log 和报告提交到 Git；体积过大时使用 release artifact，并保留校验和。

未满足以上条件时，M1 保持未完成，不进入优化收益结论。
