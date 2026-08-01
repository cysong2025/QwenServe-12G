# QwenServe-12G

面向 RTX 5070 12GB 消费级单卡的 LLM 推理优化与参数高效训练项目。

本项目不是“启动一个 vLLM 服务”的演示，而是一个可复现的系统实验：在固定硬件、模型和工作负载下，测量 KV Cache、调度参数、前缀复用、LoRA 服务与过载控制对延迟、有效吞吐、显存和模型质量的影响。

## 核心问题

> 在 RTX 5070 12GB、WSL2 和 Qwen2.5-3B-Instruct 条件下，怎样在满足延迟与质量约束的前提下，提高单卡服务的 SLO goodput？

项目最终需要交付：

- 可复现的 vLLM 基线和实验配置；
- TTFT、TPOT、E2E、吞吐、goodput、显存与质量对照数据；
- QLoRA 训练、离线评测和 LoRA 在线服务闭环；
- 一个 token-aware、SLO 感知的请求准入模块；
- 有适用边界、置信证据和原始数据支持的结论。

## 当前状态

当前完成 Stage 0 与 M1 的无 GPU 工程部分：项目章程、可行性分析、实验协议、环境自检、受控服务配置、基线负载矩阵、GPU 遥测和自动结果汇总。真实 G0/G1 数据仍需在目标 RTX 5070 上产生。

```text
docs/                 项目章程、可行性与实验协议
configs/serve/        vLLM 服务对照配置
configs/bench/        benchmark 工作负载配置
configs/matrix/       可展开的正式实验矩阵
src/qwen_serve_lab/   配置校验、命令生成与环境采集
scripts/              WSL2 初始化入口
artifacts/            环境快照和原始实验结果，不提交大文件
tests/                不依赖 GPU 的单元测试
reports/              从原始结果生成的汇总，不手工改数字
```

## 首次使用

本项目使用 Mac 进行开发与版本管理，在 Windows/WSL2 中运行 GPU 实验。两端通过 GitHub 同步受版本控制的代码、配置与报告。

Mac 开发端：

```bash
git clone git@github.com:cysong2025/QwenServe-12G.git
cd QwenServe-12G
make test
```

Windows PowerShell 中安装并更新 WSL2：

```powershell
wsl --install -d Ubuntu-24.04
wsl --update
```

进入 WSL2 后，将仓库克隆到 Linux 文件系统，不要在 `/mnt/c` 下运行：

```bash
ssh -T git@github.com
mkdir -p ~/projects
cd ~/projects
git clone git@github.com:cysong2025/QwenServe-12G.git
cd QwenServe-12G

bash scripts/bootstrap_wsl.sh
source .venv/bin/activate
make doctor
make render-baseline
make serve-baseline
```

若 `huggingface.co` 在 WSL2 中不可达，使用 Qwen 官方 ModelScope 仓库下载本地快照：

```bash
make download-model-modelscope
make render-baseline-local
make serve-baseline-local
```

默认路径为 `~/models/Qwen2.5-3B-Instruct`，可通过 `MODEL_PATH=/absolute/path` 覆盖。下载脚本使用隔离的 ModelScope CLI，并为模型根目录文件生成 `SHA256SUMS`。本地启动的 server manifest 会记录该文件的 SHA-256；同一轮正式实验不得混用 Hub 与 ModelScope 快照。

WSL2 中 vLLM V2 Model Runner 需要 pinned memory/UVA。受控服务配置会显式设置 `VLLM_WSL2_ENABLE_PIN_MEMORY=1`，并将该环境覆盖写入 server manifest；不需要手工 `export`。

RTX 5070 的 Blackwell `sm120` 架构会触发 vLLM 0.25.1 所带 FlashInfer 采样器的架构识别错误：明明高于 `sm75`，启动时仍报告 `FlashInfer requires GPUs with sm75 or higher`。两个受控 profile 固定设置 `VLLM_USE_FLASHINFER_SAMPLER=0`，仅将 top-k/top-p 采样回退到 PyTorch 原生实现；其余推理路径保持不变。该值同样记录在 server manifest 中。

`render-baseline` 只打印经过校验的命令；`serve-baseline` 启动服务并自动保存 server manifest 与完整日志。服务启动后，在另一个 WSL2 终端执行：

```bash
source .venv/bin/activate
make bench-smoke
```

smoke 通过后先执行一个正式 profile：

```bash
qsl run-matrix configs/matrix/baseline.toml --only e01_baseline_short_c1
```

确认三次重复稳定后，再运行完整的 `3 workloads x 4 concurrencies x 3 repetitions` 基线矩阵并生成报告：

```bash
make bench-baseline-matrix
make summarize
```

具体执行门槛和失败处理见 [M1 基线运行手册](docs/M1_BASELINE_RUNBOOK.md)。

详细边界和验收标准见 [项目章程](docs/PROJECT_CHARTER.md)、[可行性分析](docs/FEASIBILITY.md) 与 [实验协议](docs/EXPERIMENT_PROTOCOL.md)。

## 依据

- [Qwen2.5-3B-Instruct 配置](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/blob/main/config.json)
- [vLLM GPU 与 WSL 安装要求](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/)
- [vLLM serve 参数](https://docs.vllm.ai/en/latest/cli/serve/)
- [vLLM bench serve 参数](https://docs.vllm.ai/en/latest/cli/bench/serve/)
- [NVIDIA CUDA on WSL 指南](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)
