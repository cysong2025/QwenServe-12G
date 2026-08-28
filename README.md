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

**E01-E06 单卡推理优化里程碑已完成。** 项目在目标 RTX 5070 上完成
228 次独立正式 benchmark、22,800 个计时请求、固定正确性 canary、
FP8 自动质量评测和 50 对匿名人工复核。提交证据通过结构化审计：

```text
Overall status: PASS
Milestone status: E01_E06_COMPLETE
Unique formal benchmark runs: 228
```

E03 的长度 x 并发问题由 E01 的完整 3x4 矩阵覆盖，不重复计数。E01
原计划的 Transformers 单请求参考未执行，因此项目不声称跨引擎加速比。
E04 保留随机-token严格输出门槛的限制，E05 保留 FP8 质量回归失败，
这些负面结果均没有被重新定义为成功优化。

**E07 QLoRA/LoRA serving 实验已完整执行，机器总体门槛为 `FAIL`。** smoke、
rank-8、rank-16 训练和 Adapter 检查通过，36 次正式性能运行全部有效且错误率为
0；自动质量与用户委托的 Agent 代理盲评门槛通过，但 6 个在线成本 cell 中有
4 个因相对 TTFT 增幅超过冻结上限而失败。该结果记录为质量收益与在线成本之间的
tradeoff，不推荐当前动态 LoRA 配置作为默认部署。

**E08 token-aware 准入控制已完成协议、实现、测试与非 GPU readiness，目标 GPU
实验尚未运行。** 三种策略、三条开放到达轨迹、27 个正式 cell、精确 trace 配对、
SLO goodput 与长度公平性门槛已经冻结；readiness 为 `READY_FOR_GPU`，同时明确记录
`GPU execution: DEFERRED` 和 `Scientific result: NOT_RUN`。在 pilot 和 27 次正式
运行完成前，不把 E08 描述为完成或成功。

E08 的 WSL2 执行顺序见
[M4/E08 准入控制实验手册](docs/M4_E08_ADMISSION_RUNBOOK.md)。静态复核命令：

```bash
make test
make audit-e08-readiness
```

使用 WSL 中保留的原始 artifacts 可重建 E07 最终报告：

```bash
make finalize-e07
```

在新的 Codex Agent 中继续项目时，先阅读
[Codex Agent 交付文档](docs/CODEX_AGENT_HANDOFF.md)。其中集中记录当前分支、
实验状态、WSL2 环境问题、冻结门槛、GPU 执行顺序和禁止过度宣称的边界。

## 核心结果

| 实验 | 证据 | 结论 |
|---|---:|---|
| E01/E03 容量 | 36 runs | 12 个 profile 中 8 个通过 SLO；Long-C8/C16 吞吐增长但 goodput 下降 |
| E02 batch budget | 72 runs | 2048 相对 8192 将 Long-C4/C8 TTFT 降低 21.23%/47.74%，吞吐基本持平 |
| E04 APC | 36 runs | 约 46% actual hit 时 TTFT -24.20%；79% hit/C8 时 TTFT -55.11% 且 SLO 恢复 |
| E05 FP8 KV | 36 runs | KV 容量 2.009x，但 schema 92%->70%、匿名人工均分 3.680->3.120，不推荐默认部署 |
| E06 组合 | 48 runs | 叠加收益在 reuse50-P1024/C4 和 reuse90-P1792/C8 成立，24/24 canary 一致 |
| E07 QLoRA/LoRA | 36 runs | 自动质量和代理盲评 PASS，但 4/6 在线成本 cell 因 TTFT 回归 FAIL，不推荐当前动态 LoRA 默认部署 |
| E08 准入控制 | 0 formal runs | 协议、实现与 readiness 完成；GPU pilot/正式矩阵尚未运行，无科学结果 |

完整数据解读、部署建议、故障诊断和有效性边界见
[E01-E06 最终技术报告](docs/E01_E06_FINAL_REPORT.md)。
简历表述、面试深挖点和禁止过度宣称的边界见
[E01-E06 面试讲述指南](docs/INTERVIEW_GUIDE_E01_E06.md)。

克隆仓库后可不依赖 GPU 审计已提交证据：

```bash
make test
make audit-e01-e06
```

```text
docs/                 项目章程、可行性与实验协议
configs/serve/        vLLM 服务对照配置
configs/bench/        benchmark 工作负载配置
configs/matrix/       可展开的正式实验矩阵
configs/train/        QLoRA smoke、主实验与 rank 消融配置
configs/admission/    E08 开放到达轨迹、策略与冻结门槛
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

所有 Makefile benchmark 目标也会将该目录作为显式 `--tokenizer` 参数，避免 benchmark 客户端根据规范模型 ID 再次访问 Hugging Face。benchmark manifest 会记录 tokenizer 绝对路径和 `SHA256SUMS` 指纹；API 请求仍使用固定的 `served_model_name`。

WSL2 中 vLLM V2 Model Runner 需要 pinned memory/UVA。受控服务配置会显式设置 `VLLM_WSL2_ENABLE_PIN_MEMORY=1`，并将该环境覆盖写入 server manifest；不需要手工 `export`。

RTX 5070 的 Blackwell `sm120` 架构会触发 vLLM 0.25.1 所带 FlashInfer 采样器的架构识别错误：明明高于 `sm75`，启动时仍报告 `FlashInfer requires GPUs with sm75 or higher`。所有受控服务配置都固定设置 `VLLM_USE_FLASHINFER_SAMPLER=0`，仅将 top-k/top-p 采样回退到 PyTorch 原生实现；其余推理路径保持不变。该值同样记录在 server manifest 中。

若已有环境在运行 `vllm bench serve` 时报告 `pyarrow` 缺少 `PyExtensionType`，执行 `make repair-bench-deps`。项目将 PyArrow 固定为 20.0.0；21.0.0 删除了旧版 `datasets` 仍需使用的 API。修复后 `make doctor` 的 `vllm-bench` 检查必须 PASS。

`render-baseline` 只打印经过校验的命令；`serve-baseline` 启动服务并自动保存 server manifest 与完整日志。服务启动后，在另一个 WSL2 终端执行：

```bash
source .venv/bin/activate
make bench-smoke
```

smoke 通过后先执行一个正式 profile：

```bash
make bench-baseline
```

确认三次重复稳定后，再运行完整的 `3 workloads x 4 concurrencies x 3 repetitions` 基线矩阵并生成报告：

```bash
make bench-baseline-matrix
make summarize
```

矩阵执行会跳过相同配置哈希下已经完成三次有效重复的 profile，支持中断后安全续跑；正式汇总只读取与矩阵 TOML 哈希一致的 manifest。

具体执行门槛和失败处理见 [M1 基线运行手册](docs/M1_BASELINE_RUNBOOK.md)。

E02 的 budget 切换、启动门槛和可恢复矩阵流程见
[E02 Batch Token Budget 实验手册](docs/M2_E02_BATCH_TOKEN_RUNBOOK.md)。
完整数据解读、有效性限制和可复现结论见
[E02 实验结果](docs/E02_RESULTS.md)。

E01 基线与 E03 容量拐点的结果见
[E01/E03 实验结果](docs/E01_E03_RESULTS.md)。

E04 的 OFF/ON 配对、缓存隔离、实际 token 命中率采集和正式矩阵步骤见
[E04 Automatic Prefix Caching 实验手册](docs/M2_E04_PREFIX_CACHE_RUNBOOK.md)。
随机 token 输出不完全一致时，手册中的 correctness canary 使用固定自然语言数据集
独立验证 prompt、预期答案、OFF/ON 输出和实际 prefix cache hit，不需要重跑性能矩阵。
完整数据解读、协议偏差和可复现结论见
[E04 实验结果](docs/E04_RESULTS.md)。

E05 的 BF16/FP8 分时启动、长上下文配对矩阵、KV token capacity、自动质量门槛和
50 条匿名人工复核步骤见
[E05 FP8 KV Cache 实验手册](docs/M3_E05_FP8_KV_CACHE_RUNBOOK.md)。
完整数据解读和部署边界见 [E05 实验结果](docs/E05_RESULTS.md)。

E06 的四 cell 分时执行、组合收益门槛、因子交互和固定 canary 步骤见
[E06 组合优化实验手册](docs/M3_E06_COMBINED_RUNBOOK.md)。
完整结果和部署决策见 [E06 实验结果](docs/E06_RESULTS.md)。

E07 的研究问题、冻结质量/成本门槛见
[E07 实验协议](docs/E07_PROTOCOL.md)，训练数据来源和局限见
[E07 数据卡](docs/E07_DATA_CARD.md)，WSL2 手工执行命令见
[E07 QLoRA 与 LoRA Serving 运行手册](docs/M3_E07_QLORA_LORA_RUNBOOK.md)。
完整训练、质量、在线成本和部署结论见 [E07 实验结果](docs/E07_RESULTS.md)。
[E07 结果模板](docs/E07_RESULTS_TEMPLATE.md) 保留为报告结构参考。

详细边界和验收标准见 [项目章程](docs/PROJECT_CHARTER.md)、[可行性分析](docs/FEASIBILITY.md) 与 [实验协议](docs/EXPERIMENT_PROTOCOL.md)。

## 依据

- [Qwen2.5-3B-Instruct 配置](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/blob/main/config.json)
- [vLLM GPU 与 WSL 安装要求](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/)
- [vLLM serve 参数](https://docs.vllm.ai/en/latest/cli/serve/)
- [vLLM bench serve 参数](https://docs.vllm.ai/en/latest/cli/bench/serve/)
- [vLLM 0.25.1 Quantized KV Cache](https://docs.vllm.ai/en/v0.25.1/features/quantization/quantized_kvcache/)
- [NVIDIA CUDA on WSL 指南](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)
