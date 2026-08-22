# E01-E06 单卡 LLM 推理优化最终技术报告

## 1. 执行摘要

本里程碑研究在 RTX 5070 12GB 消费级单卡上，如何为
Qwen2.5-3B-Instruct 的 vLLM 服务选择 batch-token budget、并发度、
Automatic Prefix Caching 和 KV Cache dtype。优化目标不是单独最大化
tokens/s，而是在 P95 TTFT、P95 TPOT、错误率和生成质量门槛下提高
SLO goodput。

E01-E06 共完成 **228 次独立正式 benchmark**、**22,800 个计时
请求**，以及 E04/E06 固定 canary、E05 的 100 条 BF16/FP8 质量请求和
50 对匿名人工复核。所有正式性能运行错误率均为 0，配置哈希、seed、
重复编号和 GPU telemetry 完整。

最终决策是：

- 通用长输入服务使用 BF16 KV Cache 和较小 batch-token budget；
- APC 只在业务有稳定可复用长前缀时视为有效优化，并监控实际 token hit；
- FP8 KV Cache 虽把静态 token 容量提高到 2.009x，但当前在线 scale
  配置未通过自动质量和匿名人工质量门槛，不作为默认部署；
- 2048/256 普通长输入默认使用 C4；C8 只在高复用长前缀或有准入控制
  时使用，C16 不是本机的 SLO-safe 配置。

## 2. 背景与问题

数据中心 GPU 的公开吞吐数字不能直接解答 12GB 消费级显卡的部署问题。
在该资源上，权重、KV Cache、prefill activation、CUDA Graph private pool 和桌面
图形显存共享同一物理上限。调大并发或缓存预留可能提高 raw throughput，
也可能先导致排队、SLO 违约或 OOM。

项目回答四个核心问题：

1. 输入长度和并发度如何改变服务容量边界；
2. batch-token budget 如何权衡 prefill 延迟和吞吐；
3. APC 在什么实际 token hit 下值得开启；
4. FP8 KV Cache 的容量收益是否能通过质量门槛，以及单项优化能否叠加。

## 3. 实验环境与边界

| Component | Frozen value |
|---|---|
| GPU | NVIDIA GeForce RTX 5070, 12,227 MiB |
| Host | Windows 11 + WSL2 2.7.8 |
| Linux | Ubuntu 24.04, x86_64 |
| Driver/CUDA | 581.29 / CUDA 13.0 |
| Python/PyTorch | Python 3.12 / PyTorch 2.11.0+cu130 |
| Serving | vLLM 0.25.1 |
| Model | Qwen2.5-3B-Instruct, pinned revision |
| SLO | P95 TTFT <=1000 ms, P95 TPOT <=50 ms/token, error rate <1% |

由于 Hugging Face 443 连接在实验网络中被重置，模型通过 Qwen 官方
ModelScope 仓库下载到 WSL Linux 文件系统。服务 manifest 和 benchmark manifest
分别记录本地模型/tokenizer `SHA256SUMS` 指纹，避免实验期间隐式访问远程
模型或混用快照。

RTX 5070 `sm120` 上的 FlashInfer 路径存在 capability 检测问题。受控配置
禁用 FlashInfer sampler；E05/E06 四侧统一使用 `TRITON_ATTN`，保证后端不成为
隐藏自变量。因此结论不自动外推到其他 attention backend。

## 4. 实验与证据

| ID | Question | Formal runs | Status |
|---|---|---:|---|
| E01 | BF16 基线 | 36 | COMPLETE_WITH_PROTOCOL_DEVIATION |
| E02 | Batch-token budget sweep | 72 | COMPLETE |
| E03 | 长度 x 并发容量 | 0 additional | COVERED_BY_E01 |
| E04 | Automatic Prefix Caching | 36 | COMPLETE_WITH_LIMITATIONS |
| E05 | FP8 KV Cache | 36 | COMPLETE_WITH_QUALITY_REGRESSION |
| E06 | Budget x APC 2x2 组合 | 48 | COMPLETE |

E03 使用 E01 的同一份 36-run 矩阵，不在 228 次总数中重复计算。
E01 原计划的 Transformers 单请求参考未执行，所以只建立 vLLM 内部受控
基线，不报告 vLLM 相对 Transformers 的加速比。

证据链包含：

- TOML 配置校验和显式 vLLM 命令生成；
- 活跃服务 config SHA-256 与 benchmark 所需服务的一致性检查；
- 每次运行的环境 manifest、原始 JSON、GPU telemetry 和详细请求输出；
- 三次重复的 median `[min, max]` 汇总和每轮 SLO 门槛；
- 固定 seed、预热、冷却、请求数和质量数据集；
- 不依赖 GPU 的 `make audit-e01-e06` 提交证据审计。

## 5. 分阶段结果

### 5.1 E01/E03：并发不等于有效容量

12 个基线 profile 中 8 个通过 SLO。Short/Medium 在 C8 时通过，
C16 时 throughput 只增加 0.59%/0.29%，TTFT 却升至 1676.66/3772.68 ms。
Long-C4 的 TTFT 为 798.11 ms、goodput 1.03 req/s；Long-C8 的 throughput
增至 418.14 tok/s，但 TTFT 达到 1580.83 ms、goodput 降为 0.78 req/s。

结论：过载后 raw throughput 可继续上升或持平，SLO goodput 却已下降。

### 5.2 E02：较小 budget 改善长输入 TTFT

2048 相对 8192 budget：

- Long-C4 P95 TTFT 降低 21.23%，throughput 只下降 0.18%；
- Long-C8 P95 TTFT 降低 47.74%，goodput 提高 164.19%，throughput 提高 0.14%；
- Short/Medium 的主要 throughput 差异不足 1%；
- 16384 在任何已测负载下都没有形成优势。

Long-C8 的 2048 budget 仍因一次 TTFT 1005.24 ms 而按冻结规则标记
SLO FAIL。结论：调度粒度可以改善排队，但不能替代准入控制。

### 5.3 E04：APC 收益由实际可复用 token 决定

P1024/C4 下，0%、50%、90% 名义复用率对应 0.98%、18.78%、45.52%
实际 token hit。TTFT 收益分别为 0.43%、0.43%、24.20%，throughput 收益为
0.20%、3.84%、9.95%。本实验只能将明确收益边界定位在约 19% 与 46%
实际 hit 之间。

P1792/C8 在 78.93% hit 下将 TTFT 从 1407.15 降到 631.67 ms，
throughput 从 412.10 提高到 554.18 tok/s，使 3/3 SLO FAIL 恢复为
3/3 PASS。

随机-token逐字一致性未通过原始严格门槛，所以 E04 保留
`COMPLETE_WITH_LIMITATIONS`。补充固定自然语言 canary 在 84.89% hit 下
达到 24/24 OFF/ON 输出一致；任务正确率两侧都是 22/24。

### 5.4 E05：FP8 容量收益不等于可部署收益

FP8 E4M3 KV Cache 将可分配 token 从 96,080 提高到 193,072，比率
2.009x。最大性能信号出现在 nearmax-C16：output throughput 提高
14.77%，TTFT 降低 9.42%，但该 profile 仍不满足 SLO。

质量门槛明确失败：

- schema pass rate 从 92% 降至 70%；
- root-cause Macro-F1 从 0.7191 降至 0.5698；
- action micro-F1 从 0.5208 降至 0.3647；
- 匿名人工均分从 3.680 降至 3.120，FP8-BF16 为 -0.560；
- 25 个非平局样本中 BF16 获胜 20 个，FP8 获胜 5 个。

结论：当前在线 scale 配置不是质量敏感服务的默认选择，也不进入 E06。

### 5.5 E06：组合收益是负载相关的

2048 budget + APC 在 reuse50-P1024/C4 下使 TTFT 相对最佳单项降低
23.55%；在 reuse90-P1792/C8 容量条件下降低 10.78%，同时保持
throughput 并通过 SLO。

无复用时，组合相对 budget-only 没有收益。P1024 的 45.54% hit 下，
APC-only 已获得大部分收益，组合额外 TTFT 改善为 3.78%，低于
5% 冻结门槛。四个 cell 的固定 canary 输出 24/24 一致。

## 6. 推荐配置策略

| Traffic class | Recommended policy | Evidence |
|---|---|---|
| Short/medium, C<=8 | BF16 KV, standard budget; prioritize simplicity | E01/E02 差异小 |
| Long 2048/256, C4 | BF16 KV, 2048 batch-token budget | E02 TTFT -21.23%, throughput -0.18% |
| Stable reusable P1024/P1792 | BF16 KV, 2048 budget, APC ON | E04/E06 |
| Long C8 without stable reuse | Reject, queue, or add admission control | E01/E02 SLO FAIL |
| FP8 KV online scales | Do not default for quality-sensitive traffic | E05 quality FAIL |
| C16 closed-loop burst | Do not use as SLO-safe capacity | E01/E05 |

这是负载分类策略，不是一组对所有请求都最优的全局参数。

## 7. 关键故障诊断

### 7.1 网络与模型供应

Windows 和 WSL 到 Hugging Face 443 均被超时/重置，而 PyPI、GitHub SSH 和
ModelScope 可达。项目将代码同步固定为 GitHub SSH，模型供应改为
ModelScope 本地快照，并用 `SHA256SUMS` 防止隐式版本漂移。

### 7.2 Blackwell 后端兼容

FlashInfer 将 `sm120` 误报为不满足 `sm75`。项目将 sampler 回退到
PyTorch，E05/E06 固定 Triton attention，并把这些平台覆盖写入配置和
server manifest，而不是在终端中使用不可追溯的手工参数。

### 7.3 异步 CUDA 错误定位

E06 首次 OOM 只在异步 copy event 处报告 `cudaErrorUnknown`。通过关联
scheduler dump、显存使用和 `CUDA_LAUNCH_BLOCKING=1` 诊断，将错误定位到
6145-token 步的 130 MiB activation 分配。四组一致降低 KV Cache 预留比例后，
48 次正式运行全部完成。

## 8. 复现与审计

任意 Linux/macOS 开发环境可在不使用 GPU 的情况下验证代码和已提交证据：

```bash
make test
make audit-e01-e06
```

`audit-e01-e06` 使用 CSV/JSON 结构化解析，检查行数、profile、repetition、
错误率、请求完整性、canary、容量门槛、质量失败和 E06 因子判定。
预期输出为：

```text
Overall status: PASS
Milestone status: E01_E06_COMPLETE
Unique formal benchmark runs: 228
```

完整重跑需要 WSL2、本地模型快照和运行手册。原始 artifacts 由实验机保留且
因体积与机器路径不提交 Git；GitHub 包含可重跑配置、分析代码、机器可读汇总
和冻结结论。因此第三方可以审计已提交汇总并重跑实验，但仅凭 Git
仓库不能重建每个历史原始 JSON。这是当前证据发布的明确边界。

## 9. 有效性限制

- 只测试一张 RTX 5070、一个 3B 模型和一个 vLLM 版本；
- 性能矩阵为闭环突发负载，不等于生产 Poisson 到达流量；
- 每个 profile 3 次重复可显示稳定性，但不支持很窄的置信区间；
- E04/E06 的 prefix 负载是受控合成流量，实际业务必须监控 token hit；
- E05 使用在线 KV scale，结论不适用于经代表性数据离线校准的 FP8；
- E01 没有 Transformers 参考，不支持跨引擎性能声明；
- E07 LoRA 训练/服务和 E08 准入控制尚未执行，不属于本里程碑的完成范围。

## 10. 里程碑结论

E01-E06 推理优化里程碑已完成。项目已建立从配置、启动、环境快照、压测、
telemetry、质量门槛到自动汇总和证据审计的完整链路。它得到的不是一个
万能参数，而是一组可执行的负载分类决策，并保留了不符合假设的
E04 严格输出门槛和 E05 质量回归。

后续 E07/E08 应作为新里程碑：先补齐 QLoRA 训练与 LoRA serving 证据，再用
相同请求轨迹比较无限制、固定并发和 token-aware 准入策略。
