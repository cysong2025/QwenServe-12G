# E05 FP8 KV Cache 实验结果

## 1. 问题与状态

E05 在 RTX 5070 12GB、WSL2、Qwen2.5-3B-Instruct、vLLM 0.25.1 和
BF16 模型权重不变时，只把 KV Cache 从 BF16 改为 FP8 E4M3，并启用
vLLM 在线 KV scale 计算。实验回答三个问题：

- FP8 KV Cache 能增加多少可分配 token 容量；
- 容量压力下的延迟、吞吐和 SLO goodput 是否改善；
- 固定 AI Infra 故障诊断任务是否出现质量回归。

最终状态为 **COMPLETE_WITH_QUALITY_REGRESSION**。容量、性能、自动质量评测
和 50 条匿名人工复核均已完成；FP8 容量收益通过冻结门槛，但自动质量与人工质量
门槛均失败。因此，本实验验证了 FP8 KV Cache 的容量价值，也否定了“当前配置可在
不产生实质质量损失的情况下作为默认优化”的假设。

## 2. 证据集

性能矩阵包含 BF16/FP8、3 种长上下文形状、2 种并发和每个 profile 3 次重复，
共 12 个 profile、36 次有效 benchmark。所有正式性能运行请求错误率均为 0，
BF16/FP8 使用相同 workload、effective seed、请求数量和 telemetry 采集流程。

质量证据包含：

| 层级 | 数量 | 结果 |
|---|---:|---|
| 性能 benchmark | 36 | VALID，错误率 0% |
| 固定 AI Infra 质量集 | BF16/FP8 各 50 条 | 自动质量门槛 FAIL |
| 匿名人工复核 | 50 对 | FP8 相对 BF16 FAIL |

正式数据由以下报告承载：

- `reports/e05_kv_cache/summary.md`
- `reports/e05_kv_cache/comparison.md`
- `reports/e05_kv_cache/capacity.md`
- `reports/e05_kv_cache/quality.md`
- `reports/e05_kv_cache/human_review_summary.md`

## 3. 容量与性能

服务启动日志报告 BF16 KV Cache 可分配 96,080 tokens，FP8 E4M3 可分配
193,072 tokens，FP8/BF16 容量比为 **2.009x**，通过预先冻结的 `>=1.80x`
容量门槛。这与 KV 元素从 BF16 的 2 bytes 降至 FP8 的 1 byte 相符，同时保留了
分页、运行时元数据和显存碎片造成的实际系统开销。

性能收益不是所有负载都成立。已观察到的主要正向信号出现在接近上下文上限的
`nearmax_c16`：FP8 相对 BF16 的 output throughput 提高 14.77%，P95 TTFT
降低 9.42%，P95 TPOT 上升 2.90%。吞吐收益成立，但 TTFT 没有达到预定义的
10% 改善门槛，且该高压 profile 仍未满足冻结 SLO。

因此，2.009x 的静态 KV token capacity 不能直接解释为 2x 的在线并发能力。
prefill 激活、调度、decode 计算和 1000 ms TTFT SLO 仍会更早限制有效并发。

## 4. 自动质量结果

固定质量集覆盖 10 个平衡的 AI Infra 根因类别，并要求返回严格 JSON、正确根因、
建议动作和危险命令标记。主要结果为：

| Metric | BF16 | FP8 | 结果 |
|---|---:|---:|---|
| Schema pass rate | 92% | 70% | FP8 FAIL |
| Root-cause Macro-F1 | 0.7191 | 0.5698 | 基线与 FP8 均未达冻结门槛 |
| Action micro-F1 | 0.5208 | 0.3647 | 基线与 FP8 均未达冻结门槛 |
| Raw exact BF16/FP8 outputs | 2/50 | - | 明显存在生成差异 |

BF16 自身没有通过原定 Macro-F1 与 action-F1 绝对门槛，说明 3B Base 模型在该任务
上仍有明确能力缺口；FP8 又在 BF16 基础上进一步下降，因此不能把失败只归因于
数据集过难，也不能写成“FP8 质量与 BF16 等价”。

## 5. 匿名人工复核

人工复核在评分锁定前不读取 BF16/FP8 映射。50 对输出的结果为：

| Metric | BF16 | FP8 |
|---|---:|---:|
| Mean score | 3.680 | 3.120 |
| Preferred | 20 | 5 |
| Tie | 25 | 25 |

FP8-BF16 平均分差为 `-0.560`，而冻结门槛只允许 FP8 最多低 `0.10`，因此人工
质量状态为 **FAIL**。在 25 个非平局样本中，BF16 获胜 20 个，说明下降不是由
少量孤立样本或单一格式错误造成的。

## 6. 有效性限制

- FP8 使用 vLLM 在线 scale 计算。该流程使用启动 warmup，而不是代表性业务数据集
  的离线校准，因此结论不能推广到经过 `llm-compressor` 离线校准的 FP8 KV Cache。
- RTX 5070 `sm120` 上的 FlashInfer JIT 路径存在 capability 检测问题，正式
  BF16/FP8 两侧均固定为 `TRITON_ATTN`；结果不代表其他 attention backend。
- 质量集只有 50 条受控任务，适合发现本项目领域内的回归，不等同于通用语言模型
  benchmark。
- 性能矩阵使用突发闭环负载。静态容量增长和该矩阵中的吞吐变化不能直接外推到
  生产到达率、排队策略或多租户场景。

## 7. 决策

- 保留 FP8 KV Cache 的容量收益结论，并将其作为有质量代价的工程选项记录；
- 当前在线 scale 配置不作为质量敏感服务的默认方案；
- FP8 不进入 E06 组合优化，避免把已失败的质量变量与 APC/batch budget 混合；
- E06 只在 BF16 KV Cache 下比较 `8192/2048 batch token budget x APC OFF/ON`；
- 若后续研究离线校准，使用新的实验编号和冻结门槛，不覆盖本次 E05 证据。

## 8. 可复现结论

> 在 RTX 5070 12GB、WSL2、Qwen2.5-3B-Instruct BF16 权重、vLLM 0.25.1、
> `TRITON_ATTN` 和在线 KV scale 计算条件下，将 KV Cache 从 BF16 改为
> FP8 E4M3，使可分配 KV token 从 96,080 增至 193,072（2.009x），并在
> `nearmax_c16` 将 output throughput 提高 14.77%。但自动评测中 schema
> 通过率从 92% 降至 70%，root-cause Macro-F1 从 0.7191 降至 0.5698；
> 匿名人工均分从 3.680 降至 3.120，差值 -0.560，超过冻结容忍范围。
> 因而当前 FP8 KV Cache 配置的结论是“容量收益成立、部分性能收益成立、质量
> 门槛失败”，不推荐用于质量敏感的默认部署。

使用 WSL 中保留的原始 artifacts 可重建报告：

```bash
make summarize-e05
make compare-e05
make capacity-e05
make compare-e05-quality || true
make summarize-e05-human-review || true
```
