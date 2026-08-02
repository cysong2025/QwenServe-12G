# 实验协议

## 1. 实验原则

本协议在正式优化前冻结。优化结果不能通过改变 SLO、请求分布或统计口径来获得。任何偏离协议的实验必须在结果中标注原因。

每个正式实验都必须关联：

- 服务 profile 与 benchmark profile；
- vLLM/PyTorch/CUDA/driver/Python 版本；
- GPU 状态和模型 revision；
- Git commit 与 dirty 状态；
- UTC 时间、重复编号和原始 JSON 结果。

## 2. 指标定义

主要指标：

- TTFT：请求发出到首 token 到达；
- TPOT：生成阶段每个输出 token 的平均耗时；
- E2E：请求完整结束时间；
- goodput：同时满足全部预设 SLO 的请求数/秒；
- error rate：失败请求占比；
- peak VRAM：实验期间峰值显存。

次要指标：ITL、requests/s、input tokens/s、output tokens/s、queue time、prefix cache hit rate、GPU utilization、温度和功耗。

vLLM 输出的 `Peak concurrent requests` 按一秒时间桶统计：请求只要在该秒内活动过就会计入，顺序请求也可能落入同一桶，因此该值可能高于 `max_concurrency`。它只用于解释峰值 token 吞吐所在时间桶，不作为瞬时并发或负载约束的证据；实验并发以固定的 `max_concurrency` 和详细请求记录为准。

## 3. SLO 冻结

M1 先执行 pilot，不评价优化优劣。根据交互式服务目标和 pilot 分布冻结 SLO，初始候选值为：

```text
P95 TTFT <= 1000 ms
P95 TPOT <= 50 ms/token
error rate < 1%
```

冻结后的值写入全部正式 benchmark 配置。若后续修改，旧实验必须全部重跑。

## 4. 控制变量

固定项：

- 同一台 RTX 5070；
- 同一模型 revision、tokenizer 与 generation config；
- `temperature = 0` 的质量评测；
- 同一请求集合与随机种子；
- 同一轮次预热、冷却和重复规则；
- 关闭无关 GPU 密集应用。

自变量：

- 输入/输出长度；
- `max_num_seqs`；
- `max_num_batched_tokens`；
- 请求到达率和 burstiness；
- prefix caching 与前缀复用率；
- KV Cache dtype；
- Base/LoRA；
- 无限制、固定并发、自适应准入策略。

禁止同时改变多个自变量后把收益归因给其中一个。组合配置只在单因素实验完成后测试。

## 5. 标准负载

| 名称 | input tokens | output tokens | 用途 |
|---|---:|---:|---|
| short | 128 | 128 | 短交互与 decode 占比较高场景 |
| medium | 512 | 256 | 常规聊天负载 |
| long | 2048 | 256 | prefill 压力场景 |

并发度使用 1、4、8、16。开放负载使用逐级 request rate，直到出现明显排队、SLO 违约或错误。Prefix 实验使用长度固定的公共前缀，并设置 0%、50%、90% 复用率。

## 6. 实验清单

| ID | 实验 | 对照 | 回答问题 |
|---|---|---|---|
| E00 | 环境与 smoke | 无 | 环境是否满足实验前提 |
| E01 | 受控 BF16 基线 | Transformers 单请求参考 | vLLM 基线能力与固定开销 |
| E02 | batch token sweep | E01 | TTFT/TPOT/吞吐的权衡点 |
| E03 | 并发与长度矩阵 | E01 | 容量边界和拐点 |
| E04 | prefix reuse sweep | E01 | prefix caching 收益阈值 |
| E05 | FP8 KV Cache | E01 | 容量收益和质量风险 |
| E06 | 组合优化 | 最佳单项配置 | 优化是否可叠加 |
| E07 | Base 与 LoRA | E06 Base | 质量收益和在线成本 |
| E08 | 准入策略 | 无限制、固定并发 | 过载 goodput 与公平性 |

## 7. 重复和有效性

- smoke 使用至少 2 个 warmup 请求；正式实验至少 10 个 warmup 请求。
- 闭环并发实验每组至少 100 个有效请求；开放到达率实验的稳态阶段至少持续 3 分钟。
- 每组独立重复 3 次，轮次间冷却并记录 GPU 温度。
- 报告 P50/P95/P99、中位数、最小/最大或 bootstrap 95% CI。
- 某轮发生 OOM、服务重启或请求错误率超过 1% 时，该轮保留为失败证据，不能静默删除。

## 8. 质量门槛

FP8 KV 与 LoRA 必须运行固定质量集：

- 结构化输出 schema 通过率；
- 根因分类 Macro-F1；
- 修复步骤多标签 F1；
- 危险命令率；
- 固定 50 条样本的匿名人工评分。

性能提高但质量跌破冻结门槛时，结论应为“不适用于当前任务”，而不是成功优化。

## 9. 结果与结论格式

核心结果表必须至少包含：

| profile | workload | TTFT P95 | TPOT P95 | output tok/s | goodput | peak VRAM | error rate | quality delta |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 0 |

每条最终结论使用同一格式：

> 在【硬件/软件版本】和【工作负载】下，相比【基线】，配置【自变量】使【指标】变化【数值与波动范围】，同时【质量/错误率】为【结果】。该结论适用于【边界】，不适用于【边界】。

没有原始数据、重复证据和边界描述的陈述不能进入摘要或简历。
