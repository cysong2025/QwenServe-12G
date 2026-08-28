# E07 QLoRA 与 LoRA 在线服务实验结果

## 1. 问题与状态

E07 在 RTX 5070 12GB、WSL2、Qwen2.5-3B-Instruct 和 vLLM 0.25.1
条件下，使用 4-bit NF4 QLoRA 训练 AI Infra 事件诊断 Adapter，并比较 Base
与 rank-8 LoRA 的任务质量和在线服务成本。rank-16 只作为冻结的训练消融，不在
看到结果后替换主实验 Adapter。

实验执行状态为 **COMPLETE_WITH_ONLINE_COST_REGRESSION**：smoke、rank-8、
rank-16 训练和 Adapter 检查完成，Base/LoRA 各 18 次正式性能运行有效，自动质量
和代理盲评门槛通过；但 6 个在线成本 cell 中有 4 个失败。因此机器生成的 E07
总体门槛状态为 **FAIL**，不能表述为成功部署。

## 2. 训练证据

三次训练均使用 BF16 compute、gradient checkpointing、固定 seed 和同一训练/
验证数据来源。rank-8 与 rank-16 除 LoRA rank/alpha 外保持训练控制一致。

| Profile | Rank | Train rows | Steps / epochs | Train loss | Eval loss | Peak VRAM MiB | Status |
|---|---:|---:|---:|---:|---:|---:|---|
| smoke | 8 | 100 | 20 / 1.56 | 0.106785 | 0.335182 | 5406 | COMPLETE；save/load PASS |
| primary | 8 | 250 | 48 / 3.00 | 0.052559 | 0.006476 | 5402 | COMPLETE；Adapter PASS |
| ablation | 16 | 250 | 48 / 3.00 | 0.046409 | 0.008587 | 5538 | COMPLETE；Adapter PASS |

主 Adapter 权重 SHA-256 为
`e930d284cc29e47e88976d16699d00c5883b0a811d29a08ef85763c0e5b98fe9`。
它通过 rank、training manifest、非空权重和权重哈希检查，并实际通过 vLLM LoRA
加载门槛。rank-16 的 eval loss 高于 rank-8，且它不是冻结主实验处理，因此没有
替换 rank-8 进入质量或在线矩阵。

## 3. 质量证据

Base 与 LoRA 使用相同的 50 条冻结事件诊断 prompt、temperature 0 和配对输入。
数据集 SHA-256 与 50/50 prompt 均匹配。

| Metric | Base | LoRA | LoRA - Base | Gate |
|---|---:|---:|---:|---|
| Schema pass rate | 92.00% | 100.00% | +8.00 pp | PASS |
| Root-cause Macro-F1 | 0.7191 | 0.9366 | +0.2175 | PASS |
| Action micro-F1 | 0.5208 | 0.9400 | +0.4192 | PASS |
| Dangerous command rate | 0.00% | 0.00% | 0.00 pp | PASS |
| Blind-review mean | 3.580 | 4.760 | +1.180 | PASS |

盲评在 50 行评分锁定前没有读取 A/B 映射。解盲后，偏好计数为 Base 2、LoRA
38、平局 10。该复核由用户明确委托 Codex Agent 代为完成，而不是由独立人类
评审者完成；因此它验证了固定 rubric 下的代理盲评门槛，但不能作为独立人类偏好
研究来外推。

## 4. 在线成本

性能矩阵包含 short 128/128 与 medium 512/256、并发 1/4/8，每个 Base/LoRA
profile 3 次重复和 100 个计时请求，共 36 次正式运行。所有 profile 证据均为
`VALID`、请求错误率为 0，LoRA 绝对 TTFT/TPOT SLO 全部通过。冻结相对门槛还要求
throughput 损失不超过 20%、P95 TTFT 增幅不超过 25%、P95 TPOT 增幅不超过
20%；任一 cell 失败即总体 online cost FAIL。

| Workload | C | Base/LoRA tok/s | Delta | Base/LoRA P95 TTFT ms | Delta | Base/LoRA P95 TPOT ms | Delta | Base/LoRA goodput | Base/LoRA VRAM MiB | Cost |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| short 128/128 | 1 | 88.22 / 73.38 | -16.83% | 37.54 / 58.04 | +54.61% | 11.35 / 13.51 | +19.04% | 0.69 / 0.57 | 10472 / 11035 | FAIL |
| short 128/128 | 4 | 322.93 / 279.51 | -13.45% | 72.14 / 102.99 | +42.75% | 12.16 / 13.87 | +14.03% | 2.52 / 2.18 | 10472 / 11035 | FAIL |
| short 128/128 | 8 | 599.71 / 518.54 | -13.54% | 120.19 / 150.92 | +25.56% | 12.59 / 14.41 | +14.43% | 4.69 / 4.05 | 10500 / 11035 | FAIL |
| medium 512/256 | 1 | 86.75 / 72.42 | -16.53% | 58.94 / 74.87 | +27.02% | 11.43 / 13.66 | +19.50% | 0.34 / 0.28 | 10500 / 11048 | FAIL |
| medium 512/256 | 4 | 311.24 / 271.46 | -12.78% | 209.85 / 249.04 | +18.68% | 12.67 / 14.40 | +13.64% | 1.22 / 1.06 | 10468 / 11031 | PASS |
| medium 512/256 | 8 | 554.37 / 484.35 | -12.63% | 404.03 / 453.60 | +12.27% | 13.68 / 15.55 | +13.64% | 2.17 / 1.89 | 10597 / 11134 | PASS |

throughput 和 TPOT 的相对变化在全部 cell 内均未越过冻结上限，失败集中在 TTFT：
三个 short cell 和 `medium_c1` 超过 +25%。因此，LoRA 在该服务形态下获得了明确
质量提升，但低工作量/低并发请求的额外 Adapter 路径成本不可接受。

## 5. 部署决策

- 不把当前动态 LoRA 服务配置作为默认部署，也不放宽 +25% TTFT 门槛；
- 保留 rank-8 Adapter 的任务质量收益，可用于离线处理、受控批量任务或新的性能
  优化实验；
- `medium_c4` 和 `medium_c8` 说明较高并发能够摊薄部分固定开销，但不能用两个
  PASS cell 覆盖其余四个 FAIL cell；
- 若后续研究 Adapter 融合、静态合并、不同 LoRA backend 或准入控制，应使用新
  实验编号和预先冻结的门槛，不覆盖本次负结果。

## 6. 有效性限制

- 质量集只有 50 条受控 AI Infra 事件诊断任务，不代表通用模型能力；
- 盲评由 Codex Agent 按固定 1–5 rubric 代理完成，虽然评分前保持 A/B 盲态，
  但缺少独立人类评审者，可能高估规则一致性；
- 结果只覆盖一张 RTX 5070、当前 WSL2/CUDA/PyTorch/vLLM 组合和动态 LoRA
  加载路径；
- 性能流量为固定突发闭环矩阵，不直接代表生产到达率、多租户队列或其他上下文
  分布；
- rank-16 只完成训练消融，没有进入冻结的在线成本和质量主比较。

## 7. 可复现结论

> 在 RTX 5070 12GB、WSL2、Qwen2.5-3B-Instruct、vLLM 0.25.1 和冻结
> 50 条事件诊断任务下，rank-8 QLoRA Adapter 将 schema 通过率从 92% 提高到
> 100%，root-cause Macro-F1 从 0.7191 提高到 0.9366，action micro-F1
> 从 0.5208 提高到 0.9400；代理盲评均分从 3.580 提高到 4.760。36 次性能
> 运行均有效且错误率为 0，但 LoRA 在三个 short cell 和 `medium_c1` 的 P95
> TTFT 增幅超过冻结的 25% 上限，只有 `medium_c4/c8` 通过全部相对成本门槛。
> 因而 E07 的结论是“质量收益成立、在线成本总体失败”，当前配置不作为默认部署。

WSL 中保留的原始 artifacts 可重建并验证全部紧凑报告：

```bash
make finalize-e07
```

该目标接受比较器和最终器的退出码 2，因为科学门槛 FAIL 是有效实验结论；缺失
Adapter、原始证据、50 条评分或 36 次有效性能运行仍会使目标失败。
