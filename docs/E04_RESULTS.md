# E04 Automatic Prefix Caching 实验结果

## 1. 问题与状态

E04 在 RTX 5070 12GB、WSL2、Qwen2.5-3B-Instruct BF16、vLLM 0.25.1
和 `max_num_batched_tokens=8192` 不变时，只切换 Automatic Prefix Caching
（APC），回答三个问题：

- 固定 2048-token 输入时，共享前缀复用率达到多少才出现可观测收益；
- 固定 90% 名义复用率时，公共前缀长度如何影响 TTFT 与吞吐；
- 高复用长前缀能否改善 C8 容量边界，同时保持模型输出行为。

最终状态为 **COMPLETE_WITH_LIMITATIONS**。36 次性能 benchmark 与缓存命中率
证据完整，但随机-token工作负载没有满足预先冻结的逐字输出一致性门槛；补充的固定
自然语言 canary 在 84.89% 实际缓存命中率下得到 24/24 OFF/ON 输出一致。

## 2. 证据集

性能矩阵包含 12 个 profile、6 组 OFF/ON 条件和每个 profile 3 次重复，共 36 次
正式 benchmark。36 次运行均为 `valid=True`，请求错误率为 0，OFF/ON 使用相同
seed，且每轮 ON 都采集了 token 级缓存查询和命中计数。

证据分为三层：

| 层级 | 结果 | 解释 |
|---|---|---|
| 性能与 telemetry | COMPLETE | 36/36 有效运行，配置、seed、缓存计数和 GPU telemetry 完整 |
| 随机-token逐字复现 | FAIL | 1639/1800 输出在忽略完成顺序后相同，未达到原定 100% |
| 固定自然语言 APC 等价性 | PASS | prompt 24/24 相同，OFF/ON 输出 24/24 相同，ON hit rate 84.89% |
| canary 任务质量 | FAIL | OFF、ON 均为 22/24；两条相同的查表错误与 APC 无关 |

汇总数据见 `reports/e04_prefix_cache/runs.csv`，性能差值见
`reports/e04_prefix_cache/comparison.csv`，输出诊断和 canary 分别见
`output_diagnostics.md` 与 `correctness_canary.md`。

## 3. 性能结果

下表均为三次重复的中位数。TTFT 与 throughput delta 均以 ON 相对 OFF 计算，
负 TTFT 表示延迟降低，正 throughput 表示吞吐提高。

| Condition | C | Actual hit | P95 TTFT OFF/ON ms | TTFT delta | Output tok/s OFF/ON | Throughput delta | Peak VRAM OFF/ON MiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| reuse0_p1024 | 4 | 0.98% | 806.43/802.99 | -0.43% | 260.64/261.16 | +0.20% | 11293/11295 |
| reuse50_p1024 | 4 | 18.78% | 804.27/800.84 | -0.43% | 260.62/270.63 | +3.84% | 11369/11295 |
| reuse90_p1024 | 4 | 45.52% | 804.86/610.09 | -24.20% | 260.60/286.54 | +9.95% | 11250/11301 |
| reuse90_p256 | 4 | 12.11% | 807.90/734.78 | -9.05% | 260.57/266.88 | +2.42% | 11391/11295 |
| reuse90_p1792 | 4 | 78.94% | 804.81/436.33 | -45.79% | 260.66/309.23 | +18.63% | 11381/11299 |
| capacity_reuse90_p1792 | 8 | 78.93% | 1407.15/631.67 | -55.11% | 412.10/554.18 | +34.48% | 11725/11569 |

APC 没有形成稳定的显存惩罚；各条件 ON-OFF 峰值显存差在 -156 至 +51 MiB
之间，相对 12GB 显存属于小幅波动。

## 4. 复用率与前缀长度

固定 P1024/C4 时，0% 和 50% 名义复用对应 0.98% 和 18.78% 实际 token hit，
P95 TTFT 只变化 -0.43%，未达到预定义的 5% 收益门槛。90% 名义复用对应
45.52% 实际 hit，P95 TTFT 降低 24.20%，output throughput 提高 9.95%。

因此，本实验只能把 P1024 的收益边界定位在 **50% 与 90% 名义复用之间**，或
**18.78% 与 45.52% 实际 token hit 之间**；没有测试中间点，不能声称存在更精确
阈值。

固定 90% 名义复用/C4 时，前缀从 256 增至 1024、1792 token，实际 hit 从
12.11% 增至 45.52%、78.94%，P95 TTFT 收益从 9.05% 增至 24.20%、45.79%。
这说明 APC 收益主要由可复用 token 占总输入的比例驱动，而不只是“请求是否重复”。

## 5. 容量与 SLO

在 P1792、90% 复用的 C8 条件下，OFF 三次 P95 TTFT 都约为 1407 ms，违反
1000 ms SLO；ON 三次分别为 631.67、827.02 和 629.38 ms，全部满足 TTFT/TPOT
SLO。中位 output throughput 同时从 412.10 提高到 554.18 tok/s。

因此，在已测高复用长前缀流量下，APC 不只是减少 prefill 延迟，还把 C8 从稳定
SLO FAIL 推到 3/3 SLO PASS。该结论不适用于低复用或短公共前缀流量。

## 6. 输出与质量审计

随机-token性能矩阵使用 `temperature=0`，但只有 1/18 个 OFF/ON run pair 达到
整批逐字相同。按请求位置有 1636/1800（90.89%）一致，忽略异步完成顺序后有
1639/1800（91.06%）一致；两者只差 3 条，因此顺序不是主要原因。0% 名义复用
条件同样存在差异，不能把全部随机-token不一致直接归因于 APC。

原协议要求随机-token输出 100% 相同，所以自动 `comparison.md` 正确保留
`Evidence=INCOMPLETE` 和 `Decision=UNKNOWN`。本报告没有事后删除或放宽该规则。

为区分随机-token数值敏感性与功能回归，补充 canary 使用提交到仓库的固定自然语言
数据集，SHA-256 为
`5f3ea04b3bed995a8a71fa35c3eb7db70f439c2eaaccdd63227c9496791ee8c1`。
24/24 prompt 哈希一致、24/24 OFF/ON 输出一致，并观察到 84.89% prefix hit，
因此在该 canary 范围内没有观察到 APC 改变输出。

canary 的任务准确率为 22/24。`capacity-06` 在两侧都把 `A30` 答成 `T4`，
`incident-07` 在两侧都把 `ap-south-1` 答成 `us-east-1`。这是模型在重复结构长
上下文中的记录字段绑定错误；它限制 canary 的任务质量，但不是 APC 回归证据。

## 7. 决策与边界

- 对稳定复用的长公共前缀，建议开启 APC，并在线监控实际 token hit rate；
- P1024 流量在本机达到约 45% 实际 hit 后出现明确收益，约 19% hit 时没有 TTFT 收益；
- 90% 复用下优先保留长前缀，P1792 的收益显著大于 P256；
- C8 只在高复用长前缀条件下完成容量恢复，不能推广到普通混合流量；
- 随机-token逐字一致性原门槛失败，因此性能数字属于有补充正确性证据的工程观察，
  不是对所有输入和执行路径的 bitwise 等价证明；
- 后续 E06 可验证 `max_num_batched_tokens=2048 + APC` 是否能够叠加，但必须继续
  保留固定自然语言回归集和实际 hit rate 证据。

## 8. 可复现结论

> 在 RTX 5070 12GB、WSL2、Qwen2.5-3B-Instruct BF16、vLLM 0.25.1、
> 2048-token 输入和 8192 batch-token budget 下，APC 对低复用 P1024/C4
> 流量没有可观测 TTFT 收益；当实际 token hit 为 45.52% 时，P95 TTFT 从
> 804.86 降至 610.09 ms（-24.20%），output throughput 从 260.60 提高到
> 286.54 tok/s（+9.95%）。在 78.93% hit 的 P1792/C8 条件下，P95 TTFT
> 从 1407.15 降至 631.67 ms（-55.11%），throughput 从 412.10 提高到
> 554.18 tok/s（+34.48%），并使三次重复由 SLO FAIL 变为 3/3 PASS。
> 随机-token输出只达到 91.06% 多重集合一致率，未通过原始严格门槛；固定自然语言
> canary 在 84.89% hit 下达到 24/24 OFF/ON 输出一致，但任务准确率为 22/24。
> 因而结论是“高复用长前缀下有显著性能收益，且 canary 未发现 APC 输出回归”，
> 而不是“APC 在所有输入上 bitwise 等价”。

使用 WSL 中保留的原始 artifacts 可重建报告：

```bash
make summarize-e04
make compare-e04
make diagnose-e04
make compare-e04-canary || true
```
