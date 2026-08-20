# E02 Batch Token Budget 实验结果

## 1. 问题与证据

E02 在 RTX 5070 12GB、WSL2、Qwen2.5-3B-Instruct BF16、vLLM 0.25.1
和 `max_num_seqs=8` 不变时，比较 `max_num_batched_tokens` 为 2048、4096、
8192 和 16384 的服务行为。四组配置都显式开启 chunked prefill，关闭
prefix caching。

证据集包含 4 个 budget、6 个负载形状和每组 3 次重复，共 72 次
正式 benchmark。全部请求错误率为 0，所有 profile 都通过配置一致性、
重复完整性和 GPU telemetry 完整性检查。

原始汇总见 `reports/e02_batch_tokens/runs.csv`，自动差值见
`reports/e02_batch_tokens/comparison.md`。

## 2. 核心结果

2048 相对 8192 的主要变化：

| In/Out | C | Output tok/s delta | P95 TTFT delta | Goodput delta | Peak VRAM delta | SLO |
|---:|---:|---:|---:|---:|---:|---|
| 128/128 | 8 | +0.03% | +3.15% | +0.03% | -323 MiB | PASS |
| 512/256 | 8 | -0.14% | -2.33% | -0.14% | -164 MiB | PASS |
| 2048/256 | 4 | -0.18% | -21.23% | -0.18% | -758 MiB | PASS |
| 2048/256 | 8 | +0.14% | -47.74% | +164.19% | -878 MiB | FAIL |

在 short 和 medium 负载下，2048 和 8192 的 throughput、TPOT 和 goodput 差异
不足 1%，在本实验精度下视为等价。在 long 负载下，较小的 budget
通过更细粒度的 chunked prefill 显著降低 TTFT，同时保持近似相同的
output throughput。

4096 在 Long-C4 下得到最稳定的低 TTFT：`614.36
[613.91, 614.42] ms`，但 output throughput 比 8192 低 1.70%。16384 没有
在任何已测负载下形成可观测的性能优势。

## 3. SLO 边界

Short-C4/C8、Medium-C4/C8 和 Long-C4 在所有 budget 下都满足冻结的
P95 TTFT/TPOT SLO。Long-C8 在所有 budget 下都失败。

2048 的 Long-C8 P95 TTFT 中位数为 827.53 ms，但三次重复中有一次为
1005.24 ms。根据预先冻结的“每次重复都必须满足 P95”规则，该配置
仍为 SLO FAIL。因此 batch token budget 调整不能替代长请求的并发准入控制。

## 4. 有效性限制

4096/16384 的温度中位数比 8192 高 4.0/4.5 C，平均功耗低
3.17/3.00 W。平均 SM 时钟仅低 8/7 MHz，差异不足 0.3%，没有
明显降频证据。

运行批次状态可能影响 2% 以内的 throughput 差异，因此本报告不将
4096/16384 的小幅吞吐下降解释为确定的调度器因果效应。长输入
TTFT 的 21% 至 48% 变化远大于该混杂范围，可作为 E02 的主要结论。

## 5. 决策

- short/medium 负载下，2048 与 8192 在当前精度下等价；
- long 负载下，2048 是延迟与吞吐的首选候选，但并发应限制为 4；
- 16384 不进入后续组合优化候选；
- E04 prefix caching 继续使用 8192 作为单因素对照；
- E06 组合优化再验证 `2048 + prefix caching` 在混合长度流量下是否可叠加。

## 6. 可复现结论

> 在 RTX 5070 12GB、WSL2、Qwen2.5-3B-Instruct BF16、vLLM 0.25.1、
> `max_num_seqs=8` 和固定的 2048/256 Long-C8 闭环负载下，相比
> `max_num_batched_tokens=8192`，配置 2048 使 P95 TTFT 中位数从
> 1583.56 ms 降至 827.53 ms（-47.74%），output throughput 从 418.18
> 变为 418.76 tok/s（+0.14%），峰值显存从 11682 降至 10804 MiB。
> 但其中一次重复的 P95 TTFT 为 1005.24 ms，因此在预先冻结的
> 1000 ms SLO 下仍不能稳定承载 Long-C8。
