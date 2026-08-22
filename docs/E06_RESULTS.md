# E06 Batch Budget + APC 组合优化实验结果

## 1. 问题与状态

E06 使用 2x2 因子设计，在 BF16 权重、BF16 KV Cache 和
`TRITON_ATTN` 不变时比较：

| Cell | Batch-token budget | APC |
|---|---:|---|
| A | 8192 | OFF |
| B | 2048 | OFF |
| C | 8192 | ON |
| D | 2048 | ON |

实验回答较小 batch-token budget 和 Automatic Prefix Caching 的收益
能否叠加。最终状态为 **COMPLETE**：16 个 profile、48 次正式
benchmark 全部 `VALID`，错误率为 0，四组固定 canary 为 `PASS`。

## 2. OOM 预实验与配置修订

首次 A cell 预实验在 `gpu_memory_utilization=0.82` 下中止。CUDA 默认
异步执行，原始堆栈只在 `copy_event.synchronize()` 报告
`cudaErrorUnknown`。使用 `CUDA_LAUNCH_BLOCKING=1` 的不保存结果诊断将
根因定位为：

```text
Scheduler step: 6145 tokens
Additional allocation: 130 MiB activation tensor
Failure: torch.OutOfMemoryError
```

为不改变 `8192 vs 2048` 调度因子，四个 cell 统一将 KV Cache 预留
比例从 `0.82` 降为 `0.78`，释放约 0.48 GiB activation/CUDA Graph
余量。旧预实验整体归档且没有混入正式汇总。修订后最高峰值显存
为 11,077 MiB，48 次运行全部完成。

## 3. 组合收益

`STACKED_BENEFIT` 要求 D 相对 B/C 中的指标最优单项：throughput
下降不超过 2%，且 TTFT 至少降低 5% 或 goodput 至少提高 10%。

| Condition | C | Actual hit C/D | A/B/C/D P95 TTFT ms | D vs best | A/B/C/D tok/s | Decision |
|---|---:|---:|---:|---:|---:|---|
| reuse0 P1024 | 4 | 0.98/0.98% | 776.99/628.45/761.28/630.51 | +0.33% | 287.75/287.15/288.87/288.11 | NO_STACKED_BENEFIT |
| reuse50 P1024 | 4 | 19.27/20.26% | 768.98/755.28/755.21/577.36 | -23.55% | 288.20/287.00/298.54/299.39 | STACKED_BENEFIT |
| reuse90 P1024 | 4 | 45.54/45.54% | 767.03/581.46/485.15/466.82 | -3.78% | 288.67/289.27/319.67/317.67 | NO_STACKED_BENEFIT |
| reuse90 P1792 | 8 | 78.94/78.94% | 1513.23/845.64/670.23/597.97 | -10.78% | 447.71/447.54/618.11/618.77 | STACKED_BENEFIT |

组合收益在 `reuse50_p1024/C4` 和 `capacity_reuse90_p1792/C8` 成立。
对无复用负载，D 与 budget-only B 基本等价，说明 APC 不会从无可复用
token 的流量中创造收益。

90% 名义复用/P1024 下，D 的绝对 TTFT 最低，但 APC-only C 已获得
大部分收益，D 相对 C 只降低 3.78%，未达到预先冻结的 5% 门槛。
这是“有进一步改善，但证据不支持标记为叠加收益”，不是数据失败。

## 4. 容量场景

在 C8/P1792 容量条件下：

- A 的 P95 TTFT 为 1513.23 ms，SLO FAIL；
- B 将 TTFT 降至 845.64 ms；
- C 将 TTFT 降至 670.23 ms，throughput 提高到 618.11 tok/s；
- D 进一步将 TTFT 降至 597.97 ms，throughput 为 618.77 tok/s；
- D 相对 A 的 TTFT 降低 60.48%，throughput 提高 38.21%，goodput
  提高 137.53%。

这是 E06 最强的部署结论：高复用长前缀下，APC 提供主要吞吐收益，
较小 batch-token budget 继续改善尾延迟，两者组合将原始容量失败点恢复为
3/3 SLO PASS。

## 5. 质量与等价性

固定自然语言 canary 得到：

- dataset 和 prompt 哈希完全一致；
- 四个 cell 的 24/24 输出完全一致；
- APC 两侧都观察到 84.89% prefix hit；
- 四侧均为 22/24 任务正确，两条基础模型错误完全相同。

因此配置等价性和任务质量无回归门槛均 PASS。随机-token 性能负载的
输出哈希继续作为诊断信号，不替代固定 canary。

## 6. 部署决策

- 通用或低复用流量：优先 BF16 KV + 2048 budget，不把 APC 计为必然收益；
- 稳定模板/RAG 长前缀：使用 BF16 KV + 2048 budget + APC，并监控实际 token hit；
- 2048/256 普通流量默认不超过 C4；C8 需要高复用长前缀或后续准入控制；
- E05 的 FP8 KV 质量门槛已失败，不进入该默认部署组合。

## 7. 可复现结论

> 在 RTX 5070 12GB、WSL2、Qwen2.5-3B-Instruct BF16、vLLM 0.25.1、
> `TRITON_ATTN` 和 `gpu_memory_utilization=0.78` 下，2048 batch-token
> budget 与 APC 的组合收益是负载相关的。它在约 20% 实际 hit 的
> P1024/C4 条件下使 P95 TTFT 相对最佳单项降低 23.55%；
> 在 78.94% hit 的 P1792/C8 条件下降低 10.78%，且保持
> throughput、完成 3/3 SLO PASS。无复用时没有叠加收益；45.54%
> hit/P1024 时 APC 单项已获得大部分收益，组合的额外 TTFT 改善
> 只有 3.78%，未达冻结门槛。四组固定 canary 24/24 输出一致。

使用 WSL 中保留的原始 artifacts 可重建报告：

```bash
make summarize-e06
make compare-e06
make compare-e06-canary
```
