# E01/E03 BF16 基线与并发容量实验结果

## 1. 问题与状态

E01 在 RTX 5070 12GB、WSL2、Qwen2.5-3B-Instruct BF16 和 vLLM
0.25.1 上建立受控服务基线。矩阵同时覆盖原 E03 计划的长度与
并发容量问题，因此 E03 直接复用 E01 的 `3 workload x 4 concurrency`
证据，不重复计数。

状态为 **COMPLETE_WITH_PROTOCOL_DEVIATION**。共有 12 个 profile、36 次
正式 benchmark，全部完成 100 个请求、错误率为 0，GPU telemetry
完整。原 E01 清单中的 Transformers 单请求参考没有执行，因此本报告
不声称 vLLM 相对 Transformers 的加速比。后续实验的对照均为固定
vLLM 配置，不受该缺口影响。

## 2. 证据与固定配置

- 模型：`Qwen/Qwen2.5-3B-Instruct`，固定 revision
  `a1d308dfcc03e09da285d49d912439a655a571e8`；
- 精度：BF16 权重与 BF16 KV Cache；
- 调度：`max_num_seqs=8`、`max_num_batched_tokens=8192`、chunked prefill；
- 工作负载：128/128、512/256、2048/256 tokens；
- 并发：1、4、8、16；
- 重复：每个 profile 3 次，每次 100 个请求；
- SLO：P95 TTFT `<=1000 ms`，P95 TPOT `<=50 ms/token`，错误率 `<1%`。

机器可读数据见 `reports/baseline/runs.csv`，汇总见
`reports/baseline/summary.md`。

## 3. 基线拐点

| Workload | C | Output tok/s | P95 TTFT ms | P95 TPOT ms | Goodput req/s | SLO |
|---|---:|---:|---:|---:|---:|---|
| 128/128 | 8 | 627.27 | 114.32 | 12.07 | 4.90 | PASS |
| 128/128 | 16 | 630.99 | 1676.66 | 12.09 | 0.39 | FAIL |
| 512/256 | 8 | 583.30 | 389.07 | 13.03 | 2.28 | PASS |
| 512/256 | 16 | 584.99 | 3772.68 | 13.03 | 0.18 | FAIL |
| 2048/256 | 4 | 264.83 | 798.11 | 14.30 | 1.03 | PASS |
| 2048/256 | 8 | 418.14 | 1580.83 | 17.84 | 0.78 | FAIL |
| 2048/256 | 16 | 418.65 | 5968.28 | 17.86 | 0.07 | FAIL |

12 个 profile 中 8 个通过 SLO，4 个失败。Short/Medium 在 C8 时仍满足
SLO，提高到 C16 后 output throughput 只增长 0.59%/0.29%，但 P95
TTFT 分别上升到 1676.66/3772.68 ms，goodput 大幅下降。

Long 负载在 C4 时已接近 1000 ms TTFT 边界；C8 虽然将 output
throughput 从 264.83 提高到 418.14 tok/s，却因排队使 P95 TTFT 达到
1580.83 ms，goodput 从 1.03 降至 0.78 req/s。C16 的 throughput 基本
不再增长，TTFT 则达到 5968.28 ms。

## 4. 对 RQ1/E03 的回答

- 长输入首先放大 prefill 和 TTFT，decode TPOT 增长相对温和；
- 在 GPU 计算吞吐接近饱和后，继续提高并发主要增加排队时间；
- raw throughput 和 goodput 会在过载区间分离，因此不能仅用 tok/s
  选择并发上限；
- 对 2048/256 负载，C4 是本基线的可用闭环并发点；C8 和 C16
  需要调度优化、稳定前缀复用或准入控制。

## 5. 资源边界

基线峰值显存为 11,675 MiB，已接近 12,227 MiB 物理上限。因此
后续实验不能把 KV Cache、activation、CUDA Graph private pool 和 Windows
图形占用视为互不相干的资源。E06 预实验的 OOM 进一步验证了这一点。

## 6. 可复现结论

> 在 RTX 5070 12GB、WSL2、Qwen2.5-3B-Instruct BF16 和 vLLM
> 0.25.1 下，短/中输入的有效并发上限为 C8，C16 在 throughput
> 基本不增长时已违反 TTFT SLO。2048/256 长输入在 C4 时 P95
> TTFT 为 798.11 ms 并通过 SLO；C8 时 throughput 提高到
> 418.14 tok/s，但 TTFT 升至 1580.83 ms、goodput 降至 0.78 req/s。
> 这证明单卡服务的容量点必须由 SLO goodput 而不是原始吞吐确定。

使用 WSL 中保留的原始 artifacts 可重建汇总：

```bash
make summarize
```
