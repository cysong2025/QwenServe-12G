# E02 Batch Token Budget 实验手册

## 1. 实验问题

E02 回答一个单变量问题：在 RTX 5070 12GB、Qwen2.5-3B-Instruct BF16 和
`max_num_seqs=8` 不变时，`max_num_batched_tokens` 如何改变 TTFT、TPOT、
throughput 和 SLO goodput？

正式对照值为 `2048` 、`4096` 和 `8192`。`16384` 作为探索性上界：
启动成功后才运行矩阵；若发生 OOM，保留 server manifest 和日志作为资源边界证据。

## 2. 控制变量

四个 serve profile 除 `max_num_batched_tokens` 外必须完全相同：

```text
model=Qwen2.5-3B-Instruct
dtype=bfloat16
max_model_len=8192
gpu_memory_utilization=0.82
max_num_seqs=8
kv_cache_dtype=bfloat16
prefix_caching=false
chunked_prefill=true
temperature=0
```

`chunked_prefill=true` 必须显式写入服务命令。否则 `2048/4096` 小于
`max_model_len=8192` 时，服务行为不受本实验配置完整控制。

## 3. 工作负载

每个 budget 都运行相同的六个 profile：

| Workload | Input/Output | Concurrency |
|---|---:|---:|
| short | 128/128 | 4, 8 |
| medium | 512/256 | 4, 8 |
| long | 2048/256 | 4, 8 |

每个 profile 预热 10 个请求，正式执行 100 个请求、3 次重复，重复间冷却
30 秒。SLO 继续固定为 P95 TTFT `<=1000ms`、P95 TPOT `<=50ms/token`、
error rate `<1%`。

## 4. 执行顺序

首先拉取代码并做无 GPU 校验：

```bash
cd ~/projects/QwenServe-12G
git pull --ff-only
source .venv/bin/activate
make test
```

建议顺序为 `8192 -> 2048 -> 4096 -> 16384`。对每个 budget 都执行以下流程。

终端 1：

```bash
make render-e02-local E02_BUDGET=8192
make serve-e02-local E02_BUDGET=8192
```

确认日志出现 `Application startup complete`。终端 1 保持运行。

终端 2：

```bash
cd ~/projects/QwenServe-12G
source .venv/bin/activate

make render-e02-matrix E02_BUDGET=8192
make bench-e02-pilot E02_BUDGET=8192
make bench-e02-matrix E02_BUDGET=8192
```

pilot 完成 `medium_c8` 的三次重复。正式矩阵带有 `--skip-completed`，会复用这三次
有效结果，不会重复执行。

`run-serve` 会写入 `artifacts/server/active.json`。每个 benchmark 在发送请求前都会校验
当前 server profile 和 SHA-256 是否与 matrix 完全一致；不一致时直接失败，
防止将某个 budget 的结果错误归因给另一个服务配置。

完成后在终端 1 按 `Ctrl+C`，然后把上述命令中的值依次替换为 `2048`、
`4096` 和 `16384`。不得在同一个 server 进程上运行另一个 budget 的矩阵。

## 5. 16384 安全门槛

`16384` 启动时先查看完整的 CUDA Graph 捕获和 server ready 日志。若出现 OOM、
engine core initialization failure 或显存分配失败：

1. 不要降低其他参数后继续标记为同一 profile。
2. 保留 `artifacts/env/*server-e02_batch_tokens_16384*` 和对应 server log。
3. 跳过 `bench-e02-pilot E02_BUDGET=16384`，将结论记为当前 12GB 配置下不可行。

## 6. 汇总与退出条件

完成所有可行的 budget 后：

```bash
make summarize-e02
make compare-e02
sed -n '1,260p' reports/e02_batch_tokens/summary.md
sed -n '1,320p' reports/e02_batch_tokens/comparison.md
```

`summarize-e02` 从 manifest、原始 JSON 和 telemetry CSV 重建逐轮 `runs.csv`，
其中包含平均 SM 时钟。`compare-e02` 以 8192 为参照，自动生成各 budget 的
throughput、TTFT、TPOT、goodput 和峰值显存差值。若温度、功耗或 SM 时钟
存在显著批次差异，报告会显式标记潜在混杂因素。

E02 退出条件：

- `2048/4096/8192` 各有 6 个 profile，每个 profile 有 3 次有效重复；
- 每组结果能追溯到唯一的 serve 和 matrix SHA-256；
- 结论同时报告 throughput 变化和 SLO goodput 变化；
- 不将 `16384` 的启动失败静默删除；
- 能解释不同 input length 下最优 budget 不同的原因。
