# E04 Automatic Prefix Caching 实验手册

## 1. 实验问题

E04 回答两个边界问题：在总输入长度固定为 2048 token 时，公共前缀复用率达到多少才值得开启 Automatic Prefix Caching（APC）；在高复用率下，公共前缀长度如何改变收益？

本实验比较完全配对的 cache OFF/ON 服务。主要指标为 P95 TTFT、SLO goodput 和 output token throughput；同时记录 P95 TPOT、峰值显存和 vLLM 实际 token 级 prefix cache hit rate。仅凭名义复用率不能证明缓存生效。

## 2. 控制变量与判定规则

两个 serve profile 除 `enable_prefix_caching` 外必须完全相同：

```text
model=Qwen2.5-3B-Instruct
dtype=bfloat16
max_model_len=8192
gpu_memory_utilization=0.82
max_num_seqs=8
max_num_batched_tokens=8192
kv_cache_dtype=bfloat16
chunked_prefill=true
temperature=0
```

每个 profile 执行 100 个正式请求、3 次重复，重复间冷却 30 秒。SLO 固定为 P95 TTFT `<=1000ms`、P95 TPOT `<=50ms/token`、error rate `<1%`。

预定义 `BENEFIT` 规则：

- OFF/ON 各有 3 次有效重复；
- 每次重复使用相同 effective seed，且生成文本哈希完全一致；
- ON 的 P95 TTFT 中位数至少下降 5%；
- ON 的 output token throughput 中位数下降不超过 2%。

阈值必须在看结果之前冻结。若缓存只提高某个辅助指标但不满足上述规则，报告为 `NO_BENEFIT`，不事后修改门槛。

## 3. 工作负载矩阵

总输入固定为 2048 token，输出固定为 256 token，主矩阵并发为 C4：

| 条件 | Prefix/Suffix | Prefix 数 | 名义请求复用率 |
|---|---:|---:|---:|
| reuse0_p1024 | 1024/1024 | 100 | 0% |
| reuse50_p1024 | 1024/1024 | 50 | 50% |
| reuse90_p1024 | 1024/1024 | 10 | 90% |
| reuse90_p256 | 256/1792 | 10 | 90% |
| reuse90_p1792 | 1792/256 | 10 | 90% |

`reuse90_p1024` 同时属于复用率 sweep 和前缀长度 sweep，只执行一次。最后用 `reuse90_p1792` 在 C8 做容量验证，因此共有 12 个 OFF/ON profile、36 次正式 benchmark。

名义请求复用率定义为 `(num_prompts - num_prefixes) / num_prompts`。实际命中率使用正式请求前后 `/metrics` 中 prefix hit/query token counter 的差值计算，两者不是同一个量。

## 4. 缓存污染控制

`vllm bench serve` 内置 warmup 会提前填充待测前缀，因此 E04 固定 `num_warmups=0`。执行器在每次正式重复前发送 10 个独立的 random 请求用于稳定 CUDA 路径，然后才采集 metrics before snapshot；这些请求不进入正式指标和命中率差值。

每个工作负载和 C8 容量实验使用不同 seed 空间，每次重复再增加 100。OFF/ON 配对使用相同 seed。这样缓存开启时，前一 profile 或前一重复中保留的 KV block 不会与下一轮待测前缀重合。

不得手工修改 seed、warmup、profile 顺序或把 E04 请求发给其他正在运行的服务。

## 5. 拉取与静态检查

WSL 终端执行：

```bash
cd ~/projects/QwenServe-12G
git pull --ff-only
source .venv/bin/activate

make test
make render-e04-off-local
make render-e04-on-local
make render-e04-off-matrix
make render-e04-on-matrix
```

render 输出必须包含 `--dataset-name prefix_repetition`、对应的 `--prefix-repetition-*` 参数、`--num-warmups 0`、不同条件 seed 以及本地 `--tokenizer`。

## 6. 配对 pilot

先验证一组 OFF/ON 配对，避免完整矩阵结束后才发现 metrics 或详细输出证据缺失。

终端 1 启动 OFF 服务：

```bash
make serve-e04-off-local
```

确认 `Application startup complete` 后，终端 2 执行：

```bash
cd ~/projects/QwenServe-12G
source .venv/bin/activate
make bench-e04-off-pilot
```

完成后在终端 1 按 `Ctrl+C`，再启动 ON 服务：

```bash
make serve-e04-on-local
```

终端 2 执行：

```bash
make bench-e04-on-pilot
make summarize-e04
make compare-e04
sed -n '1,200p' reports/e04_prefix_cache/comparison.md
```

pilot 退出门槛：`reuse90_p1024` 显示 3+3 runs、`Evidence=VALID`、`Output=MATCH`，实际 token hit rate 不是 `NA`。数值是否达到 `BENEFIT` 不影响流程有效性。

## 7. 正式矩阵

如果 ON 服务仍在终端 1 运行，终端 2 继续：

```bash
make bench-e04-on-matrix
make bench-e04-on-capacity
```

`--skip-completed` 会跳过 ON pilot 的三次有效重复。完成后在终端 1 按 `Ctrl+C`，启动 OFF 服务：

```bash
make serve-e04-off-local
```

终端 2 执行：

```bash
make bench-e04-off-matrix
make bench-e04-off-capacity
```

不得让 OFF 矩阵连接 ON 服务，反之亦然。每次 benchmark 都会核对 active server profile 与 server config SHA-256，不匹配时会在发送正式请求前失败。

## 8. 汇总、审计与提交

完成 36 次正式 benchmark 后：

```bash
make summarize-e04
make compare-e04
sed -n '1,280p' reports/e04_prefix_cache/summary.md
sed -n '1,260p' reports/e04_prefix_cache/comparison.md
```

检查 profile 数和证据：

```bash
awk -F, 'NR==1 {for(i=1;i<=NF;i++){if($i=="profile")p=i;if($i=="valid")v=i}next} {profiles[$p]=1;if($v=="True")valid++} END {print "profiles=" length(profiles), "valid_runs=" valid}' reports/e04_prefix_cache/runs.csv
grep -E 'INCOMPLETE|MISMATCH|UNKNOWN' reports/e04_prefix_cache/comparison.md || true
```

只提交可重建的小型报告，不提交原始 benchmark JSON、Prometheus snapshot 或 telemetry：

```bash
git add reports/e04_prefix_cache docs/M2_E04_PREFIX_CACHE_RUNBOOK.md README.md
git commit -m "Add RTX 5070 prefix cache results"
git push origin main
```

E04 退出条件：

- 12 个 profile、36 次正式 benchmark 均可追溯到唯一配置哈希；
- 每组 OFF/ON 都是 3 次相同 seed 的配对结果；
- ON 每轮都有实际 token hit rate，所有输出均为 `MATCH`；
- 结论给出复用率阈值、前缀长度影响、C8 容量结果和不适用边界；
- 未达到收益门槛的条件和失败实验同样保留。
