# E06 Batch Budget + APC 组合优化实验手册

## 1. 实验问题

E02 表明 `max_num_batched_tokens=2048` 在长输入下可降低 TTFT，同时基本保持
output throughput；E04 表明 Automatic Prefix Caching（APC）只在实际 token
hit rate 足够高时产生明显收益。E06 回答：

> 2048 batch-token budget 与 APC 的收益能否叠加，并在相同请求轨迹上优于任一
> 单项优化？

E05 的 FP8 KV Cache 因自动质量和人工质量门槛失败，不进入 E06。四组服务都使用
BF16 权重和 BF16 KV Cache。

## 2. 2x2 因子设计

四个 cell 为：

| Cell | Batch token budget | APC | 角色 |
|---|---:|---|---|
| A | 8192 | OFF | 未优化对照 |
| B | 2048 | OFF | batch budget 单项优化 |
| C | 8192 | ON | APC 单项优化 |
| D | 2048 | ON | 组合处理 |

除两个实验因子外，四组配置共同冻结：

```text
model=Qwen/Qwen2.5-3B-Instruct
revision=a1d308dfcc03e09da285d49d912439a655a571e8
dtype=bfloat16
kv_cache_dtype=bfloat16
max_model_len=8192
gpu_memory_utilization=0.82
max_num_seqs=8
enable_chunked_prefill=true
attention_backend=TRITON_ATTN
seed=20260823
```

使用 `TRITON_ATTN` 是为了让四组共享已经在 RTX 5070 `sm120` 上验证过的后端，
避免自动后端选择成为隐藏变量。E06 不使用 FP8，也不计算 KV scales。

## 3. 工作负载与证据数量

主矩阵固定 C4、P1024，只改变名义前缀复用率：

| Condition | Prefix/Suffix | Prefixes/Prompts | Nominal reuse | C |
|---|---:|---:|---:|---:|
| reuse0_p1024 | 1024/1024 | 100/100 | 0% | 4 |
| reuse50_p1024 | 1024/1024 | 50/100 | 50% | 4 |
| reuse90_p1024 | 1024/1024 | 10/100 | 90% | 4 |

容量矩阵固定高复用长前缀：

| Condition | Prefix/Suffix | Prefixes/Prompts | Nominal reuse | C |
|---|---:|---:|---:|---:|
| capacity_reuse90_p1792 | 1792/256 | 10/100 | 90% | 8 |

每个 cell 有 4 个 profile，每个 profile 3 次重复，共：

```text
4 cells x 4 profiles x 3 repetitions = 48 benchmark runs
```

每次运行包含 100 个正式请求、10 个隔离 prewarm 请求、30 秒冷却、GPU telemetry、
per-request metrics 和 prefix-cache counter delta。四个 cell 使用完全相同的
effective seed。

## 4. 冻结判据

### 4.1 证据有效性

每个 condition 只有同时满足以下条件才标记 `VALID`：

- A/B/C/D 各有 repetition 1、2、3；
- 四组 effective seed 完全配对且没有重复；
- input/output、prefix/suffix、num_prefixes 和并发一致；
- server profile、server config SHA-256 和 benchmark config SHA-256 一致可追溯；
- 100 个请求完整、error rate `<1%`、GPU telemetry 完整；
- C/D 的 prefix query/hit token counter 完整且范围有效。

随机-token输出哈希继续报告，但不作为性能证据有效性的门槛。E04 已证明调度变化
可能造成少量随机输出差异，因此 E06 用固定自然语言 canary 单独冻结配置等价性。

### 4.2 组合收益

“最佳单项”按指标分别取 B/C 中更好的值：延迟取较低值，吞吐和 goodput 取较高值。
`STACKED_BENEFIT` 要求：

- 四 cell 证据为 `VALID`；
- D 的 output throughput 相对最佳单项下降不超过 2%；
- 并且 D 相对最佳单项满足以下至少一项：
  - P95 TTFT 降低至少 5%；
  - request goodput 提高至少 10%。

SLO 独立报告：D 的三次重复都必须满足 P95 TTFT `<=1000 ms` 和 P95 TPOT
`<=50 ms/token` 才标记 `PASS`。组合收益不能覆盖 SLO FAIL。

### 4.3 因子交互

比较器计算 APC 在两个 budget 下的相对效果，并报告：

```text
interaction = APC effect at 2048 - APC effect at 8192
```

TTFT interaction 为负表示 APC 在 2048 budget 下带来更大的延迟降低；throughput
interaction 为正表示 APC 在 2048 budget 下带来更大的吞吐提升。这是描述性
percentage-point 交互，不替代三次重复的完整证据检查。

### 4.4 固定 canary

复用 `datasets/e04_correctness_canary.json` 的 24 条固定自然语言查表任务。四个 cell
必须使用相同 dataset SHA-256 和 prompt SHA-256，且每条 A/B/C/D 输出一致；C/D
还必须观察到实际 prefix cache hit。Base 模型在四侧共同出现的相同错误会报告为
任务能力限制，但不算配置回归。

## 5. 拉取与静态检查

WSL 终端执行：

```bash
cd ~/projects/QwenServe-12G
git switch codex/e06-combined
git pull --ff-only
source .venv/bin/activate

make test
make doctor

make render-e06-bt8192-off-local
make render-e06-bt2048-off-local
make render-e06-bt8192-on-local
make render-e06-bt2048-on-local

make render-e06-bt8192-off-matrix
make render-e06-bt2048-off-matrix
make render-e06-bt8192-on-matrix
make render-e06-bt2048-on-matrix
```

确认四个 serve command 都包含 `--kv-cache-dtype bfloat16`、
`--attention-config '{"backend":"TRITON_ATTN"}'` 和相同 seed。只有 B/D 包含
`--max-num-batched-tokens 2048`，只有 C/D 包含 `--enable-prefix-caching`。

## 6. Cell A：8192/OFF

终端 1：

```bash
cd ~/projects/QwenServe-12G
source .venv/bin/activate
make serve-e06-bt8192-off-local
```

出现 `Application startup complete` 后，终端 2：

```bash
cd ~/projects/QwenServe-12G
source .venv/bin/activate
make bench-e06-bt8192-off-pilot
make run-e06-canary-bt8192-off
make bench-e06-bt8192-off-matrix
```

完成后在终端 1 按 `Ctrl+C`，确认服务退出。

## 7. Cell B：2048/OFF

终端 1：

```bash
make serve-e06-bt2048-off-local
```

终端 2：

```bash
make bench-e06-bt2048-off-pilot
make run-e06-canary-bt2048-off
make bench-e06-bt2048-off-matrix
```

完成后停止终端 1 的服务。

## 8. Cell C：8192/ON

终端 1：

```bash
make serve-e06-bt8192-on-local
```

终端 2：

```bash
make bench-e06-bt8192-on-pilot
make run-e06-canary-bt8192-on
make bench-e06-bt8192-on-matrix
```

终端输出和 manifest 必须记录 prefix cache query/hit token delta。完成后停止服务。

## 9. Cell D：2048/ON

终端 1：

```bash
make serve-e06-bt2048-on-local
```

终端 2：

```bash
make bench-e06-bt2048-on-pilot
make run-e06-canary-bt2048-on
make bench-e06-bt2048-on-matrix
```

完成后停止服务。任意时刻只能有一个 vLLM 服务占用 GPU；切换 cell 必须重启服务，
不能在同一进程中动态修改配置。

## 10. 汇总与检查

四组全部完成后：

```bash
make summarize-e06
make compare-e06
make compare-e06-canary

sed -n '1,260p' reports/e06_combined/summary.md
sed -n '1,300p' reports/e06_combined/comparison.md
sed -n '1,220p' reports/e06_combined/correctness_canary.md
```

检查数量和异常状态：

```bash
test "$(($(wc -l < reports/e06_combined/runs.csv) - 1))" -eq 48
grep -E 'INCOMPLETE|UNKNOWN' reports/e06_combined/comparison.md || true
grep -E 'FAIL' reports/e06_combined/correctness_canary.md || true
```

`compare-e06-canary` 返回 2 表示配置等价性或质量无回归门槛失败，但报告仍会生成。
保留失败证据，不删除样本或修改冻结门槛。

## 11. 中断恢复

所有 matrix target 都带 `--skip-completed`。服务配置和 matrix TOML 没有变化时，
重复执行相同命令只补齐缺失 repetition。若修改任何 serve 或 matrix 配置，SHA-256
会变化，旧结果不得混入新汇总；应先把旧 E06 manifest、results、server log 和报告
整体移动到 `artifacts/archive/e06-<timestamp>/`。

## 12. 提交与退出条件

只提交汇总报告，不提交原始 results、telemetry 或 server log：

```bash
git add \
  reports/e06_combined/runs.csv \
  reports/e06_combined/summary.md \
  reports/e06_combined/comparison.csv \
  reports/e06_combined/comparison.md \
  reports/e06_combined/correctness_canary.json \
  reports/e06_combined/correctness_canary.md

git commit -m "Add RTX 5070 E06 combined optimization results"
git push origin HEAD
```

E06 退出条件：

- 16 个 profile、48 次 benchmark 均有完整可追溯证据；
- 四个 cell 的 seed、shape、请求数和环境一致；
- APC 两侧有实际 token hit 证据；
- 4 行 factorial comparison 均为 `VALID`；
- 固定 canary 明确给出配置等价性与任务质量无回归状态；
- 最终结论同时比较 D 与 A、D 与最佳单项，并报告 SLO 和交互项；
- `NO_STACKED_BENEFIT` 或 canary FAIL 同样作为有效结论保留。
