# E05 FP8 KV Cache 实验手册

## 1. 实验问题

E05 研究在 RTX 5070 12GB、WSL2、Qwen2.5-3B-Instruct 和 vLLM 0.25.1 条件下，将 KV Cache 从 BF16 改为 FP8 E4M3 后：

- vLLM 可分配的 GPU KV token 容量能提高多少；
- 长上下文高并发下，TTFT、TPOT、output throughput 和 SLO goodput 如何变化；
- 固定 AI Infra 故障诊断任务是否出现可测量的质量退化。

这不是权重量化实验。模型权重和计算 dtype 都保持 BF16，唯一主要自变量是 KV Cache dtype。FP8 profile 使用 vLLM 在线 scale 计算；服务 seed 固定，使启动 warmup 的随机校准可复现。

## 2. 控制变量

BF16 与 FP8 serve profile 的共同配置：

```text
model=Qwen/Qwen2.5-3B-Instruct
revision=a1d308dfcc03e09da285d49d912439a655a571e8
dtype=bfloat16
max_model_len=8192
gpu_memory_utilization=0.82
max_num_seqs=16
max_num_batched_tokens=8192
enable_prefix_caching=false
enable_chunked_prefill=true
seed=20260821
```

处理变量：

| State | KV Cache dtype | Runtime scale calculation |
|---|---|---|
| BF16 | `bfloat16` | false |
| FP8 | `fp8_e4m3` | true |

vLLM 0.25.1 的在线 scale 只使用一次随机 token warmup batch，随后固定 scale。它比默认 scale `1.0` 更合理，但不等同于使用代表性数据集和 `llm-compressor` 的离线校准。最终结论必须保留这一适用边界。

### 2.1 预实验后的配置修订

2026-08-21 的 BF16 预实验使用 `max_num_batched_tokens=16384`。`long_c16`
在一次正好包含 16,384 个 prefill token 的调度步上触发 CUDA OOM：引擎只剩
556.70 MiB 可用显存，BF16 前向还需申请 344.00 MiB 激活张量，但无法完成
分配。当时 KV cache 使用率仅 23.06%，因此根因是过大的瞬时 prefill 激活峰值，
不是 KV cache 容量耗尽。

正式配置在 BF16/FP8 两侧同步冻结为 `max_num_batched_tokens=8192`，同时保留
`max_num_seqs=16`。这项修订只降低瞬时激活显存压力，不改变实验的唯一主要
自变量（KV cache dtype）。旧配置产生的 OOM 日志和 partial manifest 作为预实验
证据保留，但由于 server config SHA-256 已改变，不得混入正式配对比较。

## 3. 性能与容量判据

性能矩阵包含 3 种长度和 2 种并发：

| Workload | Input/Output | Concurrency | 用途 |
|---|---:|---:|---|
| long | 2048/256 | 8, 16 | 常规长上下文 |
| xlong | 4096/256 | 8, 16 | KV 压力增加 |
| nearmax | 7168/256 | 8, 16 | 接近 8192 token 上限 |

每个 profile 使用 100 个请求、10 个 warmup、3 次重复，轮次间冷却 30 秒。BF16/FP8 使用相同 effective seed。SLO 保持 P95 TTFT `<=1000ms`、P95 TPOT `<=50ms/token`、error rate `<1%`。

预先冻结以下解释规则：

- 容量收益要求 FP8/BF16 启动日志 token capacity ratio `>=1.80x`；
- 性能证据要求每组 BF16/FP8 各 3 次有效重复、配对 seed、100 个请求和完整 telemetry；
- 若某个 `xlong` 或 `nearmax` profile 的 FP8 goodput 提高至少 10%，或 P95 TTFT 降低至少 10%，且 output throughput 下降不超过 5%，可记为容量压力下的性能收益；
- 原始随机 token 输出哈希只作为确定性诊断，不作为质量门槛；质量由固定自然语言任务单独判断。

## 4. 质量门槛

固定数据集 `datasets/e05_ai_infra_quality.json` 含 50 条 AI Infra 故障，覆盖 10 个平衡根因类别。每条请求必须返回严格 JSON：

```json
{
  "root_cause": "allowed_label",
  "actions": ["allowed_action_1", "allowed_action_2"],
  "dangerous_command": false
}
```

自动门槛在运行前冻结：

- BF16 schema pass rate `>=90%`；
- BF16 root-cause Macro-F1 `>=0.80`；
- BF16 action micro-F1 `>=0.75`；
- dangerous command rate `<=2%`；
- FP8 的前三项相对 BF16 各自下降不得超过 `0.02`；
- FP8 dangerous command rate 不得高于 2%。

自动比较后生成 50 条匿名 A/B 人工复核表。人工门槛为 FP8 平均分不低于 BF16 平均分 `0.10` 以上。自动门槛和人工门槛均通过，才可写“质量未发生实质退化”。

## 5. 拉取与静态检查

WSL 终端执行：

```bash
cd ~/projects/QwenServe-12G
git pull --ff-only
source .venv/bin/activate

make test
make doctor
make render-e05-bf16-local
make render-e05-fp8-local
make render-e05-bf16-matrix
make render-e05-fp8-matrix
```

如果更新前已经运行过 `max_num_batched_tokens=16384` 预实验，先按第 5.1 节归档
旧证据，再启动正式实验。

FP8 render 输出必须包含：

```text
--kv-cache-dtype fp8_e4m3
--calculate-kv-scales
--seed 20260821
```

BF16 render 输出不得包含 `--calculate-kv-scales`。两侧都必须关闭 prefix caching，避免同时改变两个优化变量。

### 5.1 归档旧 E05 预实验

不要删除 OOM 证据。在 WSL 仓库根目录执行：

```bash
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
ARCHIVE="artifacts/archive/e05-preflight-16384-${STAMP}"
mkdir -p "$ARCHIVE/env" "$ARCHIVE/results" "$ARCHIVE/server"

find artifacts/env -maxdepth 1 -type f -name '*e05_*.json' \
  -exec mv -t "$ARCHIVE/env" {} +

if [ -d artifacts/results/e05_kv_cache ]; then
  mv artifacts/results/e05_kv_cache "$ARCHIVE/results/"
fi

if [ -d artifacts/results/e05_quality ]; then
  mv artifacts/results/e05_quality "$ARCHIVE/results/"
fi

find artifacts/server -maxdepth 1 -type f \
  \( -name '*e05_*.log' -o -name 'active.json' \) \
  -exec mv -t "$ARCHIVE/server" {} +

printf 'Archived preflight evidence at %s\n' "$ARCHIVE"
```

该操作只移动 E05 原始证据，不影响 E01/E02/E04。正式 BF16 和 FP8 数据必须在
修订后的配置下都从头采集，不能复用旧 `long_c8` pilot 或旧质量集。

## 6. BF16 对照实验

终端 1 启动 BF16 服务：

```bash
cd ~/projects/QwenServe-12G
source .venv/bin/activate
make serve-e05-bf16-local
```

确认出现 `Application startup complete` 后，不要关闭终端 1。终端 2 执行 pilot：

```bash
cd ~/projects/QwenServe-12G
source .venv/bin/activate
make bench-e05-bf16-pilot
```

pilot 三次请求均成功后，终端 2 运行质量集和完整矩阵：

```bash
make run-e05-quality-bf16
make bench-e05-bf16-matrix
```

`--skip-completed` 会跳过已完成的 `long_c8` pilot。完成后在终端 1 按 `Ctrl+C`，确保 BF16 服务完全退出。

## 7. FP8 处理实验

终端 1 启动 FP8 服务：

```bash
make serve-e05-fp8-local
```

先检查启动日志同时出现 FP8 KV Cache 和 KV 容量信息。若出现 unsupported dtype、attention backend、scale calculation 或 CUDA capability 错误，保留 server manifest 与日志并停止，不要自动回退到 BF16 后继续标记为 FP8。

服务就绪后，终端 2 执行：

```bash
make bench-e05-fp8-pilot
make run-e05-quality-fp8
make bench-e05-fp8-matrix
```

完成后在终端 1 按 `Ctrl+C`。BF16 与 FP8 不能同时占用同一张 GPU，也不能在服务未重启时连续运行两种 profile。

## 8. 汇总与检查

终端 2 执行：

```bash
make summarize-e05
make compare-e05
make capacity-e05
make compare-e05-quality
```

`compare-e05-quality` 返回 2 表示质量门槛失败，但报告仍会生成；这不是脚本崩溃，也不能通过删除错误样本修复。查看：

```bash
sed -n '1,260p' reports/e05_kv_cache/summary.md
sed -n '1,260p' reports/e05_kv_cache/comparison.md
sed -n '1,180p' reports/e05_kv_cache/capacity.md
sed -n '1,220p' reports/e05_kv_cache/quality.md
```

证据数量应为 12 个性能 profile、36 次有效 benchmark，以及 BF16/FP8 各 50 条质量请求。检查异常状态：

```bash
grep -E 'INCOMPLETE|UNKNOWN' reports/e05_kv_cache/comparison.md || true
grep -E 'FAIL' reports/e05_kv_cache/quality.md || true
```

## 9. 匿名人工复核

自动比较会生成：

```text
reports/e05_kv_cache/human_review.csv
reports/e05_kv_cache/human_review_key.json
```

复核者只查看 CSV 中的 incident、参考标签和匿名 output A/B，不查看 key。对每行填写：

- `preferred`：`A`、`B` 或 `TIE`；
- `score_a_1_to_5`、`score_b_1_to_5`：1 到 5 的整数；
- `notes`：可选。

完成 50 行后执行：

```bash
make summarize-e05-human-review
sed -n '1,160p' reports/e05_kv_cache/human_review_summary.md
```

若质量结果重跑，旧复核表已填写时比较器会拒绝覆盖。先将旧 CSV、key 和 summary 一起归档，再生成新表，避免人工评分与模型输出错配。

## 10. 结果提交与退出条件

只提交报告和文档，不提交 `artifacts/results`、server log 或 telemetry 原始文件：

```bash
git add \
  reports/e05_kv_cache/runs.csv \
  reports/e05_kv_cache/summary.md \
  reports/e05_kv_cache/comparison.csv \
  reports/e05_kv_cache/comparison.md \
  reports/e05_kv_cache/capacity.json \
  reports/e05_kv_cache/capacity.md \
  reports/e05_kv_cache/quality.json \
  reports/e05_kv_cache/quality.md \
  reports/e05_kv_cache/human_review.csv \
  reports/e05_kv_cache/human_review_key.json \
  reports/e05_kv_cache/human_review_summary.json \
  reports/e05_kv_cache/human_review_summary.md
git commit -m "Add RTX 5070 FP8 KV cache results"
git push origin main
```

E05 退出条件：

- 两种服务配置均有可追溯的 manifest、配置 SHA-256 和完整启动日志；
- 12 个性能 profile、36 次 benchmark 证据完整；
- 容量报告能从两侧启动日志解析 token capacity；
- 两侧质量集的 dataset SHA-256 和 50 个 prompt SHA-256 完全一致；
- 自动质量报告和匿名人工评分均完成；
- 启动失败、质量回归、SLO 失败和无收益 profile 全部保留；
- 最终结论同时说明容量、性能、质量、错误率和在线 scale 校准边界。

只有容量证据而没有质量证据时，E05 状态必须保持 `INCOMPLETE`。
