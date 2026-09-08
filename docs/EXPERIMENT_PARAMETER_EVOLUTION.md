# QwenServe-12G 实验参数演进与项目总结

> 本文回答四个问题：每轮实验调了什么参数，这些参数控制什么，为什么这样调，最终获得了什么收益或负面结论。所有数字均来自仓库内已提交的配置与报告，不把不同负载、不同统计口径的结果混在一起比较。

## 1. 项目背景与目标

QwenServe-12G 面向一台 RTX 5070 12GB 的有限资源环境，研究如何把 Qwen2.5-3B-Instruct 部署成一个可测量、可优化、可复现的 vLLM 服务。项目关注的不是单一的最高吞吐，而是：

1. 请求能否在服务等级目标（SLO）内完成；
2. 优化是否同时保持生成质量和错误率；
3. 配置在长上下文、并发和突发流量下是否仍然有效；
4. 每个结论能否通过冻结配置、seed、重复实验和报告重新验证。

固定的主要 SLO 为：

- P95 TTFT 不超过 1000 ms；
- P95 TPOT 不超过 50 ms；
- admitted error rate 低于 1%；
- SLO goodput 只统计成功完成且同时满足 TTFT、TPOT 的请求。

项目最终形成了从服务参数、缓存精度、LoRA 训练到外置准入控制的一条完整实验链，而不是只完成一次模型启动或单点 benchmark。

## 2. 固定环境与实验纪律

### 2.1 硬件与软件

| 项目 | 固定值 |
|---|---|
| GPU | NVIDIA GeForce RTX 5070，12227 MiB |
| 操作系统 | Windows + WSL2 Ubuntu 24.04 |
| 模型 | Qwen/Qwen2.5-3B-Instruct |
| 模型 revision | `a1d308dfcc03e09da285d49d912439a655a571e8` |
| 推理框架 | vLLM 0.25.1 |
| Python | 3.12 |
| 模型权重与计算精度 | BF16，除非实验明确说明 |
| 最大上下文 | 8192 tokens |
| API | OpenAI-compatible `/v1/completions` |

由于 Hugging Face 的 HTTPS 连接在该网络中不稳定，模型通过 ModelScope 下载后使用本地 snapshot。RTX 5070 的当前软件组合还需要关闭 FlashInfer sampler，并在相关实验中固定 `TRITON_ATTN`。这些属于平台兼容性措施，不是被测优化项。

### 2.2 可复现规则

- 每个正式闭环 benchmark cell 运行 3 次；
- E01-E07 的每次性能 repetition 使用 100 个计时请求；
- 同一对照实验使用相同 workload、seed 规则和请求数；
- 每个配置、数据集和 trace 保存 SHA-256；
- 参数在正式运行前冻结，不能看到结果后再偷偷修改门槛；
- 报告同时保留 PASS、FAIL 和限制，不把失败实验包装成成功；
- repetition 之间只按协议改变 seed 或重复编号，不改变该 cell 的服务参数。

## 3. 关键参数详细含义

### 3.1 服务与调度参数

| 参数 | 通俗含义 | 主要影响 |
|---|---|---|
| `max_model_len` | 单个请求允许占用的最大上下文长度 | 越大越能接长请求，但 KV Cache 容量压力越大 |
| `gpu_memory_utilization` | vLLM 计划使用的 GPU 显存比例 | 太高容易因额外 activation、CUDA graph 或碎片导致 OOM；太低会浪费 KV 容量 |
| `max_num_seqs` | vLLM 同时维护的最大活跃序列数 | 上限越高，并发容量越大，但每个请求分到的计算资源可能更少 |
| `max_num_batched_tokens` | vLLM 在一个调度 step 内最多安排多少 token 参加计算 | 大值更容易形成大 batch；小值会限制一次塞入的 prefill 工作量，常能保护长请求的 TTFT |
| `enable_chunked_prefill` | 把长 prompt 的 prefill 拆成多段调度 | 避免一个长 prompt 长时间独占整个 step |
| `kv_cache_dtype` | KV Cache 中 Key/Value 张量的存储精度 | BF16 更稳，FP8 更省显存但有量化误差 |
| `calculate_kv_scales` | 是否在运行时计算 FP8 KV 的缩放比例 | scale 决定浮点值如何映射到 FP8 可表示范围 |
| `enable_prefix_caching` | 是否复用相同前缀已经算过的 KV Cache | 前缀真实重复时可跳过部分 prefill；无复用时几乎无收益 |
| `attention_backend` | Attention 算子的实现后端 | 本项目相关受控实验固定为 `TRITON_ATTN`，减少硬件兼容变量 |
| `seed` | 控制请求构造、顺序或生成中的随机源 | 同一对照使用配对 seed，降低输入差异对结果的干扰 |

### 3.2 为什么项目里的 2048 和 8192 不冲突

项目中有三个看起来相似、实际属于不同层级的 Token 限制：

1. vLLM 的 `max_num_batched_tokens=2048`：每个调度 step 最多计算多少 token，是一个瞬时计算预算。
2. E08 的动态 `token_budget`：客户端估算当前所有在途请求的 `input + output` Token 总成本，是 admission controller 的动态容量阈值。
3. E09 的 `max_inflight_tokens=8192`：外置控制器允许同时进入服务的请求，其估算总 Token 成本不能超过 8192，是请求全生命周期的在途上限。

因此最终方案可以同时使用 vLLM step budget 2048 和外置 in-flight ceiling 8192。可以把它理解成：厨房每一轮最多处理 2048 份工作，但整个后厨同时接下的订单总量最多是 8192。

### 3.3 前缀缓存参数

| 参数 | 含义 |
|---|---|
| `prefix_len` | 每个请求中可能被复用的公共前缀长度 |
| `suffix_len` | 每个请求自己的非共享后缀长度 |
| `num_prefixes` | 100 个请求从多少种前缀中选择；越少表示名义复用越高 |
| nominal reuse | 根据前缀种类设计出来的复用比例 |
| actual token hit rate | 从 vLLM Prometheus 计数器读取的真实 Token 命中率，是判断 APC 收益的关键数据 |
| `prewarm_prompts` | 正式计时前先发送请求，让 cache 进入可观测状态 |

名义复用率不等于真实命中率。例如名义 90% 且公共前缀 1792 tokens 时，实际命中约 78.9%；名义 90% 但公共前缀只有 256 tokens 时，实际命中仅约 12.1%。

### 3.4 KV Cache 量化与 scale

BF16 通常用 2 bytes 存一个元素，FP8 通常用 1 byte，因此核心 KV 数据理论上约减半。`scale` 是量化换算系数：把原始浮点范围压缩到 FP8 能表示的范围，读取时再按比例还原。

本项目测得 FP8 KV 可用 token 容量为 BF16 的 2.009 倍，而不是精确 2.000 倍。可信原因包括：

- vLLM 按 cache block 分配容量，最后会有整数 block 向下取整；
- BF16 与 FP8 每个 block 的字节数不同，取整损失比例也不同；
- 固定开销、对齐和少量元数据不与 KV 元素宽度同比缩放；
- 因此最终可分配 token 数是离散结果，可能略高或略低于两倍。

本项目调用的是 vLLM 自带的 FP8 KV Cache 与 scale 计算功能，没有自行实现量化 kernel。

### 3.5 LoRA 与 QLoRA 参数

| 参数 | 含义 | 本项目取值 |
|---|---|---|
| `rank` | 低秩矩阵的容量，越大可训练参数越多 | 主方案 8，对照 16 |
| `alpha` | LoRA 更新的缩放强度，常与 rank 成比例 | rank 8 用 16；rank 16 用 32 |
| `dropout` | 训练时随机屏蔽 LoRA 路径，缓解过拟合 | 0.05 |
| `target_modules` | 挂载 LoRA 的线性层 | Q/K/V/O、gate/up/down projection |
| `max_seq_length` | 训练样本最大 Token 长度 | 1024 |
| `micro_batch_size` | 一次实际送入 GPU 的样本数 | 1 |
| `gradient_accumulation_steps` | 累计多次小 batch 梯度后再更新 | 正式训练 16 |
| `learning_rate` | 每次参数更新步长 | 2e-4 |
| `epochs` | 遍历训练集次数 | 3 |
| `gradient_checkpointing` | 用重新计算换显存 | 开启 |
| NF4 4-bit | 以 4-bit 形式加载冻结基座权重参与 QLoRA 训练 | 开启 |

### 3.6 准入控制参数

| 参数 | 含义 |
|---|---|
| `max_inflight` | 同时已放行、尚未结束的请求数量上限 |
| `max_inflight_tokens` | 所有在途请求估算 Token 成本之和的上限 |
| `max_queue_wait_ms` | 请求最多可在外置队列等待多久，超时即过期 |
| `poll_interval_ms` | 控制器检查可用槽位和队列的周期 |
| first-fit backfill | 队首请求暂时放不下时，向后找一个能装进剩余 Token 空间的请求，减少资源空洞 |
| Jain fairness | 不同短、中、长请求获得 SLO 成功机会是否均衡；越接近 1 越公平 |

拒绝或排队过期的请求在本实验中不会自动重试，后续也不会完成。已经放行的请求即使最终超出 SLO 仍会完成，但它的 goodput 记为 0。

## 4. 参数演进总览

| 阶段 | 核心变量 | 为什么调 | 关键结果 | 最终决策 |
|---|---|---|---|---|
| E01/E03 | workload 长度、并发 C1/C4/C8/C16 | 建立 BF16 基线并找到容量拐点 | 12 个 cell 中 8 个 SLO PASS；长请求 C8 虽吞吐更高但 TTFT 失控 | 长请求默认控制在 C4 |
| E02 | batch-token 2048/4096/8192/16384 | 限制长 prefill 一次占用的计算量 | 长 C4 的 2048 相对 8192 将 TTFT 降 21.23%，吞吐仅降 0.18% | 长负载采用 2048 |
| E04 | APC OFF/ON、前缀长度与复用率 | 找出缓存真正有收益的命中阈值 | 高复用 P1792/C8 的 TTFT 降 55.11%，吞吐升 34.48% | 只对稳定高复用流量开启 |
| E05 | BF16 KV 与 FP8 E4M3 KV | 用量化换取长上下文容量 | KV 容量 2.009 倍，但结构化质量明显下降 | 默认关闭 FP8 KV |
| E06 | 2048/8192 与 APC OFF/ON 的 2x2 组合 | 验证单项收益是否可叠加 | 高复用容量场景相对最佳单项 TTFT 再降 10.78%，恢复 SLO PASS | BF16 + 2048；高复用时再开 APC |
| E07 | Base、QLoRA rank 8/16、动态 LoRA | 补足训练能力并量化质量与在线成本 | rank 8 质量显著提高，但动态 LoRA 6 个性能 cell 中 4 个成本超门槛 | 适合离线/受控场景，不默认在线挂载 |
| E08 | 固定 C4 与动态 Token-aware AIMD | 尝试在过载时自适应并发容量 | TTFT 很低，但过早拒绝请求，goodput 比最佳基线低 10.74%/15.34% | 算法淘汰，保留失败证据 |
| E09 | C8、8192 在途 Token、800 ms 队列、backfill | 允许短暂排队并用 deadline 判断是否值得接纳 | periodic/shock burst goodput 提升 18.75%/17.43%，全部重复满足 SLO | 作为突发流量候选方案 |

## 5. 各轮实验详解

### 5.1 E01：建立 BF16 服务基线

#### 调节内容

E01 不追求优化，先固定一个可重复比较的控制组：

```toml
dtype = "bfloat16"
kv_cache_dtype = "bfloat16"
max_model_len = 8192
gpu_memory_utilization = 0.82
max_num_seqs = 8
max_num_batched_tokens = 8192
enable_prefix_caching = false
```

负载为 short 128/128、medium 512/256、long 2048/256，数字分别表示输入/输出 Token；并发测试 C1、C4、C8、C16。每个 cell 使用 100 个请求、10 个 warmup、3 次 repetition，生成参数固定 `temperature=0`、`ignore_eos=true`。

#### 调节原因

没有基线就无法判断后续优化是否真的有效。E01 固定 BF16、关闭 APC，让后续 E02、E04、E05 都能做单变量比较。

#### 收益与结论

- 12 个长度和并发 cell 中 8 个通过 SLO；
- short C8 达到约 627 tok/s，P95 TTFT 约 114 ms；
- medium C8 达到约 583 tok/s，P95 TTFT 约 389 ms；
- long C4 达到约 265 tok/s，P95 TTFT 约 798 ms并通过 SLO；
- long C8 吞吐提高到约 418 tok/s，但 P95 TTFT 升至约 1581 ms，goodput 反而下降；
- 峰值显存约 11675 MiB，说明 12GB 卡的安全余量已经很小。

真正得到的不是“并发越高越好”，而是容量边界：短、中请求可用 C8，长请求更适合 C4。

### 5.2 E03：复用 E01 矩阵分析并发与长度拐点

E03 原计划单独做长度与并发矩阵，但 E01 已完整覆盖 `3 workloads x 4 concurrencies x 3 repetitions`，因此没有重复消耗 GPU 时间，而是复用同一批 36 个正式 benchmark。

E03 的核心分析是：

- 并发增加会提高 batch 利用率和 raw throughput；
- 长 prompt 的 prefill 更重，并发增加后会排队争抢首 Token 时间；
- 当 TTFT 超过 1000 ms 时，继续提高 raw throughput 不代表用户获得更多按时完成的请求；
- 因此后续实验把 long/C4 作为稳态安全点，把 long/C8 作为需要优化的容量点。

E03 的收益是减少一次重复实验，并明确了后续 E02、E04 和 E06 要解决的目标场景。

### 5.3 E02：Batch-token budget 扫描

#### 调节内容

保持模型、BF16 KV、`max_num_seqs=8`、APC OFF 和负载不变，只扫描：

```text
max_num_batched_tokens = 2048 / 4096 / 8192 / 16384
```

正式矩阵覆盖 3 种负载、C4/C8、4 个 budget、3 次 repetition，共 72 个 benchmark。

#### 为什么从 8192 向下和向上都试

- 降到 2048/4096：限制单个调度 step 中的 prefill Token，观察能否保护 TTFT；
- 保留 8192：作为 E01 对照；
- 提高到 16384：检查更大 batch 是否能继续换来吞吐，避免只凭直觉认定“大 batch 更好”。

#### 主要结果

以 8192 为同场景参考：

| 场景 | 2048 相对 8192 的 P95 TTFT | 吞吐变化 | 显存变化 | 结论 |
|---|---:|---:|---:|---|
| short C8 | +3.15% | +0.03% | -323 MiB | 没有必要依赖小 budget |
| medium C8 | -2.33% | -0.14% | -164 MiB | 差异很小 |
| long C4 | -21.23%，794.93 -> 626.17 ms | -0.18% | -758 MiB | 明确正收益 |
| long C8 | -47.74%，1583.56 -> 827.53 ms | +0.14% | -878 MiB | 中位数明显改善，但一轮 1005.24 ms，严格门槛仍 FAIL |

4096 在 long/C4 的 TTFT 为 614.36 ms，比 2048 略低，但吞吐比 8192 低 1.70%，且运行温度、功耗存在轻微混杂。16384 没有显示额外收益，多数吞吐反而下降约 1.6% 到 1.7%。

#### 决策

- 长上下文采用 2048，因为它同时给出明显 TTFT 改善、接近不变的吞吐和更低显存；
- 8192 保留给 E04 做单变量 APC 对照；
- 4096、16384 不进入最终默认方案；
- E02 也说明小 budget 不是“算得更少”，而是把长 prefill 分散到更多调度 step 中。

### 5.4 E04：Automatic Prefix Caching 收益阈值

#### 调节内容

服务端只切换 `enable_prefix_caching=false/true`，保持 batch-token 8192。客户端构造 2048 输入、256 输出的前缀复用数据：

| 条件 | Prefix/Suffix | 名义复用 | 实际命中 | 用途 |
|---|---:|---:|---:|---|
| reuse0_p1024 C4 | 1024/1024 | 0% | 0.98% | 无复用控制组 |
| reuse50_p1024 C4 | 1024/1024 | 50% | 18.78% | 中等复用 |
| reuse90_p1024 C4 | 1024/1024 | 90% | 45.52% | 高复用、中等公共前缀 |
| reuse90_p256 C4 | 256/1792 | 90% | 12.11% | 高复用、短公共前缀 |
| reuse90_p1792 C4 | 1792/256 | 90% | 78.94% | 高复用、长公共前缀 |
| capacity p1792 C8 | 1792/256 | 90% | 78.93% | 容量恢复测试 |

#### 调节原因

APC 只能省掉“相同且命中的前缀”计算。只比较开关而不控制前缀长度与复用率，会得出不可复现的结论。因此 E04 同时改变前缀种类和公共前缀长度，并以 Prometheus 的实际命中率解释性能。

#### 主要结果

- 实际命中约 1% 时，TTFT 只下降 0.43%，基本等于无收益；
- 实际命中 45.52% 时，TTFT 下降 24.20%，吞吐提高 9.95%；
- 实际命中 78.94%、C4 时，TTFT 从 804.81 降到 436.33 ms，下降 45.79%，吞吐提高 18.63%；
- 实际命中 78.93%、C8 时，TTFT 从 1407.15 降到 631.67 ms，下降 55.11%，吞吐从 412.10 提高到 554.18 tok/s，3/3 SLO 从 FAIL 恢复为 PASS。

#### 正确性解释

随机 Token benchmark 中，OFF/ON 输出的严格位置匹配不是 100%，因此原始比较报告保留了 `INCOMPLETE/UNKNOWN`。随后增加固定自然语言 correctness canary：24/24 个 OFF/ON 输出完全一致，22/24 个任务答案正确。两个错误在 OFF 和 ON 中相同，说明是模型本身答错，不是 APC 改坏结果。

#### 决策

APC 只对稳定、高真实命中率的流量启用，例如固定 system prompt、共享长文档或多轮会话公共历史。不能把名义复用率直接当作 cache hit rate，也不应默认对无复用流量宣称收益。

### 5.5 E05：FP8 KV Cache 容量与质量权衡

#### 调节内容

对照组和处理组保持 BF16 模型权重及 BF16 计算，只改变 KV Cache：

```toml
# Control
kv_cache_dtype = "bfloat16"
calculate_kv_scales = false

# Treatment
kv_cache_dtype = "fp8_e4m3"
calculate_kv_scales = true
```

为了观察容量压力，`max_num_seqs` 提高到 16，并测试 long 2048/256、xlong 4096/256、nearmax 7168/256，在 C8/C16 下各重复 3 次，共 36 个性能 benchmark。质量部分使用独立冻结的 50 个 AI Infra 故障诊断样本和 50 对匿名输出。

#### 调节原因

KV Cache 随并发和上下文增长，是 12GB 显存上的主要容量瓶颈。FP8 理论上可把 KV 元素宽度从 16 bit 降到 8 bit，但必须同时验证性能和任务质量，不能只看显存。

#### 收益

- BF16 可用 KV 容量 96080 tokens；
- FP8 可用 KV 容量 193072 tokens；
- 容量比 2.009 倍，8192-token 参考请求的理论并发从 11.73 提高到 23.57；
- nearmax 7168/256、C16 的吞吐从 198.86 提高到 228.23 tok/s，提升 14.77%，TTFT 下降 9.42%。

#### 代价

- 结构化 schema 通过率从 92% 降到 70%；
- root-cause Macro-F1 从 0.7191 降到 0.5698；
- action micro-F1 从 0.5208 降到 0.3647；
- 匿名评分从 3.680 降到 3.120，差值 -0.560；
- 50 对中 BF16 胜 20、FP8 胜 5、平局 25；
- 所有高压性能 cell 仍然没有满足完整 SLO。

#### 决策

FP8 KV 是容量优化成功、质量门槛失败。当前 runtime scale 不代表经过代表性数据离线校准后的最好 FP8 质量，因此不能把结果外推到所有 FP8 方案；但在本项目冻结条件下，默认仍使用 BF16 KV，E06 不叠加 FP8。

### 5.6 E06：Batch-token 与 APC 的 2x2 组合实验

#### 第一轮 pilot 失败与参数修正

E06 最初沿用 `gpu_memory_utilization=0.82`。pilot 在 scheduler step 6145 附近触发 CUDA OOM，额外约 130 MiB activation 就越过显存边界。这说明 vLLM 预留 KV 显存之后，仍需给 activation、CUDA graph 和碎片留安全余量。

因此正式四个 cell 一致改为：

```text
gpu_memory_utilization: 0.82 -> 0.78
```

这大约释放 0.48 GiB 余量。旧 pilot 被归档，正式对照全部在 0.78 下重跑，避免只修改某一个 cell 破坏公平性。

#### 2x2 参数设计

| Cell | Batch tokens | APC | 角色 |
|---|---:|---|---|
| A | 8192 | OFF | 原始控制组 |
| B | 2048 | OFF | 仅 batch-token 优化 |
| C | 8192 | ON | 仅 APC 优化 |
| D | 2048 | ON | 组合优化 |

覆盖 reuse0/reuse50/reuse90 的 C4，以及高复用 P1792/C8 容量场景，共 16 profiles、48 个正式 benchmark。

#### 调节原因

E02 和 E04 分别证明了单项收益，但两个功能一起开不一定自动相加。2x2 factorial 可以区分：

- budget 的主效应；
- APC 的主效应；
- 两者是否存在正向或负向交互；
- 组合 D 是否真正优于最好的单项 B/C。

#### 主要结果

| 场景 | A/B/C/D P95 TTFT ms | D 相对最佳单项 | D 吞吐相对最佳单项 | 结论 |
|---|---|---:|---:|---|
| reuse0 C4 | 776.99/628.45/761.28/630.51 | +0.33% | -0.26% | 无叠加收益 |
| reuse50 C4 | 768.98/755.28/755.21/577.36 | -23.55% | +0.28% | 有叠加收益 |
| reuse90 P1024 C4 | 767.03/581.46/485.15/466.82 | -3.78% | -0.62% | 未达到冻结的 5% 门槛 |
| reuse90 P1792 C8 | 1513.23/845.64/670.23/597.97 | -10.78% | +0.11% | 有叠加收益，SLO PASS |

组合 D 相对原始 A，在高复用 P1792/C8 场景把 TTFT 降低 60.48%，吞吐提高 38.21%，goodput 提高 137.53%。固定 canary 中四个 cell 的 OFF/ON 输出均 24/24 一致。

#### 决策

- 通用 Base 采用 BF16 KV、batch-token 2048；
- 高复用流量再开启 APC；
- 没有复用时不为“组合开关更多”而开启 APC；
- `gpu_memory_utilization=0.78` 作为更稳妥的 12GB 运行余量。

### 5.7 E07：QLoRA 训练、rank 消融与在线 LoRA 成本

#### 数据与训练参数

任务是 AI Infra 故障归因与处置建议的结构化生成。数据按 scenario group 切分，训练 250 行、验证 100 行；E05 的 50 个冻结测试样本只用于测试，没有进入训练或验证集。

先运行 smoke，再运行两个正式 rank：

| 训练 | rank/alpha | 数据 | accumulation | epoch/step | 目的与结果 |
|---|---:|---:|---:|---:|---|
| smoke | 8/16 | 100 行 | 8 | 最多 20 steps | 验证 QLoRA、保存和加载链路；eval loss 0.335182 |
| primary | 8/16 | 250 行 | 16 | 3 epochs，48 steps | 主候选；eval loss 0.006476，峰值显存约 5402 MiB |
| ablation | 16/32 | 250 行 | 16 | 3 epochs，48 steps | 检查更大 rank；eval loss 0.008587，峰值约 5538 MiB |

其余主要参数保持一致：NF4 4-bit 基座、BF16 compute、学习率 2e-4、dropout 0.05、序列长度 1024、micro batch 1、warmup ratio 0.05、gradient checkpointing 开启、seed 20260824。

#### 为什么选择 rank 8

rank 16 的训练 loss 略低，但验证 loss 反而高于 rank 8，且占用更多显存。说明当前小数据任务不需要更大的 LoRA 容量，rank 8 是更简洁、验证效果更好的选择，而不是根据最终测试结果事后挑参数。

#### 质量收益

| 指标 | Base | LoRA rank 8 | 变化 |
|---|---:|---:|---:|
| Schema pass | 92% | 100% | +8 pp |
| Root-cause Macro-F1 | 0.7191 | 0.9366 | +0.2175 |
| Action micro-F1 | 0.5208 | 0.9400 | +0.4192 |
| 匿名平均分 | 3.580 | 4.760 | +1.180 |

匿名偏好为 Base 2、LoRA 38、tie 10。这里的复核由项目代理执行，不等同于独立外部人类偏好研究。

#### 在线成本

服务端保持 E06 Base 的 BF16、2048 budget 和 0.78 显存比例，只切换动态 LoRA。性能矩阵为 short/medium、C1/C4/C8、Base/LoRA、3 次 repetition，共 36 个 benchmark。

- LoRA 吞吐下降约 12.6% 到 16.8%；
- TPOT 增加约 13.6% 到 19.5%；
- 峰值显存增加约 0.54 GiB；
- 所有 LoRA cell 仍满足绝对 SLO；
- 但相对成本门槛要求 TTFT 增幅不超过 25%，6 个 cell 中有 4 个失败；
- short C1 的 TTFT 增加 54.61%，short C4 增加 42.75%。

#### 决策

训练和质量目标成功，动态在线部署成本目标失败。rank 8 adapter 适合离线批处理、低流量或质量优先场景；当前 12GB 实时服务不默认挂载动态 LoRA。

### 5.8 E08：第一版 Token-aware AIMD 准入算法

#### 算法参数

服务端沿用 E06 BF16 Base：`max_num_batched_tokens=2048`、APC OFF。比较三种客户端策略：

1. `unbounded`：不限制请求进入；
2. `fixed_concurrency`：最多 C4；
3. `token_aware`：最多 C8，同时动态控制在途 Token budget。

Token-aware 参数为：

```text
max_inflight = 8
initial_token_budget = 4096
min_token_budget = 2304
max_token_budget = 6144
decrease_factor = 0.80
increase_step_tokens = 256
success_window = 16
```

核心更新逻辑：

```text
如果某个已放行请求没有满足 SLO：
    budget = max(2304, 向下对齐到 256 的倍数(当前 budget * 0.8))

如果连续 16 个已放行请求满足 SLO：
    budget = min(6144, 当前 budget + 256)
```

不满足 SLO 后缩小 budget，而不是放大，是因为这表示当前在途工作过多，首 Token 已经排队过久。缩小 admission budget 会减少之后同时进入 GPU 的工作量，为现有请求留出算力。

#### 流量参数

请求类型按 short/medium/long = 50%/30%/20% 混合，对应估算成本 256/768/2304 tokens。

| Profile | 生成方式 | 平均到达率 |
|---|---|---:|
| mixed_nominal | 稳定到达 | 1 req/s |
| mixed_overload | 稳定到达 | 2 req/s |
| mixed_burst | 每 10 秒中 2 秒为 6 req/s，其余 8 秒为 1 req/s | 2 req/s |

每个 profile 持续 180 秒，3 个 seed，3 种策略复用完全相同的 trace，共 27 个正式 cell。

#### 为什么算法失败

| Profile | 最佳基线 goodput | Token-aware | 相对变化 | Token TTFT |
|---|---:|---:|---:|---:|
| mixed_burst | 1.500 | 1.339 | -10.74% | 243.2 ms |
| mixed_nominal | 0.956 | 0.889 | -6.98% | 248.4 ms |
| mixed_overload | 1.811 | 1.533 | -15.34% | 244.2 ms |

它把 admitted 请求的 TTFT 压得很低，却没有提高 goodput，原因是：

- admission 当下就拒绝放不下的请求，没有给短暂突发排队等待的机会；
- SLO 结果要等请求完成后才能反馈，控制信号天然滞后；
- 一次失败就乘 0.8，下降快；恢复则每 16 次成功只加 256，上升慢；
- 用 `input + output` 估算的请求成本在整个生命周期内保持不变，没有反映 prefill 完成后真实成本下降；
- 结果是 GPU 仍有能力时，控制器已经拒绝了过多本来可能按时完成的请求。

E08 的价值在于定位了错误机制：低 TTFT 不等于高 goodput，准入不能只靠完成后的 AIMD 反馈，也不能把所有超限请求立即丢弃。

### 5.9 E09：第二版 Deadline-aware 有界队列算法

#### 从 E08 得到的校准信号

对 E08 的请求级数据按请求到达前的 active 数量分桶：0 到 7 个先前 active 请求时，SLO yield 为 100%；active=8 时约 83.9%，active=9 时约 58.9%，active=10 时约 40%。这说明 C4 太保守，完全 unbounded 又会过载，C8 附近存在可利用容量。

#### 算法参数

E09 取消 AIMD 动态 budget，改为可解释的固定约束：

```text
max_inflight = 8
max_inflight_tokens = 8192
max_queue_wait_ms = 800
poll_interval_ms = 5
queue_order = deadline/FIFO + first-fit backfill
```

请求处理流程：

1. 请求到达后先进入外置有界队列，而不是立即拒绝；
2. 同时满足请求数上限 C8 和在途 Token 上限 8192 才能放行；
3. TTFT 从原始到达时刻计算，因此排队时间不会被隐藏；
4. 队首请求太大、暂时装不下时，向后查找能放入剩余 Token 空间的请求；
5. 等待超过 800 ms 的请求过期，不再送给 GPU，避免消耗资源后仍必然错过 TTFT SLO。

#### 为什么使用 800 ms

TTFT SLO 是 1000 ms。队列不能把全部 1000 ms 都花掉，还要为真正的 prefill、调度和网络开销留余量。800 ms 是一个有界等待期限，而不是承诺所有等待 800 ms 的请求都一定成功。

#### Holdout 流量

E09 不复用 E08 的正式 seed，使用新的 3 个 holdout seed。每个 profile 180 秒：

| Profile | 流量构造 | 作用 |
|---|---|---|
| holdout_nominal | 稳定 1 req/s | 检查低负载不能无故变慢 |
| holdout_periodic_burst | 每 10 秒中 2 秒 6 req/s，8 秒 1 req/s | 周期突发 |
| holdout_shock_burst | 每 10 秒中 1 秒 11 req/s，9 秒 1 req/s | 更尖锐但同均值的瞬时冲击 |

三种策略、三个 profile、三个 seed，共 27 个 holdout cell。策略执行顺序按 Latin rotation 轮换，减少机器温度和时间顺序偏差。

#### 结果

| Profile | 最佳基线 goodput | Deadline goodput | 收益 | Deadline P95 TTFT | P95 queue | Jain fairness |
|---|---:|---:|---:|---:|---:|---:|
| nominal | 1.006 | 1.006 | 0.00% | 240.5 ms | 0.2 ms | 1.000 |
| periodic burst | 1.422 | 1.689 | +18.75% | 723.4 ms | 639.7 ms | 0.997 |
| shock burst | 1.339 | 1.572 | +17.43% | 739.2 ms | 660.7 ms | 0.998 |

两种 gated burst 的全部 repetition 都满足 TTFT/TPOT SLO、公平性和错误率门槛。unbounded 在 periodic/shock 下的 P95 TTFT 分别达到 2440.9/3120.4 ms。Deadline-aware 没有提升低负载硬件算力，nominal 收益为 0；它的收益来自突发时允许短暂排队、充分使用 C8 容量，并阻止已经没有时间预算的请求继续占用 GPU。

#### 决策与边界

E09 通过冻结门槛，成为突发流量的候选准入方案。但结论只覆盖这两个 holdout burst profile，不外推到无限持续过载、其他模型、其他 GPU 或生产故障场景。

## 6. 实验规模与统计单位

| 阶段 | 正式执行规模 | 补充证据 |
|---|---:|---|
| E01/E03 | 3 workloads x 4 concurrencies x 3 reps = 36 | E03 复用 E01，不重复计数 |
| E02 | 4 budgets x 3 workloads x 2 concurrencies x 3 reps = 72 | 每个 cell 100 请求 |
| E04 | 12 profiles x 3 reps = 36 | 24-case correctness canary |
| E05 | 2 KV dtypes x 3 workloads x 2 concurrencies x 3 reps = 36 | 50-case 自动质量 + 50 对匿名复核 |
| E06 | 4 factorial cells x 4 workload conditions x 3 reps = 48 | 四状态 correctness canary |
| E07 | 2 states x 2 workloads x 3 concurrencies x 3 reps = 36 | 3 次训练、50-case 质量、50 对匿名复核 |
| E08 | 3 profiles x 3 policies x 3 seeds = 27 | 2650 个唯一正式到达事件被三策略重放 |
| E09 | 3 profiles x 3 policies x 3 seeds = 27 | 2721 个唯一 holdout 到达事件被三策略重放 |

合计 318 个正式执行单元。其中：

- E01-E06 共 228 次 benchmark、22,800 个计时请求；
- E07 增加 36 次在线 benchmark、3,600 个计时请求；
- E08/E09 是开放到达率实验，请求数量由 trace 和持续时间决定，统计单位不同；
- 318 不是 318 个独立 profile，也不能把开放流量 cell 与闭环 benchmark 当成完全相同的样本。

## 7. 最终推荐配置与使用条件

### 7.1 通用实时 Base

```text
模型权重/计算: BF16
KV Cache: BF16
gpu_memory_utilization: 0.78
max_model_len: 8192
max_num_seqs: 8
max_num_batched_tokens: 2048
chunked prefill: ON
attention backend: TRITON_ATTN
FlashInfer sampler: OFF
```

### 7.2 按负载启用的策略

| 负载 | 推荐策略 |
|---|---|
| 短/中请求、普通并发 | BF16 Base，C8 以内 |
| 长请求、无明显前缀复用 | batch-token 2048，稳态优先 C4 |
| 长公共前缀且真实 hit rate 高 | 2048 + APC；C8 需有命中率和 SLO 监控 |
| 需要更多 KV 容量但质量敏感 | 不默认使用当前 FP8 KV |
| AI Infra 结构化诊断，质量优先 | rank 8 adapter 可用于离线或受控部署 |
| 周期或瞬时突发 | Deadline-aware：C8 + 8192 in-flight tokens + 800 ms queue + backfill |
| 持续过载 | 当前仍是开放问题，不能直接引用 E09 burst 结论 |

## 8. 项目最终结论

### 8.1 得到的正向结论

1. **调度预算比盲目提高并发更重要。** 长请求把 batch-token 从 8192 降到 2048，可在吞吐基本不变时显著降低 TTFT。
2. **APC 收益由真实 Token 命中率决定。** 高命中、长公共前缀场景可以同时提高吞吐并恢复容量 SLO；无复用时收益接近零。
3. **优化项需要做组合实验。** E06 证明 2048 与 APC 只在部分负载上可叠加，不能用“两个单项都有效”推导“组合一定更好”。
4. **容量收益不能替代质量门槛。** FP8 KV 的 2.009 倍容量是真实收益，但当前质量退化使它不能成为默认配置。
5. **小参数训练也要测在线成本。** QLoRA rank 8 明显提高任务质量，但动态 LoRA 的 TTFT、TPOT、吞吐和显存成本同样真实。
6. **准入优化的目标应是 goodput，不是只让被接纳请求更快。** E08 低延迟但拒绝过多；E09 用短暂排队与 deadline 取得更高按时完成率。

### 8.2 失败实验带来的工程价值

- E05 阻止了“FP8 容量翻倍，所以应该默认开启”的错误决策；
- E06 pilot 暴露 12GB 卡上 0.82 显存比例缺少 activation 安全余量；
- E07 阻止了“质量更好，所以在线部署也一定更好”的推断；
- E08 明确了纯 AIMD、立即拒绝和静态 Token 成本模型的问题，为 E09 提供设计依据。

### 8.3 项目能力总结

这个项目最终交付的不是某个孤立脚本，而是一套单 GPU AI Infra 实验系统：

- 配置驱动的 vLLM 服务启动与矩阵执行；
- TTFT、TPOT、吞吐、goodput、错误率、VRAM、功耗和 cache hit telemetry；
- BF16、FP8 KV、APC、QLoRA/LoRA 的性能与质量联合评测；
- trace 驱动的开放到达率回放和外置准入控制；
- seed、配置哈希、数据哈希、重复实验、冻结门槛和审计报告；
- 失败诊断、参数修正、holdout 验证和明确的结论边界。

截至 E09，仓库记录 318 个正式执行单元，并由 91 项不依赖 GPU 的单元/静态测试保护实验代码与报告逻辑。项目证明了在 12GB 消费级 GPU 上，也可以完成接近真实 AI Infra 工作方式的容量规划、推理优化、训练评测和流量治理闭环。

## 9. 证据索引

- 通用实验协议：[EXPERIMENT_PROTOCOL.md](EXPERIMENT_PROTOCOL.md)
- E01/E03 基线：[E01_E03_RESULTS.md](E01_E03_RESULTS.md)
- E02 batch-token：[E02_RESULTS.md](E02_RESULTS.md) / [comparison.md](../reports/e02_batch_tokens/comparison.md)
- E04 APC：[E04_RESULTS.md](E04_RESULTS.md) / [correctness_canary.md](../reports/e04_prefix_cache/correctness_canary.md)
- E05 FP8 KV：[E05_RESULTS.md](E05_RESULTS.md) / [capacity.md](../reports/e05_kv_cache/capacity.md) / [quality.md](../reports/e05_kv_cache/quality.md)
- E06 组合实验：[E06_RESULTS.md](E06_RESULTS.md) / [comparison.md](../reports/e06_combined/comparison.md)
- E07 QLoRA/LoRA：[E07_RESULTS.md](E07_RESULTS.md) / [quality.md](../reports/e07_lora/quality.md) / [comparison.md](../reports/e07_lora/comparison.md)
- E08 第一版准入：[E08_RESULTS.md](E08_RESULTS.md) / [comparison.md](../reports/e08_admission/comparison.md)
- E09 第二版准入：[E09_PROTOCOL.md](E09_PROTOCOL.md) / [E09_RESULTS.md](E09_RESULTS.md) / [comparison.md](../reports/e09_deadline_admission/comparison.md)
- E01-E06 汇总审计：[audit.md](../reports/e01_e06/audit.md)

## 10. 面试时的一分钟总结

我在 RTX 5070 12GB 和 WSL2 上搭建了一个配置驱动的 Qwen2.5-3B vLLM 实验平台，不只测吞吐，还用 TTFT、TPOT、SLO goodput、显存和质量门槛做联合决策。我先通过长度和并发矩阵定位长请求瓶颈，再分别扫描 batch-token、APC 和 FP8 KV，并用 2x2 实验验证组合收益。随后用 4-bit QLoRA 训练 AI Infra 故障诊断 adapter，证明质量提高但动态 LoRA 在线成本过高。最后我实现了两版外置准入控制，第一版 Token-aware AIMD 因拒绝过早导致 goodput 下降，第二版改为 C8、8192 在途 Token、800 ms 有界队列与 backfill，在新的 holdout 突发流量上相对最佳基线把 SLO goodput 提高 17.43% 到 18.75%。整个项目保留配置哈希、seed、重复实验、请求级证据和失败结论，可完整复现每项决策。
