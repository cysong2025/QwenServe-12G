# 可行性分析

## 1. 结论

项目在 RTX 5070 12GB 上可行，但必须采用“推理与训练分时运行、训练使用 QLoRA、推理从 8K 上下文起步”的边界。3B BF16 在线服务、单 Adapter LoRA serving、受控并发压测和 QLoRA SFT 属于核心可交付；全参数训练、训练服务同时常驻和 32K 高并发不属于可承诺范围。

## 2. 推理显存预算

Qwen2.5-3B-Instruct 的配置为 36 层、16 个 attention heads、2 个 KV heads，`head_dim = 2048 / 16 = 128`。

BF16 KV Cache 的理论占用为：

```text
bytes/token
= layers * K/V * kv_heads * head_dim * bytes_per_element
= 36 * 2 * 2 * 128 * 2
= 36,864 bytes
= 36 KiB/token
```

因此：

| 活跃 token 数 | 理论 BF16 KV | 理论 FP8 KV |
|---:|---:|---:|
| 4,096 | 144 MiB | 72 MiB |
| 8,192 | 288 MiB | 144 MiB |
| 32,768 | 1,152 MiB | 576 MiB |

这些数字只表示 KV Tensor，不包含权重、CUDA Graph、allocator 碎片、临时工作区和 Windows 图形占用。实际容量必须以 vLLM 启动日志、`/metrics` 和 `nvidia-smi` 为准。

首个受控配置使用：

- `max_model_len = 8192`
- `gpu_memory_utilization = 0.82`
- `max_num_seqs = 8`
- BF16 权重和 BF16 KV Cache
- 明确关闭 prefix caching，作为可解释基线

若启动失败，按顺序降低 `gpu_memory_utilization` 到 0.78、`max_num_seqs` 到 4、`max_model_len` 到 4096。不能一开始就替换成更小模型，否则失去原定研究对象。

## 3. 训练可行性

3B 全参数 AdamW 训练需要权重、梯度、优化器状态和激活，远超 12GB。QLoRA 将基础权重量化并仅训练低秩参数，配合以下约束具备可行性：

- 4-bit NF4 基础权重，BF16 compute；
- micro batch size 1；
- gradient accumulation 8 或 16；
- sequence length 从 1024 起步，2048 作为资源门槛实验；
- gradient checkpointing；
- rank 8 为基线，rank 16 为消融；
- 训练时关闭 vLLM 服务，结束后再加载 Adapter。

训练速度不能提前承诺。M3 首先用 100 条样本完成 overfit/smoke，确认 loss、保存与加载链路，再启动完整训练。

## 4. 可观测性可行性

vLLM 原生 benchmark 和 `/metrics` 足以采集请求级及服务级指标。WSL2 下不把 DCGM 作为硬依赖，GPU 指标首先使用 `nvidia-smi` 轮询采集，以降低环境兼容风险。

可稳定采集：

- TTFT、TPOT、ITL、E2E；
- 请求与 token 吞吐；
- goodput 和错误率；
- GPU 名称、驱动、显存、温度、功耗和利用率；
- vLLM、PyTorch、CUDA、Python、模型 revision 与 Git commit。

## 5. 风险和处理

| 风险 | 影响 | 控制措施 |
|---|---|---|
| Blackwell 与 wheel/driver 不匹配 | 无法导入或运行 kernel | 使用 WSL2、最新 Windows NVIDIA 驱动和支持 Blackwell 的 vLLM wheel；不在 WSL 安装 Linux 驱动。 |
| Windows 桌面占用显存 | 启动或压测 OOM | 从 0.82 显存利用率起步，记录空闲显存，实验时关闭 GPU 密集应用。 |
| 版本快速变化 | 参数和结果不可复现 | 固定 vLLM 版本、模型 revision，并保存 `uv pip freeze`。 |
| FP8 KV 无校准导致质量下降 | 结论失真 | 质量评测与性能评测同时执行，未经质量门槛不得宣布优化有效。 |
| benchmark 热态和温度漂移 | 重复结果不稳定 | 固定预热、重复三次、轮次间冷却并记录温度/功耗。 |
| 领域数据泄漏 | 质量提升虚高 | 按来源分组划分数据，测试集人工核验并单独冻结。 |
| 项目范围过大 | 无法形成完整结论 | M0-M2 推理研究优先，训练闭环次之，UI/K8s 不进入核心路径。 |

## 6. 可行性闸门

- G0：WSL2 内 `nvidia-smi`、PyTorch CUDA、vLLM 自检全部通过。
- G1：3B BF16 服务连续完成 smoke 与 30 分钟稳定性测试。
- G2：基线实验三次重复的关键指标变异可解释，否则先修复实验环境。
- G3：QLoRA 100 样本 smoke 能保存 Adapter，并由 vLLM 成功加载。
- G4：只有当 G0-G3 通过后，才实现和评价准入控制。
