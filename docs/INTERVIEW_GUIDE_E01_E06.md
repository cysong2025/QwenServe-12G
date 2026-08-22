# E01-E06 面试讲述指南

## 1. 30 秒项目介绍

> 我在 RTX 5070 12GB 和 WSL2 上搭建了一套可复现的 vLLM 推理实验
> 平台，研究 Qwen2.5-3B-Instruct 的并发容量、batch-token 调度、
> Automatic Prefix Caching 和 FP8 KV Cache。E01-E06 共完成 228 次
> 正式 benchmark。项目不只报告 tokens/s，而是用 TTFT/TPOT SLO、
> goodput、GPU telemetry、固定 canary 和匿名人工质量门槛决定优化
> 是否可部署。最终得到基于负载类型的配置策略，同时保留了
> FP8 容量增长但质量门槛失败的负面结果。

## 2. 简历要点

- 设计并实现 RTX 5070 12GB 单卡 vLLM 实验平台，以 TOML 固定配置、
  SHA-256 关联模型/服务/压测、自动采集 GPU telemetry，完成 228 次
  三重复正式 benchmark 和可中断恢复的证据汇总。
- 定量评估 batch-token budget 与 APC：2048 budget 相对 8192 将
  Long-C4/C8 P95 TTFT 降低 21.23%/47.74%；高复用 P1792/C8 下组合
  配置相对最佳单项再降低 10.78% TTFT，并将 SLO FAIL 恢复为 PASS。
- 建立 FP8 KV 自动质量与匿名人工门槛：验证 KV token 容量提高
  2.009x，但 schema 通过率由 92% 降至 70%、人工均分由 3.680 降至
  3.120，因此拒绝将其作为质量敏感服务的默认优化。

## 3. 值得深挖的技术点

### 为什么用 goodput 而不是只用 throughput？

Long-C8 相对 C4 的 output throughput 更高，但 TTFT 违反 1000 ms SLO，
goodput 反而下降。这说明原始吞吐会把过载排队包装成“容量增长”。

### 如何保证对照有效？

配置文件产生 SHA-256，benchmark 启动前检查当前活跃服务的 profile 和
config hash。每轮记录 effective seed、模型/tokenizer 指纹、环境、重复编号和
telemetry。修改配置后旧证据会因 hash 不匹配而不能继续汇总。

### 如何定位 E06 服务退出？

表面错误是 `copy_event.synchronize()` 处的 `cudaErrorUnknown`。因为 CUDA 异步
执行，该堆栈只是错误上报点。使用不保存结果的 `CUDA_LAUNCH_BLOCKING=1`
复现后，定位到 6145-token 调度步中的 130 MiB activation OOM。四个
对照 cell 统一将 KV Cache 预留比例从 0.82 降到 0.78，并归档旧证据后全部重跑。

### 为什么 FP8 不能只看容量？

KV 元素减半使静态 token 容量接近 2x，但 prefill activation、调度、decode
计算和 SLO 仍会限制有效并发。更重要的是，当前在线 scale 导致结构化输出、
根因分类和人工可用性同时下降。容量证据和部署决策必须分开。

## 4. 不应使用的表述

- 不说“vLLM 相对 Transformers 加速 X%”，因为 Transformers 参考没有执行；
- 不说“APC 必然提升性能”，因为低 actual hit 下收益不明显；
- 不说“FP8 KV 无损将并发提高 2x”，因为 2.009x 是静态 token 容量且质量失败；
- 不说“E06 组合在所有负载下最优”，叠加门槛只在 2/4 条件成立；
- 不把 228 次 benchmark 写成 228 个 profile，它包含 76 个独立 profile 的三次重复。

## 5. 现场演示

无 GPU 时可演示代码和证据审计：

```bash
make test
make audit-e01-e06
sed -n '1,220p' reports/e01_e06/audit.md
```

有目标 GPU 时，使用 `make render-...` 先展示受控服务命令，再根据对应
runbook 执行 pilot、canary 和可恢复矩阵。
