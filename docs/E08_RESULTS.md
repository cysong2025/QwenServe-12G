# E08 Token-aware 准入控制实验结果

## 1. 问题与状态

E08 在 RTX 5070 12GB、WSL2、Qwen2.5-3B-Instruct 和 vLLM 0.25.1
条件下，比较 `unbounded`、固定并发 4 和 token-aware SLO-feedback 三种准入
策略。三种策略对同一组开放到达 trace 做配对回放，冻结问题是 token-aware 是否
能在过载下提高 goodput，同时保持绝对 TTFT/TPOT SLO 与长度公平性。

实验执行状态为 **COMPLETE_WITH_FROZEN_GATE_FAILURE**：3 个 pilot 和 27 个
正式 cell 已完成，证据完整性通过，但机器最终状态为 **FAIL**。这是一项有效的
负面实验结论，不能表述为 token-aware 策略成功提升了过载 goodput。

## 2. 证据完整性

- 27/27 正式结果均为 `valid: true`，覆盖 3 个 profile、3 个策略和 3 次重复；
- 27 个逻辑 cell 全部唯一，无重复或缺失；
- 9 个 profile/repetition 组内三策略的 `trace_sha256` 全部一致；
- 270/270 untimed warmup 完成，所有 cell 的 `pending_after_drain` 为 0；
- 最大 admitted error rate 为 0.361%，低于冻结的 1% 有效性上限；
- 实验配置和服务配置 SHA-256 分别为
  `c936169cb314f1f59b9e91303002b6757d732b0de4f9b41aec2d2d3577aa0dd3` 和
  `2d47ae6083111e82b17f13e9676b2df92b8f27c42873a341db6285bc223495ba`。

原始请求、控制器状态和 GPU telemetry 保留在 WSL2 的忽略目录中；Git 只提交由
比较器生成的紧凑报告。

## 3. 冻结结果

下表为三次重复的中位数；绝对 SLO 门槛仍逐次检查，而不是只检查中位数。

| Profile | Unbounded/fixed/token goodput | Token vs best baseline | Unbounded/fixed/token P95 TTFT ms | Unbounded/fixed/token Jain | Token SLO | Fairness | Gate |
|---|---:|---:|---:|---:|---|---|---|
| mixed_burst | 1.500 / 0.994 / 1.339 | -10.74% | 2597.8 / 242.5 / 243.2 | 0.999 / 0.994 / 0.972 | PASS | PASS | FAIL |
| mixed_nominal | 0.956 / 0.850 / 0.889 | -6.98% | 251.1 / 250.1 / 248.4 | 1.000 / 1.000 / 0.994 | PASS | PASS | 诊断项 |
| mixed_overload | 1.811 / 1.189 / 1.533 | -15.34% | 900.5 / 248.3 / 244.2 | 1.000 / 0.997 / 0.957 | FAIL | PASS | FAIL |

两个过载 profile 都没有达到“token-aware goodput 至少比最佳基线高 10%”的冻结
门槛。`mixed_overload` 的 token-aware/r2 还出现 P95 TPOT 63.37 ms，超过 50 ms
上限；r1/r3 分别约 14.12/13.82 ms，因此逐次绝对 SLO 判为 FAIL。token-aware
在两个过载 profile 的 Jain 指数均大于 0.80，且相对 fixed-C4 的回退不超过
0.05，所以公平性门槛通过。

## 4. 系统含义

- token-aware 和 fixed-C4 都显著压低了 burst 的 TTFT 尾延迟，但通过拒绝请求
  换取延迟稳定性，最终 SLO goodput 仍低于 unbounded；
- 当前 token-aware 控制器在 overload/r2 将预算降到 3072 tokens，却仍发生
  TPOT 尾延迟越线，说明冻结 AIMD 反馈在该 trace 上不够稳定；
- unbounded 在两个过载 profile 获得最高中位 goodput，但 burst 的 P95 TTFT
  约 2.6 秒，不能把更高 goodput 解读为全面更好的延迟行为；
- fixed-C4 提供稳定的约 242–248 ms P95 TTFT，但拒绝更激进，goodput 最低。

## 5. 部署决策

- 不把当前 token-aware 策略作为默认部署，也不放宽 +10% goodput、1000/50 ms
  绝对 SLO或公平性门槛；
- 若业务优先约束 TTFT，fixed-C4 或 token-aware 可作为后续研究起点，但需要新的
  实验编号、预先冻结的效用函数和拒绝成本，不能覆盖本次 FAIL；
- 若继续优化控制器，应研究分 workload 预算、队列等待估计和更稳定的反馈窗口，
  并重新执行完整配对矩阵。

## 6. 有效性限制

- 结果只覆盖一张 RTX 5070、当前 WSL2/CUDA/PyTorch/vLLM 组合和一个 3B 模型；
- trace 是受控的 Poisson/周期突发合成流量，不等同于生产多租户到达过程；
- workload mix 固定为 short/medium/long 的 50/30/20，拒绝成本只通过 SLO yield
  进入 goodput，没有业务优先级或用户价值权重；
- 三次重复可检查一致性，但不足以刻画更长时间尺度的负载漂移。

## 7. 可复现结论

> 在冻结的 RTX 5070 单实例 E08 矩阵中，27 次正式运行全部有效且精确配对。
> token-aware 保持了长度公平性并显著降低 burst TTFT，但其 burst 和 steady
> overload 中位 goodput 分别比最佳基线低 10.74% 和 15.34%，且 steady
> overload 的第二次重复出现 63.37 ms P95 TPOT，超过 50 ms 上限。因此 E08
> 执行完整、科学结论为 FAIL，当前策略不作为默认部署。

WSL 中保留的原始 artifacts 可重建紧凑报告：

```bash
make compare-e08; status=$?; test "$status" -eq 0 -o "$status" -eq 2
```

退出码 2 表示完整证据下的冻结科学门槛失败，不代表比较器故障。
