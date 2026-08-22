# E05 KV Cache Capacity

Generated at: 2026-08-21T18:27:15.970355+00:00

Capacity gate: **PASS** (required ratio >= 1.80x)
Capacity values are parsed from vLLM startup logs under identical memory and scheduler controls.

| State | KV dtype | GPU KV tokens | Reference request tokens | Maximum concurrency |
|---|---|---:|---:|---:|
| bf16 | bfloat16 | 96080 | 8192 | 11.73x |
| fp8 | fp8_e4m3 | 193072 | 8192 | 23.57x |

FP8/BF16 token capacity ratio: **2.009x**
FP8/BF16 reported concurrency ratio: **2.009x**
