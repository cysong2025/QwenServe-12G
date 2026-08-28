# M4 / E08 Token-aware Admission Runbook

## 1. Scope and frozen question

E08 asks whether a single-instance, token-aware SLO-feedback admission policy
improves overload goodput without unacceptable mixed-length unfairness. It
compares three policies on the exact same open-arrival request trace:

- `unbounded`: send every request to vLLM;
- `fixed_concurrency`: admit only while fewer than four requests are active;
- `token_aware`: admit by input-plus-output token cost, at most eight active
  requests, with an AIMD token budget driven by the frozen TTFT/TPOT SLO.

The serving control is `configs/serve/e06_bt2048_apc_off.toml`: Base model,
BF16 KV cache, batch-token budget 2048, APC off, and no LoRA. E07's negative
online-cost result is not modified or reinterpreted by E08.

## 2. Frozen profiles and gates

All formal profiles use short/medium/long shapes 128/128, 512/256, and 2048/256
in an exact 50/30/20 cycle. Arrival times are generated once per profile and
repetition, written as hash-authenticated trace files, and replayed unchanged
for all three policies.

| Profile | Arrival process | Mean rate | Timed phase | Role |
|---|---|---:|---:|---|
| `mixed_nominal` | seeded Poisson | 1.0 req/s | 180 s | non-overload diagnostic |
| `mixed_overload` | seeded Poisson | 2.0 req/s | 180 s | overload gate |
| `mixed_burst` | 2 s at 6.0 req/s, then 8 s at 1.0 req/s | 2.0 req/s | 180 s | burst overload gate |

Each cell has ten untimed warmups and three independent repetitions. The
27-cell formal matrix uses a predeclared Latin rotation of policy order across
repetitions and a 30-second cooldown between cells.

E08 is `PASS` only if both overload profiles have complete valid evidence and:

- token-aware P95 TTFT <= 1000 ms and P95 TPOT <= 50 ms in every repetition;
- token-aware median goodput is at least 10% above the better of unbounded and
  fixed-C4;
- token-aware median Jain index over per-class SLO yield is at least 0.80;
- token-aware Jain index is no more than 0.05 below fixed-C4.

The admitted-request error-rate validity limit remains below 1%. Rejections are
explicit policy decisions, not request errors, and remain in the denominator
of SLO yield. A failed scientific gate is retained as a valid negative result;
do not retune these values after formal execution.

## 3. Preflight on WSL2

Run in the WSL project checkout:

```bash
cd ~/projects/QwenServe-12G
git fetch origin
git switch codex/e08-admission
git pull --ff-only origin codex/e08-admission
source .venv/bin/activate

make test
make prepare-e08-traces
make audit-e08-readiness
sed -n '1,220p' reports/e08_admission/readiness.md
```

Continue only when readiness says `READY_FOR_GPU`, `GPU execution: DEFERRED`,
and `Scientific result: NOT_RUN`. The readiness report is not experiment
completion evidence.

Confirm the machine is idle before starting vLLM:

```bash
pgrep -af 'vllm serve' || true
nvidia-smi
```

## 4. Terminal 1: controlled server

Keep this terminal running for pilot and formal execution:

```bash
cd ~/projects/QwenServe-12G
source .venv/bin/activate
make serve-e08-local
```

Wait for `Application startup complete`. In another terminal verify the model
endpoint before starting a cell:

```bash
curl -fsS http://127.0.0.1:8000/v1/models | python -m json.tool
```

Do not start LoRA serving, training, or another GPU workload concurrently.

## 5. Terminal 2: pilot

Run all three policies against the same 30-second pilot trace:

```bash
cd ~/projects/QwenServe-12G
source .venv/bin/activate

make bench-e08-pilot-unbounded
sleep 30
make bench-e08-pilot-fixed
sleep 30
make bench-e08-pilot-token
```

Inspect the three pilot documents. They must have `valid: true`, zero pending
requests after drain, ten completed warmups, matching `trace_sha256`, and no
unexpected admitted-request errors. Pilot data does not enter the scientific
comparison.

```bash
find artifacts/results/e08_admission/runs/pilot -name '*-e08-*.json' -print
```

If pilot evidence is invalid, stop and diagnose it. Do not change a frozen
arrival rate, SLO, policy budget, mix, or gate in response to pilot performance.

## 6. Terminal 2: formal matrix

The following command runs the balanced 27-cell matrix and stops on invalid
evidence. Expect at least 81 minutes of timed traffic plus warmup, drain, and
cooldown time.

```bash
make bench-e08-matrix
```

If the shell disconnects but vLLM remains healthy, the preserved result files
are the source of truth. Resume only completed cells with:

```bash
make bench-e08-matrix-resume
```

The runner refuses to overwrite or silently rerun an existing cell. If a cell
is invalid, retain it and diagnose the earliest request/server error before any
new formal attempt. Do not delete failed evidence.

## 7. Build the result report

After all 27 formal cells exist:

```bash
make compare-e08; status=$?; test "$status" -eq 0 -o "$status" -eq 2
sed -n '1,240p' reports/e08_admission/comparison.md
python -m json.tool reports/e08_admission/final.json | sed -n '1,260p'
```

Exit code 2 is expected when complete evidence fails a scientific gate. It
does not authorize changing the gate or describing E08 as successful. Missing,
duplicate, hash-mismatched, invalid, or non-paired evidence is an experiment
integrity problem and must be resolved separately from the scientific result.

Stop Terminal 1 with `Ctrl+C` only after report generation is complete.
