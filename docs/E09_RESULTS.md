# E09 Deadline-aware Admission Results

## Outcome

E09 completed on the Windows/WSL2 RTX 5070 host and the frozen scientific gate
is **PASS**. All 27 formal cells are valid, all nine repetition trace groups
pair exactly across policies, and both gated burst holdouts exceed the required
10% median SLO-goodput gain over the better of unbounded and fixed-C4 while
meeting the absolute latency and fairness gates.

This is the project's first repeated, holdout-validated positive admission
result. The deployment claim is deliberately bounded to bursty open-arrival
traffic under the frozen server and workload mix.

## Frozen identity and evidence

- Experiment config SHA-256:
  `18935a9eee1ac06a05d312aed049bc78bf66b6b5bcb8ab7fddf6298df85b5635`
- Server config SHA-256:
  `2d47ae6083111e82b17f13e9676b2df92b8f27c42873a341db6285bc223495ba`
- Server: Qwen2.5-3B-Instruct Base, BF16 KV, batch-token 2048, APC off
- Formal matrix: 3 profiles x 3 policies x 3 independent repetitions = 27
  cells
- Policies: unbounded, immediate fixed-C4, and deadline-aware C8 with an
  8192-token dispatch ceiling and 800 ms external queue deadline
- SLO: P95 TTFT <= 1000 ms and P95 TPOT <= 50 ms
- Workload mix: 50% 128/128, 30% 512/256, 20% 2048/256 input/output tokens

External queue waiting remains part of arrival-to-first-token latency. Queue
expiry and admission rejection count as zero SLO yield and zero goodput.

## Formal comparison

| Profile | Scope | Offered rate | Goodput unbounded | Goodput fixed-C4 | Goodput deadline | Gain vs best baseline | Deadline P95 TTFT/TPOT | Fairness | Gate |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| holdout_nominal | diagnostic | 1.006 | 1.006 | 0.861 | 1.006 | 0.00% | 240.5 / 13.4 ms | 1.000 | N/A |
| holdout_periodic_burst | gated | 2.072 | 1.422 | 1.050 | 1.689 | **+18.75%** | 723.4 / 14.7 ms | 0.997 | **PASS** |
| holdout_shock_burst | gated | 2.094 | 1.339 | 0.994 | 1.572 | **+17.43%** | 739.2 / 14.3 ms | 0.998 | **PASS** |

The periodic and shock traces retain 45.70% and 56.43% measured offered-rate
headroom over the best baseline, respectively, above the frozen 15% feasibility
gate.

Every deadline-aware repetition passes the absolute SLO:

- periodic P95 TTFT: 718.8, 733.4, and 723.4 ms; P95 TPOT: 15.1, 14.1, and
  14.7 ms;
- shock P95 TTFT: 607.6, 742.4, and 739.2 ms; P95 TPOT: 14.9, 14.3, and
  14.3 ms.

Admitted error rate is zero in all formal cells. Deadline-aware median Jain
SLO-yield fairness is 0.9973 on periodic burst and 0.9978 on shock burst, both
above 0.80 and within 0.05 of fixed-C4.

## Mechanism and interpretation

Unbounded serving accepts every arrival but lets burst queues grow inside the
server. Its median P95 TTFT reaches 2440.9 ms on periodic burst and 3120.4 ms on
shock burst, so substantial completed throughput misses the 1000 ms TTFT SLO.

Immediate fixed-C4 keeps admitted latency low but rejects too early: its gated
median goodput is only 1.050 and 0.994 requests/s. The deadline-aware controller
uses the measured safe C8 region, waits only while an arrival can still meet its
deadline, and expires stale work before it consumes GPU service. It therefore
converts available offered headroom into SLO goodput while preserving class
fairness.

The nominal diagnostic shows no artificial benefit: deadline-aware and
unbounded both deliver 1.006 requests/s. The controller is useful under burst
congestion, not as a claim of increased physical capacity at low load.

## Deployment decision

Adopt deadline-aware admission as the default front-door policy for the frozen
bursty mixed workload on this RTX 5070 serving profile. Keep these boundaries:

- do not generalize the gains to other GPUs, models, token mixes, SLOs, or
  serving configs without a new controlled experiment;
- do not claim a steady-overload win from E09; the E08 steady-overload policy
  remains a frozen negative result;
- preserve arrival-origin TTFT, rejected-as-zero-yield accounting, and the
  800 ms queue deadline in any reproduction;
- use the nominal path as a no-regression diagnostic, not a gain claim.

## Artifact locations

Compact committed evidence:

- `reports/e09_deadline_admission/runs.csv`
- `reports/e09_deadline_admission/comparison.csv`
- `reports/e09_deadline_admission/comparison.md`
- `reports/e09_deadline_admission/final.json`

Request-level run JSON, deterministic traces, telemetry, and server logs remain
on WSL2 at
`~/projects/QwenServe-12G-e09/artifacts/results/e09_deadline_admission/`.
