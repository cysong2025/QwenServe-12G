# E09 Deadline-aware Admission Protocol

## Status and research question

E08 is a frozen, valid negative result. Its token-aware AIMD policy reduced
goodput by 10.74% on the periodic burst and by 15.34% on steady overload, so
E08 does not complete the project objective.

E09 asks a narrower and feasible question: on bursty open-arrival traffic with
measured offered headroom, can a bounded external queue stop hopeless requests
from consuming GPU time and raise SLO goodput by at least 10% over the better of
unbounded serving and immediate fixed-C4 admission?

## Calibration evidence and frozen controller

The E08 periodic-burst unbounded runs are calibration data, not E09 holdout
data. Reconstructing overlap from their request-level arrival and completion
times found 100% SLO yield when a request arrived with zero through seven prior
requests active. Yield fell to 83.9% with eight prior requests, 58.9% with nine,
40.0% with ten, and 6.4% with twelve.

E09 revision A therefore freezes:

- at most eight dispatched requests;
- at most 8192 estimated input-plus-output tokens dispatched;
- an 800 ms external queue deadline;
- FIFO deadline order with first-fit backfill when the oldest request cannot fit
  the token ceiling;
- latency measured from the original trace arrival, so external waiting is part
  of TTFT;
- rejected and expired requests counted as zero SLO yield and zero goodput.

## Independence and profiles

The calibration seed is `20261217`. Formal holdout seeds are `20270117`,
`20270217`, and `20270317`; they are disjoint from E08 and calibration seeds.
All policies in a repetition consume the exact same authenticated trace.

The 180-second formal profiles are:

- `holdout_nominal`: steady 1 request/s, diagnostic only;
- `holdout_periodic_burst`: 2 seconds at 6 requests/s and 8 seconds at
  1 request/s, mean 2 requests/s, gated;
- `holdout_shock_burst`: 1 second at 11 requests/s and 9 seconds at
  1 request/s, mean 2 requests/s, gated.

The workload cycle remains 50% 128/128, 30% 512/256, and 20% 2048/256
input/output tokens. Arrival times remain open-loop and are never delayed by
client completions.

## Policies and order

The policies are `unbounded`, immediate `fixed_concurrency` at C4, and
`deadline_aware`. The three repetitions use a complete Latin rotation so each
policy occurs once in each order position. Every cell has ten warm-up requests
and a 30-second cooldown separates cells.

## Frozen scientific gate

Both gated burst profiles must independently satisfy every condition:

1. all 27 formal cells exist exactly once, are valid, hash-bound, and have exact
   trace parity across policies;
2. the measured trace offered-rate ceiling is at least 15% above the measured
   best baseline median goodput;
3. deadline-aware median request goodput is at least 10% above the better of
   unbounded and fixed-C4 median goodput;
4. deadline-aware P95 TTFT is at most 1000 ms and P95 TPOT is at most 50 ms in
   every repetition;
5. admitted error rate is below 1% and token shapes are exact;
6. median Jain fairness over per-class SLO yield is at least 0.80 and no more
   than 0.05 below fixed-C4.

The nominal profile must have complete valid evidence but is not a gain gate.
Readiness, calibration, or an incomplete matrix can never be reported as E09
completion. A formal `FAIL` remains a negative result and leaves the overall
project objective incomplete.

## Freeze boundary

The calibration burst may be run before any formal cell. If it motivates a
parameter change, that change requires a new committed config hash and newly
generated traces before formal execution. Once any formal E09 cell is run, the
config, controller, traces, gates, and report logic are frozen. No post-hoc
parameter tuning, profile removal, seed replacement, or gate relaxation is
allowed.
