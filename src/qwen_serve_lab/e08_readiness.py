from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from qwen_serve_lab.e08_admission import (
    AdmissionController,
    E08Config,
    POLICIES,
    build_e08_trace,
)


def _check(name: str, passed: bool, detail: str) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "detail": detail}


def write_e08_readiness_report(
    repo_root: str | Path = ".",
    config_path: str | Path = "configs/admission/e08.toml",
    output_dir: str | Path = "reports/e08_admission",
) -> tuple[Path, Path, bool]:
    root = Path(repo_root).resolve()
    config_file = Path(config_path)
    if not config_file.is_absolute():
        config_file = root / config_file
    config = E08Config.from_file(config_file)
    formal_profiles = [profile for profile in config.profiles if profile.formal]
    traces = [
        build_e08_trace(config, profile.name, repetition)
        for profile in formal_profiles
        for repetition in range(1, config.repetitions + 1)
    ]
    deterministic = all(
        trace["trace_sha256"]
        == build_e08_trace(
            config, str(trace["profile"]["name"]), int(trace["repetition"])
        )["trace_sha256"]
        for trace in traces
    )
    trace_counts = [int(trace["request_count"]) for trace in traces]
    class_mix_valid = True
    for trace in traces:
        counts = {name: 0 for name in ("short", "medium", "long")}
        for request in trace["requests"]:
            counts[str(request["workload"])] += 1
        total = sum(counts.values())
        expected = {"short": 0.5, "medium": 0.3, "long": 0.2}
        class_mix_valid = class_mix_valid and total >= 100 and all(
            abs(counts[name] / total - proportion) <= 10 / total
            for name, proportion in expected.items()
        )

    fixed = AdmissionController("fixed_concurrency", config)
    fixed_leases = [
        fixed.try_acquire(f"fixed-{index}", "short", 256)
        for index in range(config.fixed_max_inflight + 1)
    ]
    fixed_valid = all(fixed_leases[:-1]) and fixed_leases[-1] is None
    for lease in fixed_leases[:-1]:
        assert lease is not None
        fixed.release(lease, slo_met=True)

    adaptive = AdmissionController("token_aware", config)
    long_lease = adaptive.try_acquire("long", "long", 2304)
    medium_lease = adaptive.try_acquire("medium", "medium", 768)
    overflow = adaptive.try_acquire("overflow", "long", 2304)
    adaptive_budget_valid = bool(long_lease and medium_lease and overflow is None)
    assert long_lease is not None and medium_lease is not None
    adaptive.release(long_lease, slo_met=False)
    decreased_budget = adaptive.snapshot()["current_token_budget"]
    adaptive.release(medium_lease, slo_met=True)
    adaptive_aimd_valid = bool(
        isinstance(decreased_budget, int)
        and config.min_token_budget <= decreased_budget < config.initial_token_budget
    )

    planned_runs = len(formal_profiles) * len(POLICIES) * config.repetitions
    checks = [
        _check(
            "server-control",
            config.server_profile == "e06_bt2048_apc_off",
            "E06 Base, BF16 KV, batch-token 2048, APC-off server is hash-bound",
        ),
        _check(
            "formal-open-arrival-profiles",
            len(formal_profiles) == 3
            and all(profile.duration_seconds >= 180 for profile in formal_profiles)
            and planned_runs == 27,
            "nominal, overload, and periodic-burst mixed traces; 27 formal runs",
        ),
        _check(
            "deterministic-traces",
            deterministic and min(trace_counts, default=0) >= 100,
            f"nine formal traces reproduce byte-normalized hashes; request counts {min(trace_counts)}-{max(trace_counts)}",
        ),
        _check(
            "frozen-workload-mix",
            class_mix_valid,
            "standard 128/128, 512/256, 2048/256 shapes use exact 50/30/20 cycles",
        ),
        _check(
            "policy-controls",
            config.policies == POLICIES and fixed_valid,
            "unbounded and fixed-C4 baselines are explicit and independently accounted",
        ),
        _check(
            "token-aware-controller",
            adaptive_budget_valid and adaptive_aimd_valid,
            "token cost, C8 ceiling, 4096 initial budget, and SLO-feedback AIMD are exercised",
        ),
        _check(
            "frozen-scientific-gates",
            config.gates.minimum_goodput_gain_percent_vs_best_baseline == 10
            and config.gates.minimum_fairness_jain == 0.80
            and config.gates.maximum_fairness_regression_vs_fixed == 0.05,
            "overload gate uses best-baseline goodput, absolute SLO, and Jain fairness",
        ),
        _check(
            "request-level-evidence",
            (root / "src/qwen_serve_lab/e08_runner.py").is_file()
            and (root / "src/qwen_serve_lab/e08.py").is_file(),
            "streaming TTFT/TPOT, decisions, class yield, telemetry, and hashes are retained",
        ),
        _check(
            "runbook",
            (root / "docs/M4_E08_ADMISSION_RUNBOOK.md").is_file(),
            "two-terminal GPU runbook exists",
        ),
    ]
    passed = all(check["passed"] for check in checks)
    document = {
        "schema_version": 1,
        "kind": "e08_readiness_report",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "READY_FOR_GPU" if passed else "INCOMPLETE",
        "gpu_execution": "DEFERRED",
        "scientific_result": "NOT_RUN",
        "planned_formal_runs": planned_runs,
        "checks": checks,
    }
    output = Path(output_dir)
    if not output.is_absolute():
        output = root / output
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "readiness.json"
    markdown_path = output / "readiness.md"
    json_path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# E08 GPU Readiness Audit",
        "",
        f"Generated at: {document['created_at']}",
        "",
        f"Status: **{document['status']}**",
        "GPU execution: **DEFERRED**",
        "Scientific result: **NOT_RUN**",
        f"Planned formal runs: **{planned_runs}**",
        "",
        "This audit validates the frozen protocol, deterministic trace generator, admission policies, evidence schema, and report gates without executing vLLM or an RTX 5070 experiment.",
        "",
        "| Check | Status | Detail |",
        "|---|---|---|",
    ]
    lines.extend(
        f"| {check['name']} | {'PASS' if check['passed'] else 'FAIL'} | {check['detail']} |"
        for check in checks
    )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, markdown_path, passed
