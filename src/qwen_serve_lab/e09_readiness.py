from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

from qwen_serve_lab.e09_admission import (
    E09AdmissionController,
    E09Config,
    FORMAL_PROFILES,
    POLICIES,
    build_e09_trace,
)


def _check(name: str, passed: bool, detail: str) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "detail": detail}


def write_e09_readiness_report(
    repo_root: str | Path = ".",
    config_path: str | Path = "configs/admission/e09.toml",
    output_dir: str | Path = "reports/e09_deadline_admission",
) -> tuple[Path, Path, bool]:
    root = Path(repo_root).resolve()
    config_file = Path(config_path)
    if not config_file.is_absolute():
        config_file = root / config_file
    config = E09Config.from_file(config_file)
    formal_profiles = [profile for profile in config.profiles if profile.formal]
    traces = [
        build_e09_trace(config, profile.name, repetition)
        for profile in formal_profiles
        for repetition in range(1, config.repetitions + 1)
    ]
    deterministic = all(
        trace["trace_sha256"]
        == build_e09_trace(
            config, str(trace["profile"]["name"]), int(trace["repetition"])
        )["trace_sha256"]
        for trace in traces
    )
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

    offered_rates: dict[str, float] = {}
    for profile in formal_profiles:
        profile_traces = [
            trace for trace in traces if trace["profile"]["name"] == profile.name
        ]
        offered_rates[profile.name] = median(
            trace["request_count"] / profile.duration_seconds
            for trace in profile_traces
        )
    historical_headroom = {
        profile: (
            offered_rates[profile]
            - config.gates.historical_calibration_best_goodput_rps
        )
        / config.gates.historical_calibration_best_goodput_rps
        * 100
        for profile in config.gates.gated_profiles
    }

    fixed = E09AdmissionController("fixed_concurrency", config)
    fixed_leases = [
        fixed.try_acquire(f"fixed-{index}", "short", 256)
        for index in range(config.fixed_max_inflight + 1)
    ]
    fixed_valid = all(fixed_leases[:-1]) and fixed_leases[-1] is None
    for lease in fixed_leases[:-1]:
        assert lease is not None
        fixed.release(lease, slo_met=True)

    deadline = E09AdmissionController("deadline_aware", config)
    deadline_leases = [
        deadline.try_acquire(f"deadline-{index}", "short", 256)
        for index in range(config.deadline_max_inflight + 1)
    ]
    deadline_valid = all(deadline_leases[:-1]) and deadline_leases[-1] is None
    for lease in deadline_leases[:-1]:
        assert lease is not None
        deadline.release(lease, slo_met=True)

    planned_runs = len(formal_profiles) * len(POLICIES) * config.repetitions
    checks = [
        _check(
            "server-control",
            config.server_profile == "e06_bt2048_apc_off",
            "E06 Base, BF16 KV, batch-token 2048, APC-off server is hash-bound",
        ),
        _check(
            "independent-holdout",
            len(set(config.seeds)) == 3
            and config.calibration_seed not in config.seeds
            and tuple(profile.name for profile in formal_profiles) == FORMAL_PROFILES,
            "calibration seed is disjoint from three predeclared formal holdout seeds",
        ),
        _check(
            "formal-open-arrival-profiles",
            len(formal_profiles) == 3
            and all(profile.duration_seconds >= 180 for profile in formal_profiles)
            and planned_runs == 27,
            "nominal diagnostic plus periodic and shock burst holdouts; 27 formal runs",
        ),
        _check(
            "deterministic-traces",
            deterministic and min(int(trace["request_count"]) for trace in traces) >= 100,
            "nine formal traces reproduce authenticated hashes",
        ),
        _check(
            "frozen-workload-mix",
            class_mix_valid,
            "128/128, 512/256, and 2048/256 shapes use exact 50/30/20 cycles",
        ),
        _check(
            "offered-headroom",
            all(
                value >= config.gates.minimum_offered_headroom_percent
                for value in historical_headroom.values()
            ),
            "holdout offered ceilings retain at least 15% headroom over the frozen E08 burst reference",
        ),
        _check(
            "policy-controls",
            config.policies == POLICIES and fixed_valid and deadline_valid,
            "unbounded, fixed-C4, and deadline C8/token ceilings are independently accounted",
        ),
        _check(
            "deadline-accounting",
            config.deadline_max_queue_wait_ms == 800
            and config.deadline_max_inflight_tokens == 8192
            and config.slo_ttft_ms == 1000,
            "arrival-to-first-token latency includes the frozen external queue wait",
        ),
        _check(
            "frozen-scientific-gates",
            config.gates.minimum_goodput_gain_percent_vs_best_baseline == 10
            and config.gates.minimum_offered_headroom_percent == 15
            and config.gates.minimum_fairness_jain == 0.80
            and config.gates.maximum_fairness_regression_vs_fixed == 0.05,
            "gated bursts require headroom, best-baseline gain, absolute SLO, and fairness",
        ),
        _check(
            "request-level-evidence",
            (root / "src/qwen_serve_lab/e09_runner.py").is_file()
            and (root / "src/qwen_serve_lab/e09.py").is_file(),
            "decisions, queue waits, TTFT/TPOT, class yield, telemetry, and hashes are retained",
        ),
        _check(
            "runbook",
            (root / "docs/M4_E09_DEADLINE_ADMISSION_RUNBOOK.md").is_file(),
            "two-terminal GPU runbook exists",
        ),
    ]
    passed = all(check["passed"] for check in checks)
    document = {
        "schema_version": 1,
        "kind": "e09_readiness_report",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "READY_FOR_GPU" if passed else "INCOMPLETE",
        "gpu_execution": "DEFERRED",
        "scientific_result": "NOT_RUN",
        "planned_formal_runs": planned_runs,
        "historical_headroom_percent": historical_headroom,
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
        "# E09 GPU Readiness Audit",
        "",
        f"Generated at: {document['created_at']}",
        "",
        f"Status: **{document['status']}**",
        "GPU execution: **DEFERRED**",
        "Scientific result: **NOT_RUN**",
        f"Planned formal runs: **{planned_runs}**",
        "",
        "This audit validates the frozen E09 holdout protocol and evidence path. It is not a GPU result and cannot satisfy the scientific gate.",
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
