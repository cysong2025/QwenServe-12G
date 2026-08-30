from __future__ import annotations

import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

from qwen_serve_lab.e09_admission import E09Config, POLICIES
from qwen_serve_lab.results import ResultError


def _number(data: dict[str, Any], key: str) -> float:
    value = data.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ResultError(f"E09 result field {key} must be numeric")
    return float(value)


def _integer(data: dict[str, Any], key: str) -> int:
    value = data.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ResultError(f"E09 result field {key} must be an integer")
    return value


def load_e09_runs(
    result_root: str | Path, config: E09Config
) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for path in sorted(Path(result_root).rglob("*-e09-*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if (
            not isinstance(data, dict)
            or data.get("kind") != "e09_admission_run"
            or data.get("experiment_config_sha256") != config.source_sha256
            or data.get("formal") is not True
        ):
            continue
        data["_path"] = str(path)
        runs.append(data)
    if not runs:
        raise ResultError("No formal E09 admission results found")
    return runs


def _flatten_run(run: dict[str, Any]) -> dict[str, Any]:
    summary = run.get("summary")
    controller = run.get("controller")
    telemetry = run.get("telemetry")
    if not all(isinstance(value, dict) for value in (summary, controller, telemetry)):
        raise ResultError(f"E09 run {run.get('_path')} lacks summary evidence")
    assert isinstance(summary, dict)
    assert isinstance(controller, dict)
    assert isinstance(telemetry, dict)
    class_metrics = summary.get("class_metrics")
    if not isinstance(class_metrics, dict):
        raise ResultError(f"E09 run {run.get('_path')} lacks class metrics")
    duration = _number(run, "offered_duration_seconds")
    arrivals = _integer(summary, "arrivals")
    return {
        "result_file": run.get("_path"),
        "profile": run.get("profile"),
        "policy": run.get("policy"),
        "repetition": run.get("repetition"),
        "seed": run.get("seed"),
        "experiment_config_sha256": run.get("experiment_config_sha256"),
        "server_config_sha256": run.get("server_config_sha256"),
        "trace_sha256": run.get("trace_sha256"),
        "offered_duration_seconds": duration,
        "offered_rate_rps": arrivals / duration,
        "arrivals": arrivals,
        "admitted": summary.get("admitted"),
        "rejected": summary.get("rejected"),
        "completed": summary.get("completed"),
        "failed": summary.get("failed"),
        "admitted_error_rate": summary.get("admitted_error_rate"),
        "request_throughput": summary.get("request_throughput"),
        "request_goodput": summary.get("request_goodput"),
        "slo_yield": summary.get("slo_yield"),
        "p95_ttft_ms": summary.get("p95_ttft_ms"),
        "p95_tpot_ms": summary.get("p95_tpot_ms"),
        "p95_e2e_ms": summary.get("p95_e2e_ms"),
        "p95_arrival_lag_ms": summary.get("p95_arrival_lag_ms"),
        "p95_admitted_queue_wait_ms": summary.get("p95_admitted_queue_wait_ms"),
        "max_admitted_queue_wait_ms": summary.get("max_admitted_queue_wait_ms"),
        "deadline_rejections": summary.get("deadline_rejections"),
        "fairness_jain": class_metrics.get("jain_slo_yield"),
        "short_slo_yield": (
            class_metrics.get("short", {}).get("slo_yield")
            if isinstance(class_metrics.get("short"), dict)
            else None
        ),
        "medium_slo_yield": (
            class_metrics.get("medium", {}).get("slo_yield")
            if isinstance(class_metrics.get("medium"), dict)
            else None
        ),
        "long_slo_yield": (
            class_metrics.get("long", {}).get("slo_yield")
            if isinstance(class_metrics.get("long"), dict)
            else None
        ),
        "peak_inflight": controller.get("peak_active"),
        "peak_inflight_tokens": controller.get("peak_inflight_tokens"),
        "peak_queue_depth": controller.get("peak_queue_depth"),
        "backfilled": controller.get("backfilled"),
        "peak_memory_used_mib": telemetry.get("peak_memory_used_mib"),
        "mean_gpu_utilization_percent": telemetry.get(
            "mean_gpu_utilization_percent"
        ),
        "max_temperature_c": telemetry.get("max_temperature_c"),
        "valid": run.get("valid"),
    }


def compare_e09_runs(
    runs: list[dict[str, Any]], config: E09Config
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], bool]:
    expected_profiles = {profile.name for profile in config.profiles if profile.formal}
    expected_keys = {
        (profile, policy, repetition)
        for profile in expected_profiles
        for policy in POLICIES
        for repetition in range(1, config.repetitions + 1)
    }
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    flat_runs: list[dict[str, Any]] = []
    for run in runs:
        flat = _flatten_run(run)
        key = (str(flat["profile"]), str(flat["policy"]), _integer(flat, "repetition"))
        grouped[key].append(flat)
        flat_runs.append(flat)
    duplicates = sorted(key for key, values in grouped.items() if len(values) != 1)
    if duplicates:
        raise ResultError(f"E09 contains duplicate formal cells: {duplicates}")
    missing = sorted(expected_keys - set(grouped))
    unexpected = sorted(set(grouped) - expected_keys)
    if unexpected:
        raise ResultError(f"E09 contains unexpected formal cells: {unexpected}")

    rows: list[dict[str, Any]] = []
    overall_passed = not missing
    for profile_name in sorted(expected_profiles):
        policy_runs = {
            policy: [
                grouped[(profile_name, policy, repetition)][0]
                for repetition in range(1, config.repetitions + 1)
                if (profile_name, policy, repetition) in grouped
            ]
            for policy in POLICIES
        }
        trace_parity = all(
            len(
                {
                    grouped[(profile_name, policy, repetition)][0]["trace_sha256"]
                    for policy in POLICIES
                    if (profile_name, policy, repetition) in grouped
                }
            )
            == 1
            for repetition in range(1, config.repetitions + 1)
            if all(
                (profile_name, policy, repetition) in grouped for policy in POLICIES
            )
        )
        evidence_valid = bool(
            all(len(policy_runs[policy]) == config.repetitions for policy in POLICIES)
            and trace_parity
            and all(
                run.get("valid") is True
                and run.get("experiment_config_sha256") == config.source_sha256
                and run.get("server_config_sha256") == config.server_config_sha256
                and _number(run, "offered_duration_seconds") >= 180
                for values in policy_runs.values()
                for run in values
            )
        )

        def med(policy: str, key: str) -> float | None:
            values = policy_runs[policy]
            if not values:
                return None
            raw = [run.get(key) for run in values]
            if any(
                not isinstance(value, (int, float)) or isinstance(value, bool)
                for value in raw
            ):
                return None
            return median(float(value) for value in raw)

        goodputs = {policy: med(policy, "request_goodput") for policy in POLICIES}
        fairness = {policy: med(policy, "fairness_jain") for policy in POLICIES}
        baseline_values = [
            value
            for value in (goodputs["unbounded"], goodputs["fixed_concurrency"])
            if value is not None
        ]
        best_baseline = max(baseline_values) if baseline_values else None
        deadline_goodput = goodputs["deadline_aware"]
        goodput_gain = (
            (deadline_goodput - best_baseline) / best_baseline * 100
            if deadline_goodput is not None and best_baseline not in (None, 0)
            else None
        )
        offered_rate = med("deadline_aware", "offered_rate_rps")
        offered_headroom = (
            (offered_rate - best_baseline) / best_baseline * 100
            if offered_rate is not None and best_baseline not in (None, 0)
            else None
        )
        deadline_slo = bool(
            len(policy_runs["deadline_aware"]) == config.repetitions
            and all(
                isinstance(run.get("p95_ttft_ms"), (int, float))
                and not isinstance(run.get("p95_ttft_ms"), bool)
                and float(run["p95_ttft_ms"]) <= config.slo_ttft_ms
                and isinstance(run.get("p95_tpot_ms"), (int, float))
                and not isinstance(run.get("p95_tpot_ms"), bool)
                and float(run["p95_tpot_ms"]) <= config.slo_tpot_ms
                and isinstance(run.get("max_admitted_queue_wait_ms"), (int, float))
                and float(run["max_admitted_queue_wait_ms"])
                <= config.deadline_max_queue_wait_ms
                for run in policy_runs["deadline_aware"]
            )
        )
        deadline_fairness = fairness["deadline_aware"]
        fixed_fairness = fairness["fixed_concurrency"]
        fairness_passed = bool(
            deadline_fairness is not None
            and fixed_fairness is not None
            and deadline_fairness >= config.gates.minimum_fairness_jain
            and deadline_fairness
            >= fixed_fairness - config.gates.maximum_fairness_regression_vs_fixed
        )
        headroom_passed = bool(
            offered_headroom is not None
            and offered_headroom >= config.gates.minimum_offered_headroom_percent
        )
        gated = profile_name in config.gates.gated_profiles
        scientific_passed = bool(
            evidence_valid
            and headroom_passed
            and deadline_slo
            and fairness_passed
            and goodput_gain is not None
            and goodput_gain
            >= config.gates.minimum_goodput_gain_percent_vs_best_baseline
        )
        if gated:
            overall_passed = overall_passed and scientific_passed
        else:
            overall_passed = overall_passed and evidence_valid
        rows.append(
            {
                "profile": profile_name,
                "evidence": "VALID" if evidence_valid else "INCOMPLETE",
                "trace_parity": "PASS" if trace_parity else "FAIL",
                "offered_rate_rps": offered_rate,
                "offered_headroom_percent": offered_headroom,
                "unbounded_goodput": goodputs["unbounded"],
                "fixed_goodput": goodputs["fixed_concurrency"],
                "deadline_goodput": deadline_goodput,
                "best_baseline_goodput": best_baseline,
                "deadline_goodput_gain_percent": goodput_gain,
                "unbounded_p95_ttft_ms": med("unbounded", "p95_ttft_ms"),
                "fixed_p95_ttft_ms": med("fixed_concurrency", "p95_ttft_ms"),
                "deadline_p95_ttft_ms": med("deadline_aware", "p95_ttft_ms"),
                "unbounded_p95_tpot_ms": med("unbounded", "p95_tpot_ms"),
                "fixed_p95_tpot_ms": med("fixed_concurrency", "p95_tpot_ms"),
                "deadline_p95_tpot_ms": med("deadline_aware", "p95_tpot_ms"),
                "deadline_p95_queue_wait_ms": med(
                    "deadline_aware", "p95_admitted_queue_wait_ms"
                ),
                "unbounded_fairness_jain": fairness["unbounded"],
                "fixed_fairness_jain": fixed_fairness,
                "deadline_fairness_jain": deadline_fairness,
                "offered_headroom_gate": "PASS" if headroom_passed else "FAIL",
                "deadline_absolute_slo": "PASS" if deadline_slo else "FAIL",
                "deadline_fairness": "PASS" if fairness_passed else "FAIL",
                "gate_scope": "GATED_BURST" if gated else "DIAGNOSTIC",
                "scientific_gate": (
                    "PASS" if scientific_passed else "FAIL"
                ) if gated else "NOT_APPLICABLE",
            }
        )
    return (
        sorted(flat_runs, key=lambda row: (row["profile"], row["policy"], row["repetition"])),
        rows,
        overall_passed,
    )


def _fmt(value: Any, digits: int = 3) -> str:
    return "NA" if not isinstance(value, (int, float)) else f"{value:.{digits}f}"


def write_e09_comparison(
    result_root: str | Path = "artifacts/results/e09_deadline_admission/runs",
    config_path: str | Path = "configs/admission/e09.toml",
    output_dir: str | Path = "reports/e09_deadline_admission",
) -> tuple[Path, Path, Path, Path, bool]:
    config = E09Config.from_file(config_path)
    flat_runs, comparisons, passed = compare_e09_runs(
        load_e09_runs(result_root, config), config
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    runs_path = output / "runs.csv"
    comparison_path = output / "comparison.csv"
    markdown_path = output / "comparison.md"
    final_path = output / "final.json"
    with runs_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat_runs[0]))
        writer.writeheader()
        writer.writerows(flat_runs)
    with comparison_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparisons[0]))
        writer.writeheader()
        writer.writerows(comparisons)
    status = "PASS" if passed else "FAIL"
    document = {
        "schema_version": 1,
        "kind": "e09_final_report",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "experiment_config_sha256": config.source_sha256,
        "server_config_sha256": config.server_config_sha256,
        "formal_runs": len(flat_runs),
        "gates": {
            "minimum_goodput_gain_percent_vs_best_baseline": config.gates.minimum_goodput_gain_percent_vs_best_baseline,
            "minimum_offered_headroom_percent": config.gates.minimum_offered_headroom_percent,
            "minimum_fairness_jain": config.gates.minimum_fairness_jain,
            "maximum_fairness_regression_vs_fixed": config.gates.maximum_fairness_regression_vs_fixed,
            "absolute_slo_ttft_ms": config.slo_ttft_ms,
            "absolute_slo_tpot_ms": config.slo_tpot_ms,
        },
        "profiles": comparisons,
    }
    final_path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# E09 Deadline-aware Admission Comparison",
        "",
        f"Generated at: {document['created_at']}",
        "",
        f"Overall frozen gate: **{status}**",
        "",
        "Each gated holdout burst requires three valid exact-trace repetitions per policy, at least 15% measured offered headroom, deadline-aware P95 TTFT/TPOT within 1000/50 ms in every repetition, median goodput at least 10% above the better of unbounded and fixed-C4, Jain SLO-yield fairness at least 0.80, and no more than 0.05 fairness regression versus fixed-C4.",
        "",
        "| Profile | Evidence | Offered/headroom | Goodput unbounded/fixed/deadline | Gain | P95 TTFT unbounded/fixed/deadline ms | Queue P95 ms | Fairness fixed/deadline | SLO | Fairness | Gate |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|---|---|",
    ]
    for row in comparisons:
        lines.append(
            "| {profile} | {evidence} | {offered}/{headroom}% | {ug}/{fg}/{dg} | {gain}% | {ut}/{ft}/{dt} | {queue} | {ff}/{df} | {slo} | {fair} | {gate} |".format(
                profile=row["profile"],
                evidence=row["evidence"],
                offered=_fmt(row["offered_rate_rps"]),
                headroom=_fmt(row["offered_headroom_percent"], 2),
                ug=_fmt(row["unbounded_goodput"]),
                fg=_fmt(row["fixed_goodput"]),
                dg=_fmt(row["deadline_goodput"]),
                gain=_fmt(row["deadline_goodput_gain_percent"], 2),
                ut=_fmt(row["unbounded_p95_ttft_ms"], 1),
                ft=_fmt(row["fixed_p95_ttft_ms"], 1),
                dt=_fmt(row["deadline_p95_ttft_ms"], 1),
                queue=_fmt(row["deadline_p95_queue_wait_ms"], 1),
                ff=_fmt(row["fixed_fairness_jain"]),
                df=_fmt(row["deadline_fairness_jain"]),
                slo=row["deadline_absolute_slo"],
                fair=row["deadline_fairness"],
                gate=row["scientific_gate"],
            )
        )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return runs_path, comparison_path, markdown_path, final_path, passed
