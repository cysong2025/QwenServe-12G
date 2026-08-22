from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any

from qwen_serve_lab.config import ServeConfig
from qwen_serve_lab.results import ResultError


E02_PROFILE = re.compile(r"^e02_bt(?P<budget>\d+)_")
E06_SERVERS = (
    "e06_bt8192_apc_off",
    "e06_bt2048_apc_off",
    "e06_bt8192_apc_on",
    "e06_bt2048_apc_on",
)


def _load_csv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except OSError as exc:
        raise ResultError(f"Cannot read audit CSV {path}: {exc}") from exc
    if not rows:
        raise ResultError(f"Audit CSV is empty: {path}")
    return rows


def _load_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ResultError(f"Cannot read audit JSON {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ResultError(f"Audit JSON root must be an object: {path}")
    return data


def _as_float(row: dict[str, str], key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ResultError(f"Audit field {key!r} must be numeric") from exc


def _as_int(row: dict[str, str], key: str) -> int:
    value = _as_float(row, key)
    if not value.is_integer():
        raise ResultError(f"Audit field {key!r} must be an integer")
    return int(value)


def _is_true(row: dict[str, str], key: str) -> bool:
    return row.get(key, "").strip().lower() == "true"


def _profile_groups(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    groups: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        profile = row.get("profile", "")
        groups.setdefault(profile, []).append(row)
    return groups


def _run_checks(
    rows: list[dict[str, str]], expected_runs: int, expected_profiles: int
) -> tuple[dict[str, bool], int]:
    groups = _profile_groups(rows)
    checks = {
        "run_count": len(rows) == expected_runs,
        "profile_count": len(groups) == expected_profiles,
        "three_repetitions_per_profile": all(
            len(profile_rows) == 3
            and {_as_int(row, "repetition") for row in profile_rows} == {1, 2, 3}
            for profile_rows in groups.values()
        ),
        "all_runs_valid": all(_is_true(row, "valid") for row in rows),
        "all_error_rates_below_one_percent": all(
            _as_float(row, "error_rate") < 0.01 for row in rows
        ),
        "all_requests_complete": all(
            _as_int(row, "completed") == 100 and _as_int(row, "failed") == 0
            for row in rows
        ),
    }
    return checks, len(groups)


def _slo_profile_counts(rows: list[dict[str, str]]) -> tuple[int, int]:
    groups = _profile_groups(rows)
    passed = sum(
        all(
            _as_float(row, "p95_ttft_ms") <= _as_float(row, "slo_ttft_ms")
            and _as_float(row, "p95_tpot_ms") <= _as_float(row, "slo_tpot_ms")
            for row in profile_rows
        )
        for profile_rows in groups.values()
    )
    return passed, len(groups) - passed


def _stage_complete(stage: dict[str, Any]) -> bool:
    checks = stage.get("checks", {})
    return isinstance(checks, dict) and all(checks.values())


def audit_e01_e06(repo_root: str | Path = ".") -> dict[str, Any]:
    root = Path(repo_root).resolve()
    reports = root / "reports"

    e01_rows = _load_csv(reports / "baseline/runs.csv")
    e01_checks, e01_profiles = _run_checks(e01_rows, 36, 12)
    e01_pass, e01_fail = _slo_profile_counts(e01_rows)
    e01_shapes = {
        (_as_int(row, "input_len"), _as_int(row, "output_len"))
        for row in e01_rows
    }
    e01_concurrencies = {_as_int(row, "max_concurrency") for row in e01_rows}
    e01_checks.update({
        "baseline_shapes_complete": e01_shapes
        == {(128, 128), (512, 256), (2048, 256)},
        "baseline_concurrencies_complete": e01_concurrencies == {1, 4, 8, 16},
        "expected_slo_boundary": (e01_pass, e01_fail) == (8, 4),
    })
    e01 = {
        "id": "E01",
        "status": "COMPLETE_WITH_PROTOCOL_DEVIATION",
        "benchmark_runs": len(e01_rows),
        "profiles": e01_profiles,
        "slo_pass_profiles": e01_pass,
        "slo_fail_profiles": e01_fail,
        "checks": e01_checks,
        "protocol_deviation": "Transformers single-request reference not executed",
    }
    e03 = {
        "id": "E03",
        "status": "COVERED_BY_E01",
        "benchmark_runs": 0,
        "profiles": 0,
        "shared_evidence": "E01 length-by-concurrency matrix",
        "checks": {
            "three_workload_shapes": len(e01_shapes) == 3,
            "four_concurrency_levels": len(e01_concurrencies) == 4,
            "shared_runs_not_double_counted": True,
        },
    }

    e02_rows = _load_csv(reports / "e02_batch_tokens/runs.csv")
    e02_comparison = _load_csv(reports / "e02_batch_tokens/comparison.csv")
    e02_checks, e02_profiles = _run_checks(e02_rows, 72, 24)
    e02_budgets = {
        int(match.group("budget"))
        for row in e02_rows
        if (match := E02_PROFILE.match(row.get("profile", ""))) is not None
    }
    e02_checks.update({
        "four_budgets": e02_budgets == {2048, 4096, 8192, 16384},
        "comparison_rows": len(e02_comparison) == 24,
        "comparison_evidence_valid": all(
            row.get("evidence_status") == "VALID" for row in e02_comparison
        ),
        "long_c8_is_slo_boundary": sum(
            _as_int(row, "input_len") == 2048
            and _as_int(row, "max_concurrency") == 8
            and row.get("slo_status") == "FAIL"
            for row in e02_comparison
        )
        == 4,
    })
    e02 = {
        "id": "E02",
        "status": "COMPLETE",
        "benchmark_runs": len(e02_rows),
        "profiles": e02_profiles,
        "budgets": sorted(e02_budgets),
        "checks": e02_checks,
    }

    e04_rows = _load_csv(reports / "e04_prefix_cache/runs.csv")
    e04_comparison = _load_csv(reports / "e04_prefix_cache/comparison.csv")
    e04_canary = _load_json(
        reports / "e04_prefix_cache/correctness_canary.json"
    )
    e04_checks, e04_profiles = _run_checks(e04_rows, 36, 12)
    e04_checks.update({
        "paired_comparison_rows": len(e04_comparison) == 6,
        "strict_random_output_gate_preserved": all(
            row.get("evidence") == "INCOMPLETE" for row in e04_comparison
        ),
        "canary_dataset_matches": e04_canary.get("dataset_sha256_match") is True,
        "canary_outputs_match": e04_canary.get("output_matches") == 24,
        "apc_equivalence_passes": e04_canary.get("apc_equivalence_status")
        == "PASS",
        "cache_exercised": e04_canary.get("on_cache_exercised") is True,
        "shared_base_errors_preserved": e04_canary.get("off_expected_matches")
        == e04_canary.get("on_expected_matches")
        == 22,
    })
    e04 = {
        "id": "E04",
        "status": "COMPLETE_WITH_LIMITATIONS",
        "benchmark_runs": len(e04_rows),
        "profiles": e04_profiles,
        "canary_output_matches": "24/24",
        "canary_task_accuracy": "22/24",
        "checks": e04_checks,
    }

    e05_rows = _load_csv(reports / "e05_kv_cache/runs.csv")
    e05_capacity = _load_json(reports / "e05_kv_cache/capacity.json")
    e05_quality = _load_json(reports / "e05_kv_cache/quality.json")
    e05_human = _load_json(
        reports / "e05_kv_cache/human_review_summary.json"
    )
    e05_checks, e05_profiles = _run_checks(e05_rows, 36, 12)
    capacity_ratio = float(e05_capacity.get("fp8_to_bf16_token_capacity_ratio", 0))
    e05_checks.update({
        "capacity_evidence_valid": e05_capacity.get("evidence_status") == "VALID",
        "capacity_gate_passes": e05_capacity.get("capacity_status") == "PASS"
        and capacity_ratio >= 1.8,
        "quality_dataset_complete": e05_quality.get("cases") == 50
        and e05_quality.get("dataset_sha256_match") is True,
        "automated_regression_preserved": e05_quality.get("automated_status")
        == "FAIL",
        "human_review_complete": e05_human.get("cases") == 50,
        "human_regression_preserved": e05_human.get("status") == "FAIL"
        and float(e05_human.get("fp8_minus_bf16", 0)) <= -0.10,
    })
    e05 = {
        "id": "E05",
        "status": "COMPLETE_WITH_QUALITY_REGRESSION",
        "benchmark_runs": len(e05_rows),
        "profiles": e05_profiles,
        "fp8_capacity_ratio": capacity_ratio,
        "automated_quality_status": e05_quality.get("automated_status"),
        "human_quality_status": e05_human.get("status"),
        "checks": e05_checks,
    }

    e06_rows = _load_csv(reports / "e06_combined/runs.csv")
    e06_comparison = _load_csv(reports / "e06_combined/comparison.csv")
    e06_canary = _load_json(reports / "e06_combined/correctness_canary.json")
    e06_checks, e06_profiles = _run_checks(e06_rows, 48, 16)
    e06_server_configs = [
        ServeConfig.from_file(root / f"configs/serve/{name}.toml")
        for name in E06_SERVERS
    ]
    stacked_conditions = {
        row.get("condition", "")
        for row in e06_comparison
        if row.get("decision") == "STACKED_BENEFIT"
    }
    e06_checks.update({
        "four_factorial_conditions": len(e06_comparison) == 4,
        "factorial_evidence_valid": all(
            row.get("evidence") == "VALID" for row in e06_comparison
        ),
        "combined_slo_passes": all(
            row.get("combined_slo") == "PASS" for row in e06_comparison
        ),
        "expected_stacked_conditions": stacked_conditions
        == {"reuse50_p1024", "capacity_reuse90_p1792"},
        "canary_passes": e06_canary.get("overall_status") == "PASS",
        "canary_outputs_match": e06_canary.get(
            "output_matches_across_all_cells"
        )
        == 24,
        "apc_cells_exercised_cache": e06_canary.get(
            "on_cells_exercised_cache"
        )
        is True,
        "oom_headroom_revision_frozen": {
            config.gpu_memory_utilization for config in e06_server_configs
        }
        == {0.78},
    })
    e06 = {
        "id": "E06",
        "status": "COMPLETE",
        "benchmark_runs": len(e06_rows),
        "profiles": e06_profiles,
        "stacked_benefit_conditions": sorted(stacked_conditions),
        "canary_output_matches": "24/24",
        "checks": e06_checks,
    }

    stages = [e01, e02, e03, e04, e05, e06]
    total_runs = sum(
        int(stage["benchmark_runs"])
        for stage in stages
        if stage["id"] != "E03"
    )
    complete = all(_stage_complete(stage) for stage in stages) and total_runs == 228
    return {
        "schema_version": 1,
        "kind": "e01_e06_evidence_audit",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "overall_status": "PASS" if complete else "FAIL",
        "milestone_status": "E01_E06_COMPLETE" if complete else "INCOMPLETE",
        "total_benchmark_runs": total_runs,
        "stages": stages,
    }


def _check_summary(stage: dict[str, Any]) -> str:
    checks = stage["checks"]
    passed = sum(bool(value) for value in checks.values())
    return f"{passed}/{len(checks)}"


def _render_markdown(data: dict[str, Any]) -> str:
    lines = [
        "# E01-E06 Evidence Audit",
        "",
        f"Generated at: {data['created_at']}",
        "",
        f"Overall status: **{data['overall_status']}**",
        f"Milestone status: **{data['milestone_status']}**",
        f"Unique formal benchmark runs: **{data['total_benchmark_runs']}**",
        "",
        "A PASS means the committed evidence is complete and internally consistent. "
        "Expected negative findings, including the E05 quality regression, remain failures "
        "of their optimization gates and are not rewritten as successful optimizations.",
        "",
        "| Experiment | Status | Unique runs | Profiles | Checks |",
        "|---|---|---:|---:|---:|",
    ]
    for stage in data["stages"]:
        lines.append(
            f"| {stage['id']} | {stage['status']} | "
            f"{stage['benchmark_runs']} | {stage['profiles']} | "
            f"{_check_summary(stage)} |"
        )
    lines.extend([
        "",
        "## Frozen Findings",
        "",
        "- E01/E03: 8 of 12 baseline profiles pass SLO; C16 and Long-C8 expose the queueing boundary.",
        "  The planned Transformers single-request reference was not executed, so no vLLM-versus-Transformers speedup is claimed.",
        "- E02: all four Long-C8 budgets fail the every-repetition SLO gate; smaller budgets reduce long-input TTFT but do not replace admission control.",
        "- E04: APC functional equivalence passes 24/24 canary cases, while the strict random-output gate remains incomplete and task accuracy remains 22/24 on both sides.",
        f"- E05: FP8 KV capacity is {data['stages'][4]['fp8_capacity_ratio']:.3f}x, but automated and blinded-human quality gates fail.",
        "- E06: stacked benefit is validated for reuse50_p1024/C4 and capacity_reuse90_p1792/C8; the four-cell canary passes 24/24.",
        "",
    ])
    return "\n".join(lines)


def write_e01_e06_audit(
    repo_root: str | Path = ".", output_dir: str | Path = "reports/e01_e06"
) -> tuple[Path, Path, bool]:
    data = audit_e01_e06(repo_root)
    root = Path(repo_root).resolve()
    output = Path(output_dir)
    if not output.is_absolute():
        output = root / output
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "audit.json"
    markdown_path = output / "audit.md"
    json_path.write_text(
        json.dumps(data, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    markdown_path.write_text(_render_markdown(data), encoding="utf-8")
    return json_path, markdown_path, data["overall_status"] == "PASS"
