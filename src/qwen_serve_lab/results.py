from __future__ import annotations

import csv
import json
import statistics
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from qwen_serve_lab.telemetry import summarize_telemetry


class ResultError(ValueError):
    """Raised when result evidence is missing or internally inconsistent."""


def _number(data: dict[str, Any], key: str) -> float:
    value = data.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ResultError(f"Result field {key!r} must be numeric")
    return float(value)


def _integer(data: dict[str, Any], key: str) -> int:
    value = data.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ResultError(f"Result field {key!r} must be an integer")
    return value


def _string(data: dict[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value:
        raise ResultError(f"Result field {key!r} must be a non-empty string")
    return value


def _metadata_number(data: dict[str, Any], key: str) -> float:
    value = data.get(key)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError as exc:
            raise ResultError(f"Result metadata {key!r} must be numeric") from exc
    return _number(data, key)


@dataclass(frozen=True)
class ResultRecord:
    manifest: str
    result_file: str
    profile: str
    server_profile: str
    repetition: int
    benchmark_config_sha256: str
    server_config_sha256: str
    input_len: int
    output_len: int
    slo_ttft_ms: float
    slo_tpot_ms: float
    max_concurrency: int
    completed: int
    failed: int
    error_rate: float
    request_throughput: float
    request_goodput: float
    output_throughput: float
    total_token_throughput: float
    p50_ttft_ms: float
    p95_ttft_ms: float
    p99_ttft_ms: float
    p50_tpot_ms: float
    p95_tpot_ms: float
    p99_tpot_ms: float
    p95_e2el_ms: float
    peak_memory_used_mib: float | None
    mean_gpu_utilization_percent: float | None
    max_temperature_c: float | None
    mean_power_draw_w: float | None
    valid: bool


def parse_vllm_result(
    result_path: str | Path,
    manifest_path: str | Path,
    telemetry_summary: dict[str, Any] | None = None,
) -> ResultRecord:
    path = Path(result_path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ResultError(f"Cannot read vLLM result {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ResultError(f"Expected one JSON object in {path}")

    completed = _integer(data, "completed")
    failed = _integer(data, "failed")
    total_requests = completed + failed
    if total_requests <= 0:
        raise ResultError(f"No requests recorded in {path}")
    error_rate = failed / total_requests
    telemetry = telemetry_summary or {}
    repetition_raw = _string(data, "repetition")
    try:
        repetition = int(repetition_raw)
        input_len = int(_string(data, "input_len"))
        output_len = int(_string(data, "output_len"))
    except ValueError as exc:
        raise ResultError(f"Invalid integer metadata in {path}") from exc

    record = ResultRecord(
        manifest=str(Path(manifest_path)),
        result_file=str(path),
        profile=_string(data, "profile"),
        server_profile=_string(data, "server_profile"),
        repetition=repetition,
        benchmark_config_sha256=_string(data, "benchmark_config_sha256"),
        server_config_sha256=_string(data, "server_config_sha256"),
        input_len=input_len,
        output_len=output_len,
        slo_ttft_ms=_metadata_number(data, "slo_ttft_ms"),
        slo_tpot_ms=_metadata_number(data, "slo_tpot_ms"),
        max_concurrency=_integer(data, "max_concurrency"),
        completed=completed,
        failed=failed,
        error_rate=error_rate,
        request_throughput=_number(data, "request_throughput"),
        request_goodput=_number(data, "request_goodput"),
        output_throughput=_number(data, "output_throughput"),
        total_token_throughput=_number(data, "total_token_throughput"),
        p50_ttft_ms=_number(data, "p50_ttft_ms"),
        p95_ttft_ms=_number(data, "p95_ttft_ms"),
        p99_ttft_ms=_number(data, "p99_ttft_ms"),
        p50_tpot_ms=_number(data, "p50_tpot_ms"),
        p95_tpot_ms=_number(data, "p95_tpot_ms"),
        p99_tpot_ms=_number(data, "p99_tpot_ms"),
        p95_e2el_ms=_number(data, "p95_e2el_ms"),
        peak_memory_used_mib=telemetry.get("peak_memory_used_mib"),
        mean_gpu_utilization_percent=telemetry.get(
            "mean_gpu_utilization_percent"
        ),
        max_temperature_c=telemetry.get("max_temperature_c"),
        mean_power_draw_w=telemetry.get("mean_power_draw_w"),
        valid=error_rate < 0.01,
    )
    if record.input_len <= 0 or record.output_len <= 0:
        raise ResultError(f"Invalid workload lengths in {path}")
    return record


def _resolve_evidence_path(project_root: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else project_root / path


def load_records_from_manifests(
    manifest_dir: str | Path,
    profile_prefix: str | None = None,
    benchmark_config_sha256: str | None = None,
) -> list[ResultRecord]:
    records: list[ResultRecord] = []
    for manifest_path in sorted(Path(manifest_dir).rglob("*.json")):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(manifest, dict) or not isinstance(manifest.get("runs"), list):
            continue
        profile = manifest.get("profile")
        if profile_prefix and (
            not isinstance(profile, str) or not profile.startswith(profile_prefix)
        ):
            continue
        if (
            benchmark_config_sha256 is not None
            and manifest.get("benchmark_config_sha256")
            != benchmark_config_sha256
        ):
            continue
        environment = manifest.get("environment")
        if not isinstance(environment, dict) or not isinstance(
            environment.get("project_root"), str
        ):
            raise ResultError(f"Manifest lacks environment.project_root: {manifest_path}")
        project_root = Path(environment["project_root"])

        for run in manifest["runs"]:
            if not isinstance(run, dict) or run.get("returncode") != 0:
                continue
            telemetry_raw = run.get("telemetry")
            telemetry_summary = None
            if isinstance(telemetry_raw, str):
                telemetry_path = _resolve_evidence_path(project_root, telemetry_raw)
                if telemetry_path.is_file():
                    telemetry_summary = summarize_telemetry(telemetry_path)
            result_files = run.get("result_files")
            if not isinstance(result_files, list) or len(result_files) != 1:
                raise ResultError(
                    f"Successful run must reference exactly one result file: {manifest_path}"
                )
            result_path = _resolve_evidence_path(project_root, str(result_files[0]))
            records.append(
                parse_vllm_result(
                    result_path,
                    manifest_path,
                    telemetry_summary=telemetry_summary,
                )
            )
    return records


def _median(records: list[ResultRecord], field: str) -> float:
    return float(statistics.median(getattr(record, field) for record in records))


def _min_max(records: list[ResultRecord], field: str) -> tuple[float, float]:
    values = [float(getattr(record, field)) for record in records]
    return min(values), max(values)


def _format(value: float | None, digits: int = 2) -> str:
    return "NA" if value is None else f"{value:.{digits}f}"


def _format_median_range(
    median: float, value_range: tuple[float, float], digits: int = 2
) -> str:
    return (
        f"{median:.{digits}f} "
        f"[{value_range[0]:.{digits}f}, {value_range[1]:.{digits}f}]"
    )


def aggregate_records(records: list[ResultRecord]) -> list[dict[str, Any]]:
    grouped: dict[str, list[ResultRecord]] = {}
    for record in records:
        grouped.setdefault(record.profile, []).append(record)

    aggregates: list[dict[str, Any]] = []
    for profile, group in sorted(grouped.items()):
        benchmark_hashes = {record.benchmark_config_sha256 for record in group}
        server_hashes = {record.server_config_sha256 for record in group}
        shapes = {
            (
                record.input_len,
                record.output_len,
                record.max_concurrency,
                record.slo_ttft_ms,
                record.slo_tpot_ms,
            )
            for record in group
        }
        consistent = (
            len(benchmark_hashes) == 1
            and len(server_hashes) == 1
            and len(shapes) == 1
        )
        input_len, output_len, max_concurrency, slo_ttft_ms, slo_tpot_ms = sorted(
            shapes
        )[0]
        peak_values = [
            record.peak_memory_used_mib
            for record in group
            if record.peak_memory_used_mib is not None
        ]
        telemetry_complete = len(peak_values) == len(group)
        evidence_valid = (
            consistent
            and telemetry_complete
            and len(group) >= 3
            and all(record.valid for record in group)
        )
        slo_compliant = evidence_valid and all(
            record.p95_ttft_ms <= record.slo_ttft_ms
            and record.p95_tpot_ms <= record.slo_tpot_ms
            for record in group
        )
        aggregates.append(
            {
                "profile": profile,
                "runs": len(group),
                "input_len": input_len,
                "output_len": output_len,
                "max_concurrency": max_concurrency,
                "slo_ttft_ms": slo_ttft_ms,
                "slo_tpot_ms": slo_tpot_ms,
                "request_goodput": _median(group, "request_goodput"),
                "request_goodput_range": _min_max(group, "request_goodput"),
                "output_throughput": _median(group, "output_throughput"),
                "output_throughput_range": _min_max(group, "output_throughput"),
                "p95_ttft_ms": _median(group, "p95_ttft_ms"),
                "p95_ttft_ms_range": _min_max(group, "p95_ttft_ms"),
                "p95_tpot_ms": _median(group, "p95_tpot_ms"),
                "p95_tpot_ms_range": _min_max(group, "p95_tpot_ms"),
                "p95_e2el_ms": _median(group, "p95_e2el_ms"),
                "error_rate": _median(group, "error_rate"),
                "peak_memory_used_mib": max(peak_values) if peak_values else None,
                "evidence_status": "VALID" if evidence_valid else "INCOMPLETE",
                "slo_status": (
                    "PASS" if slo_compliant else "FAIL" if evidence_valid else "UNKNOWN"
                ),
            }
        )
    aggregates.sort(
        key=lambda row: (
            row["input_len"],
            row["output_len"],
            row["max_concurrency"],
            row["profile"],
        )
    )
    return aggregates


def write_reports(
    records: list[ResultRecord], output_dir: str | Path
) -> tuple[Path, Path]:
    if not records:
        raise ResultError("No valid benchmark records found in manifests")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "runs.csv"
    markdown_path = output / "summary.md"

    row_dicts = [asdict(record) for record in records]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row_dicts[0]))
        writer.writeheader()
        writer.writerows(row_dicts)

    lines = [
        "# Benchmark Summary",
        "",
        f"Generated at: {datetime.now(timezone.utc).isoformat()}",
        "",
        "Metrics are median [minimum, maximum] across repetitions; peak VRAM is the maximum observed sample.",
        "Evidence is VALID when three consistent runs have complete telemetry and error rate below 1%; SLO PASS requires every repetition to meet both P95 limits.",
        "",
        "| Profile | Runs | In/Out | C | SLO TTFT/TPOT ms | Goodput req/s | Output tok/s | P95 TTFT ms | P95 TPOT ms | P95 E2E ms | Peak VRAM MiB | Error rate | Evidence | SLO |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in aggregate_records(records):
        lines.append(
            "| {profile} | {runs} | {input_len}/{output_len} | {max_concurrency} | {slo} | "
            "{goodput} | {throughput} | {ttft} | {tpot} | {e2el} | {vram} | "
            "{error_rate} | {evidence_status} | {slo_status} |".format(
                profile=row["profile"],
                runs=row["runs"],
                input_len=row["input_len"],
                output_len=row["output_len"],
                max_concurrency=row["max_concurrency"],
                slo=f"{row['slo_ttft_ms']:g}/{row['slo_tpot_ms']:g}",
                goodput=_format_median_range(
                    row["request_goodput"], row["request_goodput_range"]
                ),
                throughput=_format_median_range(
                    row["output_throughput"], row["output_throughput_range"]
                ),
                ttft=_format_median_range(
                    row["p95_ttft_ms"], row["p95_ttft_ms_range"]
                ),
                tpot=_format_median_range(
                    row["p95_tpot_ms"], row["p95_tpot_ms_range"]
                ),
                e2el=_format(row["p95_e2el_ms"]),
                vram=_format(row["peak_memory_used_mib"], 0),
                error_rate=_format(row["error_rate"] * 100, 2) + "%",
                evidence_status=row["evidence_status"],
                slo_status=row["slo_status"],
            )
        )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path, markdown_path
