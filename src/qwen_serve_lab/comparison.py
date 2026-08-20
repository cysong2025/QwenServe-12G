from __future__ import annotations

import csv
import re
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from qwen_serve_lab.results import ResultError


E02_PROFILE_PATTERN = re.compile(r"^e02_bt(?P<budget>\d+)_.+$")

INTEGER_FIELDS = (
    "repetition",
    "input_len",
    "output_len",
    "max_concurrency",
    "completed",
    "failed",
)
FLOAT_FIELDS = (
    "slo_ttft_ms",
    "slo_tpot_ms",
    "error_rate",
    "request_goodput",
    "output_throughput",
    "p95_ttft_ms",
    "p95_tpot_ms",
    "peak_memory_used_mib",
    "mean_gpu_utilization_percent",
    "max_temperature_c",
    "mean_power_draw_w",
    "mean_sm_clock_mhz",
)


def _optional_float(row: dict[str, str], key: str) -> float | None:
    raw = row.get(key, "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError as exc:
        raise ResultError(f"CSV field {key!r} must be numeric") from exc


def load_e02_runs(path: str | Path) -> list[dict[str, Any]]:
    csv_path = Path(path)
    try:
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            raw_rows = list(csv.DictReader(handle))
    except OSError as exc:
        raise ResultError(f"Cannot read E02 runs CSV {csv_path}: {exc}") from exc

    if not raw_rows:
        raise ResultError(f"E02 runs CSV is empty: {csv_path}")

    rows: list[dict[str, Any]] = []
    for raw in raw_rows:
        profile = raw.get("profile", "")
        match = E02_PROFILE_PATTERN.fullmatch(profile)
        if match is None:
            raise ResultError(f"Unexpected E02 profile name: {profile!r}")
        try:
            row: dict[str, Any] = {
                **raw,
                "budget": int(match.group("budget")),
                "valid": raw.get("valid", "").lower() == "true",
            }
            for key in INTEGER_FIELDS:
                row[key] = int(raw[key])
            for key in FLOAT_FIELDS:
                row[key] = _optional_float(raw, key)
        except (KeyError, ValueError) as exc:
            raise ResultError(f"Invalid E02 CSV row for {profile}: {exc}") from exc
        rows.append(row)
    return rows


def _numeric(rows: list[dict[str, Any]], key: str) -> list[float]:
    return [float(row[key]) for row in rows if row.get(key) is not None]


def _median(rows: list[dict[str, Any]], key: str) -> float | None:
    values = _numeric(rows, key)
    return float(statistics.median(values)) if values else None


def _maximum(rows: list[dict[str, Any]], key: str) -> float | None:
    values = _numeric(rows, key)
    return max(values) if values else None


def _pct_delta(value: float | None, reference: float | None) -> float | None:
    if value is None or reference is None or reference == 0:
        return None
    return (value - reference) / reference * 100


def _format(value: float | None, digits: int = 2) -> str:
    return "NA" if value is None else f"{value:.{digits}f}"


def _format_delta(value: float | None) -> str:
    return "NA" if value is None else f"{value:+.2f}%"


def aggregate_e02_runs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, int, int, int], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            row["input_len"],
            row["output_len"],
            row["max_concurrency"],
            row["budget"],
        )
        grouped.setdefault(key, []).append(row)

    aggregates: list[dict[str, Any]] = []
    for (input_len, output_len, concurrency, budget), group in sorted(grouped.items()):
        consistent = (
            len({row["benchmark_config_sha256"] for row in group}) == 1
            and len({row["server_config_sha256"] for row in group}) == 1
            and {row["repetition"] for row in group} == {1, 2, 3}
        )
        evidence_valid = (
            len(group) == 3
            and consistent
            and all(row["valid"] for row in group)
            and len(_numeric(group, "peak_memory_used_mib")) == len(group)
        )
        slo_pass = evidence_valid and all(
            row["p95_ttft_ms"] is not None
            and row["p95_tpot_ms"] is not None
            and row["p95_ttft_ms"] <= row["slo_ttft_ms"]
            and row["p95_tpot_ms"] <= row["slo_tpot_ms"]
            for row in group
        )
        aggregates.append(
            {
                "input_len": input_len,
                "output_len": output_len,
                "max_concurrency": concurrency,
                "budget": budget,
                "runs": len(group),
                "request_goodput": _median(group, "request_goodput"),
                "output_throughput": _median(group, "output_throughput"),
                "p95_ttft_ms": _median(group, "p95_ttft_ms"),
                "p95_tpot_ms": _median(group, "p95_tpot_ms"),
                "peak_memory_used_mib": _maximum(group, "peak_memory_used_mib"),
                "mean_gpu_utilization_percent": _median(
                    group, "mean_gpu_utilization_percent"
                ),
                "max_temperature_c": _median(group, "max_temperature_c"),
                "mean_power_draw_w": _median(group, "mean_power_draw_w"),
                "mean_sm_clock_mhz": _median(group, "mean_sm_clock_mhz"),
                "evidence_status": "VALID" if evidence_valid else "INCOMPLETE",
                "slo_status": (
                    "PASS" if slo_pass else "FAIL" if evidence_valid else "UNKNOWN"
                ),
            }
        )
    return aggregates


def _telemetry_by_budget(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["budget"], []).append(row)
    return [
        {
            "budget": budget,
            "runs": len(group),
            "mean_gpu_utilization_percent": _median(
                group, "mean_gpu_utilization_percent"
            ),
            "max_temperature_c": _median(group, "max_temperature_c"),
            "mean_power_draw_w": _median(group, "mean_power_draw_w"),
            "mean_sm_clock_mhz": _median(group, "mean_sm_clock_mhz"),
        }
        for budget, group in sorted(grouped.items())
    ]


def build_e02_comparison(
    rows: list[dict[str, Any]], reference_budget: int = 8192
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    aggregates = aggregate_e02_runs(rows)
    references = {
        (row["input_len"], row["output_len"], row["max_concurrency"]): row
        for row in aggregates
        if row["budget"] == reference_budget
    }
    shapes = {
        (row["input_len"], row["output_len"], row["max_concurrency"])
        for row in aggregates
    }
    missing = sorted(shapes - references.keys())
    if missing:
        raise ResultError(
            f"Reference budget {reference_budget} is missing shapes: {missing}"
        )

    compared: list[dict[str, Any]] = []
    for row in aggregates:
        shape = (row["input_len"], row["output_len"], row["max_concurrency"])
        reference = references[shape]
        compared.append(
            {
                **row,
                "output_throughput_delta_percent": _pct_delta(
                    row["output_throughput"], reference["output_throughput"]
                ),
                "p95_ttft_delta_percent": _pct_delta(
                    row["p95_ttft_ms"], reference["p95_ttft_ms"]
                ),
                "p95_tpot_delta_percent": _pct_delta(
                    row["p95_tpot_ms"], reference["p95_tpot_ms"]
                ),
                "request_goodput_delta_percent": _pct_delta(
                    row["request_goodput"], reference["request_goodput"]
                ),
                "peak_vram_delta_mib": (
                    row["peak_memory_used_mib"]
                    - reference["peak_memory_used_mib"]
                    if row["peak_memory_used_mib"] is not None
                    and reference["peak_memory_used_mib"] is not None
                    else None
                ),
            }
        )
    return compared, _telemetry_by_budget(rows)


def write_e02_comparison(
    runs_csv: str | Path,
    output_dir: str | Path,
    reference_budget: int = 8192,
) -> tuple[Path, Path]:
    rows = load_e02_runs(runs_csv)
    compared, telemetry = build_e02_comparison(rows, reference_budget)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "comparison.csv"
    markdown_path = output / "comparison.md"

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(compared[0]))
        writer.writeheader()
        writer.writerows(compared)

    lines = [
        "# E02 Batch Token Budget Comparison",
        "",
        f"Generated at: {datetime.now(timezone.utc).isoformat()}",
        "",
        f"Reference budget: `{reference_budget}`. Positive throughput/goodput deltas are better; negative latency/VRAM deltas are better.",
        "",
        "| In/Out | C | Budget | Output tok/s | Delta | P95 TTFT ms | Delta | P95 TPOT ms | Delta | Goodput req/s | Delta | Peak VRAM MiB | Delta MiB | Evidence | SLO |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in compared:
        lines.append(
            "| {shape} | {concurrency} | {budget} | {throughput} | {throughput_delta} | "
            "{ttft} | {ttft_delta} | {tpot} | {tpot_delta} | {goodput} | "
            "{goodput_delta} | {vram} | {vram_delta} | {evidence} | {slo} |".format(
                shape=f"{row['input_len']}/{row['output_len']}",
                concurrency=row["max_concurrency"],
                budget=row["budget"],
                throughput=_format(row["output_throughput"]),
                throughput_delta=_format_delta(
                    row["output_throughput_delta_percent"]
                ),
                ttft=_format(row["p95_ttft_ms"]),
                ttft_delta=_format_delta(row["p95_ttft_delta_percent"]),
                tpot=_format(row["p95_tpot_ms"]),
                tpot_delta=_format_delta(row["p95_tpot_delta_percent"]),
                goodput=_format(row["request_goodput"]),
                goodput_delta=_format_delta(
                    row["request_goodput_delta_percent"]
                ),
                vram=_format(row["peak_memory_used_mib"], 0),
                vram_delta=_format(row["peak_vram_delta_mib"], 0),
                evidence=row["evidence_status"],
                slo=row["slo_status"],
            )
        )

    lines.extend(
        [
            "",
            "## Run-State Telemetry",
            "",
            "Values are medians across all repetitions and workload shapes for each budget; temperature is the per-run maximum.",
            "",
            "| Budget | Runs | GPU util % | Max temp C | Mean power W | Mean SM clock MHz |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in telemetry:
        lines.append(
            "| {budget} | {runs} | {util} | {temp} | {power} | {clock} |".format(
                budget=row["budget"],
                runs=row["runs"],
                util=_format(row["mean_gpu_utilization_percent"]),
                temp=_format(row["max_temperature_c"]),
                power=_format(row["mean_power_draw_w"]),
                clock=_format(row["mean_sm_clock_mhz"], 0),
            )
        )

    reference_telemetry = next(
        (row for row in telemetry if row["budget"] == reference_budget), None
    )
    warnings: list[str] = []
    if reference_telemetry is not None:
        for row in telemetry:
            if row["budget"] == reference_budget:
                continue
            temp_delta = (
                row["max_temperature_c"] - reference_telemetry["max_temperature_c"]
                if row["max_temperature_c"] is not None
                and reference_telemetry["max_temperature_c"] is not None
                else None
            )
            power_delta = (
                row["mean_power_draw_w"] - reference_telemetry["mean_power_draw_w"]
                if row["mean_power_draw_w"] is not None
                and reference_telemetry["mean_power_draw_w"] is not None
                else None
            )
            clock_delta = (
                row["mean_sm_clock_mhz"] - reference_telemetry["mean_sm_clock_mhz"]
                if row["mean_sm_clock_mhz"] is not None
                and reference_telemetry["mean_sm_clock_mhz"] is not None
                else None
            )
            if (
                (temp_delta is not None and abs(temp_delta) >= 3)
                or (power_delta is not None and abs(power_delta) >= 2)
                or (clock_delta is not None and abs(clock_delta) >= 30)
            ):
                warnings.append(
                    f"- Budget {row['budget']} differs from {reference_budget} in run-state telemetry "
                    f"(temperature {_format(temp_delta)} C, power {_format(power_delta)} W, "
                    f"SM clock {_format(clock_delta, 0)} MHz); small performance deltas may be confounded."
                )
    if warnings:
        lines.extend(["", "## Validity Notes", "", *warnings])

    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path, markdown_path
