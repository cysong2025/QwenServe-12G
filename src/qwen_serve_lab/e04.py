from __future__ import annotations

import csv
import re
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from qwen_serve_lab.results import ResultError


PROFILE_PATTERN = re.compile(
    r"^e04_(?P<state>off|on)_(?P<condition>.+)_c(?P<concurrency>\d+)$"
)
INTEGER_FIELDS = (
    "repetition",
    "effective_seed",
    "input_len",
    "output_len",
    "max_concurrency",
    "prefix_len",
    "suffix_len",
    "num_prefixes",
    "completed",
    "failed",
)
FLOAT_FIELDS = (
    "nominal_reuse_percent",
    "slo_ttft_ms",
    "slo_tpot_ms",
    "error_rate",
    "request_goodput",
    "output_throughput",
    "p95_ttft_ms",
    "p95_tpot_ms",
    "peak_memory_used_mib",
    "prefix_cache_query_tokens",
    "prefix_cache_hit_tokens",
    "prefix_cache_hit_rate_percent",
)
REQUIRED_FLOAT_FIELDS = (
    "nominal_reuse_percent",
    "slo_ttft_ms",
    "slo_tpot_ms",
    "error_rate",
    "request_goodput",
    "output_throughput",
    "p95_ttft_ms",
    "p95_tpot_ms",
)


def _required_int(row: dict[str, str], key: str) -> int:
    raw = row.get(key, "").strip()
    if not raw:
        raise ResultError(f"E04 CSV field {key!r} is required")
    try:
        return int(raw)
    except ValueError as exc:
        raise ResultError(f"E04 CSV field {key!r} must be an integer") from exc


def _optional_float(row: dict[str, str], key: str) -> float | None:
    raw = row.get(key, "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError as exc:
        raise ResultError(f"E04 CSV field {key!r} must be numeric") from exc


def _required_bool(row: dict[str, str], key: str) -> bool:
    raw = row.get(key, "").strip().lower()
    if raw not in {"true", "false"}:
        raise ResultError(f"E04 CSV field {key!r} must be true or false")
    return raw == "true"


def load_e04_runs(path: str | Path) -> list[dict[str, Any]]:
    csv_path = Path(path)
    try:
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            raw_rows = list(csv.DictReader(handle))
    except OSError as exc:
        raise ResultError(f"Cannot read E04 runs CSV {csv_path}: {exc}") from exc
    if not raw_rows:
        raise ResultError(f"E04 runs CSV is empty: {csv_path}")

    rows: list[dict[str, Any]] = []
    for raw in raw_rows:
        profile = raw.get("profile", "")
        match = PROFILE_PATTERN.fullmatch(profile)
        if match is None:
            raise ResultError(f"Unexpected E04 profile name: {profile!r}")
        row: dict[str, Any] = {
            **raw,
            "state": match.group("state"),
            "condition": match.group("condition"),
            "valid": _required_bool(raw, "valid"),
            "prefix_cache_enabled": _required_bool(
                raw, "prefix_cache_enabled"
            ),
        }
        for key in INTEGER_FIELDS:
            row[key] = _required_int(raw, key)
        for key in FLOAT_FIELDS:
            row[key] = _optional_float(raw, key)
        missing = [key for key in REQUIRED_FLOAT_FIELDS if row[key] is None]
        if missing:
            raise ResultError(
                f"E04 CSV row {profile} lacks required metrics: "
                + ", ".join(missing)
            )
        if row["max_concurrency"] != int(match.group("concurrency")):
            raise ResultError(f"Concurrency mismatch in E04 profile {profile}")
        rows.append(row)
    return rows


def _values(rows: list[dict[str, Any]], key: str) -> list[float]:
    return [float(row[key]) for row in rows if row.get(key) is not None]


def _median(rows: list[dict[str, Any]], key: str) -> float | None:
    values = _values(rows, key)
    return float(statistics.median(values)) if values else None


def _maximum(rows: list[dict[str, Any]], key: str) -> float | None:
    values = _values(rows, key)
    return max(values) if values else None


def _range(rows: list[dict[str, Any]], key: str) -> tuple[float, float] | None:
    values = _values(rows, key)
    return (min(values), max(values)) if values else None


def _pct_delta(value: float | None, reference: float | None) -> float | None:
    if value is None or reference is None or reference == 0:
        return None
    return (value - reference) / reference * 100


def compare_e04_runs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], dict[str, list[dict[str, Any]]]] = {}
    for row in rows:
        key = (row["condition"], row["max_concurrency"])
        grouped.setdefault(key, {}).setdefault(row["state"], []).append(row)

    comparisons: list[dict[str, Any]] = []
    for (condition, concurrency), states in sorted(grouped.items()):
        off = states.get("off", [])
        on = states.get("on", [])
        all_rows = off + on
        shapes = {
            (
                row["input_len"],
                row["output_len"],
                row["prefix_len"],
                row["suffix_len"],
                row["num_prefixes"],
                row["nominal_reuse_percent"],
            )
            for row in all_rows
        }
        shape_consistent = len(shapes) == 1
        if not shapes:
            continue
        shape = next(iter(shapes))

        off_by_seed = {row["effective_seed"]: row for row in off}
        on_by_seed = {row["effective_seed"]: row for row in on}
        seeds_match = (
            len(off_by_seed) == len(off)
            and len(on_by_seed) == len(on)
            and set(off_by_seed) == set(on_by_seed)
        )
        output_match = seeds_match and all(
            off_by_seed[seed].get("generated_texts_sha256")
            and off_by_seed[seed].get("generated_texts_sha256")
            == on_by_seed[seed].get("generated_texts_sha256")
            for seed in off_by_seed
        )
        cache_states_correct = all(
            not row["prefix_cache_enabled"] for row in off
        ) and all(row["prefix_cache_enabled"] for row in on)
        on_metrics_complete = all(
            row["prefix_cache_query_tokens"] is not None
            and row["prefix_cache_query_tokens"] > 0
            and row["prefix_cache_hit_tokens"] is not None
            and row["prefix_cache_hit_tokens"] >= 0
            and row["prefix_cache_hit_rate_percent"] is not None
            and 0 <= row["prefix_cache_hit_rate_percent"] <= 100
            for row in on
        )
        telemetry_complete = len(
            _values(all_rows, "peak_memory_used_mib")
        ) == len(all_rows)
        request_count_complete = all(
            row["completed"] + row["failed"] == 100 for row in all_rows
        )
        evidence_valid = (
            len(off) == 3
            and len(on) == 3
            and shape_consistent
            and seeds_match
            and output_match
            and cache_states_correct
            and on_metrics_complete
            and telemetry_complete
            and request_count_complete
            and all(row["valid"] for row in all_rows)
        )

        off_ttft = _median(off, "p95_ttft_ms")
        on_ttft = _median(on, "p95_ttft_ms")
        off_tpot = _median(off, "p95_tpot_ms")
        on_tpot = _median(on, "p95_tpot_ms")
        off_throughput = _median(off, "output_throughput")
        on_throughput = _median(on, "output_throughput")
        off_goodput = _median(off, "request_goodput")
        on_goodput = _median(on, "request_goodput")
        ttft_delta = _pct_delta(on_ttft, off_ttft)
        throughput_delta = _pct_delta(on_throughput, off_throughput)
        benefit = (
            evidence_valid
            and ttft_delta is not None
            and ttft_delta <= -5
            and throughput_delta is not None
            and throughput_delta >= -2
        )
        on_slo_pass = evidence_valid and all(
            row["p95_ttft_ms"] is not None
            and row["p95_tpot_ms"] is not None
            and row["slo_ttft_ms"] is not None
            and row["slo_tpot_ms"] is not None
            and row["p95_ttft_ms"] <= row["slo_ttft_ms"]
            and row["p95_tpot_ms"] <= row["slo_tpot_ms"]
            for row in on
        )
        hit_range = _range(on, "prefix_cache_hit_rate_percent")
        comparisons.append(
            {
                "condition": condition,
                "max_concurrency": concurrency,
                "input_len": shape[0],
                "output_len": shape[1],
                "prefix_len": shape[2],
                "suffix_len": shape[3],
                "num_prefixes": shape[4],
                "nominal_reuse_percent": shape[5],
                "runs_off": len(off),
                "runs_on": len(on),
                "actual_hit_rate_percent": _median(
                    on, "prefix_cache_hit_rate_percent"
                ),
                "actual_hit_rate_min_percent": (
                    hit_range[0] if hit_range is not None else None
                ),
                "actual_hit_rate_max_percent": (
                    hit_range[1] if hit_range is not None else None
                ),
                "off_p95_ttft_ms": off_ttft,
                "on_p95_ttft_ms": on_ttft,
                "p95_ttft_delta_percent": ttft_delta,
                "off_p95_tpot_ms": off_tpot,
                "on_p95_tpot_ms": on_tpot,
                "p95_tpot_delta_percent": _pct_delta(on_tpot, off_tpot),
                "off_output_throughput": off_throughput,
                "on_output_throughput": on_throughput,
                "output_throughput_delta_percent": throughput_delta,
                "off_request_goodput": off_goodput,
                "on_request_goodput": on_goodput,
                "request_goodput_delta_percent": _pct_delta(
                    on_goodput, off_goodput
                ),
                "off_peak_vram_mib": _maximum(off, "peak_memory_used_mib"),
                "on_peak_vram_mib": _maximum(on, "peak_memory_used_mib"),
                "output_match": output_match,
                "evidence": "VALID" if evidence_valid else "INCOMPLETE",
                "on_slo": "PASS" if on_slo_pass else (
                    "FAIL" if evidence_valid else "UNKNOWN"
                ),
                "decision": "BENEFIT" if benefit else (
                    "NO_BENEFIT" if evidence_valid else "UNKNOWN"
                ),
            }
        )
    return comparisons


def _format(value: float | None, digits: int = 2) -> str:
    return "NA" if value is None else f"{value:.{digits}f}"


def _format_delta(value: float | None) -> str:
    return "NA" if value is None else f"{value:+.2f}%"


def write_e04_comparison(
    runs_csv: str | Path, output_dir: str | Path
) -> tuple[Path, Path]:
    comparisons = compare_e04_runs(load_e04_runs(runs_csv))
    if not comparisons:
        raise ResultError("No E04 OFF/ON comparisons could be built")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "comparison.csv"
    markdown_path = output / "comparison.md"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparisons[0]))
        writer.writeheader()
        writer.writerows(comparisons)

    lines = [
        "# E04 Automatic Prefix Caching Comparison",
        "",
        f"Generated at: {datetime.now(timezone.utc).isoformat()}",
        "",
        "OFF and ON use paired seeds. BENEFIT requires valid evidence, at least 5% lower P95 TTFT, no more than 2% output-throughput regression, and identical generated outputs.",
        "",
        "| Condition | C | Prefix/Suffix | Nominal reuse | Actual token hit rate | OFF/ON P95 TTFT ms | TTFT delta | OFF/ON output tok/s | Throughput delta | Output | Evidence | SLO | Decision |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|",
    ]
    for row in comparisons:
        hit_rate = _format(row["actual_hit_rate_percent"])
        if row["actual_hit_rate_min_percent"] is not None:
            hit_rate += (
                f" [{row['actual_hit_rate_min_percent']:.2f}, "
                f"{row['actual_hit_rate_max_percent']:.2f}]"
            )
        lines.append(
            "| {condition} | {concurrency} | {prefix}/{suffix} | {reuse}% | "
            "{hit_rate} | {off_ttft}/{on_ttft} | {ttft_delta} | "
            "{off_tp}/{on_tp} | {tp_delta} | {output} | {evidence} | "
            "{slo} | {decision} |".format(
                condition=row["condition"],
                concurrency=row["max_concurrency"],
                prefix=row["prefix_len"],
                suffix=row["suffix_len"],
                reuse=_format(row["nominal_reuse_percent"], 0),
                hit_rate=hit_rate,
                off_ttft=_format(row["off_p95_ttft_ms"]),
                on_ttft=_format(row["on_p95_ttft_ms"]),
                ttft_delta=_format_delta(row["p95_ttft_delta_percent"]),
                off_tp=_format(row["off_output_throughput"]),
                on_tp=_format(row["on_output_throughput"]),
                tp_delta=_format_delta(row["output_throughput_delta_percent"]),
                output="MATCH" if row["output_match"] else "MISMATCH",
                evidence=row["evidence"],
                slo=row["on_slo"],
                decision=row["decision"],
            )
        )

    valid_c4_p1024 = sorted(
        (
            row
            for row in comparisons
            if row["max_concurrency"] == 4
            and row["prefix_len"] == 1024
            and row["condition"].startswith("reuse")
            and row["evidence"] == "VALID"
        ),
        key=lambda row: row["nominal_reuse_percent"],
    )
    threshold = next(
        (row for row in valid_c4_p1024 if row["decision"] == "BENEFIT"), None
    )
    lines.extend(["", "## Data-dependent conclusion", ""])
    if threshold is None:
        lines.append(
            "No validated C4/P1024 reuse threshold currently satisfies the predefined benefit rule."
        )
    else:
        lines.append(
            "The first validated C4/P1024 benefit point is nominal reuse "
            f"{threshold['nominal_reuse_percent']:.0f}% with actual token hit rate "
            f"{threshold['actual_hit_rate_percent']:.2f}%."
        )
    valid_length_sweep = [
        row
        for row in comparisons
        if row["max_concurrency"] == 4
        and row["nominal_reuse_percent"] == 90
        and row["condition"].startswith("reuse")
        and row["evidence"] == "VALID"
        and row["p95_ttft_delta_percent"] is not None
    ]
    if valid_length_sweep:
        best_length = min(
            valid_length_sweep,
            key=lambda row: row["p95_ttft_delta_percent"],
        )
        lines.append(
            "At C4 and 90% nominal reuse, the largest P95 TTFT reduction occurs "
            f"at prefix length {best_length['prefix_len']} tokens "
            f"({_format_delta(best_length['p95_ttft_delta_percent'])})."
        )
    else:
        lines.append("The C4/90% prefix-length sweep is not yet complete.")

    capacity = next(
        (
            row
            for row in comparisons
            if row["condition"].startswith("capacity_")
        ),
        None,
    )
    if capacity is not None and capacity["evidence"] == "VALID":
        lines.append(
            "The C8 capacity validation is "
            f"{capacity['decision']} with P95 TTFT delta "
            f"{_format_delta(capacity['p95_ttft_delta_percent'])}, output-throughput "
            f"delta {_format_delta(capacity['output_throughput_delta_percent'])}, "
            f"and SLO {capacity['on_slo']}."
        )
    else:
        lines.append("The C8 capacity validation is not yet complete.")
    lines.extend([
        "",
        "Actual hit rate is calculated from the per-run delta of vLLM prefix-cache hit/query token counters. Peak VRAM is the maximum sampled value across three repetitions.",
    ])
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path, markdown_path
