from __future__ import annotations

import csv
import re
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from qwen_serve_lab.results import ResultError


PROFILE_PATTERN = re.compile(
    r"^e06_bt(?P<budget>2048|8192)_(?P<apc>off|on)_"
    r"(?P<condition>.+)_c(?P<concurrency>\d+)$"
)
CONTROL = "bt8192_off"
BUDGET_ONLY = "bt2048_off"
APC_ONLY = "bt8192_on"
COMBINED = "bt2048_on"
CELLS = (CONTROL, BUDGET_ONLY, APC_ONLY, COMBINED)
EXPECTED_SERVER_PROFILES = {
    CONTROL: "e06_bt8192_apc_off",
    BUDGET_ONLY: "e06_bt2048_apc_off",
    APC_ONLY: "e06_bt8192_apc_on",
    COMBINED: "e06_bt2048_apc_on",
}
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
OPTIONAL_FLOAT_FIELDS = (
    "peak_memory_used_mib",
    "prefix_cache_query_tokens",
    "prefix_cache_hit_tokens",
    "prefix_cache_hit_rate_percent",
)


def _required_int(row: dict[str, str], key: str) -> int:
    raw = row.get(key, "").strip()
    try:
        return int(raw)
    except ValueError as exc:
        raise ResultError(f"E06 CSV field {key!r} must be an integer") from exc


def _required_float(row: dict[str, str], key: str) -> float:
    raw = row.get(key, "").strip()
    try:
        return float(raw)
    except ValueError as exc:
        raise ResultError(f"E06 CSV field {key!r} must be numeric") from exc


def _optional_float(row: dict[str, str], key: str) -> float | None:
    raw = row.get(key, "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError as exc:
        raise ResultError(f"E06 CSV field {key!r} must be numeric") from exc


def _required_bool(row: dict[str, str], key: str) -> bool:
    raw = row.get(key, "").strip().lower()
    if raw not in {"true", "false"}:
        raise ResultError(f"E06 CSV field {key!r} must be true or false")
    return raw == "true"


def load_e06_runs(path: str | Path) -> list[dict[str, Any]]:
    csv_path = Path(path)
    try:
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            raw_rows = list(csv.DictReader(handle))
    except OSError as exc:
        raise ResultError(f"Cannot read E06 runs CSV {csv_path}: {exc}") from exc
    if not raw_rows:
        raise ResultError(f"E06 runs CSV is empty: {csv_path}")

    rows: list[dict[str, Any]] = []
    for raw in raw_rows:
        profile = raw.get("profile", "")
        match = PROFILE_PATTERN.fullmatch(profile)
        if match is None:
            raise ResultError(f"Unexpected E06 profile name: {profile!r}")
        cell = f"bt{match.group('budget')}_{match.group('apc')}"
        row: dict[str, Any] = {
            **raw,
            "cell": cell,
            "budget": int(match.group("budget")),
            "apc": match.group("apc"),
            "condition": match.group("condition"),
            "valid": _required_bool(raw, "valid"),
            "prefix_cache_enabled": _required_bool(
                raw, "prefix_cache_enabled"
            ),
        }
        for key in INTEGER_FIELDS:
            row[key] = _required_int(raw, key)
        for key in REQUIRED_FLOAT_FIELDS:
            row[key] = _required_float(raw, key)
        for key in OPTIONAL_FLOAT_FIELDS:
            row[key] = _optional_float(raw, key)
        if row["max_concurrency"] != int(match.group("concurrency")):
            raise ResultError(f"Concurrency mismatch in E06 profile {profile}")
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


def _pct_delta(value: float | None, reference: float | None) -> float | None:
    if value is None or reference is None or reference == 0:
        return None
    return (value - reference) / reference * 100


def _difference(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return left - right


def _best_higher(left: float | None, right: float | None) -> float | None:
    values = [value for value in (left, right) if value is not None]
    return max(values) if values else None


def _best_lower(left: float | None, right: float | None) -> float | None:
    values = [value for value in (left, right) if value is not None]
    return min(values) if values else None


def _one_nonempty_value(rows: list[dict[str, Any]], key: str) -> bool:
    values = {row.get(key) for row in rows}
    return bool(
        len(values) == 1
        and all(isinstance(value, str) and value for value in values)
    )


def compare_e06_runs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], dict[str, list[dict[str, Any]]]] = {}
    for row in rows:
        key = (row["condition"], row["max_concurrency"])
        grouped.setdefault(key, {}).setdefault(row["cell"], []).append(row)

    comparisons: list[dict[str, Any]] = []
    for (condition, concurrency), states in sorted(grouped.items()):
        cell_rows = {cell: states.get(cell, []) for cell in CELLS}
        all_rows = [row for cell in CELLS for row in cell_rows[cell]]
        if not all_rows:
            continue
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
        shape = next(iter(shapes))
        rows_by_seed = {
            cell: {row["effective_seed"]: row for row in cell_rows[cell]}
            for cell in CELLS
        }
        seed_sets = [set(rows_by_seed[cell]) for cell in CELLS]
        seeds_paired = bool(
            all(
                len(rows_by_seed[cell]) == len(cell_rows[cell])
                for cell in CELLS
            )
            and len({frozenset(seeds) for seeds in seed_sets}) == 1
        )
        repetitions_complete = all(
            len(cell_rows[cell]) == 3
            and {row["repetition"] for row in cell_rows[cell]} == {1, 2, 3}
            for cell in CELLS
        )
        profiles_correct = all(
            all(
                row.get("server_profile") == EXPECTED_SERVER_PROFILES[cell]
                for row in cell_rows[cell]
            )
            for cell in CELLS
        )
        hashes_consistent = all(
            _one_nonempty_value(cell_rows[cell], "benchmark_config_sha256")
            and _one_nonempty_value(cell_rows[cell], "server_config_sha256")
            for cell in CELLS
        )
        cache_states_correct = all(
            row["prefix_cache_enabled"] == cell.endswith("_on")
            for cell in CELLS
            for row in cell_rows[cell]
        )
        on_metrics_complete = all(
            row["prefix_cache_query_tokens"] is not None
            and row["prefix_cache_query_tokens"] > 0
            and row["prefix_cache_hit_tokens"] is not None
            and row["prefix_cache_hit_tokens"] >= 0
            and row["prefix_cache_hit_rate_percent"] is not None
            and 0 <= row["prefix_cache_hit_rate_percent"] <= 100
            for cell in (APC_ONLY, COMBINED)
            for row in cell_rows[cell]
        )
        request_counts_complete = all(
            row["completed"] + row["failed"] == 100 for row in all_rows
        )
        telemetry_complete = all(
            row["peak_memory_used_mib"] is not None for row in all_rows
        )
        evidence_valid = bool(
            repetitions_complete
            and len(shapes) == 1
            and seeds_paired
            and profiles_correct
            and hashes_consistent
            and cache_states_correct
            and on_metrics_complete
            and request_counts_complete
            and telemetry_complete
            and all(row["valid"] and row["error_rate"] < 0.01 for row in all_rows)
        )

        metrics: dict[str, dict[str, float | None]] = {}
        for cell in CELLS:
            metrics[cell] = {
                "ttft": _median(cell_rows[cell], "p95_ttft_ms"),
                "tpot": _median(cell_rows[cell], "p95_tpot_ms"),
                "throughput": _median(cell_rows[cell], "output_throughput"),
                "goodput": _median(cell_rows[cell], "request_goodput"),
                "vram": _maximum(cell_rows[cell], "peak_memory_used_mib"),
                "hit_rate": _median(
                    cell_rows[cell], "prefix_cache_hit_rate_percent"
                ),
            }

        best_single_ttft = _best_lower(
            metrics[BUDGET_ONLY]["ttft"], metrics[APC_ONLY]["ttft"]
        )
        best_single_tpot = _best_lower(
            metrics[BUDGET_ONLY]["tpot"], metrics[APC_ONLY]["tpot"]
        )
        best_single_throughput = _best_higher(
            metrics[BUDGET_ONLY]["throughput"],
            metrics[APC_ONLY]["throughput"],
        )
        best_single_goodput = _best_higher(
            metrics[BUDGET_ONLY]["goodput"], metrics[APC_ONLY]["goodput"]
        )
        combined_vs_best_ttft = _pct_delta(
            metrics[COMBINED]["ttft"], best_single_ttft
        )
        combined_vs_best_tpot = _pct_delta(
            metrics[COMBINED]["tpot"], best_single_tpot
        )
        combined_vs_best_throughput = _pct_delta(
            metrics[COMBINED]["throughput"], best_single_throughput
        )
        combined_vs_best_goodput = _pct_delta(
            metrics[COMBINED]["goodput"], best_single_goodput
        )
        stacked_benefit = bool(
            evidence_valid
            and combined_vs_best_throughput is not None
            and combined_vs_best_throughput >= -2
            and (
                combined_vs_best_ttft is not None
                and combined_vs_best_ttft <= -5
                or combined_vs_best_goodput is not None
                and combined_vs_best_goodput >= 10
            )
        )
        combined_slo_pass = evidence_valid and all(
            row["p95_ttft_ms"] <= row["slo_ttft_ms"]
            and row["p95_tpot_ms"] <= row["slo_tpot_ms"]
            for row in cell_rows[COMBINED]
        )

        baseline_by_seed = rows_by_seed[CONTROL]
        output_matches: dict[str, int] = {}
        for cell in (BUDGET_ONLY, APC_ONLY, COMBINED):
            output_matches[cell] = sum(
                bool(
                    baseline_by_seed.get(seed, {}).get("generated_texts_sha256")
                    and baseline_by_seed[seed].get("generated_texts_sha256")
                    == rows_by_seed[cell].get(seed, {}).get(
                        "generated_texts_sha256"
                    )
                )
                for seed in set(baseline_by_seed) & set(rows_by_seed[cell])
            )

        apc_effect_8192_ttft = _pct_delta(
            metrics[APC_ONLY]["ttft"], metrics[CONTROL]["ttft"]
        )
        apc_effect_2048_ttft = _pct_delta(
            metrics[COMBINED]["ttft"], metrics[BUDGET_ONLY]["ttft"]
        )
        apc_effect_8192_throughput = _pct_delta(
            metrics[APC_ONLY]["throughput"], metrics[CONTROL]["throughput"]
        )
        apc_effect_2048_throughput = _pct_delta(
            metrics[COMBINED]["throughput"],
            metrics[BUDGET_ONLY]["throughput"],
        )

        result: dict[str, Any] = {
            "condition": condition,
            "max_concurrency": concurrency,
            "input_len": shape[0],
            "output_len": shape[1],
            "prefix_len": shape[2],
            "suffix_len": shape[3],
            "num_prefixes": shape[4],
            "nominal_reuse_percent": shape[5],
            "runs_per_cell": min(len(cell_rows[cell]) for cell in CELLS),
        }
        for cell in CELLS:
            result.update(
                {
                    f"{cell}_p95_ttft_ms": metrics[cell]["ttft"],
                    f"{cell}_p95_tpot_ms": metrics[cell]["tpot"],
                    f"{cell}_output_throughput": metrics[cell]["throughput"],
                    f"{cell}_request_goodput": metrics[cell]["goodput"],
                    f"{cell}_peak_vram_mib": metrics[cell]["vram"],
                }
            )
        result.update(
            {
                "bt8192_on_hit_rate_percent": metrics[APC_ONLY]["hit_rate"],
                "bt2048_on_hit_rate_percent": metrics[COMBINED]["hit_rate"],
                "combined_vs_baseline_ttft_percent": _pct_delta(
                    metrics[COMBINED]["ttft"], metrics[CONTROL]["ttft"]
                ),
                "combined_vs_baseline_throughput_percent": _pct_delta(
                    metrics[COMBINED]["throughput"],
                    metrics[CONTROL]["throughput"],
                ),
                "combined_vs_baseline_goodput_percent": _pct_delta(
                    metrics[COMBINED]["goodput"], metrics[CONTROL]["goodput"]
                ),
                "combined_vs_best_single_ttft_percent": combined_vs_best_ttft,
                "combined_vs_best_single_tpot_percent": combined_vs_best_tpot,
                "combined_vs_best_single_throughput_percent": (
                    combined_vs_best_throughput
                ),
                "combined_vs_best_single_goodput_percent": combined_vs_best_goodput,
                "apc_effect_bt8192_ttft_percent": apc_effect_8192_ttft,
                "apc_effect_bt2048_ttft_percent": apc_effect_2048_ttft,
                "ttft_interaction_percentage_points": _difference(
                    apc_effect_2048_ttft, apc_effect_8192_ttft
                ),
                "apc_effect_bt8192_throughput_percent": (
                    apc_effect_8192_throughput
                ),
                "apc_effect_bt2048_throughput_percent": (
                    apc_effect_2048_throughput
                ),
                "throughput_interaction_percentage_points": _difference(
                    apc_effect_2048_throughput,
                    apc_effect_8192_throughput,
                ),
                "bt2048_off_output_matches_control": (
                    f"{output_matches[BUDGET_ONLY]}/3"
                ),
                "bt8192_on_output_matches_control": f"{output_matches[APC_ONLY]}/3",
                "bt2048_on_output_matches_control": f"{output_matches[COMBINED]}/3",
                "evidence": "VALID" if evidence_valid else "INCOMPLETE",
                "combined_slo": (
                    "PASS"
                    if combined_slo_pass
                    else "FAIL"
                    if evidence_valid
                    else "UNKNOWN"
                ),
                "decision": (
                    "STACKED_BENEFIT"
                    if stacked_benefit
                    else "NO_STACKED_BENEFIT"
                    if evidence_valid
                    else "UNKNOWN"
                ),
            }
        )
        comparisons.append(result)
    return comparisons


def _format(value: float | None, digits: int = 2) -> str:
    return "NA" if value is None else f"{value:.{digits}f}"


def _format_delta(value: float | None) -> str:
    return "NA" if value is None else f"{value:+.2f}%"


def write_e06_comparison(
    runs_csv: str | Path, output_dir: str | Path
) -> tuple[Path, Path]:
    comparisons = compare_e06_runs(load_e06_runs(runs_csv))
    if not comparisons:
        raise ResultError("No E06 factorial comparisons were produced")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "comparison.csv"
    markdown_path = output / "comparison.md"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparisons[0]))
        writer.writeheader()
        writer.writerows(comparisons)

    lines = [
        "# E06 Combined Optimization Factorial Comparison",
        "",
        f"Generated at: {datetime.now(timezone.utc).isoformat()}",
        "",
        "Cells are A=8192/OFF, B=2048/OFF, C=8192/ON, and D=2048/ON. "
        "STACKED_BENEFIT requires valid four-cell evidence, D throughput no more "
        "than 2% below the better single treatment, and either at least 5% lower "
        "P95 TTFT or at least 10% higher goodput than the better single treatment.",
        "Random-token output hashes are diagnostic only; fixed canary "
        "equivalence is reported separately.",
        "",
        "| Condition | C | Reuse | APC hit C/D | P95 TTFT A/B/C/D ms | "
        "D vs best | Output tok/s A/B/C/D | D vs best | Goodput D vs best | "
        "Output matches B/C/D vs A | Evidence | D SLO | Decision |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|",
    ]
    for row in comparisons:
        lines.append(
            "| {condition} | {c} | {reuse}% | {hit_c}/{hit_d} | "
            "{a_ttft}/{b_ttft}/{c_ttft}/{d_ttft} | {ttft_delta} | "
            "{a_tp}/{b_tp}/{c_tp}/{d_tp} | {tp_delta} | {goodput_delta} | "
            "{match_b}/{match_c}/{match_d} | {evidence} | {slo} | {decision} |".format(
                condition=row["condition"],
                c=row["max_concurrency"],
                reuse=_format(row["nominal_reuse_percent"], 0),
                hit_c=_format(row["bt8192_on_hit_rate_percent"]),
                hit_d=_format(row["bt2048_on_hit_rate_percent"]),
                a_ttft=_format(row["bt8192_off_p95_ttft_ms"]),
                b_ttft=_format(row["bt2048_off_p95_ttft_ms"]),
                c_ttft=_format(row["bt8192_on_p95_ttft_ms"]),
                d_ttft=_format(row["bt2048_on_p95_ttft_ms"]),
                ttft_delta=_format_delta(
                    row["combined_vs_best_single_ttft_percent"]
                ),
                a_tp=_format(row["bt8192_off_output_throughput"]),
                b_tp=_format(row["bt2048_off_output_throughput"]),
                c_tp=_format(row["bt8192_on_output_throughput"]),
                d_tp=_format(row["bt2048_on_output_throughput"]),
                tp_delta=_format_delta(
                    row["combined_vs_best_single_throughput_percent"]
                ),
                goodput_delta=_format_delta(
                    row["combined_vs_best_single_goodput_percent"]
                ),
                match_b=row["bt2048_off_output_matches_control"],
                match_c=row["bt8192_on_output_matches_control"],
                match_d=row["bt2048_on_output_matches_control"],
                evidence=row["evidence"],
                slo=row["combined_slo"],
                decision=row["decision"],
            )
        )

    lines.extend(
        [
            "",
            "## Factorial interaction",
            "",
            "Interaction is the APC percentage effect at budget 2048 minus the "
            "APC percentage effect at budget 8192. Negative TTFT interaction "
            "and positive throughput interaction indicate that APC becomes "
            "more effective with the smaller scheduler budget.",
            "",
            "| Condition | APC TTFT effect 8192 | APC TTFT effect 2048 | "
            "TTFT interaction | APC throughput effect 8192 | APC throughput "
            "effect 2048 | Throughput interaction |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in comparisons:
        lines.append(
            "| {condition} | {ttft_8192} | {ttft_2048} | {ttft_i} pp | "
            "{tp_8192} | {tp_2048} | {tp_i} pp |".format(
                condition=row["condition"],
                ttft_8192=_format_delta(row["apc_effect_bt8192_ttft_percent"]),
                ttft_2048=_format_delta(row["apc_effect_bt2048_ttft_percent"]),
                ttft_i=_format(row["ttft_interaction_percentage_points"]),
                tp_8192=_format_delta(
                    row["apc_effect_bt8192_throughput_percent"]
                ),
                tp_2048=_format_delta(
                    row["apc_effect_bt2048_throughput_percent"]
                ),
                tp_i=_format(row["throughput_interaction_percentage_points"]),
            )
        )

    valid = [row for row in comparisons if row["evidence"] == "VALID"]
    benefits = [row for row in valid if row["decision"] == "STACKED_BENEFIT"]
    capacity = next(
        (row for row in comparisons if row["condition"].startswith("capacity_")),
        None,
    )
    no_reuse = next(
        (row for row in comparisons if row["condition"].startswith("reuse0_")),
        None,
    )
    lines.extend(["", "## Data-dependent conclusion", ""])
    if benefits:
        names = ", ".join(
            f"{row['condition']}/c{row['max_concurrency']}" for row in benefits
        )
        lines.append(f"Validated stacked benefit is present at: {names}.")
    elif valid:
        lines.append(
            "No validated condition beats the better single treatment under "
            "the frozen stacked-benefit rule."
        )
    else:
        lines.append("The four-cell factorial evidence is not yet complete.")
    if no_reuse is not None and no_reuse["evidence"] == "VALID":
        no_reuse_ttft_delta = _pct_delta(
            no_reuse["bt2048_on_p95_ttft_ms"],
            no_reuse["bt2048_off_p95_ttft_ms"],
        )
        no_reuse_throughput_delta = _pct_delta(
            no_reuse["bt2048_on_output_throughput"],
            no_reuse["bt2048_off_output_throughput"],
        )
        lines.append(
            "For the no-reuse control, combined D versus budget-only B "
            "changes P95 TTFT by "
            f"{_format_delta(no_reuse_ttft_delta)} "
            "and output throughput by "
            f"{_format_delta(no_reuse_throughput_delta)}."
        )
    else:
        lines.append("The no-reuse overhead control is not yet complete.")
    if capacity is not None and capacity["evidence"] == "VALID":
        lines.append(
            "The C8/P1792 capacity condition is "
            f"{capacity['decision']} with combined SLO {capacity['combined_slo']}."
        )
    else:
        lines.append("The C8/P1792 capacity condition is not yet complete.")

    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path, markdown_path
