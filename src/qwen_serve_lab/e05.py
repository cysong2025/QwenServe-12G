from __future__ import annotations

import csv
import json
import re
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from qwen_serve_lab.config import config_sha256
from qwen_serve_lab.results import ResultError


PROFILE_PATTERN = re.compile(
    r"^e05_(?P<state>bf16|fp8)_(?P<workload>.+)_c(?P<concurrency>\d+)$"
)
KV_TOKENS_PATTERN = re.compile(r"GPU KV cache size:\s*([\d,]+) tokens", re.I)
CONCURRENCY_PATTERN = re.compile(
    r"Maximum concurrency for\s*([\d,]+) tokens per request:\s*([\d.]+)x",
    re.I,
)
EXPECTED_SERVER_PROFILES = {
    "bf16": "e05_kv_bf16",
    "fp8": "e05_kv_fp8",
}
EXPECTED_SERVER_CONFIGS = {
    "bf16": "configs/serve/e05_kv_bf16.toml",
    "fp8": "configs/serve/e05_kv_fp8.toml",
}
EXPECTED_KV_DTYPES = {"bf16": "bfloat16", "fp8": "fp8_e4m3"}


def _required_int(row: dict[str, str], key: str) -> int:
    raw = row.get(key, "").strip()
    try:
        return int(raw)
    except ValueError as exc:
        raise ResultError(f"E05 CSV field {key!r} must be an integer") from exc


def _required_float(row: dict[str, str], key: str) -> float:
    raw = row.get(key, "").strip()
    try:
        return float(raw)
    except ValueError as exc:
        raise ResultError(f"E05 CSV field {key!r} must be numeric") from exc


def _optional_float(row: dict[str, str], key: str) -> float | None:
    raw = row.get(key, "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError as exc:
        raise ResultError(f"E05 CSV field {key!r} must be numeric") from exc


def _required_bool(row: dict[str, str], key: str) -> bool:
    raw = row.get(key, "").strip().lower()
    if raw not in {"true", "false"}:
        raise ResultError(f"E05 CSV field {key!r} must be true or false")
    return raw == "true"


def load_e05_runs(path: str | Path) -> list[dict[str, Any]]:
    csv_path = Path(path)
    try:
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            raw_rows = list(csv.DictReader(handle))
    except OSError as exc:
        raise ResultError(f"Cannot read E05 runs CSV {csv_path}: {exc}") from exc
    if not raw_rows:
        raise ResultError(f"E05 runs CSV is empty: {csv_path}")

    rows: list[dict[str, Any]] = []
    for raw in raw_rows:
        profile = raw.get("profile", "")
        match = PROFILE_PATTERN.fullmatch(profile)
        if match is None:
            raise ResultError(f"Unexpected E05 profile name: {profile!r}")
        row: dict[str, Any] = {
            **raw,
            "state": match.group("state"),
            "workload": match.group("workload"),
            "valid": _required_bool(raw, "valid"),
        }
        for key in (
            "repetition",
            "effective_seed",
            "input_len",
            "output_len",
            "max_concurrency",
            "completed",
            "failed",
        ):
            row[key] = _required_int(raw, key)
        for key in (
            "slo_ttft_ms",
            "slo_tpot_ms",
            "error_rate",
            "request_goodput",
            "output_throughput",
            "p95_ttft_ms",
            "p95_tpot_ms",
        ):
            row[key] = _required_float(raw, key)
        row["peak_memory_used_mib"] = _optional_float(
            raw, "peak_memory_used_mib"
        )
        if row["max_concurrency"] != int(match.group("concurrency")):
            raise ResultError(f"Concurrency mismatch in E05 profile {profile}")
        rows.append(row)
    return rows


def _median(rows: list[dict[str, Any]], key: str) -> float:
    return float(statistics.median(float(row[key]) for row in rows))


def _maximum(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return max(values) if values else None


def _pct_delta(value: float, reference: float) -> float | None:
    if reference == 0:
        return None
    return (value - reference) / reference * 100


def compare_e05_runs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], dict[str, list[dict[str, Any]]]] = {}
    for row in rows:
        key = (row["workload"], row["max_concurrency"])
        grouped.setdefault(key, {}).setdefault(row["state"], []).append(row)

    comparisons: list[dict[str, Any]] = []
    for (workload, concurrency), states in sorted(grouped.items()):
        bf16 = states.get("bf16", [])
        fp8 = states.get("fp8", [])
        all_rows = bf16 + fp8
        if not all_rows:
            continue
        shapes = {
            (row["input_len"], row["output_len"], row["max_concurrency"])
            for row in all_rows
        }
        bf16_by_seed = {row["effective_seed"]: row for row in bf16}
        fp8_by_seed = {row["effective_seed"]: row for row in fp8}
        paired_seeds = (
            len(bf16_by_seed) == len(bf16)
            and len(fp8_by_seed) == len(fp8)
            and set(bf16_by_seed) == set(fp8_by_seed)
        )
        exact_output_pairs = sum(
            bool(
                bf16_by_seed[seed].get("generated_texts_sha256")
                and bf16_by_seed[seed].get("generated_texts_sha256")
                == fp8_by_seed[seed].get("generated_texts_sha256")
            )
            for seed in set(bf16_by_seed) & set(fp8_by_seed)
        )
        evidence_valid = bool(
            len(bf16) == 3
            and len(fp8) == 3
            and {row["repetition"] for row in bf16} == {1, 2, 3}
            and {row["repetition"] for row in fp8} == {1, 2, 3}
            and len(shapes) == 1
            and paired_seeds
            and len({row["benchmark_config_sha256"] for row in bf16}) == 1
            and len({row["benchmark_config_sha256"] for row in fp8}) == 1
            and len({row["server_config_sha256"] for row in bf16}) == 1
            and len({row["server_config_sha256"] for row in fp8}) == 1
            and all(row["completed"] + row["failed"] == 100 for row in all_rows)
            and all(row["peak_memory_used_mib"] is not None for row in all_rows)
            and all(row["valid"] for row in all_rows)
        )
        shape = next(iter(shapes))
        bf16_ttft = _median(bf16, "p95_ttft_ms") if bf16 else 0.0
        fp8_ttft = _median(fp8, "p95_ttft_ms") if fp8 else 0.0
        bf16_tpot = _median(bf16, "p95_tpot_ms") if bf16 else 0.0
        fp8_tpot = _median(fp8, "p95_tpot_ms") if fp8 else 0.0
        bf16_throughput = _median(bf16, "output_throughput") if bf16 else 0.0
        fp8_throughput = _median(fp8, "output_throughput") if fp8 else 0.0
        bf16_goodput = _median(bf16, "request_goodput") if bf16 else 0.0
        fp8_goodput = _median(fp8, "request_goodput") if fp8 else 0.0
        throughput_delta = _pct_delta(fp8_throughput, bf16_throughput)
        ttft_delta = _pct_delta(fp8_ttft, bf16_ttft)
        goodput_delta = _pct_delta(fp8_goodput, bf16_goodput)
        goodput_improved = bool(
            (goodput_delta is not None and goodput_delta >= 10)
            or (bf16_goodput == 0 and fp8_goodput > 0)
        )
        capacity_pressure_workload = workload in {"xlong", "nearmax"}
        performance_benefit = bool(
            evidence_valid
            and capacity_pressure_workload
            and throughput_delta is not None
            and throughput_delta >= -5
            and (
                goodput_improved
                or (ttft_delta is not None and ttft_delta <= -10)
            )
        )
        fp8_slo_pass = evidence_valid and all(
            row["p95_ttft_ms"] <= row["slo_ttft_ms"]
            and row["p95_tpot_ms"] <= row["slo_tpot_ms"]
            for row in fp8
        )
        comparisons.append(
            {
                "workload": workload,
                "input_len": shape[0],
                "output_len": shape[1],
                "max_concurrency": concurrency,
                "runs_bf16": len(bf16),
                "runs_fp8": len(fp8),
                "exact_output_pairs": f"{exact_output_pairs}/{len(bf16_by_seed)}",
                "bf16_output_throughput": bf16_throughput,
                "fp8_output_throughput": fp8_throughput,
                "output_throughput_delta_percent": _pct_delta(
                    fp8_throughput, bf16_throughput
                ),
                "bf16_p95_ttft_ms": bf16_ttft,
                "fp8_p95_ttft_ms": fp8_ttft,
                "p95_ttft_delta_percent": ttft_delta,
                "bf16_p95_tpot_ms": bf16_tpot,
                "fp8_p95_tpot_ms": fp8_tpot,
                "p95_tpot_delta_percent": _pct_delta(fp8_tpot, bf16_tpot),
                "bf16_request_goodput": bf16_goodput,
                "fp8_request_goodput": fp8_goodput,
                "request_goodput_delta_percent": goodput_delta,
                "bf16_peak_vram_mib": _maximum(bf16, "peak_memory_used_mib"),
                "fp8_peak_vram_mib": _maximum(fp8, "peak_memory_used_mib"),
                "evidence": "VALID" if evidence_valid else "INCOMPLETE",
                "fp8_slo": (
                    "PASS" if fp8_slo_pass else "FAIL" if evidence_valid else "UNKNOWN"
                ),
                "performance_signal": (
                    "BENEFIT"
                    if performance_benefit
                    else "NO_BENEFIT"
                    if evidence_valid and capacity_pressure_workload
                    else "CONTROL"
                    if evidence_valid
                    else "UNKNOWN"
                ),
            }
        )
    return comparisons


def _format(value: float | None, digits: int = 2) -> str:
    return "NA" if value is None else f"{value:.{digits}f}"


def _format_delta(value: float | None) -> str:
    return "NA" if value is None else f"{value:+.2f}%"


def write_e05_comparison(
    runs_csv: str | Path, output_dir: str | Path
) -> tuple[Path, Path]:
    comparisons = compare_e05_runs(load_e05_runs(runs_csv))
    if not comparisons:
        raise ResultError("No E05 performance comparisons were produced")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "comparison.csv"
    markdown_path = output / "comparison.md"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparisons[0]))
        writer.writeheader()
        writer.writerows(comparisons)

    lines = [
        "# E05 FP8 KV Cache Performance Comparison",
        "",
        f"Generated at: {datetime.now(timezone.utc).isoformat()}",
        "",
        "BF16 and FP8 use paired seeds and identical workload controls. Quality is evaluated separately and is not inferred from this table.",
        "",
        "| Workload | In/Out | C | BF16 tok/s | FP8 tok/s | Delta | BF16 P95 TTFT | FP8 P95 TTFT | Delta | BF16 P95 TPOT | FP8 P95 TPOT | Delta | BF16 goodput | FP8 goodput | Delta | BF16/FP8 VRAM MiB | Exact output pairs | Evidence | FP8 SLO | Signal |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|",
    ]
    for row in comparisons:
        lines.append(
            "| {workload} | {shape} | {c} | {bt} | {ft} | {td} | {bttft} | {fttft} | {ttftd} | {btpot} | {ftpot} | {tpotd} | {bg} | {fg} | {gd} | {bv}/{fv} | {outputs} | {evidence} | {slo} | {signal} |".format(
                workload=row["workload"],
                shape=f"{row['input_len']}/{row['output_len']}",
                c=row["max_concurrency"],
                bt=_format(row["bf16_output_throughput"]),
                ft=_format(row["fp8_output_throughput"]),
                td=_format_delta(row["output_throughput_delta_percent"]),
                bttft=_format(row["bf16_p95_ttft_ms"]),
                fttft=_format(row["fp8_p95_ttft_ms"]),
                ttftd=_format_delta(row["p95_ttft_delta_percent"]),
                btpot=_format(row["bf16_p95_tpot_ms"]),
                ftpot=_format(row["fp8_p95_tpot_ms"]),
                tpotd=_format_delta(row["p95_tpot_delta_percent"]),
                bg=_format(row["bf16_request_goodput"]),
                fg=_format(row["fp8_request_goodput"]),
                gd=_format_delta(row["request_goodput_delta_percent"]),
                bv=_format(row["bf16_peak_vram_mib"], 0),
                fv=_format(row["fp8_peak_vram_mib"], 0),
                outputs=row["exact_output_pairs"],
                evidence=row["evidence"],
                slo=row["fp8_slo"],
                signal=row["performance_signal"],
            )
        )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path, markdown_path


def parse_kv_capacity_log(text: str) -> dict[str, float | int]:
    token_matches = KV_TOKENS_PATTERN.findall(text)
    concurrency_matches = CONCURRENCY_PATTERN.findall(text)
    if not token_matches or not concurrency_matches:
        raise ResultError(
            "Server log lacks GPU KV cache size or maximum concurrency evidence"
        )
    model_len, concurrency = concurrency_matches[-1]
    capacity = {
        "gpu_kv_cache_tokens": int(token_matches[-1].replace(",", "")),
        "reference_request_tokens": int(model_len.replace(",", "")),
        "maximum_concurrency": float(concurrency),
    }
    if any(value <= 0 for value in capacity.values()):
        raise ResultError("Server log reports non-positive KV cache capacity")
    return capacity


def _resolve_log_path(manifest: dict[str, Any], manifest_path: Path) -> Path:
    raw_log = manifest.get("log")
    if not isinstance(raw_log, str) or not raw_log:
        raise ResultError(f"Server manifest has no log path: {manifest_path}")
    log_path = Path(raw_log)
    if log_path.is_absolute():
        return log_path
    environment = manifest.get("environment")
    if isinstance(environment, dict) and isinstance(
        environment.get("project_root"), str
    ):
        return Path(environment["project_root"]) / log_path
    return manifest_path.parent.parent.parent / log_path


def _latest_capacity(
    manifest_dir: str | Path, state: str
) -> dict[str, Any]:
    expected_profile = EXPECTED_SERVER_PROFILES[state]
    project_root = Path(__file__).resolve().parents[2]
    expected_config_sha256 = config_sha256(
        project_root / EXPECTED_SERVER_CONFIGS[state]
    )
    candidates: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted(Path(manifest_dir).rglob("*.json"), reverse=True):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if (
            isinstance(data, dict)
            and data.get("kind") == "server"
            and data.get("profile") == expected_profile
            and data.get("server_config_sha256") == expected_config_sha256
        ):
            candidates.append((path, data))
    for manifest_path, manifest in candidates:
        log_path = _resolve_log_path(manifest, manifest_path)
        try:
            capacity = parse_kv_capacity_log(log_path.read_text(encoding="utf-8"))
        except (OSError, ResultError):
            continue
        config = manifest.get("effective_config")
        kv_cache_dtype = (
            config.get("kv_cache_dtype") if isinstance(config, dict) else None
        )
        if kv_cache_dtype != EXPECTED_KV_DTYPES[state]:
            continue
        return {
            "state": state,
            "server_profile": expected_profile,
            "manifest": str(manifest_path),
            "log": str(log_path),
            "server_config_sha256": manifest.get("server_config_sha256"),
            "kv_cache_dtype": kv_cache_dtype,
            **capacity,
        }
    raise ResultError(f"No parseable E05 {state} server capacity log found")


def write_e05_capacity_report(
    manifest_dir: str | Path, output_dir: str | Path
) -> tuple[Path, Path]:
    rows = [_latest_capacity(manifest_dir, state) for state in ("bf16", "fp8")]
    bf16, fp8 = rows
    if bf16["reference_request_tokens"] != fp8["reference_request_tokens"]:
        raise ResultError("E05 capacity logs use different reference request lengths")
    token_ratio = fp8["gpu_kv_cache_tokens"] / bf16["gpu_kv_cache_tokens"]
    concurrency_ratio = fp8["maximum_concurrency"] / bf16["maximum_concurrency"]
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "capacity.json"
    markdown_path = output / "capacity.md"
    document = {
        "schema_version": 1,
        "kind": "e05_kv_capacity_comparison",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "rows": rows,
        "fp8_to_bf16_token_capacity_ratio": token_ratio,
        "fp8_to_bf16_maximum_concurrency_ratio": concurrency_ratio,
        "evidence_status": "VALID",
        "capacity_status": "PASS" if token_ratio >= 1.80 else "FAIL",
    }
    json_path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# E05 KV Cache Capacity",
        "",
        f"Generated at: {document['created_at']}",
        "",
        f"Capacity gate: **{document['capacity_status']}** (required ratio >= 1.80x)",
        "Capacity values are parsed from vLLM startup logs under identical memory and scheduler controls.",
        "",
        "| State | KV dtype | GPU KV tokens | Reference request tokens | Maximum concurrency |",
        "|---|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['state']} | {row['kv_cache_dtype']} | {row['gpu_kv_cache_tokens']} | {row['reference_request_tokens']} | {row['maximum_concurrency']:.2f}x |"
        )
    lines.extend(
        [
            "",
            f"FP8/BF16 token capacity ratio: **{token_ratio:.3f}x**",
            f"FP8/BF16 reported concurrency ratio: **{concurrency_ratio:.3f}x**",
        ]
    )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, markdown_path
