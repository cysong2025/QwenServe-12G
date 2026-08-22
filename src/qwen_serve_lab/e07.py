from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

from qwen_serve_lab.results import ResultError


PROFILE_PATTERN = re.compile(r"^e07_(?P<state>base|lora)_(?P<workload>short|medium)_c(?P<c>1|4|8)$")
EXPECTED_WORKLOADS = {("short", 1), ("short", 4), ("short", 8), ("medium", 1), ("medium", 4), ("medium", 8)}


def _float(row: dict[str, str], key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ResultError(f"E07 runs CSV has invalid {key}") from exc


def _int(row: dict[str, str], key: str) -> int:
    try:
        return int(row[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ResultError(f"E07 runs CSV has invalid {key}") from exc


def _valid(row: dict[str, str]) -> bool:
    return row.get("valid", "").lower() == "true"


def _median(rows: list[dict[str, str]], key: str) -> float:
    return median(_float(row, key) for row in rows)


def _maximum(rows: list[dict[str, str]], key: str) -> float:
    return max(_float(row, key) for row in rows)


def _delta(treatment: float, control: float) -> float | None:
    if control == 0:
        return None
    return (treatment - control) / control * 100


def load_e07_runs(path: str | Path) -> list[dict[str, str]]:
    try:
        with Path(path).open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except OSError as exc:
        raise ResultError(f"Cannot read E07 runs CSV: {exc}") from exc
    selected = [row for row in rows if PROFILE_PATTERN.match(row.get("profile", ""))]
    if not selected:
        raise ResultError("No E07 Base/LoRA performance rows found")
    return selected


def compare_e07_runs(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], dict[str, list[dict[str, str]]]] = defaultdict(
        lambda: {"base": [], "lora": []}
    )
    for row in rows:
        match = PROFILE_PATTERN.match(row.get("profile", ""))
        if match is None:
            continue
        grouped[(match.group("workload"), int(match.group("c")))][
            match.group("state")
        ].append(row)
    if set(grouped) != EXPECTED_WORKLOADS:
        missing = sorted(EXPECTED_WORKLOADS - set(grouped))
        raise ResultError(f"E07 performance matrix is incomplete; missing {missing}")

    comparisons: list[dict[str, Any]] = []
    for workload, concurrency in sorted(grouped):
        states = grouped[(workload, concurrency)]
        base = states["base"]
        lora = states["lora"]
        base_seeds = {_int(row, "effective_seed") for row in base}
        lora_seeds = {_int(row, "effective_seed") for row in lora}
        input_valid = bool(
            len(base) == 3
            and len(lora) == 3
            and all(_valid(row) for row in base + lora)
            and all(_int(row, "completed") == 100 for row in base + lora)
            and all(_int(row, "failed") == 0 for row in base + lora)
            and all(_float(row, "error_rate") < 0.01 for row in base + lora)
            and base_seeds == lora_seeds
            and len(base_seeds) == 3
            and len({row.get("benchmark_config_sha256") for row in base}) == 1
            and len({row.get("benchmark_config_sha256") for row in lora}) == 1
            and len({row.get("server_config_sha256") for row in base}) == 1
            and len({row.get("server_config_sha256") for row in lora}) == 1
            and {
                (_int(row, "input_len"), _int(row, "output_len"))
                for row in base + lora
            }
            == ({(128, 128)} if workload == "short" else {(512, 256)})
            and all(
                _int(row, "max_concurrency") == concurrency
                for row in base + lora
            )
        )
        base_throughput = _median(base, "output_throughput") if base else 0.0
        lora_throughput = _median(lora, "output_throughput") if lora else 0.0
        base_ttft = _median(base, "p95_ttft_ms") if base else 0.0
        lora_ttft = _median(lora, "p95_ttft_ms") if lora else 0.0
        base_tpot = _median(base, "p95_tpot_ms") if base else 0.0
        lora_tpot = _median(lora, "p95_tpot_ms") if lora else 0.0
        throughput_delta = _delta(lora_throughput, base_throughput)
        ttft_delta = _delta(lora_ttft, base_ttft)
        tpot_delta = _delta(lora_tpot, base_tpot)
        lora_slo = bool(
            input_valid
            and all(
                _float(row, "p95_ttft_ms") <= _float(row, "slo_ttft_ms")
                and _float(row, "p95_tpot_ms") <= _float(row, "slo_tpot_ms")
                for row in lora
            )
        )
        acceptable_cost = bool(
            input_valid
            and lora_slo
            and throughput_delta is not None
            and throughput_delta >= -20
            and ttft_delta is not None
            and ttft_delta <= 25
            and tpot_delta is not None
            and tpot_delta <= 20
        )
        comparisons.append(
            {
                "workload": workload,
                "input_len": _int(base[0], "input_len") if base else None,
                "output_len": _int(base[0], "output_len") if base else None,
                "max_concurrency": concurrency,
                "base_runs": len(base),
                "lora_runs": len(lora),
                "paired_seeds": len(base_seeds & lora_seeds),
                "base_output_throughput": base_throughput,
                "lora_output_throughput": lora_throughput,
                "output_throughput_delta_percent": throughput_delta,
                "base_p95_ttft_ms": base_ttft,
                "lora_p95_ttft_ms": lora_ttft,
                "p95_ttft_delta_percent": ttft_delta,
                "base_p95_tpot_ms": base_tpot,
                "lora_p95_tpot_ms": lora_tpot,
                "p95_tpot_delta_percent": tpot_delta,
                "base_goodput": _median(base, "request_goodput") if base else 0.0,
                "lora_goodput": _median(lora, "request_goodput") if lora else 0.0,
                "base_peak_vram_mib": _maximum(base, "peak_memory_used_mib") if base else 0.0,
                "lora_peak_vram_mib": _maximum(lora, "peak_memory_used_mib") if lora else 0.0,
                "evidence": "VALID" if input_valid else "INCOMPLETE",
                "lora_slo": "PASS" if lora_slo else "FAIL" if input_valid else "UNKNOWN",
                "online_cost": "PASS" if acceptable_cost else "FAIL" if input_valid else "UNKNOWN",
            }
        )
    return comparisons


def _fmt(value: float | None, digits: int = 2) -> str:
    return "NA" if value is None else f"{value:.{digits}f}"


def _pct(value: float | None) -> str:
    return "NA" if value is None else f"{value:+.2f}%"


def write_e07_comparison(
    runs_csv: str | Path,
    output_dir: str | Path,
) -> tuple[Path, Path, bool]:
    comparisons = compare_e07_runs(load_e07_runs(runs_csv))
    passed = all(row["online_cost"] == "PASS" for row in comparisons)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "comparison.csv"
    markdown_path = output / "comparison.md"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparisons[0]))
        writer.writeheader()
        writer.writerows(comparisons)
    lines = [
        "# E07 Base vs LoRA Online Cost",
        "",
        f"Generated at: {datetime.now(timezone.utc).isoformat()}",
        "",
        f"Overall online-cost gate: **{'PASS' if passed else 'FAIL'}**",
        "",
        "Each cell requires three valid paired repetitions. The frozen cost gate requires LoRA SLO PASS, output throughput loss <= 20%, P95 TTFT increase <= 25%, and P95 TPOT increase <= 20%.",
        "",
        "| Workload | In/Out | C | Base/LoRA tok/s | Delta | Base/LoRA P95 TTFT | Delta | Base/LoRA P95 TPOT | Delta | Base/LoRA goodput | Base/LoRA VRAM MiB | Evidence | LoRA SLO | Cost |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|",
    ]
    for row in comparisons:
        lines.append(
            "| {workload} | {shape} | {c} | {bt}/{lt} | {td} | {bttft}/{lttft} | {ttftd} | {btpot}/{ltpot} | {tpotd} | {bg}/{lg} | {bv}/{lv} | {evidence} | {slo} | {cost} |".format(
                workload=row["workload"],
                shape=f"{row['input_len']}/{row['output_len']}",
                c=row["max_concurrency"],
                bt=_fmt(row["base_output_throughput"]),
                lt=_fmt(row["lora_output_throughput"]),
                td=_pct(row["output_throughput_delta_percent"]),
                bttft=_fmt(row["base_p95_ttft_ms"]),
                lttft=_fmt(row["lora_p95_ttft_ms"]),
                ttftd=_pct(row["p95_ttft_delta_percent"]),
                btpot=_fmt(row["base_p95_tpot_ms"]),
                ltpot=_fmt(row["lora_p95_tpot_ms"]),
                tpotd=_pct(row["p95_tpot_delta_percent"]),
                bg=_fmt(row["base_goodput"]),
                lg=_fmt(row["lora_goodput"]),
                bv=_fmt(row["base_peak_vram_mib"], 0),
                lv=_fmt(row["lora_peak_vram_mib"], 0),
                evidence=row["evidence"],
                slo=row["lora_slo"],
                cost=row["online_cost"],
            )
        )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path, markdown_path, passed


def write_e07_final_report(output_dir: str | Path) -> tuple[Path, Path, bool]:
    output = Path(output_dir)
    required = {
        "adapter": output / "adapter.json",
        "quality": output / "quality.json",
        "human": output / "human_review_summary.json",
        "performance": output / "comparison.csv",
    }
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        raise ResultError("E07 final report is missing: " + ", ".join(missing))
    try:
        adapter = json.loads(required["adapter"].read_text(encoding="utf-8"))
        quality = json.loads(required["quality"].read_text(encoding="utf-8"))
        human = json.loads(required["human"].read_text(encoding="utf-8"))
        with required["performance"].open("r", encoding="utf-8", newline="") as handle:
            performance = list(csv.DictReader(handle))
    except (OSError, json.JSONDecodeError) as exc:
        raise ResultError(f"Cannot read E07 final evidence: {exc}") from exc
    gates = {
        "adapter": adapter.get("passed") is True,
        "automated_quality": quality.get("automated_status") == "PASS",
        "human_quality": human.get("status") == "PASS",
        "online_cost": bool(performance) and all(row.get("online_cost") == "PASS" for row in performance),
    }
    passed = all(gates.values())
    document = {
        "schema_version": 1,
        "kind": "e07_final_report",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if passed else "FAIL",
        "gates": gates,
        "adapter": adapter,
        "quality": quality,
        "human_review": human,
        "performance_cells": len(performance),
    }
    json_path = output / "final.json"
    markdown_path = output / "final.md"
    json_path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(
        "\n".join(
            [
                "# E07 Final Report",
                "",
                f"Generated at: {document['created_at']}",
                "",
                f"Overall status: **{document['status']}**",
                "",
                "| Gate | Status |",
                "|---|---|",
                *[
                    f"| {name} | {'PASS' if value else 'FAIL'} |"
                    for name, value in gates.items()
                ],
                "",
                "A PASS requires a validated Adapter, automated and blinded-human quality gains, and acceptable online cost in every frozen performance cell.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return json_path, markdown_path, passed
