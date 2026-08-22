from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from qwen_serve_lab.e04_canary import (
    load_canary_dataset,
    request_canary_completion,
)
from qwen_serve_lab.config import config_sha256
from qwen_serve_lab.prometheus import (
    fetch_metrics,
    prefix_delta,
    prefix_snapshot,
)
from qwen_serve_lab.results import ResultError


CONTROL = "bt8192_off"
STATES = (CONTROL, "bt2048_off", "bt8192_on", "bt2048_on")
EXPECTED_SERVER_PROFILES = {
    CONTROL: "e06_bt8192_apc_off",
    "bt2048_off": "e06_bt2048_apc_off",
    "bt8192_on": "e06_bt8192_apc_on",
    "bt2048_on": "e06_bt2048_apc_on",
}
EXPECTED_SERVER_CONFIGS = {
    "bt8192_off": "configs/serve/e06_bt8192_apc_off.toml",
    "bt2048_off": "configs/serve/e06_bt2048_apc_off.toml",
    "bt8192_on": "configs/serve/e06_bt8192_apc_on.toml",
    "bt2048_on": "configs/serve/e06_bt2048_apc_on.toml",
}


def _load_active_server(path: str | Path, state: str) -> dict[str, Any]:
    marker_path = Path(path)
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ResultError(
            f"Cannot read controlled server marker {marker_path}: {exc}"
        ) from exc
    expected_profile = EXPECTED_SERVER_PROFILES[state]
    if not isinstance(marker, dict) or marker.get("profile") != expected_profile:
        actual = marker.get("profile") if isinstance(marker, dict) else None
        raise ResultError(
            f"E06 canary {state} requires active server {expected_profile}; "
            f"found {actual}"
        )
    project_root = Path(__file__).resolve().parents[2]
    expected_sha256 = config_sha256(project_root / EXPECTED_SERVER_CONFIGS[state])
    if marker.get("server_config_sha256") != expected_sha256:
        raise ResultError(
            f"E06 canary {state} requires server config SHA-256 "
            f"{expected_sha256}; found {marker.get('server_config_sha256')}"
        )
    pid = marker.get("pid")
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
        raise ResultError("Controlled server marker contains an invalid pid")
    try:
        os.kill(pid, 0)
    except ProcessLookupError as exc:
        raise ResultError(
            f"Controlled server marker is stale; process {pid} is absent"
        ) from exc
    except PermissionError:
        pass
    return marker


def run_e06_canary(
    state: str,
    dataset_path: str | Path,
    result_root: str | Path,
    base_url: str = "http://127.0.0.1:8000",
    served_model_name: str = "qwen2.5-3b-instruct",
    active_server_path: str | Path = "artifacts/server/active.json",
    timeout_seconds: float = 120,
    request_completion: Callable[
        [str, str, str, float], dict[str, Any]
    ] = request_canary_completion,
    metrics_fetcher: Callable[[str], str] = fetch_metrics,
) -> tuple[Path, bool]:
    if state not in EXPECTED_SERVER_PROFILES:
        raise ResultError(f"E06 canary state must be one of: {', '.join(STATES)}")
    marker = _load_active_server(active_server_path, state)
    dataset_sha256, cases = load_canary_dataset(dataset_path, experiment="E06")
    metrics_before = prefix_snapshot(metrics_fetcher(base_url))
    results: list[dict[str, Any]] = []

    for case in cases:
        try:
            response = request_completion(
                base_url,
                served_model_name,
                case["prompt"],
                timeout_seconds,
            )
            choice = response["choices"][0]
            generated = choice["text"].strip()
            results.append(
                {
                    "id": case["id"],
                    "group": case["group"],
                    "prompt_sha256": case["prompt_sha256"],
                    "expected": case["expected"],
                    "generated": generated,
                    "expected_match": generated == case["expected"],
                    "finish_reason": choice.get("finish_reason"),
                    "usage": response.get("usage"),
                    "error": None,
                }
            )
        except ResultError as exc:
            results.append(
                {
                    "id": case["id"],
                    "group": case["group"],
                    "prompt_sha256": case["prompt_sha256"],
                    "expected": case["expected"],
                    "generated": None,
                    "expected_match": False,
                    "finish_reason": None,
                    "usage": None,
                    "error": str(exc),
                }
            )

    metrics_after = prefix_snapshot(metrics_fetcher(base_url))
    metrics = prefix_delta(metrics_before, metrics_after)
    completed = sum(result["error"] is None for result in results)
    expected_matches = sum(result["expected_match"] for result in results)
    valid = completed == len(cases)
    if state.endswith("_on"):
        valid = valid and bool(
            isinstance(metrics.get("query_tokens"), (int, float))
            and metrics["query_tokens"] > 0
            and isinstance(metrics.get("hit_tokens"), (int, float))
            and metrics["hit_tokens"] > 0
        )
    document = {
        "schema_version": 1,
        "kind": "e06_correctness_canary",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "state": state,
        "server_profile": marker["profile"],
        "server_config_sha256": marker.get("server_config_sha256"),
        "dataset": str(Path(dataset_path)),
        "dataset_sha256": dataset_sha256,
        "base_url": base_url,
        "served_model_name": served_model_name,
        "request_count": len(cases),
        "completed": completed,
        "failed": len(cases) - completed,
        "expected_matches": expected_matches,
        "prefix_metrics": metrics,
        "valid": valid,
        "results": results,
    }
    output_dir = Path(result_root) / state
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_path = output_dir / f"{timestamp}-e06-canary-{state}.json"
    output_path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output_path, valid


def _latest_canary(result_root: str | Path, state: str) -> Path:
    candidates = sorted(
        (Path(result_root) / state).glob(f"*-e06-canary-{state}.json")
    )
    if not candidates:
        raise ResultError(f"No E06 {state} correctness canary result found")
    return candidates[-1]


def _load_canary_result(path: str | Path, state: str) -> dict[str, Any]:
    result_path = Path(path)
    try:
        data = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ResultError(
            f"Cannot read E06 canary result {result_path}: {exc}"
        ) from exc
    if (
        not isinstance(data, dict)
        or data.get("kind") != "e06_correctness_canary"
        or data.get("state") != state
    ):
        raise ResultError(f"Unexpected E06 canary {state} result: {result_path}")
    return data


def compare_e06_canary(
    result_root: str | Path,
    output_dir: str | Path,
    result_paths: dict[str, str | Path] | None = None,
) -> tuple[Path, Path, bool]:
    paths = {
        state: (
            Path(result_paths[state])
            if result_paths is not None and state in result_paths
            else _latest_canary(result_root, state)
        )
        for state in STATES
    }
    documents = {
        state: _load_canary_result(paths[state], state) for state in STATES
    }
    rows_by_state = {
        state: {
            row["id"]: row
            for row in documents[state].get("results", [])
            if isinstance(row, dict) and isinstance(row.get("id"), str)
        }
        for state in STATES
    }
    identifiers = sorted(
        set().union(*(set(rows_by_state[state]) for state in STATES))
    )
    row_evidence: list[dict[str, Any]] = []
    for identifier in identifiers:
        state_rows = {
            state: rows_by_state[state].get(identifier, {}) for state in STATES
        }
        control_row = state_rows[CONTROL]
        prompt_match = bool(
            control_row.get("prompt_sha256")
            and all(
                row.get("prompt_sha256") == control_row.get("prompt_sha256")
                for row in state_rows.values()
            )
        )
        output_match = bool(
            control_row.get("generated") is not None
            and all(
                row.get("generated") == control_row.get("generated")
                for row in state_rows.values()
            )
        )
        row_evidence.append(
            {
                "id": identifier,
                "group": control_row.get("group"),
                "expected": control_row.get("expected"),
                "prompt_match": prompt_match,
                "output_match": output_match,
                **{
                    f"{state}_generated": state_rows[state].get("generated")
                    for state in STATES
                },
                **{
                    f"{state}_expected_match": bool(
                        state_rows[state].get("expected_match")
                    )
                    for state in STATES
                },
            }
        )

    dataset_hashes = {documents[state].get("dataset_sha256") for state in STATES}
    dataset_hashes_valid = bool(
        len(dataset_hashes) == 1
        and all(isinstance(value, str) and value for value in dataset_hashes)
    )
    prompt_matches = sum(row["prompt_match"] for row in row_evidence)
    output_matches = sum(row["output_match"] for row in row_evidence)
    expected_matches = {
        state: sum(row[f"{state}_expected_match"] for row in row_evidence)
        for state in STATES
    }
    cache_hit_rates: dict[str, float | None] = {}
    cache_exercised = True
    for state in ("bt8192_on", "bt2048_on"):
        metrics = documents[state].get("prefix_metrics", {})
        exercised = bool(
            isinstance(metrics, dict)
            and isinstance(metrics.get("query_tokens"), (int, float))
            and metrics["query_tokens"] > 0
            and isinstance(metrics.get("hit_tokens"), (int, float))
            and metrics["hit_tokens"] > 0
        )
        cache_exercised = cache_exercised and exercised
        cache_hit_rates[state] = (
            metrics.get("hit_rate_percent") if isinstance(metrics, dict) else None
        )

    count = len(row_evidence)
    input_evidence_valid = bool(
        count > 0
        and dataset_hashes_valid
        and all(documents[state].get("valid") is True for state in STATES)
        and prompt_matches == count
    )
    equivalence_passed = bool(
        input_evidence_valid and output_matches == count and cache_exercised
    )
    quality_no_regression = bool(
        input_evidence_valid
        and all(
            expected_matches[state] >= expected_matches[CONTROL]
            for state in STATES
        )
    )
    passed = equivalence_passed and quality_no_regression
    summary = {
        "schema_version": 1,
        "kind": "e06_correctness_canary_comparison",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "results_by_state": {state: str(paths[state]) for state in STATES},
        "dataset_sha256_match": dataset_hashes_valid,
        "cases": count,
        "prompt_matches": prompt_matches,
        "output_matches_across_all_cells": output_matches,
        "expected_matches": expected_matches,
        "prefix_cache_hit_rate_percent": cache_hit_rates,
        "on_cells_exercised_cache": cache_exercised,
        "configuration_equivalence_status": (
            "PASS" if equivalence_passed else "FAIL"
        ),
        "task_quality_no_regression_status": (
            "PASS" if quality_no_regression else "FAIL"
        ),
        "overall_status": "PASS" if passed else "FAIL",
        "status": "PASS" if passed else "FAIL",
        "results": row_evidence,
    }
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    json_path = destination / "correctness_canary.json"
    markdown_path = destination / "correctness_canary.md"
    json_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    def _hit_rate(state: str) -> str:
        value = cache_hit_rates[state]
        return f"{value:.2f}%" if isinstance(value, (int, float)) else "NA"

    lines = [
        "# E06 Correctness Canary",
        "",
        f"Generated at: {summary['created_at']}",
        "",
        f"Overall status: **{summary['overall_status']}**",
        "Configuration equivalence: "
        f"**{summary['configuration_equivalence_status']}**",
        "Task quality no-regression: "
        f"**{summary['task_quality_no_regression_status']}**",
        "",
        f"Dataset SHA-256 match: {'YES' if summary['dataset_sha256_match'] else 'NO'}",
        f"Prompt matches: {prompt_matches}/{count}",
        f"Outputs matching across all four cells: {output_matches}/{count}",
        "Expected-answer matches A/B/C/D: "
        + "/".join(str(expected_matches[state]) for state in STATES),
        "APC hit rate C/D: "
        f"{_hit_rate('bt8192_on')}/{_hit_rate('bt2048_on')}",
        "",
        "A=8192/OFF, B=2048/OFF, C=8192/ON, D=2048/ON. The gate requires "
        "all four outputs to match for every case and both APC cells to record "
        "cache hits. Base-model mistakes shared by all cells are reported but "
        "are not treated as configuration regressions.",
        "",
        "| Case | Group | Prompt | Expected A/B/C/D | All outputs match |",
        "|---|---|---|---:|---|",
    ]
    for row in row_evidence:
        expected_cells = "/".join(
            "PASS" if row[f"{state}_expected_match"] else "FAIL"
            for state in STATES
        )
        lines.append(
            "| {id} | {group} | {prompt} | {expected} | {output} |".format(
                id=row["id"],
                group=row["group"],
                prompt="MATCH" if row["prompt_match"] else "MISMATCH",
                expected=expected_cells,
                output="MATCH" if row["output_match"] else "MISMATCH",
            )
        )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, markdown_path, passed
