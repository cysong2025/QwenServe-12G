from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from qwen_serve_lab.prometheus import (
    fetch_metrics,
    prefix_delta,
    prefix_snapshot,
)
from qwen_serve_lab.results import ResultError


EXPECTED_SERVER_PROFILES = {
    "off": "e04_prefix_off_bf16",
    "on": "e04_prefix_on_bf16",
}


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _load_dataset(path: str | Path) -> tuple[str, list[dict[str, str]]]:
    dataset_path = Path(path)
    try:
        raw = dataset_path.read_bytes()
        data = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise ResultError(f"Cannot read E04 canary dataset {dataset_path}: {exc}") from exc
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise ResultError("E04 canary dataset must use schema_version 1")
    groups = data.get("groups")
    if not isinstance(groups, list) or not groups:
        raise ResultError("E04 canary dataset must contain non-empty groups")

    cases: list[dict[str, str]] = []
    identifiers: set[str] = set()
    for group in groups:
        if not isinstance(group, dict):
            raise ResultError("E04 canary group must be an object")
        name = group.get("name")
        prefix = group.get("shared_prefix")
        group_cases = group.get("cases")
        if not isinstance(name, str) or not name:
            raise ResultError("E04 canary group name must be non-empty")
        if not isinstance(prefix, str) or not prefix:
            raise ResultError(f"E04 canary group {name} has no shared_prefix")
        if not isinstance(group_cases, list) or len(group_cases) < 2:
            raise ResultError(f"E04 canary group {name} needs at least two cases")
        for case in group_cases:
            if not isinstance(case, dict):
                raise ResultError(f"E04 canary group {name} contains an invalid case")
            case_id = case.get("id")
            question = case.get("question")
            expected = case.get("expected")
            if not all(isinstance(value, str) and value for value in (
                case_id,
                question,
                expected,
            )):
                raise ResultError(f"E04 canary group {name} has incomplete case data")
            assert isinstance(case_id, str)
            assert isinstance(question, str)
            assert isinstance(expected, str)
            if case_id in identifiers:
                raise ResultError(f"Duplicate E04 canary case id: {case_id}")
            identifiers.add(case_id)
            prompt = f"{prefix}\n\nQuestion: {question}\nAnswer:"
            cases.append(
                {
                    "id": case_id,
                    "group": name,
                    "prompt": prompt,
                    "prompt_sha256": _sha256_bytes(prompt.encode("utf-8")),
                    "expected": expected,
                }
            )
    return _sha256_bytes(raw), cases


def _load_active_server(path: str | Path, state: str) -> dict[str, Any]:
    marker_path = Path(path)
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ResultError(f"Cannot read controlled server marker {marker_path}: {exc}") from exc
    expected_profile = EXPECTED_SERVER_PROFILES[state]
    if not isinstance(marker, dict) or marker.get("profile") != expected_profile:
        actual = marker.get("profile") if isinstance(marker, dict) else None
        raise ResultError(
            f"E04 canary {state} requires active server {expected_profile}; found {actual}"
        )
    pid = marker.get("pid")
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
        raise ResultError("Controlled server marker contains an invalid pid")
    try:
        os.kill(pid, 0)
    except ProcessLookupError as exc:
        raise ResultError(f"Controlled server marker is stale; process {pid} is absent") from exc
    except PermissionError:
        pass
    return marker


def _request_completion(
    base_url: str,
    served_model_name: str,
    prompt: str,
    timeout_seconds: float,
) -> dict[str, Any]:
    payload = json.dumps(
        {
            "model": served_model_name,
            "prompt": prompt,
            "max_tokens": 16,
            "temperature": 0,
            "seed": 20260821,
            "stop": ["\n"],
            "stream": False,
        }
    ).encode("utf-8")
    request = Request(
        base_url.rstrip("/") + "/v1/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            result = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ResultError(f"E04 canary completion request failed: {exc}") from exc
    if not isinstance(result, dict):
        raise ResultError("E04 canary completion response is not an object")
    choices = result.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise ResultError("E04 canary completion response must contain one choice")
    choice = choices[0]
    if not isinstance(choice, dict) or not isinstance(choice.get("text"), str):
        raise ResultError("E04 canary completion response has no generated text")
    return result


def run_e04_canary(
    state: str,
    dataset_path: str | Path,
    result_root: str | Path,
    base_url: str = "http://127.0.0.1:8000",
    served_model_name: str = "qwen2.5-3b-instruct",
    active_server_path: str | Path = "artifacts/server/active.json",
    timeout_seconds: float = 120,
    request_completion: Callable[[str, str, str, float], dict[str, Any]] = _request_completion,
    metrics_fetcher: Callable[[str], str] = fetch_metrics,
) -> tuple[Path, bool]:
    if state not in EXPECTED_SERVER_PROFILES:
        raise ResultError("E04 canary state must be 'off' or 'on'")
    marker = _load_active_server(active_server_path, state)
    dataset_sha256, cases = _load_dataset(dataset_path)
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
    if state == "on":
        valid = valid and bool(
            isinstance(metrics.get("query_tokens"), (int, float))
            and metrics["query_tokens"] > 0
            and isinstance(metrics.get("hit_tokens"), (int, float))
            and metrics["hit_tokens"] > 0
        )
    document = {
        "schema_version": 1,
        "kind": "e04_correctness_canary",
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
    output_path = output_dir / f"{timestamp}-e04-canary-{state}.json"
    output_path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output_path, valid


def _latest_canary(result_root: str | Path, state: str) -> Path:
    candidates = sorted((Path(result_root) / state).glob("*-e04-canary-*.json"))
    if not candidates:
        raise ResultError(f"No E04 {state} correctness canary result found")
    return candidates[-1]


def _load_canary_result(path: str | Path, state: str) -> dict[str, Any]:
    result_path = Path(path)
    try:
        data = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ResultError(f"Cannot read E04 canary result {result_path}: {exc}") from exc
    if (
        not isinstance(data, dict)
        or data.get("kind") != "e04_correctness_canary"
        or data.get("state") != state
    ):
        raise ResultError(f"Unexpected E04 canary {state} result: {result_path}")
    return data


def compare_e04_canary(
    result_root: str | Path,
    output_dir: str | Path,
    off_result: str | Path | None = None,
    on_result: str | Path | None = None,
) -> tuple[Path, Path, bool]:
    off_path = Path(off_result) if off_result else _latest_canary(result_root, "off")
    on_path = Path(on_result) if on_result else _latest_canary(result_root, "on")
    off = _load_canary_result(off_path, "off")
    on = _load_canary_result(on_path, "on")
    off_rows = {row["id"]: row for row in off.get("results", [])}
    on_rows = {row["id"]: row for row in on.get("results", [])}
    identifiers = sorted(set(off_rows) | set(on_rows))
    row_evidence: list[dict[str, Any]] = []
    for identifier in identifiers:
        off_row = off_rows.get(identifier, {})
        on_row = on_rows.get(identifier, {})
        prompt_match = bool(
            off_row.get("prompt_sha256")
            and off_row.get("prompt_sha256") == on_row.get("prompt_sha256")
        )
        output_match = bool(
            off_row.get("generated") is not None
            and off_row.get("generated") == on_row.get("generated")
        )
        row_evidence.append(
            {
                "id": identifier,
                "group": off_row.get("group") or on_row.get("group"),
                "prompt_match": prompt_match,
                "expected": off_row.get("expected") or on_row.get("expected"),
                "off_generated": off_row.get("generated"),
                "on_generated": on_row.get("generated"),
                "off_expected_match": bool(off_row.get("expected_match")),
                "on_expected_match": bool(on_row.get("expected_match")),
                "output_match": output_match,
            }
        )

    prompt_matches = sum(row["prompt_match"] for row in row_evidence)
    off_expected_matches = sum(row["off_expected_match"] for row in row_evidence)
    on_expected_matches = sum(row["on_expected_match"] for row in row_evidence)
    output_matches = sum(row["output_match"] for row in row_evidence)
    on_metrics = on.get("prefix_metrics", {})
    on_cache_exercised = bool(
        isinstance(on_metrics, dict)
        and isinstance(on_metrics.get("query_tokens"), (int, float))
        and on_metrics["query_tokens"] > 0
        and isinstance(on_metrics.get("hit_tokens"), (int, float))
        and on_metrics["hit_tokens"] > 0
    )
    count = len(row_evidence)
    input_evidence_valid = bool(
        count > 0
        and off.get("dataset_sha256") == on.get("dataset_sha256")
        and off.get("valid") is True
        and on.get("valid") is True
        and prompt_matches == count
    )
    equivalence_passed = bool(
        input_evidence_valid
        and output_matches == count
        and on_cache_exercised
    )
    quality_passed = bool(
        input_evidence_valid
        and off_expected_matches == count
        and on_expected_matches == count
    )
    passed = equivalence_passed and quality_passed
    summary = {
        "schema_version": 1,
        "kind": "e04_correctness_canary_comparison",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "off_result": str(off_path),
        "on_result": str(on_path),
        "dataset_sha256_match": off.get("dataset_sha256") == on.get("dataset_sha256"),
        "cases": count,
        "prompt_matches": prompt_matches,
        "off_expected_matches": off_expected_matches,
        "on_expected_matches": on_expected_matches,
        "output_matches": output_matches,
        "on_prefix_cache_hit_rate_percent": (
            on_metrics.get("hit_rate_percent") if isinstance(on_metrics, dict) else None
        ),
        "on_cache_exercised": on_cache_exercised,
        "apc_equivalence_status": "PASS" if equivalence_passed else "FAIL",
        "task_quality_status": "PASS" if quality_passed else "FAIL",
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
    hit_rate = summary["on_prefix_cache_hit_rate_percent"]
    hit_rate_text = f"{hit_rate:.2f}%" if isinstance(hit_rate, (int, float)) else "NA"
    lines = [
        "# E04 Correctness Canary",
        "",
        f"Generated at: {summary['created_at']}",
        "",
        f"Overall status: **{summary['overall_status']}**",
        f"APC equivalence: **{summary['apc_equivalence_status']}**",
        f"Task quality: **{summary['task_quality_status']}**",
        "",
        f"Dataset SHA-256 match: {'YES' if summary['dataset_sha256_match'] else 'NO'}",
        f"Prompt matches: {prompt_matches}/{count}",
        f"OFF expected-answer matches: {off_expected_matches}/{count}",
        f"ON expected-answer matches: {on_expected_matches}/{count}",
        f"OFF/ON output matches: {output_matches}/{count}",
        f"ON prefix cache hit rate: {hit_rate_text}",
        "",
        "| Case | Group | Prompt | OFF expected | ON expected | OFF/ON output |",
        "|---|---|---|---|---|---|",
    ]
    for row in row_evidence:
        lines.append(
            "| {id} | {group} | {prompt} | {off} | {on} | {output} |".format(
                id=row["id"],
                group=row["group"],
                prompt="MATCH" if row["prompt_match"] else "MISMATCH",
                off="PASS" if row["off_expected_match"] else "FAIL",
                on="PASS" if row["on_expected_match"] else "FAIL",
                output="MATCH" if row["output_match"] else "MISMATCH",
            )
        )
    failures = [
        row
        for row in row_evidence
        if not row["off_expected_match"] or not row["on_expected_match"]
    ]
    if failures:
        lines.extend(
            [
                "",
                "## Task Quality Failures",
                "",
                "| Case | Expected | OFF generated | ON generated |",
                "|---|---|---|---|",
            ]
        )
        for row in failures:
            cells = [
                str(row[key]).replace("|", "\\|").replace("\n", "<br>")
                for key in ("id", "expected", "off_generated", "on_generated")
            ]
            lines.append("| " + " | ".join(cells) + " |")
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, markdown_path, passed
