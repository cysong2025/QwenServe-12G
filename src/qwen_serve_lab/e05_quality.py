from __future__ import annotations

import csv
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from qwen_serve_lab.config import config_sha256
from qwen_serve_lab.results import ResultError


EXPECTED_SERVER_PROFILES = {
    "bf16": "e05_kv_bf16",
    "fp8": "e05_kv_fp8",
}
EXPECTED_SERVER_CONFIGS = {
    "bf16": "configs/serve/e05_kv_bf16.toml",
    "fp8": "configs/serve/e05_kv_fp8.toml",
}
DANGEROUS_PATTERNS = (
    re.compile(r"\brm\s+-rf\b", re.I),
    re.compile(r"\bmkfs(?:\.|\s)", re.I),
    re.compile(r"\bdd\s+if=", re.I),
    re.compile(r"\bchmod\s+-R\s+777\b", re.I),
    re.compile(r"\bcurl\b[^\n|]*\|\s*(?:ba)?sh\b", re.I),
)
SYSTEM_PROMPT = (
    "You are an AI infrastructure incident triage classifier. Return exactly one "
    "JSON object, with no Markdown or explanation. Use exactly these keys: "
    "root_cause, actions, dangerous_command. Select one allowed root_cause, "
    "select exactly two allowed action labels, and set dangerous_command to false."
)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def load_e05_quality_dataset(
    path: str | Path,
) -> tuple[str, dict[str, Any], list[dict[str, Any]]]:
    dataset_path = Path(path)
    try:
        raw = dataset_path.read_bytes()
        data = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise ResultError(f"Cannot read E05 quality dataset {dataset_path}: {exc}") from exc
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise ResultError("E05 quality dataset must use schema_version 1")
    labels = data.get("root_cause_labels")
    actions = data.get("action_labels")
    cases = data.get("cases")
    if not isinstance(labels, list) or len(labels) < 2 or not all(
        isinstance(item, str) and item for item in labels
    ):
        raise ResultError("E05 quality root_cause_labels are invalid")
    if len(set(labels)) != len(labels):
        raise ResultError("E05 quality root_cause_labels must be unique")
    if not isinstance(actions, list) or len(actions) < 2 or not all(
        isinstance(item, str) and item for item in actions
    ):
        raise ResultError("E05 quality action_labels are invalid")
    if len(set(actions)) != len(actions):
        raise ResultError("E05 quality action_labels must be unique")
    if not isinstance(cases, list) or len(cases) != 50:
        raise ResultError("E05 quality dataset must contain exactly 50 cases")

    identifiers: set[str] = set()
    label_counts = {label: 0 for label in labels}
    normalized: list[dict[str, Any]] = []
    for case in cases:
        if not isinstance(case, dict):
            raise ResultError("E05 quality case must be an object")
        identifier = case.get("id")
        incident = case.get("incident")
        expected_root = case.get("root_cause")
        expected_actions = case.get("actions")
        if not isinstance(identifier, str) or not identifier:
            raise ResultError("E05 quality case id must be non-empty")
        if identifier in identifiers:
            raise ResultError(f"Duplicate E05 quality case id: {identifier}")
        identifiers.add(identifier)
        if not isinstance(incident, str) or not incident:
            raise ResultError(f"E05 quality case {identifier} has no incident text")
        if expected_root not in labels:
            raise ResultError(f"E05 quality case {identifier} has invalid root cause")
        if (
            not isinstance(expected_actions, list)
            or len(expected_actions) != 2
            or len(set(expected_actions)) != 2
            or any(action not in actions for action in expected_actions)
        ):
            raise ResultError(f"E05 quality case {identifier} needs two valid actions")
        label_counts[expected_root] += 1
        user_prompt = (
            f"Allowed root causes: {json.dumps(labels)}\n"
            f"Allowed actions: {json.dumps(actions)}\n"
            f"Incident:\n{incident}"
        )
        normalized.append(
            {
                "id": identifier,
                "incident": incident,
                "expected_root_cause": expected_root,
                "expected_actions": expected_actions,
                "user_prompt": user_prompt,
                "prompt_sha256": _sha256(
                    (SYSTEM_PROMPT + "\n" + user_prompt).encode("utf-8")
                ),
            }
        )
    if len(set(label_counts.values())) != 1:
        raise ResultError("E05 quality cases must be balanced across root causes")
    metadata = {
        "root_cause_labels": labels,
        "action_labels": actions,
        "label_counts": label_counts,
    }
    return _sha256(raw), metadata, normalized


def _load_active_server(
    path: str | Path, state: str, expected_config_sha256: str
) -> dict[str, Any]:
    marker_path = Path(path)
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ResultError(f"Cannot read controlled server marker {marker_path}: {exc}") from exc
    expected = EXPECTED_SERVER_PROFILES[state]
    if not isinstance(marker, dict) or marker.get("profile") != expected:
        actual = marker.get("profile") if isinstance(marker, dict) else None
        raise ResultError(f"E05 quality {state} requires active server {expected}; found {actual}")
    if marker.get("server_config_sha256") != expected_config_sha256:
        raise ResultError(
            f"E05 quality {state} active server config hash does not match "
            f"{EXPECTED_SERVER_CONFIGS[state]}"
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


def _request_chat_completion(
    base_url: str,
    served_model_name: str,
    system_prompt: str,
    user_prompt: str,
    seed: int,
    timeout_seconds: float,
) -> dict[str, Any]:
    payload = json.dumps(
        {
            "model": served_model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": 128,
            "temperature": 0,
            "seed": seed,
            "stream": False,
        }
    ).encode("utf-8")
    request = Request(
        base_url.rstrip("/") + "/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            result = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ResultError(f"E05 quality request failed: {exc}") from exc
    if not isinstance(result, dict):
        raise ResultError("E05 quality response is not an object")
    choices = result.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise ResultError("E05 quality response must contain one choice")
    choice = choices[0]
    message = choice.get("message") if isinstance(choice, dict) else None
    if not isinstance(message, dict) or not isinstance(message.get("content"), str):
        raise ResultError("E05 quality response has no message content")
    return result


def _strict_parse(
    generated: str, labels: set[str], actions: set[str]
) -> tuple[dict[str, Any] | None, str | None]:
    try:
        parsed = json.loads(generated)
    except json.JSONDecodeError as exc:
        return None, f"invalid_json: {exc.msg}"
    if not isinstance(parsed, dict) or set(parsed) != {
        "root_cause",
        "actions",
        "dangerous_command",
    }:
        return None, "schema_keys"
    if parsed.get("root_cause") not in labels:
        return None, "root_cause_enum"
    predicted_actions = parsed.get("actions")
    if (
        not isinstance(predicted_actions, list)
        or len(predicted_actions) != 2
        or len(set(predicted_actions)) != 2
        or any(action not in actions for action in predicted_actions)
    ):
        return None, "actions_schema"
    if not isinstance(parsed.get("dangerous_command"), bool):
        return None, "dangerous_command_type"
    return parsed, None


def _contains_dangerous_command(generated: str, parsed: dict[str, Any] | None) -> bool:
    return bool(
        (parsed is not None and parsed.get("dangerous_command") is True)
        or any(pattern.search(generated) for pattern in DANGEROUS_PATTERNS)
    )


def run_e05_quality(
    state: str,
    dataset_path: str | Path,
    result_root: str | Path,
    base_url: str = "http://127.0.0.1:8000",
    served_model_name: str = "qwen2.5-3b-instruct",
    active_server_path: str | Path = "artifacts/server/active.json",
    timeout_seconds: float = 120,
    request_completion: Callable[
        [str, str, str, str, int, float], dict[str, Any]
    ] = _request_chat_completion,
) -> tuple[Path, bool]:
    if state not in EXPECTED_SERVER_PROFILES:
        raise ResultError("E05 quality state must be 'bf16' or 'fp8'")
    project_root = Path(__file__).resolve().parents[2]
    expected_config_sha256 = config_sha256(
        project_root / EXPECTED_SERVER_CONFIGS[state]
    )
    marker = _load_active_server(
        active_server_path, state, expected_config_sha256
    )
    dataset_sha256, metadata, cases = load_e05_quality_dataset(dataset_path)
    labels = set(metadata["root_cause_labels"])
    actions = set(metadata["action_labels"])
    results: list[dict[str, Any]] = []
    for index, case in enumerate(cases):
        seed = 20260821 + index
        try:
            response = request_completion(
                base_url,
                served_model_name,
                SYSTEM_PROMPT,
                case["user_prompt"],
                seed,
                timeout_seconds,
            )
            choice = response["choices"][0]
            generated = choice["message"]["content"].strip()
            parsed, schema_error = _strict_parse(generated, labels, actions)
            results.append(
                {
                    "id": case["id"],
                    "incident": case["incident"],
                    "prompt_sha256": case["prompt_sha256"],
                    "seed": seed,
                    "expected_root_cause": case["expected_root_cause"],
                    "expected_actions": case["expected_actions"],
                    "generated": generated,
                    "parsed": parsed,
                    "schema_pass": parsed is not None,
                    "schema_error": schema_error,
                    "dangerous_command_detected": _contains_dangerous_command(
                        generated, parsed
                    ),
                    "finish_reason": choice.get("finish_reason"),
                    "usage": response.get("usage"),
                    "error": None,
                }
            )
        except ResultError as exc:
            results.append(
                {
                    "id": case["id"],
                    "incident": case["incident"],
                    "prompt_sha256": case["prompt_sha256"],
                    "seed": seed,
                    "expected_root_cause": case["expected_root_cause"],
                    "expected_actions": case["expected_actions"],
                    "generated": None,
                    "parsed": None,
                    "schema_pass": False,
                    "schema_error": None,
                    "dangerous_command_detected": False,
                    "finish_reason": None,
                    "usage": None,
                    "error": str(exc),
                }
            )
    completed = sum(row["error"] is None for row in results)
    document = {
        "schema_version": 1,
        "kind": "e05_quality_run",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "state": state,
        "server_profile": marker["profile"],
        "server_config_sha256": marker.get("server_config_sha256"),
        "dataset": str(Path(dataset_path)),
        "dataset_sha256": dataset_sha256,
        "root_cause_labels": metadata["root_cause_labels"],
        "action_labels": metadata["action_labels"],
        "base_url": base_url,
        "served_model_name": served_model_name,
        "request_count": len(cases),
        "completed": completed,
        "failed": len(cases) - completed,
        "valid": completed == len(cases),
        "results": results,
    }
    output_dir = Path(result_root) / state
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_path = output_dir / f"{timestamp}-e05-quality-{state}.json"
    output_path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output_path, bool(document["valid"])


def _latest_result(result_root: str | Path, state: str) -> Path:
    candidates = sorted((Path(result_root) / state).glob("*-e05-quality-*.json"))
    if not candidates:
        raise ResultError(f"No E05 {state} quality result found")
    return candidates[-1]


def _load_result(path: str | Path, state: str) -> dict[str, Any]:
    result_path = Path(path)
    try:
        data = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ResultError(f"Cannot read E05 quality result {result_path}: {exc}") from exc
    if (
        not isinstance(data, dict)
        or data.get("kind") != "e05_quality_run"
        or data.get("state") != state
    ):
        raise ResultError(f"Unexpected E05 {state} quality result: {result_path}")
    return data


def _root_macro_f1(rows: list[dict[str, Any]], labels: list[str]) -> float:
    scores: list[float] = []
    for label in labels:
        tp = fp = fn = 0
        for row in rows:
            expected = row.get("expected_root_cause")
            parsed = row.get("parsed")
            predicted = parsed.get("root_cause") if isinstance(parsed, dict) else None
            tp += int(expected == label and predicted == label)
            fp += int(expected != label and predicted == label)
            fn += int(expected == label and predicted != label)
        denominator = 2 * tp + fp + fn
        scores.append(2 * tp / denominator if denominator else 0.0)
    return sum(scores) / len(scores)


def _action_micro_f1(rows: list[dict[str, Any]]) -> float:
    tp = fp = fn = 0
    for row in rows:
        expected = set(row.get("expected_actions", []))
        parsed = row.get("parsed")
        predicted = set(parsed.get("actions", [])) if isinstance(parsed, dict) else set()
        tp += len(expected & predicted)
        fp += len(predicted - expected)
        fn += len(expected - predicted)
    denominator = 2 * tp + fp + fn
    return 2 * tp / denominator if denominator else 0.0


def _metrics(result: dict[str, Any]) -> dict[str, float]:
    rows = result.get("results")
    labels = result.get("root_cause_labels")
    if not isinstance(rows, list) or not rows or not isinstance(labels, list):
        raise ResultError("E05 quality result lacks rows or root-cause labels")
    count = len(rows)
    return {
        "schema_pass_rate": sum(bool(row.get("schema_pass")) for row in rows) / count,
        "root_cause_macro_f1": _root_macro_f1(rows, labels),
        "action_micro_f1": _action_micro_f1(rows),
        "dangerous_command_rate": sum(
            bool(row.get("dangerous_command_detected")) for row in rows
        )
        / count,
    }


def _write_human_review_template(
    bf16: dict[str, Any], fp8: dict[str, Any], output_dir: Path
) -> tuple[Path, Path]:
    csv_path = output_dir / "human_review.csv"
    key_path = output_dir / "human_review_key.json"
    if csv_path.exists() != key_path.exists():
        raise ResultError("E05 human review CSV/key pair is incomplete")
    if csv_path.exists():
        try:
            existing_key = json.loads(key_path.read_text(encoding="utf-8"))
            with csv_path.open("r", encoding="utf-8", newline="") as handle:
                existing_rows = list(csv.DictReader(handle))
        except (OSError, json.JSONDecodeError) as exc:
            raise ResultError(f"Cannot inspect existing E05 human review: {exc}") from exc
        same_results = bool(
            isinstance(existing_key, dict)
            and existing_key.get("bf16_result") == bf16.get("source_path")
            and existing_key.get("fp8_result") == fp8.get("source_path")
        )
        if same_results:
            return csv_path, key_path
        review_started = any(
            row.get("preferred", "").strip()
            or row.get("score_a_1_to_5", "").strip()
            or row.get("score_b_1_to_5", "").strip()
            for row in existing_rows
        )
        if review_started:
            raise ResultError(
                "Existing E05 human review contains scores for different result files; "
                "archive it before generating a new review"
            )
    bf16_rows = {row["id"]: row for row in bf16["results"]}
    fp8_rows = {row["id"]: row for row in fp8["results"]}
    review_rows: list[dict[str, Any]] = []
    key: dict[str, str] = {}
    for identifier in sorted(bf16_rows):
        control = bf16_rows[identifier]
        treatment = fp8_rows[identifier]
        bf16_is_a = hashlib.sha256(identifier.encode("utf-8")).digest()[0] % 2 == 0
        a_state, b_state = ("bf16", "fp8") if bf16_is_a else ("fp8", "bf16")
        a_row, b_row = (control, treatment) if bf16_is_a else (treatment, control)
        key[identifier] = a_state
        review_rows.append(
            {
                "id": identifier,
                "incident": control.get("incident"),
                "expected_root_cause": control.get("expected_root_cause"),
                "expected_actions": json.dumps(control.get("expected_actions")),
                "output_a": a_row.get("generated"),
                "output_b": b_row.get("generated"),
                "preferred": "",
                "score_a_1_to_5": "",
                "score_b_1_to_5": "",
                "notes": "",
            }
        )
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(review_rows[0]))
        writer.writeheader()
        writer.writerows(review_rows)
    key_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "bf16_result": bf16.get("source_path"),
                "fp8_result": fp8.get("source_path"),
                "a_state_by_id": key,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return csv_path, key_path


def compare_e05_quality(
    result_root: str | Path,
    output_dir: str | Path,
    bf16_result: str | Path | None = None,
    fp8_result: str | Path | None = None,
) -> tuple[Path, Path, bool]:
    bf16_path = Path(bf16_result) if bf16_result else _latest_result(result_root, "bf16")
    fp8_path = Path(fp8_result) if fp8_result else _latest_result(result_root, "fp8")
    bf16 = _load_result(bf16_path, "bf16")
    fp8 = _load_result(fp8_path, "fp8")
    bf16["source_path"] = str(bf16_path)
    fp8["source_path"] = str(fp8_path)
    bf16_rows = {row["id"]: row for row in bf16.get("results", [])}
    fp8_rows = {row["id"]: row for row in fp8.get("results", [])}
    identifiers = sorted(set(bf16_rows) | set(fp8_rows))
    prompt_matches = sum(
        bool(
            bf16_rows.get(identifier, {}).get("prompt_sha256")
            and bf16_rows[identifier].get("prompt_sha256")
            == fp8_rows.get(identifier, {}).get("prompt_sha256")
        )
        for identifier in identifiers
    )
    raw_output_matches = sum(
        bool(
            bf16_rows.get(identifier, {}).get("generated") is not None
            and bf16_rows[identifier].get("generated")
            == fp8_rows.get(identifier, {}).get("generated")
        )
        for identifier in identifiers
    )
    input_valid = bool(
        len(identifiers) == 50
        and bf16.get("valid") is True
        and fp8.get("valid") is True
        and bf16.get("dataset_sha256") == fp8.get("dataset_sha256")
        and prompt_matches == len(identifiers)
    )
    bf16_metrics = _metrics(bf16)
    fp8_metrics = _metrics(fp8)
    baseline_valid = bool(
        bf16_metrics["schema_pass_rate"] >= 0.90
        and bf16_metrics["root_cause_macro_f1"] >= 0.80
        and bf16_metrics["action_micro_f1"] >= 0.75
        and bf16_metrics["dangerous_command_rate"] <= 0.02
    )
    no_material_regression = bool(
        fp8_metrics["schema_pass_rate"] >= bf16_metrics["schema_pass_rate"] - 0.02
        and fp8_metrics["root_cause_macro_f1"]
        >= bf16_metrics["root_cause_macro_f1"] - 0.02
        and fp8_metrics["action_micro_f1"] >= bf16_metrics["action_micro_f1"] - 0.02
        and fp8_metrics["dangerous_command_rate"]
        <= max(0.02, bf16_metrics["dangerous_command_rate"])
    )
    automated_pass = input_valid and baseline_valid and no_material_regression
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    review_csv, review_key = _write_human_review_template(bf16, fp8, output)
    json_path = output / "quality.json"
    markdown_path = output / "quality.md"
    summary = {
        "schema_version": 1,
        "kind": "e05_quality_comparison",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "bf16_result": str(bf16_path),
        "fp8_result": str(fp8_path),
        "dataset_sha256_match": bf16.get("dataset_sha256") == fp8.get("dataset_sha256"),
        "cases": len(identifiers),
        "prompt_matches": prompt_matches,
        "raw_output_matches": raw_output_matches,
        "bf16": bf16_metrics,
        "fp8": fp8_metrics,
        "baseline_quality_status": "PASS" if baseline_valid else "FAIL",
        "no_material_regression_status": "PASS" if no_material_regression else "FAIL",
        "automated_status": "PASS" if automated_pass else "FAIL",
        "human_review_status": "PENDING",
        "overall_status": "PENDING" if automated_pass else "FAIL",
        "human_review_csv": str(review_csv),
        "human_review_key": str(review_key),
    }
    json_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# E05 FP8 KV Cache Quality",
        "",
        f"Generated at: {summary['created_at']}",
        "",
        f"Automated status: **{summary['automated_status']}**",
        f"Human review: **{summary['human_review_status']}**",
        f"Overall status: **{summary['overall_status']}**",
        "",
        f"Dataset SHA-256 match: {'YES' if summary['dataset_sha256_match'] else 'NO'}",
        f"Prompt matches: {prompt_matches}/{len(identifiers)}",
        f"Raw BF16/FP8 output matches: {raw_output_matches}/{len(identifiers)}",
        "",
        "| State | Schema pass | Root cause Macro-F1 | Action micro-F1 | Dangerous command rate |",
        "|---|---:|---:|---:|---:|",
        f"| BF16 | {bf16_metrics['schema_pass_rate']:.2%} | {bf16_metrics['root_cause_macro_f1']:.4f} | {bf16_metrics['action_micro_f1']:.4f} | {bf16_metrics['dangerous_command_rate']:.2%} |",
        f"| FP8 | {fp8_metrics['schema_pass_rate']:.2%} | {fp8_metrics['root_cause_macro_f1']:.4f} | {fp8_metrics['action_micro_f1']:.4f} | {fp8_metrics['dangerous_command_rate']:.2%} |",
        "",
        "Frozen automated gate: BF16 schema >= 90%, root Macro-F1 >= 0.80, action micro-F1 >= 0.75, dangerous commands <= 2%; FP8 may drop at most 0.02 on each quality score and may not exceed a 2% dangerous-command rate.",
        "",
        f"Complete the blinded 50-case review in `{review_csv}` before making the final E05 quality claim.",
    ]
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, markdown_path, automated_pass


def summarize_e05_human_review(
    review_csv: str | Path,
    review_key: str | Path,
    output_dir: str | Path,
) -> tuple[Path, Path, bool]:
    try:
        with Path(review_csv).open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        key = json.loads(Path(review_key).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ResultError(f"Cannot read E05 human review evidence: {exc}") from exc
    mapping = key.get("a_state_by_id") if isinstance(key, dict) else None
    if len(rows) != 50 or not isinstance(mapping, dict):
        raise ResultError("E05 human review requires 50 rows and a valid blind key")
    scores = {"bf16": [], "fp8": []}
    preferences = {"bf16": 0, "fp8": 0, "tie": 0}
    for row in rows:
        identifier = row.get("id", "")
        a_state = mapping.get(identifier)
        if a_state not in {"bf16", "fp8"}:
            raise ResultError(f"Missing blind key for E05 review case {identifier}")
        b_state = "fp8" if a_state == "bf16" else "bf16"
        try:
            score_a = int(row.get("score_a_1_to_5", ""))
            score_b = int(row.get("score_b_1_to_5", ""))
        except ValueError as exc:
            raise ResultError(f"Review scores are incomplete for {identifier}") from exc
        if score_a not in range(1, 6) or score_b not in range(1, 6):
            raise ResultError(f"Review scores must be 1-5 for {identifier}")
        preferred = row.get("preferred", "").strip().upper()
        if preferred not in {"A", "B", "TIE"}:
            raise ResultError(f"Review preference must be A, B, or TIE for {identifier}")
        scores[a_state].append(score_a)
        scores[b_state].append(score_b)
        if preferred == "TIE":
            preferences["tie"] += 1
        else:
            preferences[a_state if preferred == "A" else b_state] += 1
    bf16_mean = sum(scores["bf16"]) / 50
    fp8_mean = sum(scores["fp8"]) / 50
    passed = fp8_mean >= bf16_mean - 0.10
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "human_review_summary.json"
    markdown_path = output / "human_review_summary.md"
    summary = {
        "schema_version": 1,
        "kind": "e05_human_review_summary",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "cases": 50,
        "bf16_mean_score": bf16_mean,
        "fp8_mean_score": fp8_mean,
        "fp8_minus_bf16": fp8_mean - bf16_mean,
        "preferences": preferences,
        "status": "PASS" if passed else "FAIL",
    }
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(
        "\n".join(
            [
                "# E05 Human Review Summary",
                "",
                f"Generated at: {summary['created_at']}",
                "",
                f"Status: **{summary['status']}**",
                f"BF16 mean score: {bf16_mean:.3f}",
                f"FP8 mean score: {fp8_mean:.3f}",
                f"FP8 - BF16: {fp8_mean - bf16_mean:+.3f}",
                f"Preferences: BF16 {preferences['bf16']}, FP8 {preferences['fp8']}, tie {preferences['tie']}",
                "",
                "Frozen gate: FP8 mean score must be no more than 0.10 below BF16.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return json_path, markdown_path, passed
