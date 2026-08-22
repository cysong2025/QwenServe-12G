from __future__ import annotations

import csv
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from qwen_serve_lab.config import config_sha256
from qwen_serve_lab.e05_quality import (
    SYSTEM_PROMPT,
    _contains_dangerous_command,
    _metrics,
    _request_chat_completion,
    _strict_parse,
    load_e05_quality_dataset,
)
from qwen_serve_lab.results import ResultError


EXPECTED_SERVER_PROFILES = {"base": "e07_base", "lora": "e07_lora"}
EXPECTED_SERVER_CONFIGS = {
    "base": "configs/serve/e07_base.toml",
    "lora": "configs/serve/e07_lora.toml",
}
SERVED_MODEL_NAMES = {
    "base": "qwen2.5-3b-instruct",
    "lora": "ai-infra-triage-r8",
}


def _load_active_server(
    path: str | Path, state: str, expected_config_sha256: str
) -> dict[str, Any]:
    marker_path = Path(path)
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ResultError(f"Cannot read E07 active server marker: {exc}") from exc
    expected_profile = EXPECTED_SERVER_PROFILES[state]
    if not isinstance(marker, dict) or marker.get("profile") != expected_profile:
        actual = marker.get("profile") if isinstance(marker, dict) else None
        raise ResultError(
            f"E07 quality {state} requires {expected_profile}; found {actual}"
        )
    if marker.get("server_config_sha256") != expected_config_sha256:
        raise ResultError(f"E07 {state} active server config hash is stale")
    if state == "lora" and (
        marker.get("lora_name") != SERVED_MODEL_NAMES["lora"]
        or not marker.get("lora_manifest_sha256")
        or not marker.get("lora_weights_sha256")
    ):
        raise ResultError("E07 LoRA server marker lacks validated Adapter evidence")
    pid = marker.get("pid")
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
        raise ResultError("E07 active server marker contains an invalid pid")
    try:
        os.kill(pid, 0)
    except ProcessLookupError as exc:
        raise ResultError(f"E07 active server process {pid} is absent") from exc
    except PermissionError:
        pass
    return marker


def run_e07_quality(
    state: str,
    dataset_path: str | Path,
    result_root: str | Path,
    base_url: str = "http://127.0.0.1:8000",
    active_server_path: str | Path = "artifacts/server/active.json",
    timeout_seconds: float = 120,
    request_completion: Callable[
        [str, str, str, str, int, float], dict[str, Any]
    ] = _request_chat_completion,
) -> tuple[Path, bool]:
    if state not in EXPECTED_SERVER_PROFILES:
        raise ResultError("E07 quality state must be 'base' or 'lora'")
    project_root = Path(__file__).resolve().parents[2]
    expected_hash = config_sha256(project_root / EXPECTED_SERVER_CONFIGS[state])
    marker = _load_active_server(active_server_path, state, expected_hash)
    dataset_sha256, metadata, cases = load_e05_quality_dataset(dataset_path)
    labels = set(metadata["root_cause_labels"])
    actions = set(metadata["action_labels"])
    served_model_name = SERVED_MODEL_NAMES[state]
    results: list[dict[str, Any]] = []
    for index, case in enumerate(cases):
        seed = 20260824 + index
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
        "kind": "e07_quality_run",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "state": state,
        "server_profile": marker["profile"],
        "server_config_sha256": marker.get("server_config_sha256"),
        "lora_manifest_sha256": marker.get("lora_manifest_sha256"),
        "lora_weights_sha256": marker.get("lora_weights_sha256"),
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
    output = Path(result_root) / state
    output.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_path = output / f"{timestamp}-e07-quality-{state}.json"
    output_path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output_path, bool(document["valid"])


def _latest_result(result_root: str | Path, state: str) -> Path:
    paths = sorted((Path(result_root) / state).glob("*-e07-quality-*.json"))
    if not paths:
        raise ResultError(f"No E07 {state} quality result found")
    return paths[-1]


def _load_result(path: Path, state: str) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ResultError(f"Cannot read E07 quality result {path}: {exc}") from exc
    if (
        not isinstance(data, dict)
        or data.get("kind") != "e07_quality_run"
        or data.get("state") != state
    ):
        raise ResultError(f"Unexpected E07 {state} quality result")
    data["source_path"] = str(path)
    return data


def _write_human_review_template(
    base: dict[str, Any], lora: dict[str, Any], output_dir: Path
) -> tuple[Path, Path]:
    csv_path = output_dir / "human_review.csv"
    key_path = output_dir / "human_review_key.json"
    if csv_path.exists() != key_path.exists():
        raise ResultError("E07 human review CSV/key pair is incomplete")
    if csv_path.exists():
        try:
            key = json.loads(key_path.read_text(encoding="utf-8"))
            with csv_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
        except (OSError, json.JSONDecodeError) as exc:
            raise ResultError(f"Cannot inspect E07 human review: {exc}") from exc
        if (
            isinstance(key, dict)
            and key.get("base_result") == base.get("source_path")
            and key.get("lora_result") == lora.get("source_path")
        ):
            return csv_path, key_path
        if any(
            row.get("preferred", "").strip()
            or row.get("score_a_1_to_5", "").strip()
            or row.get("score_b_1_to_5", "").strip()
            for row in rows
        ):
            raise ResultError(
                "Existing E07 review contains scores for different results; archive it first"
            )
    base_rows = {row["id"]: row for row in base["results"]}
    lora_rows = {row["id"]: row for row in lora["results"]}
    review_rows: list[dict[str, Any]] = []
    mapping: dict[str, str] = {}
    for identifier in sorted(base_rows):
        control = base_rows[identifier]
        treatment = lora_rows[identifier]
        base_is_a = hashlib.sha256(("e07-" + identifier).encode()).digest()[0] % 2 == 0
        a_state, b_state = ("base", "lora") if base_is_a else ("lora", "base")
        a_row, b_row = (control, treatment) if base_is_a else (treatment, control)
        mapping[identifier] = a_state
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
                "base_result": base.get("source_path"),
                "lora_result": lora.get("source_path"),
                "a_state_by_id": mapping,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return csv_path, key_path


def _write_quality_report(summary: dict[str, Any], output_dir: Path) -> tuple[Path, Path]:
    json_path = output_dir / "quality.json"
    markdown_path = output_dir / "quality.md"
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    base = summary["base"]
    lora = summary["lora"]
    lines = [
        "# E07 Base vs LoRA Quality",
        "",
        f"Generated at: {summary['created_at']}",
        "",
        f"Automated status: **{summary['automated_status']}**",
        f"Human review: **{summary['human_review_status']}**",
        f"Overall status: **{summary['overall_status']}**",
        "",
        f"Dataset SHA-256 match: {'YES' if summary['dataset_sha256_match'] else 'NO'}",
        f"Prompt matches: {summary['prompt_matches']}/{summary['cases']}",
        "",
        "| State | Schema pass | Root cause Macro-F1 | Action micro-F1 | Dangerous command rate |",
        "|---|---:|---:|---:|---:|",
        f"| Base | {base['schema_pass_rate']:.2%} | {base['root_cause_macro_f1']:.4f} | {base['action_micro_f1']:.4f} | {base['dangerous_command_rate']:.2%} |",
        f"| LoRA | {lora['schema_pass_rate']:.2%} | {lora['root_cause_macro_f1']:.4f} | {lora['action_micro_f1']:.4f} | {lora['dangerous_command_rate']:.2%} |",
        "",
        "Frozen automated gate: LoRA schema >= 98%, root Macro-F1 >= 0.90, action micro-F1 >= 0.85, dangerous command rate = 0%; versus Base, schema may not regress, root Macro-F1 must improve >= 0.10, and action micro-F1 must improve >= 0.20.",
    ]
    human = summary.get("human_review")
    if isinstance(human, dict):
        lines.extend(
            [
                "",
                "## Blinded human review",
                "",
                f"Base mean score: {human['base_mean_score']:.3f}",
                f"LoRA mean score: {human['lora_mean_score']:.3f}",
                f"LoRA - Base: {human['lora_minus_base']:+.3f}",
                "",
                "Frozen human gate: LoRA mean score must exceed Base by at least 0.30.",
            ]
        )
    else:
        lines.extend(
            [
                "",
                f"Complete the blinded review in `{summary['human_review_csv']}` before making the final E07 claim.",
            ]
        )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, markdown_path


def compare_e07_quality(
    result_root: str | Path,
    output_dir: str | Path,
    base_result: str | Path | None = None,
    lora_result: str | Path | None = None,
) -> tuple[Path, Path, bool]:
    base_path = Path(base_result) if base_result else _latest_result(result_root, "base")
    lora_path = Path(lora_result) if lora_result else _latest_result(result_root, "lora")
    base = _load_result(base_path, "base")
    lora = _load_result(lora_path, "lora")
    base_rows = {row["id"]: row for row in base.get("results", [])}
    lora_rows = {row["id"]: row for row in lora.get("results", [])}
    identifiers = sorted(set(base_rows) | set(lora_rows))
    prompt_matches = sum(
        bool(
            base_rows.get(identifier, {}).get("prompt_sha256")
            and base_rows[identifier].get("prompt_sha256")
            == lora_rows.get(identifier, {}).get("prompt_sha256")
        )
        for identifier in identifiers
    )
    input_valid = bool(
        len(identifiers) == 50
        and base.get("valid") is True
        and lora.get("valid") is True
        and base.get("server_profile") == EXPECTED_SERVER_PROFILES["base"]
        and lora.get("server_profile") == EXPECTED_SERVER_PROFILES["lora"]
        and base.get("server_config_sha256")
        == config_sha256(
            Path(__file__).resolve().parents[2] / EXPECTED_SERVER_CONFIGS["base"]
        )
        and lora.get("server_config_sha256")
        == config_sha256(
            Path(__file__).resolve().parents[2] / EXPECTED_SERVER_CONFIGS["lora"]
        )
        and base.get("dataset_sha256") == lora.get("dataset_sha256")
        and prompt_matches == 50
        and lora.get("lora_manifest_sha256")
        and lora.get("lora_weights_sha256")
    )
    base_metrics = _metrics(base)
    lora_metrics = _metrics(lora)
    absolute_gate = bool(
        lora_metrics["schema_pass_rate"] >= 0.98
        and lora_metrics["root_cause_macro_f1"] >= 0.90
        and lora_metrics["action_micro_f1"] >= 0.85
        and lora_metrics["dangerous_command_rate"] == 0
    )
    improvement_gate = bool(
        lora_metrics["schema_pass_rate"] >= base_metrics["schema_pass_rate"]
        and lora_metrics["root_cause_macro_f1"]
        >= base_metrics["root_cause_macro_f1"] + 0.10
        and lora_metrics["action_micro_f1"]
        >= base_metrics["action_micro_f1"] + 0.20
    )
    passed = input_valid and absolute_gate and improvement_gate
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    review_csv, review_key = _write_human_review_template(base, lora, output)
    summary = {
        "schema_version": 1,
        "kind": "e07_quality_comparison",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "base_result": str(base_path),
        "lora_result": str(lora_path),
        "lora_manifest_sha256": lora.get("lora_manifest_sha256"),
        "lora_weights_sha256": lora.get("lora_weights_sha256"),
        "dataset_sha256_match": base.get("dataset_sha256") == lora.get("dataset_sha256"),
        "cases": len(identifiers),
        "prompt_matches": prompt_matches,
        "base": base_metrics,
        "lora": lora_metrics,
        "absolute_quality_status": "PASS" if absolute_gate else "FAIL",
        "quality_improvement_status": "PASS" if improvement_gate else "FAIL",
        "automated_status": "PASS" if passed else "FAIL",
        "human_review_status": "PENDING",
        "overall_status": "PENDING" if passed else "FAIL",
        "human_review_csv": str(review_csv),
        "human_review_key": str(review_key),
    }
    json_path, markdown_path = _write_quality_report(summary, output)
    return json_path, markdown_path, passed


def summarize_e07_human_review(
    review_csv: str | Path,
    review_key: str | Path,
    output_dir: str | Path,
) -> tuple[Path, Path, bool]:
    try:
        with Path(review_csv).open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        key = json.loads(Path(review_key).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ResultError(f"Cannot read E07 human review evidence: {exc}") from exc
    mapping = key.get("a_state_by_id") if isinstance(key, dict) else None
    if len(rows) != 50 or not isinstance(mapping, dict):
        raise ResultError("E07 human review requires 50 rows and a valid blind key")
    scores = {"base": [], "lora": []}
    preferences = {"base": 0, "lora": 0, "tie": 0}
    for row in rows:
        identifier = row.get("id", "")
        a_state = mapping.get(identifier)
        if a_state not in {"base", "lora"}:
            raise ResultError(f"Missing E07 blind key for {identifier}")
        b_state = "lora" if a_state == "base" else "base"
        try:
            score_a = int(row.get("score_a_1_to_5", ""))
            score_b = int(row.get("score_b_1_to_5", ""))
        except ValueError as exc:
            raise ResultError(f"E07 review scores are incomplete for {identifier}") from exc
        if score_a not in range(1, 6) or score_b not in range(1, 6):
            raise ResultError(f"E07 review scores must be 1-5 for {identifier}")
        preferred = row.get("preferred", "").strip().upper()
        if preferred not in {"A", "B", "TIE"}:
            raise ResultError(f"E07 preference must be A, B, or TIE for {identifier}")
        scores[a_state].append(score_a)
        scores[b_state].append(score_b)
        if preferred == "TIE":
            preferences["tie"] += 1
        else:
            preferences[a_state if preferred == "A" else b_state] += 1
    base_mean = sum(scores["base"]) / 50
    lora_mean = sum(scores["lora"]) / 50
    passed = lora_mean >= base_mean + 0.30
    summary = {
        "schema_version": 1,
        "kind": "e07_human_review_summary",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "cases": 50,
        "base_mean_score": base_mean,
        "lora_mean_score": lora_mean,
        "lora_minus_base": lora_mean - base_mean,
        "preferences": preferences,
        "status": "PASS" if passed else "FAIL",
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "human_review_summary.json"
    markdown_path = output / "human_review_summary.md"
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(
        "\n".join(
            [
                "# E07 Human Review Summary",
                "",
                f"Generated at: {summary['created_at']}",
                "",
                f"Status: **{summary['status']}**",
                f"Base mean score: {base_mean:.3f}",
                f"LoRA mean score: {lora_mean:.3f}",
                f"LoRA - Base: {lora_mean - base_mean:+.3f}",
                f"Preferences: Base {preferences['base']}, LoRA {preferences['lora']}, tie {preferences['tie']}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    quality_path = output / "quality.json"
    if quality_path.is_file():
        try:
            quality = json.loads(quality_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ResultError(f"Cannot finalize E07 quality report: {exc}") from exc
        quality["human_review"] = summary
        quality["human_review_status"] = summary["status"]
        quality["overall_status"] = (
            "PASS"
            if quality.get("automated_status") == "PASS" and passed
            else "FAIL"
        )
        quality["finalized_at"] = summary["created_at"]
        _write_quality_report(quality, output)
    return json_path, markdown_path, passed
