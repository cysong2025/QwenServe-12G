from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from qwen_serve_lab.e05_quality import SYSTEM_PROMPT, load_e05_quality_dataset
from qwen_serve_lab.results import ResultError


EXPECTED_ROOT_CAUSES = 10
EXPECTED_TRAIN_ROWS = 250
EXPECTED_VALIDATION_ROWS = 100


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalized_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def _portable_path(path: str | Path) -> str:
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(resolved)


def _load_source(path: str | Path) -> tuple[Path, dict[str, Any]]:
    source_path = Path(path)
    try:
        data = json.loads(source_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ResultError(f"Cannot read E07 SFT source {source_path}: {exc}") from exc
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise ResultError("E07 SFT source must use schema_version 1")
    prefixes = data.get("render_prefixes")
    categories = data.get("categories")
    if (
        not isinstance(prefixes, list)
        or len(prefixes) != 5
        or not all(isinstance(item, str) and item for item in prefixes)
    ):
        raise ResultError("E07 SFT source requires five render prefixes")
    if not isinstance(categories, list) or len(categories) != EXPECTED_ROOT_CAUSES:
        raise ResultError("E07 SFT source requires ten root-cause categories")
    return source_path, data


def _build_user_prompt(labels: list[str], actions: list[str], incident: str) -> str:
    return (
        f"Allowed root causes: {json.dumps(labels)}\n"
        f"Allowed actions: {json.dumps(actions)}\n"
        f"Incident:\n{incident}"
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def prepare_e07_dataset(
    source_path: str | Path,
    test_path: str | Path,
    output_dir: str | Path,
) -> tuple[Path, Path, Path]:
    source_file, source = _load_source(source_path)
    test_sha256, test_metadata, test_cases = load_e05_quality_dataset(test_path)
    labels = list(test_metadata["root_cause_labels"])
    actions = list(test_metadata["action_labels"])
    test_incidents = {
        _normalized_text(case["incident"]): case["id"] for case in test_cases
    }

    seen_roots: set[str] = set()
    seen_source_incidents: set[str] = set()
    rows_by_split: dict[str, list[dict[str, Any]]] = {
        "train": [],
        "validation": [],
    }
    prefixes = source["render_prefixes"]
    for category in source["categories"]:
        if not isinstance(category, dict):
            raise ResultError("Every E07 SFT category must be an object")
        root = category.get("root_cause")
        expected_actions = category.get("actions")
        if root not in labels or root in seen_roots:
            raise ResultError(f"Invalid or duplicate E07 root cause: {root}")
        if (
            not isinstance(expected_actions, list)
            or len(expected_actions) != 2
            or len(set(expected_actions)) != 2
            or any(action not in actions for action in expected_actions)
        ):
            raise ResultError(f"E07 category {root} needs two valid actions")
        seen_roots.add(root)
        for split, expected_count in (("train", 5), ("validation", 2)):
            incidents = category.get(split)
            if (
                not isinstance(incidents, list)
                or len(incidents) != expected_count
                or not all(isinstance(item, str) and item for item in incidents)
            ):
                raise ResultError(
                    f"E07 category {root} requires {expected_count} {split} incidents"
                )
            for scenario_index, incident in enumerate(incidents, start=1):
                normalized_source = _normalized_text(incident)
                if normalized_source in seen_source_incidents:
                    raise ResultError("Duplicate E07 source incident")
                if normalized_source in test_incidents:
                    raise ResultError(
                        "E07 train/validation incident overlaps frozen test case "
                        f"{test_incidents[normalized_source]}"
                    )
                seen_source_incidents.add(normalized_source)
                for view_index, prefix in enumerate(prefixes, start=1):
                    rendered_incident = prefix + incident
                    user_prompt = _build_user_prompt(
                        labels, actions, rendered_incident
                    )
                    response = json.dumps(
                        {
                            "root_cause": root,
                            "actions": expected_actions,
                            "dangerous_command": False,
                        },
                        separators=(",", ":"),
                    )
                    identifier = (
                        f"{split}-{root}-{scenario_index:02d}-v{view_index}"
                    )
                    rows_by_split[split].append(
                        {
                            "id": identifier,
                            "split": split,
                            "source_group": (
                                f"{split}-{root}-{scenario_index:02d}"
                            ),
                            "root_cause": root,
                            "actions": expected_actions,
                            "messages": [
                                {"role": "system", "content": SYSTEM_PROMPT},
                                {"role": "user", "content": user_prompt},
                                {"role": "assistant", "content": response},
                            ],
                        }
                    )

    if set(labels) != seen_roots:
        raise ResultError("E07 source label set does not match the frozen test set")
    train = rows_by_split["train"]
    validation = rows_by_split["validation"]
    if len(train) != EXPECTED_TRAIN_ROWS or len(validation) != EXPECTED_VALIDATION_ROWS:
        raise ResultError("E07 generated split sizes do not match the frozen protocol")
    train_groups = {row["source_group"] for row in train}
    validation_groups = {row["source_group"] for row in validation}
    if train_groups & validation_groups:
        raise ResultError("E07 train and validation source groups overlap")
    for split, rows in rows_by_split.items():
        counts = Counter(row["root_cause"] for row in rows)
        if len(set(counts.values())) != 1:
            raise ResultError(f"E07 {split} split is not class-balanced")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    train_path = output / "train.jsonl"
    validation_path = output / "validation.jsonl"
    manifest_path = output / "dataset_manifest.json"
    _write_jsonl(train_path, train)
    _write_jsonl(validation_path, validation)
    manifest = {
        "schema_version": 1,
        "kind": "e07_dataset_manifest",
        "source": _portable_path(source_file),
        "source_sha256": _sha256(source_file),
        "frozen_test": _portable_path(test_path),
        "frozen_test_sha256": test_sha256,
        "split_policy": "scenario-grouped; five deterministic render views per source incident",
        "test_policy": "E05 50-case set is test-only and never materialized in train/validation",
        "train_rows": len(train),
        "validation_rows": len(validation),
        "train_source_groups": len(train_groups),
        "validation_source_groups": len(validation_groups),
        "root_cause_labels": labels,
        "action_labels": actions,
        "train_sha256": _sha256(train_path),
        "validation_sha256": _sha256(validation_path),
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return train_path, validation_path, manifest_path


def audit_e07_dataset(output_dir: str | Path) -> dict[str, Any]:
    output = Path(output_dir)
    manifest_path = output / "dataset_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ResultError(f"Cannot read E07 dataset manifest: {exc}") from exc
    checks = {
        "kind": manifest.get("kind") == "e07_dataset_manifest",
        "train_rows": manifest.get("train_rows") == EXPECTED_TRAIN_ROWS,
        "validation_rows": (
            manifest.get("validation_rows") == EXPECTED_VALIDATION_ROWS
        ),
        "root_causes": len(manifest.get("root_cause_labels", [])) == 10,
        "train_hash": (
            manifest.get("train_sha256") == _sha256(output / "train.jsonl")
        ),
        "validation_hash": (
            manifest.get("validation_sha256")
            == _sha256(output / "validation.jsonl")
        ),
    }
    return {"passed": all(checks.values()), "checks": checks, "manifest": manifest}
