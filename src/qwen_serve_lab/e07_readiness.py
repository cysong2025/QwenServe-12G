from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from qwen_serve_lab.commands import build_serve_command
from qwen_serve_lab.config import BenchmarkMatrix, ServeConfig
from qwen_serve_lab.e07_data import audit_e07_dataset, prepare_e07_dataset
from qwen_serve_lab.e07_training import E07TrainingConfig


def _check(name: str, passed: bool, detail: str) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "detail": detail}


def write_e07_readiness_report(
    repo_root: str | Path = ".",
    output_dir: str | Path = "reports/e07_lora",
) -> tuple[Path, Path, bool]:
    root = Path(repo_root).resolve()
    prepare_e07_dataset(
        root / "datasets/e07_ai_infra_sft_source.json",
        root / "datasets/e05_ai_infra_quality.json",
        root / "datasets/e07_sft",
    )
    dataset = audit_e07_dataset(root / "datasets/e07_sft")
    smoke = E07TrainingConfig.from_file(
        root / "configs/train/e07_qlora_smoke_rank8.toml"
    )
    rank8 = E07TrainingConfig.from_file(
        root / "configs/train/e07_qlora_rank8.toml"
    )
    rank16 = E07TrainingConfig.from_file(
        root / "configs/train/e07_qlora_rank16.toml"
    )
    rank8_payload = asdict(rank8)
    rank16_payload = asdict(rank16)
    rank_differences = {
        key
        for key in rank8_payload
        if rank8_payload[key] != rank16_payload[key]
    }
    base_server = ServeConfig.from_file(root / "configs/serve/e07_base.toml")
    lora_server = ServeConfig.from_file(root / "configs/serve/e07_lora.toml")
    base_matrix = BenchmarkMatrix.from_file(root / "configs/matrix/e07_base.toml")
    lora_matrix = BenchmarkMatrix.from_file(root / "configs/matrix/e07_lora.toml")

    base_payload = asdict(base_server)
    lora_payload = asdict(lora_server)
    ignored_server_fields = {
        "profile_name",
        "description",
        "source_path",
        "source_sha256",
        "enable_lora",
        "lora_name",
        "lora_path",
        "max_loras",
        "max_lora_rank",
        "lora_manifest_sha256",
        "lora_weights_sha256",
    }
    server_controls_equal = all(
        base_payload[key] == lora_payload[key]
        for key in base_payload
        if key not in ignored_server_fields
    )
    base_cells = {
        (
            config.profile_name.removeprefix("e07_base_"),
            config.input_len,
            config.output_len,
            config.max_concurrency,
            config.seed,
        )
        for config in base_matrix.configs
    }
    lora_cells = {
        (
            config.profile_name.removeprefix("e07_lora_"),
            config.input_len,
            config.output_len,
            config.max_concurrency,
            config.seed,
        )
        for config in lora_matrix.configs
    }
    command = build_serve_command(lora_server)
    checks = [
        _check(
            "dataset",
            dataset["passed"],
            "250 train and 100 validation rows; frozen 50-case set remains test-only",
        ),
        _check(
            "smoke-profile",
            smoke.is_smoke and smoke.rank == 8 and smoke.max_seq_length == 1024,
            "100-example rank-8 smoke with bounded steps",
        ),
        _check(
            "rank-ablation",
            rank8.rank == 8
            and rank16.rank == 16
            and rank_differences
            == {
                "profile_name",
                "description",
                "source_path",
                "source_sha256",
                "output_dir",
                "rank",
                "alpha",
            },
            "rank 8/16 profiles differ only in rank, alpha, description, and output",
        ),
        _check(
            "server-control",
            server_controls_equal
            and not base_server.enable_lora
            and lora_server.enable_lora,
            "Base and LoRA serving controls match outside the treatment fields",
        ),
        _check(
            "lora-command",
            "--enable-lora" in command
            and "--lora-modules" in command
            and "ai-infra-triage-r8=artifacts/adapters/e07/rank8" in command,
            "vLLM command explicitly names the rank-8 Adapter",
        ),
        _check(
            "performance-matrix",
            len(base_matrix.configs) == 6
            and len(lora_matrix.configs) == 6
            and base_cells == lora_cells
            and all(config.repetitions == 3 for config in base_matrix.configs + lora_matrix.configs),
            "six paired cells per state and 36 total planned repetitions",
        ),
        _check(
            "quality-protocol",
            (root / "datasets/e05_ai_infra_quality.json").is_file()
            and (root / "src/qwen_serve_lab/e07_quality.py").is_file(),
            "fixed 50-case automated and blinded-human comparison is wired",
        ),
        _check(
            "training-dependencies",
            all(
                pin in (root / "constraints/e07-train.txt").read_text()
                for pin in (
                    "accelerate==1.12.0",
                    "bitsandbytes==0.49.1",
                    "peft==0.18.1",
                )
            ),
            "QLoRA-only packages are pinned without replacing torch/transformers",
        ),
        _check(
            "runbook",
            (root / "docs/M3_E07_QLORA_LORA_RUNBOOK.md").is_file()
            and (root / "docs/E07_DATA_CARD.md").is_file(),
            "operator runbook and data card are present",
        ),
    ]
    passed = all(check["passed"] for check in checks)
    document = {
        "schema_version": 1,
        "kind": "e07_readiness_report",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "READY_FOR_GPU" if passed else "INCOMPLETE",
        "gpu_execution": "DEFERRED",
        "planned_formal_benchmark_runs": 36,
        "checks": checks,
    }
    output = root / output_dir
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "readiness.json"
    markdown_path = output / "readiness.md"
    json_path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# E07 GPU Readiness Audit",
        "",
        f"Generated at: {document['created_at']}",
        "",
        f"Status: **{document['status']}**",
        "GPU execution: **DEFERRED**",
        "Planned formal benchmark runs: **36**",
        "",
        "This report validates protocol, data, configs, commands, and analysis code only. It is not evidence that QLoRA training or LoRA serving succeeded on the target GPU.",
        "",
        "| Check | Status | Detail |",
        "|---|---|---|",
    ]
    lines.extend(
        f"| {check['name']} | {'PASS' if check['passed'] else 'FAIL'} | {check['detail']} |"
        for check in checks
    )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, markdown_path, passed
