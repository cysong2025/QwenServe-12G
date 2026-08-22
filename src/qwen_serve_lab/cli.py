from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from qwen_serve_lab.commands import (
    build_benchmark_command,
    build_prewarm_command,
    build_serve_command,
    build_serve_environment,
    render_shell_command,
)
from qwen_serve_lab.comparison import write_e02_comparison
from qwen_serve_lab.e04 import (
    write_e04_comparison,
    write_e04_output_diagnostics,
)
from qwen_serve_lab.e04_canary import compare_e04_canary, run_e04_canary
from qwen_serve_lab.e05 import (
    write_e05_capacity_report,
    write_e05_comparison,
)
from qwen_serve_lab.e05_quality import (
    compare_e05_quality,
    run_e05_quality,
    summarize_e05_human_review,
)
from qwen_serve_lab.e06 import write_e06_comparison
from qwen_serve_lab.e06_canary import compare_e06_canary, run_e06_canary
from qwen_serve_lab.e07 import write_e07_comparison, write_e07_final_report
from qwen_serve_lab.e07_data import prepare_e07_dataset
from qwen_serve_lab.e07_quality import (
    compare_e07_quality,
    run_e07_quality,
    summarize_e07_human_review,
)
from qwen_serve_lab.e07_readiness import write_e07_readiness_report
from qwen_serve_lab.e07_training import (
    E07TrainingConfig,
    run_e07_training,
    write_e07_adapter_report,
)
from qwen_serve_lab.final_audit import write_e01_e06_audit
from qwen_serve_lab.config import (
    BenchmarkConfig,
    BenchmarkMatrix,
    ConfigError,
    ServeConfig,
    config_sha256,
)
from qwen_serve_lab.environment import (
    checks_as_dict,
    collect_environment,
    run_doctor,
    write_json,
)
from qwen_serve_lab.telemetry import NvidiaSmiSampler, summarize_telemetry
from qwen_serve_lab.prometheus import (
    MetricsError,
    fetch_metrics,
    prefix_delta,
    prefix_snapshot,
)
from qwen_serve_lab.results import (
    ResultError,
    generated_texts_sha256,
    load_records_from_manifests,
    write_reports,
)


ACTIVE_SERVER_PATH = Path("artifacts/server/active.json")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="qsl")
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser("doctor", help="Validate the target WSL2 runtime")
    doctor.add_argument("--json", action="store_true", dest="as_json")

    collect = subparsers.add_parser("collect-env", help="Write an environment snapshot")
    collect.add_argument("--output", required=True, type=Path)

    serve = subparsers.add_parser("render-serve", help="Render a validated vLLM command")
    serve.add_argument("config", type=Path)
    serve.add_argument("--model-path", type=Path)
    serve.add_argument("--adapter-path", type=Path)

    run_serve = subparsers.add_parser(
        "run-serve", help="Run vLLM while capturing logs and a server manifest"
    )
    run_serve.add_argument("config", type=Path)
    run_serve.add_argument("--model-path", type=Path)
    run_serve.add_argument("--adapter-path", type=Path)

    benchmark = subparsers.add_parser(
        "render-bench", help="Render one validated vLLM benchmark command"
    )
    benchmark.add_argument("config", type=Path)
    benchmark.add_argument("--repetition", type=int, default=1)
    benchmark.add_argument("--tokenizer-path", type=Path)

    run = subparsers.add_parser(
        "run-bench", help="Snapshot the environment and execute benchmark repetitions"
    )
    run.add_argument("config", type=Path)
    run.add_argument("--tokenizer-path", type=Path)

    render_matrix = subparsers.add_parser(
        "render-matrix", help="Render every effective benchmark in a matrix"
    )
    render_matrix.add_argument("config", type=Path)
    render_matrix.add_argument("--tokenizer-path", type=Path)

    run_matrix = subparsers.add_parser(
        "run-matrix", help="Execute a benchmark matrix with manifests and telemetry"
    )
    run_matrix.add_argument("config", type=Path)
    run_matrix.add_argument("--tokenizer-path", type=Path)
    run_matrix.add_argument(
        "--skip-completed",
        action="store_true",
        help="Skip profiles with all valid repetitions for the same config hashes",
    )
    run_matrix.add_argument(
        "--only",
        action="append",
        default=[],
        help="Run only an exact expanded profile name; may be repeated",
    )

    summarize = subparsers.add_parser(
        "summarize", help="Build CSV and Markdown reports from run manifests"
    )
    summarize.add_argument(
        "--manifest-dir", type=Path, default=Path("artifacts/env")
    )
    summarize.add_argument("--output-dir", type=Path, required=True)
    summarize.add_argument("--profile-prefix")
    summarize.add_argument(
        "--benchmark-config",
        type=Path,
        help="Include only manifests matching this config file's SHA-256",
    )
    compare_e02 = subparsers.add_parser(
        "compare-e02", help="Compare E02 budgets from a summarized runs CSV"
    )
    compare_e02.add_argument(
        "--runs-csv",
        type=Path,
        default=Path("reports/e02_batch_tokens/runs.csv"),
    )
    compare_e02.add_argument(
        "--output-dir", type=Path, default=Path("reports/e02_batch_tokens")
    )
    compare_e02.add_argument("--reference-budget", type=int, default=8192)
    compare_e04 = subparsers.add_parser(
        "compare-e04", help="Compare paired E04 prefix-cache OFF/ON runs"
    )
    compare_e04.add_argument(
        "--runs-csv",
        type=Path,
        default=Path("reports/e04_prefix_cache/runs.csv"),
    )
    compare_e04.add_argument(
        "--output-dir", type=Path, default=Path("reports/e04_prefix_cache")
    )
    diagnose_e04 = subparsers.add_parser(
        "diagnose-e04", help="Measure paired E04 generated-output overlap"
    )
    diagnose_e04.add_argument(
        "--runs-csv",
        type=Path,
        default=Path("reports/e04_prefix_cache/runs.csv"),
    )
    diagnose_e04.add_argument(
        "--output-dir", type=Path, default=Path("reports/e04_prefix_cache")
    )
    run_e04_canary_parser = subparsers.add_parser(
        "run-e04-canary", help="Run the fixed E04 correctness canary"
    )
    run_e04_canary_parser.add_argument("--state", choices=("off", "on"), required=True)
    run_e04_canary_parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("datasets/e04_correctness_canary.json"),
    )
    run_e04_canary_parser.add_argument(
        "--result-root",
        type=Path,
        default=Path("artifacts/results/e04_correctness_canary"),
    )
    run_e04_canary_parser.add_argument(
        "--base-url", default="http://127.0.0.1:8000"
    )
    run_e04_canary_parser.add_argument(
        "--served-model-name", default="qwen2.5-3b-instruct"
    )
    compare_e04_canary_parser = subparsers.add_parser(
        "compare-e04-canary", help="Compare latest E04 OFF/ON canary results"
    )
    compare_e04_canary_parser.add_argument(
        "--result-root",
        type=Path,
        default=Path("artifacts/results/e04_correctness_canary"),
    )
    compare_e04_canary_parser.add_argument(
        "--output-dir", type=Path, default=Path("reports/e04_prefix_cache")
    )
    compare_e05 = subparsers.add_parser(
        "compare-e05", help="Compare paired E05 BF16/FP8 performance runs"
    )
    compare_e05.add_argument(
        "--runs-csv",
        type=Path,
        default=Path("reports/e05_kv_cache/runs.csv"),
    )
    compare_e05.add_argument(
        "--output-dir", type=Path, default=Path("reports/e05_kv_cache")
    )
    capacity_e05 = subparsers.add_parser(
        "capacity-e05", help="Parse E05 KV capacity from server startup logs"
    )
    capacity_e05.add_argument(
        "--manifest-dir", type=Path, default=Path("artifacts/env")
    )
    capacity_e05.add_argument(
        "--output-dir", type=Path, default=Path("reports/e05_kv_cache")
    )
    run_e05_quality_parser = subparsers.add_parser(
        "run-e05-quality", help="Run the fixed 50-case E05 quality set"
    )
    run_e05_quality_parser.add_argument(
        "--state", choices=("bf16", "fp8"), required=True
    )
    run_e05_quality_parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("datasets/e05_ai_infra_quality.json"),
    )
    run_e05_quality_parser.add_argument(
        "--result-root",
        type=Path,
        default=Path("artifacts/results/e05_quality"),
    )
    run_e05_quality_parser.add_argument(
        "--base-url", default="http://127.0.0.1:8000"
    )
    run_e05_quality_parser.add_argument(
        "--served-model-name", default="qwen2.5-3b-instruct"
    )
    compare_e05_quality_parser = subparsers.add_parser(
        "compare-e05-quality", help="Compare latest BF16/FP8 E05 quality runs"
    )
    compare_e05_quality_parser.add_argument(
        "--result-root",
        type=Path,
        default=Path("artifacts/results/e05_quality"),
    )
    compare_e05_quality_parser.add_argument(
        "--output-dir", type=Path, default=Path("reports/e05_kv_cache")
    )
    human_e05 = subparsers.add_parser(
        "summarize-e05-human-review", help="Unblind and summarize E05 human scores"
    )
    human_e05.add_argument(
        "--review-csv",
        type=Path,
        default=Path("reports/e05_kv_cache/human_review.csv"),
    )
    human_e05.add_argument(
        "--review-key",
        type=Path,
        default=Path("reports/e05_kv_cache/human_review_key.json"),
    )
    human_e05.add_argument(
        "--output-dir", type=Path, default=Path("reports/e05_kv_cache")
    )
    compare_e06 = subparsers.add_parser(
        "compare-e06", help="Compare the four-cell E06 factorial experiment"
    )
    compare_e06.add_argument(
        "--runs-csv",
        type=Path,
        default=Path("reports/e06_combined/runs.csv"),
    )
    compare_e06.add_argument(
        "--output-dir", type=Path, default=Path("reports/e06_combined")
    )
    run_e06_canary_parser = subparsers.add_parser(
        "run-e06-canary", help="Run one E06 factorial correctness canary"
    )
    run_e06_canary_parser.add_argument(
        "--state",
        choices=("bt8192_off", "bt2048_off", "bt8192_on", "bt2048_on"),
        required=True,
    )
    run_e06_canary_parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("datasets/e04_correctness_canary.json"),
    )
    run_e06_canary_parser.add_argument(
        "--result-root",
        type=Path,
        default=Path("artifacts/results/e06_correctness_canary"),
    )
    run_e06_canary_parser.add_argument(
        "--base-url", default="http://127.0.0.1:8000"
    )
    run_e06_canary_parser.add_argument(
        "--served-model-name", default="qwen2.5-3b-instruct"
    )
    compare_e06_canary_parser = subparsers.add_parser(
        "compare-e06-canary", help="Compare all four E06 correctness canaries"
    )
    compare_e06_canary_parser.add_argument(
        "--result-root",
        type=Path,
        default=Path("artifacts/results/e06_correctness_canary"),
    )
    compare_e06_canary_parser.add_argument(
        "--output-dir", type=Path, default=Path("reports/e06_combined")
    )
    audit_e01_e06_parser = subparsers.add_parser(
        "audit-e01-e06", help="Audit all committed E01-E06 evidence"
    )
    audit_e01_e06_parser.add_argument("--root", type=Path, default=Path("."))
    audit_e01_e06_parser.add_argument(
        "--output-dir", type=Path, default=Path("reports/e01_e06")
    )
    prepare_e07 = subparsers.add_parser(
        "prepare-e07-data", help="Build and audit grouped E07 SFT splits"
    )
    prepare_e07.add_argument(
        "--source",
        type=Path,
        default=Path("datasets/e07_ai_infra_sft_source.json"),
    )
    prepare_e07.add_argument(
        "--test",
        type=Path,
        default=Path("datasets/e05_ai_infra_quality.json"),
    )
    prepare_e07.add_argument(
        "--output-dir", type=Path, default=Path("datasets/e07_sft")
    )
    render_e07_train = subparsers.add_parser(
        "render-e07-train", help="Validate and render one E07 training invocation"
    )
    render_e07_train.add_argument("config", type=Path)
    render_e07_train.add_argument("--model-path", type=Path, required=True)
    train_e07 = subparsers.add_parser(
        "train-e07", help="Run one controlled E07 QLoRA training profile"
    )
    train_e07.add_argument("config", type=Path)
    train_e07.add_argument("--model-path", type=Path, required=True)
    inspect_e07 = subparsers.add_parser(
        "inspect-e07-adapter", help="Validate an E07 Adapter and write its report"
    )
    inspect_e07.add_argument(
        "--adapter-dir", type=Path, default=Path("artifacts/adapters/e07/rank8")
    )
    inspect_e07.add_argument("--expected-rank", type=int, default=8)
    inspect_e07.add_argument(
        "--output-dir", type=Path, default=Path("reports/e07_lora")
    )
    run_e07_quality_parser = subparsers.add_parser(
        "run-e07-quality", help="Run the fixed E07 Base/LoRA quality set"
    )
    run_e07_quality_parser.add_argument(
        "--state", choices=("base", "lora"), required=True
    )
    run_e07_quality_parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("datasets/e05_ai_infra_quality.json"),
    )
    run_e07_quality_parser.add_argument(
        "--result-root",
        type=Path,
        default=Path("artifacts/results/e07_quality"),
    )
    compare_e07_quality_parser = subparsers.add_parser(
        "compare-e07-quality", help="Compare E07 Base and LoRA quality"
    )
    compare_e07_quality_parser.add_argument(
        "--result-root",
        type=Path,
        default=Path("artifacts/results/e07_quality"),
    )
    compare_e07_quality_parser.add_argument(
        "--output-dir", type=Path, default=Path("reports/e07_lora")
    )
    human_e07 = subparsers.add_parser(
        "summarize-e07-human-review", help="Unblind and summarize E07 human review"
    )
    human_e07.add_argument(
        "--review-csv",
        type=Path,
        default=Path("reports/e07_lora/human_review.csv"),
    )
    human_e07.add_argument(
        "--review-key",
        type=Path,
        default=Path("reports/e07_lora/human_review_key.json"),
    )
    human_e07.add_argument(
        "--output-dir", type=Path, default=Path("reports/e07_lora")
    )
    compare_e07_parser = subparsers.add_parser(
        "compare-e07", help="Compare paired E07 Base/LoRA performance runs"
    )
    compare_e07_parser.add_argument(
        "--runs-csv",
        type=Path,
        default=Path("reports/e07_lora/runs.csv"),
    )
    compare_e07_parser.add_argument(
        "--output-dir", type=Path, default=Path("reports/e07_lora")
    )
    finalize_e07_parser = subparsers.add_parser(
        "finalize-e07", help="Apply all E07 Adapter, quality, and cost gates"
    )
    finalize_e07_parser.add_argument(
        "--output-dir", type=Path, default=Path("reports/e07_lora")
    )
    readiness_e07_parser = subparsers.add_parser(
        "audit-e07-readiness", help="Audit E07 code and protocol without a GPU"
    )
    readiness_e07_parser.add_argument("--root", type=Path, default=Path("."))
    readiness_e07_parser.add_argument(
        "--output-dir", type=Path, default=Path("reports/e07_lora")
    )
    return parser


def _doctor(as_json: bool) -> int:
    checks = run_doctor()
    if as_json:
        print(json.dumps(checks_as_dict(checks), ensure_ascii=False, indent=2))
    else:
        for check in checks:
            status = "PASS" if check.passed else ("WARN" if not check.required else "FAIL")
            print(f"[{status}] {check.name}: {check.detail}")
    return 0 if all(check.passed for check in checks if check.required) else 1


def _effective_config(config: BenchmarkConfig) -> dict[str, object]:
    payload = asdict(config)
    for key, value in list(payload.items()):
        if isinstance(value, Path):
            payload[key] = str(value)
    return payload


def _effective_serve_config(config: ServeConfig) -> dict[str, object]:
    payload = asdict(config)
    for key, value in list(payload.items()):
        if isinstance(value, Path):
            payload[key] = str(value)
    return payload


def _load_serve_config(
    config_path: Path,
    model_path: Path | None,
    adapter_path: Path | None = None,
    validate_adapter: bool = False,
) -> ServeConfig:
    config = ServeConfig.from_file(config_path)
    if model_path is not None:
        config = config.with_local_model(model_path)
    if adapter_path is not None:
        config = config.with_lora_adapter(adapter_path)
    elif validate_adapter and config.enable_lora:
        config = config.with_lora_adapter()
    return config


def _write_active_server(config: ServeConfig, pid: int) -> dict[str, object]:
    marker = {
        "schema_version": 1,
        "profile": config.profile_name,
        "server_config": str(config.source_path),
        "server_config_sha256": config.source_sha256,
        "pid": pid,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "lora_name": config.lora_name,
        "lora_path": str(config.lora_path) if config.lora_path else None,
        "lora_manifest_sha256": config.lora_manifest_sha256,
        "lora_weights_sha256": config.lora_weights_sha256,
    }
    write_json(marker, ACTIVE_SERVER_PATH)
    return marker


def _clear_active_server(pid: int) -> None:
    try:
        marker = json.loads(ACTIVE_SERVER_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    if isinstance(marker, dict) and marker.get("pid") == pid:
        ACTIVE_SERVER_PATH.unlink(missing_ok=True)


def _verify_active_server(config: BenchmarkConfig) -> dict[str, object]:
    try:
        marker = json.loads(ACTIVE_SERVER_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(
            "No valid controlled server marker; start the matching server with "
            "a make serve-* target"
        ) from exc
    if not isinstance(marker, dict):
        raise ConfigError("Controlled server marker must contain a JSON object")

    expected = (config.server_profile, config.server_config_sha256)
    actual = (marker.get("profile"), marker.get("server_config_sha256"))
    if actual != expected:
        raise ConfigError(
            "Active server does not match benchmark config: "
            f"expected {expected[0]} ({expected[1]}), "
            f"found {actual[0]} ({actual[1]})"
        )

    pid = marker.get("pid")
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
        raise ConfigError("Controlled server marker contains an invalid pid")
    try:
        os.kill(pid, 0)
    except ProcessLookupError as exc:
        raise ConfigError(
            f"Controlled server marker is stale; process {pid} is not running"
        ) from exc
    except PermissionError:
        pass
    return marker


def _run_server(
    config_path: Path,
    model_path: Path | None = None,
    adapter_path: Path | None = None,
) -> int:
    config = _load_serve_config(
        config_path,
        model_path,
        adapter_path,
        validate_adapter=True,
    )
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    command = build_serve_command(config)
    environment_overrides = build_serve_environment(config)
    log_path = Path("artifacts/server") / f"{timestamp}-{config.profile_name}.log"
    manifest_path = (
        Path("artifacts/env") / f"{timestamp}-server-{config.profile_name}.json"
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "kind": "server",
        "profile": config.profile_name,
        "server_config": str(config.source_path),
        "server_config_sha256": config.source_sha256,
        "effective_config": _effective_serve_config(config),
        "environment": collect_environment(),
        "environment_overrides": environment_overrides,
        "command": render_shell_command(command, environment_overrides),
        "log": str(log_path),
        "returncode": None,
        "stopped_by_user": False,
        "unexpected_exit": None,
    }
    write_json(manifest, manifest_path)
    print(f"Server manifest: {manifest_path}")
    print(f"Server log: {log_path}")
    print(render_shell_command(command, environment_overrides))

    process: subprocess.Popen[str] | None = None
    try:
        with log_path.open("w", encoding="utf-8") as log_handle:
            process_environment = os.environ.copy()
            process_environment.update(environment_overrides)
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=process_environment,
            )
            _write_active_server(config, process.pid)
            if process.stdout is not None:
                for line in process.stdout:
                    print(line, end="")
                    log_handle.write(line)
                    log_handle.flush()
            returncode = process.wait()
    except KeyboardInterrupt:
        manifest["stopped_by_user"] = True
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                returncode = process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                returncode = process.wait()
        else:
            returncode = 130
    except OSError as exc:
        returncode = 127
        manifest["execution_error"] = str(exc)
    finally:
        if process is not None:
            _clear_active_server(process.pid)

    manifest["returncode"] = returncode
    manifest["unexpected_exit"] = not manifest["stopped_by_user"]
    write_json(manifest, manifest_path)
    if manifest["unexpected_exit"] and returncode == 0:
        print(
            "Model server exited without a user stop; treating the clean process "
            "exit as a service failure",
            file=sys.stderr,
        )
        return 1
    return returncode


def _execute_benchmark(
    config: BenchmarkConfig, repetitions: list[int] | None = None
) -> int:
    active_server = _verify_active_server(config)
    config.result_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    selected_repetitions = (
        list(range(1, config.repetitions + 1))
        if repetitions is None
        else list(repetitions)
    )
    expected_repetitions = set(range(1, config.repetitions + 1))
    if (
        not selected_repetitions
        or len(selected_repetitions) != len(set(selected_repetitions))
        or not set(selected_repetitions).issubset(expected_repetitions)
    ):
        raise ConfigError(
            "Benchmark repetitions must be unique values between 1 and "
            f"{config.repetitions}"
        )
    commands = [
        build_benchmark_command(config, repetition=index)
        for index in selected_repetitions
    ]
    prewarm_commands = [
        build_prewarm_command(config, repetition=index)
        for index in selected_repetitions
    ]
    manifest = {
        "schema_version": 1,
        "profile": config.profile_name,
        "benchmark_config": str(config.source_path),
        "benchmark_config_sha256": config.source_sha256,
        "server_config": str(config.server_config_path),
        "server_config_sha256": config.server_config_sha256,
        "active_server": active_server,
        "effective_config": _effective_config(config),
        "environment": collect_environment(),
        "commands": [render_shell_command(command) for command in commands],
        "prewarm_commands": [
            render_shell_command(command) if command is not None else None
            for command in prewarm_commands
        ],
        "requested_repetitions": selected_repetitions,
        "runs": [],
    }
    manifest_path = Path("artifacts/env") / f"{timestamp}-{config.profile_name}.json"
    write_json(manifest, manifest_path)
    print(f"Environment manifest: {manifest_path}")

    run_items = list(
        zip(selected_repetitions, prewarm_commands, commands, strict=True)
    )
    for position, (index, prewarm_command, command) in enumerate(
        run_items, start=1
    ):
        print(f"Running repetition {index}/{config.repetitions}")
        if position > 1:
            try:
                _verify_active_server(config)
            except ConfigError as exc:
                manifest["runs"].append(
                    {
                        "repetition": index,
                        "effective_seed": config.seed_for_repetition(index),
                        "returncode": 2,
                        "execution_error": str(exc),
                        "failed_stage": "server_check",
                        "prewarm_returncode": None,
                        "result_files": [],
                    }
                )
                write_json(manifest, manifest_path)
                print(str(exc), file=sys.stderr)
                return 2
        prewarm_returncode = None
        prewarm_execution_error = None
        if prewarm_command is not None:
            print("Running isolated prewarm workload")
            print(render_shell_command(prewarm_command))
            try:
                prewarm_completed = subprocess.run(prewarm_command, check=False)
            except OSError as exc:
                prewarm_execution_error = str(exc)
                prewarm_completed = subprocess.CompletedProcess(prewarm_command, 127)
            prewarm_returncode = prewarm_completed.returncode
            if prewarm_returncode != 0:
                run_record = {
                    "repetition": index,
                    "effective_seed": config.seed_for_repetition(index),
                    "returncode": prewarm_returncode,
                    "execution_error": prewarm_execution_error,
                    "failed_stage": "prewarm",
                    "prewarm_returncode": prewarm_returncode,
                    "result_files": [],
                }
                manifest["runs"].append(run_record)
                write_json(manifest, manifest_path)
                print(
                    f"Prewarm for repetition {index} failed with exit code "
                    f"{prewarm_returncode}",
                    file=sys.stderr,
                )
                return prewarm_returncode

        metrics_before = None
        metrics_before_path = None
        if config.dataset_name == "prefix_repetition":
            metrics_before_path = (
                config.result_dir
                / f"metrics-before-{config.profile_name}-r{index}-{timestamp}.prom"
            )
            try:
                metrics_text = fetch_metrics(config.base_url)
                metrics_before_path.write_text(metrics_text, encoding="utf-8")
                metrics_before = prefix_snapshot(metrics_text)
            except (MetricsError, OSError) as exc:
                run_record = {
                    "repetition": index,
                    "effective_seed": config.seed_for_repetition(index),
                    "returncode": 2,
                    "execution_error": str(exc),
                    "failed_stage": "metrics_before",
                    "prewarm_returncode": prewarm_returncode,
                    "result_files": [],
                }
                manifest["runs"].append(run_record)
                write_json(manifest, manifest_path)
                print(str(exc), file=sys.stderr)
                return 2

        print(render_shell_command(command))
        telemetry_path = (
            config.result_dir
            / f"telemetry-{config.profile_name}-r{index}-{timestamp}.csv"
        )
        result_files_before = set(config.result_dir.glob("*.json"))
        execution_error = None
        with NvidiaSmiSampler(telemetry_path):
            try:
                completed = subprocess.run(command, check=False)
            except OSError as exc:
                execution_error = str(exc)
                completed = subprocess.CompletedProcess(command, 127)
        result_files_after = set(config.result_dir.glob("*.json"))
        new_result_files = sorted(result_files_after - result_files_before)
        evidence_error = None
        prefix_metrics = None
        metrics_after_path = None
        if config.dataset_name == "prefix_repetition":
            metrics_after_path = (
                config.result_dir
                / f"metrics-after-{config.profile_name}-r{index}-{timestamp}.prom"
            )
            try:
                metrics_text = fetch_metrics(config.base_url)
                metrics_after_path.write_text(metrics_text, encoding="utf-8")
                metrics_after = prefix_snapshot(metrics_text)
                assert metrics_before is not None
                prefix_metrics = prefix_delta(metrics_before, metrics_after)
                if config.server_prefix_caching_enabled and not (
                    isinstance(prefix_metrics.get("query_tokens"), (int, float))
                    and prefix_metrics["query_tokens"] > 0
                    and isinstance(prefix_metrics.get("hit_tokens"), (int, float))
                ):
                    raise MetricsError(
                        "Prefix caching is enabled but token counters are unavailable"
                    )
            except (MetricsError, OSError) as exc:
                evidence_error = str(exc)

        output_hash = None
        if completed.returncode == 0 and len(new_result_files) == 1:
            try:
                output_hash = generated_texts_sha256(new_result_files[0])
            except ResultError as exc:
                if config.dataset_name == "prefix_repetition":
                    evidence_error = str(exc)
        elif (
            completed.returncode == 0
            and config.dataset_name == "prefix_repetition"
        ):
            evidence_error = (
                "A successful prefix benchmark must create exactly one result "
                f"file; found {len(new_result_files)}"
            )
        run_record = {
            "repetition": index,
            "effective_seed": config.seed_for_repetition(index),
            "returncode": completed.returncode,
            "execution_error": execution_error,
            "evidence_error": evidence_error,
            "prewarm_returncode": prewarm_returncode,
            "telemetry": str(telemetry_path),
            "telemetry_summary": summarize_telemetry(telemetry_path),
            "result_files": [str(path) for path in new_result_files],
            "metrics_before": (
                str(metrics_before_path) if metrics_before_path is not None else None
            ),
            "metrics_after": (
                str(metrics_after_path) if metrics_after_path is not None else None
            ),
            "prefix_metrics": prefix_metrics,
            "generated_texts_sha256": output_hash,
        }
        manifest["runs"].append(run_record)
        write_json(manifest, manifest_path)
        if completed.returncode != 0:
            print(
                f"Benchmark repetition {index} failed with exit code "
                f"{completed.returncode}: {execution_error or 'see benchmark output'}",
                file=sys.stderr,
            )
            return completed.returncode
        if evidence_error is not None:
            print(
                f"Evidence collection for repetition {index} failed: "
                f"{evidence_error}",
                file=sys.stderr,
            )
            return 2
        if position < len(run_items) and config.cooldown_seconds:
            time.sleep(config.cooldown_seconds)
    return 0


def _load_benchmark_config(
    config_path: Path, tokenizer_path: Path | None
) -> BenchmarkConfig:
    config = BenchmarkConfig.from_file(config_path)
    if tokenizer_path is not None:
        config = config.with_local_tokenizer(tokenizer_path)
    return config


def _run_benchmark(config_path: Path, tokenizer_path: Path | None) -> int:
    return _execute_benchmark(
        _load_benchmark_config(config_path, tokenizer_path)
    )


def _run_matrix(
    config_path: Path,
    selected: list[str],
    tokenizer_path: Path | None,
    skip_completed: bool,
) -> int:
    matrix = BenchmarkMatrix.from_file(config_path)
    configs = list(matrix.configs)
    if tokenizer_path is not None:
        configs = [
            config.with_local_tokenizer(tokenizer_path) for config in configs
        ]
    if selected:
        selected_set = set(selected)
        known = {config.profile_name for config in configs}
        unknown = sorted(selected_set - known)
        if unknown:
            raise ConfigError(f"Unknown matrix profile(s): {', '.join(unknown)}")
        configs = [config for config in configs if config.profile_name in selected_set]

    work_items = [
        (config, list(range(1, config.repetitions + 1))) for config in configs
    ]
    if skip_completed and Path("artifacts/env").is_dir():
        records = load_records_from_manifests("artifacts/env")
        pending: list[tuple[BenchmarkConfig, list[int]]] = []
        for config in configs:
            repetitions = {
                record.repetition
                for record in records
                if record.profile == config.profile_name
                and record.benchmark_config_sha256 == config.source_sha256
                and record.server_config_sha256 == config.server_config_sha256
                and record.input_len == config.input_len
                and record.output_len == config.output_len
                and record.max_concurrency == config.max_concurrency
                and record.completed + record.failed == config.num_prompts
                and record.valid
            }
            expected = set(range(1, config.repetitions + 1))
            if expected.issubset(repetitions):
                print(
                    f"Skipping completed matrix profile: {config.profile_name} "
                    f"({config.repetitions}/{config.repetitions} valid repetitions)"
                )
            else:
                missing = sorted(expected - repetitions)
                if repetitions:
                    completed = ",".join(str(value) for value in sorted(repetitions))
                    missing_text = ",".join(str(value) for value in missing)
                    print(
                        f"Resuming matrix profile: {config.profile_name}; "
                        f"valid repetitions={completed}, missing={missing_text}"
                    )
                pending.append((config, missing))
        work_items = pending

    for index, (config, repetitions) in enumerate(work_items, start=1):
        print(f"Matrix profile {index}/{len(work_items)}: {config.profile_name}")
        returncode = _execute_benchmark(config, repetitions=repetitions)
        if returncode != 0:
            return returncode
        if index < len(work_items) and config.cooldown_seconds:
            time.sleep(config.cooldown_seconds)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "doctor":
            return _doctor(args.as_json)
        if args.command == "collect-env":
            output = write_json(collect_environment(), args.output)
            print(output)
            return 0
        if args.command == "render-serve":
            config = _load_serve_config(
                args.config, args.model_path, args.adapter_path
            )
            print(
                render_shell_command(
                    build_serve_command(config), build_serve_environment(config)
                )
            )
            return 0
        if args.command == "run-serve":
            return _run_server(args.config, args.model_path, args.adapter_path)
        if args.command == "render-bench":
            config = _load_benchmark_config(args.config, args.tokenizer_path)
            print(
                render_shell_command(
                    build_benchmark_command(config, repetition=args.repetition)
                )
            )
            return 0
        if args.command == "run-bench":
            return _run_benchmark(args.config, args.tokenizer_path)
        if args.command == "render-matrix":
            matrix = BenchmarkMatrix.from_file(args.config)
            configs = list(matrix.configs)
            if args.tokenizer_path is not None:
                configs = [
                    config.with_local_tokenizer(args.tokenizer_path)
                    for config in configs
                ]
            for config in configs:
                command = build_benchmark_command(config, repetition=1)
                print(f"[{config.profile_name}] {render_shell_command(command)}")
            return 0
        if args.command == "run-matrix":
            return _run_matrix(
                args.config,
                args.only,
                args.tokenizer_path,
                args.skip_completed,
            )
        if args.command == "summarize":
            benchmark_hash = (
                config_sha256(args.benchmark_config)
                if args.benchmark_config is not None
                else None
            )
            records = load_records_from_manifests(
                args.manifest_dir,
                profile_prefix=args.profile_prefix,
                benchmark_config_sha256=benchmark_hash,
            )
            csv_path, markdown_path = write_reports(records, args.output_dir)
            print(csv_path)
            print(markdown_path)
            return 0
        if args.command == "compare-e02":
            csv_path, markdown_path = write_e02_comparison(
                args.runs_csv,
                args.output_dir,
                reference_budget=args.reference_budget,
            )
            print(csv_path)
            print(markdown_path)
            return 0
        if args.command == "compare-e04":
            csv_path, markdown_path = write_e04_comparison(
                args.runs_csv,
                args.output_dir,
            )
            print(csv_path)
            print(markdown_path)
            return 0
        if args.command == "diagnose-e04":
            csv_path, markdown_path = write_e04_output_diagnostics(
                args.runs_csv,
                args.output_dir,
            )
            print(csv_path)
            print(markdown_path)
            return 0
        if args.command == "run-e04-canary":
            output_path, valid = run_e04_canary(
                state=args.state,
                dataset_path=args.dataset,
                result_root=args.result_root,
                base_url=args.base_url,
                served_model_name=args.served_model_name,
            )
            print(output_path)
            return 0 if valid else 2
        if args.command == "compare-e04-canary":
            json_path, markdown_path, passed = compare_e04_canary(
                result_root=args.result_root,
                output_dir=args.output_dir,
            )
            print(json_path)
            print(markdown_path)
            return 0 if passed else 2
        if args.command == "compare-e05":
            csv_path, markdown_path = write_e05_comparison(
                args.runs_csv, args.output_dir
            )
            print(csv_path)
            print(markdown_path)
            return 0
        if args.command == "capacity-e05":
            json_path, markdown_path = write_e05_capacity_report(
                args.manifest_dir, args.output_dir
            )
            print(json_path)
            print(markdown_path)
            return 0
        if args.command == "run-e05-quality":
            output_path, valid = run_e05_quality(
                state=args.state,
                dataset_path=args.dataset,
                result_root=args.result_root,
                base_url=args.base_url,
                served_model_name=args.served_model_name,
            )
            print(output_path)
            return 0 if valid else 2
        if args.command == "compare-e05-quality":
            json_path, markdown_path, passed = compare_e05_quality(
                result_root=args.result_root,
                output_dir=args.output_dir,
            )
            print(json_path)
            print(markdown_path)
            return 0 if passed else 2
        if args.command == "summarize-e05-human-review":
            json_path, markdown_path, passed = summarize_e05_human_review(
                args.review_csv,
                args.review_key,
                args.output_dir,
            )
            print(json_path)
            print(markdown_path)
            return 0 if passed else 2
        if args.command == "compare-e06":
            csv_path, markdown_path = write_e06_comparison(
                args.runs_csv, args.output_dir
            )
            print(csv_path)
            print(markdown_path)
            return 0
        if args.command == "run-e06-canary":
            output_path, valid = run_e06_canary(
                state=args.state,
                dataset_path=args.dataset,
                result_root=args.result_root,
                base_url=args.base_url,
                served_model_name=args.served_model_name,
            )
            print(output_path)
            return 0 if valid else 2
        if args.command == "compare-e06-canary":
            json_path, markdown_path, passed = compare_e06_canary(
                result_root=args.result_root,
                output_dir=args.output_dir,
            )
            print(json_path)
            print(markdown_path)
            return 0 if passed else 2
        if args.command == "audit-e01-e06":
            json_path, markdown_path, passed = write_e01_e06_audit(
                repo_root=args.root,
                output_dir=args.output_dir,
            )
            print(json_path)
            print(markdown_path)
            return 0 if passed else 2
        if args.command == "prepare-e07-data":
            paths = prepare_e07_dataset(args.source, args.test, args.output_dir)
            for path in paths:
                print(path)
            return 0
        if args.command == "render-e07-train":
            config = E07TrainingConfig.from_file(args.config)
            command = [
                sys.executable,
                "-m",
                "qwen_serve_lab.cli",
                "train-e07",
                str(args.config),
                "--model-path",
                str(args.model_path),
            ]
            print(f"# profile={config.profile_name} rank={config.rank}")
            print(render_shell_command(command, {"PYTHONPATH": "src"}))
            return 0
        if args.command == "train-e07":
            manifest_path = run_e07_training(args.config, args.model_path)
            print(manifest_path)
            return 0
        if args.command == "inspect-e07-adapter":
            json_path, markdown_path, passed = write_e07_adapter_report(
                args.adapter_dir, args.output_dir, args.expected_rank
            )
            print(json_path)
            print(markdown_path)
            return 0 if passed else 2
        if args.command == "run-e07-quality":
            output_path, valid = run_e07_quality(
                args.state, args.dataset, args.result_root
            )
            print(output_path)
            return 0 if valid else 2
        if args.command == "compare-e07-quality":
            json_path, markdown_path, passed = compare_e07_quality(
                args.result_root, args.output_dir
            )
            print(json_path)
            print(markdown_path)
            return 0 if passed else 2
        if args.command == "summarize-e07-human-review":
            json_path, markdown_path, passed = summarize_e07_human_review(
                args.review_csv, args.review_key, args.output_dir
            )
            print(json_path)
            print(markdown_path)
            return 0 if passed else 2
        if args.command == "compare-e07":
            csv_path, markdown_path, passed = write_e07_comparison(
                args.runs_csv, args.output_dir
            )
            print(csv_path)
            print(markdown_path)
            return 0 if passed else 2
        if args.command == "finalize-e07":
            json_path, markdown_path, passed = write_e07_final_report(
                args.output_dir
            )
            print(json_path)
            print(markdown_path)
            return 0 if passed else 2
        if args.command == "audit-e07-readiness":
            json_path, markdown_path, passed = write_e07_readiness_report(
                args.root, args.output_dir
            )
            print(json_path)
            print(markdown_path)
            return 0 if passed else 2
    except (ConfigError, MetricsError, ResultError) as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
