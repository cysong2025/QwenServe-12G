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
    build_serve_command,
    build_serve_environment,
    render_shell_command,
)
from qwen_serve_lab.config import (
    BenchmarkConfig,
    BenchmarkMatrix,
    ConfigError,
    ServeConfig,
)
from qwen_serve_lab.environment import (
    checks_as_dict,
    collect_environment,
    run_doctor,
    write_json,
)
from qwen_serve_lab.telemetry import NvidiaSmiSampler, summarize_telemetry
from qwen_serve_lab.results import (
    ResultError,
    load_records_from_manifests,
    write_reports,
)


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

    run_serve = subparsers.add_parser(
        "run-serve", help="Run vLLM while capturing logs and a server manifest"
    )
    run_serve.add_argument("config", type=Path)
    run_serve.add_argument("--model-path", type=Path)

    benchmark = subparsers.add_parser(
        "render-bench", help="Render one validated vLLM benchmark command"
    )
    benchmark.add_argument("config", type=Path)
    benchmark.add_argument("--repetition", type=int, default=1)

    run = subparsers.add_parser(
        "run-bench", help="Snapshot the environment and execute benchmark repetitions"
    )
    run.add_argument("config", type=Path)

    render_matrix = subparsers.add_parser(
        "render-matrix", help="Render every effective benchmark in a matrix"
    )
    render_matrix.add_argument("config", type=Path)

    run_matrix = subparsers.add_parser(
        "run-matrix", help="Execute a benchmark matrix with manifests and telemetry"
    )
    run_matrix.add_argument("config", type=Path)
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


def _load_serve_config(config_path: Path, model_path: Path | None) -> ServeConfig:
    config = ServeConfig.from_file(config_path)
    if model_path is not None:
        config = config.with_local_model(model_path)
    return config


def _run_server(config_path: Path, model_path: Path | None = None) -> int:
    config = _load_serve_config(config_path, model_path)
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

    manifest["returncode"] = returncode
    write_json(manifest, manifest_path)
    return returncode


def _execute_benchmark(config: BenchmarkConfig) -> int:
    config.result_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    commands = [
        build_benchmark_command(config, repetition=index)
        for index in range(1, config.repetitions + 1)
    ]
    manifest = {
        "schema_version": 1,
        "profile": config.profile_name,
        "benchmark_config": str(config.source_path),
        "benchmark_config_sha256": config.source_sha256,
        "server_config": str(config.server_config_path),
        "server_config_sha256": config.server_config_sha256,
        "effective_config": _effective_config(config),
        "environment": collect_environment(),
        "commands": [render_shell_command(command) for command in commands],
        "runs": [],
    }
    manifest_path = Path("artifacts/env") / f"{timestamp}-{config.profile_name}.json"
    write_json(manifest, manifest_path)
    print(f"Environment manifest: {manifest_path}")

    for index, command in enumerate(commands, start=1):
        print(f"Running repetition {index}/{config.repetitions}")
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
        run_record = {
            "repetition": index,
            "returncode": completed.returncode,
            "execution_error": execution_error,
            "telemetry": str(telemetry_path),
            "telemetry_summary": summarize_telemetry(telemetry_path),
            "result_files": [str(path) for path in new_result_files],
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
        if index < config.repetitions and config.cooldown_seconds:
            time.sleep(config.cooldown_seconds)
    return 0


def _run_benchmark(config_path: Path) -> int:
    return _execute_benchmark(BenchmarkConfig.from_file(config_path))


def _run_matrix(config_path: Path, selected: list[str]) -> int:
    matrix = BenchmarkMatrix.from_file(config_path)
    configs = list(matrix.configs)
    if selected:
        selected_set = set(selected)
        known = {config.profile_name for config in configs}
        unknown = sorted(selected_set - known)
        if unknown:
            raise ConfigError(f"Unknown matrix profile(s): {', '.join(unknown)}")
        configs = [config for config in configs if config.profile_name in selected_set]

    for index, config in enumerate(configs, start=1):
        print(f"Matrix profile {index}/{len(configs)}: {config.profile_name}")
        returncode = _execute_benchmark(config)
        if returncode != 0:
            return returncode
        if index < len(configs) and config.cooldown_seconds:
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
            config = _load_serve_config(args.config, args.model_path)
            print(
                render_shell_command(
                    build_serve_command(config), build_serve_environment(config)
                )
            )
            return 0
        if args.command == "run-serve":
            return _run_server(args.config, args.model_path)
        if args.command == "render-bench":
            config = BenchmarkConfig.from_file(args.config)
            print(
                render_shell_command(
                    build_benchmark_command(config, repetition=args.repetition)
                )
            )
            return 0
        if args.command == "run-bench":
            return _run_benchmark(args.config)
        if args.command == "render-matrix":
            matrix = BenchmarkMatrix.from_file(args.config)
            for config in matrix.configs:
                command = build_benchmark_command(config, repetition=1)
                print(f"[{config.profile_name}] {render_shell_command(command)}")
            return 0
        if args.command == "run-matrix":
            return _run_matrix(args.config, args.only)
        if args.command == "summarize":
            records = load_records_from_manifests(
                args.manifest_dir, profile_prefix=args.profile_prefix
            )
            csv_path, markdown_path = write_reports(records, args.output_dir)
            print(csv_path)
            print(markdown_path)
            return 0
    except (ConfigError, ResultError) as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
