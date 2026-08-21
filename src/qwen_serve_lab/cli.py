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


def _write_active_server(config: ServeConfig, pid: int) -> dict[str, object]:
    marker = {
        "schema_version": 1,
        "profile": config.profile_name,
        "server_config": str(config.source_path),
        "server_config_sha256": config.source_sha256,
        "pid": pid,
        "started_at": datetime.now(timezone.utc).isoformat(),
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
    write_json(manifest, manifest_path)
    return returncode


def _execute_benchmark(config: BenchmarkConfig) -> int:
    active_server = _verify_active_server(config)
    config.result_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    commands = [
        build_benchmark_command(config, repetition=index)
        for index in range(1, config.repetitions + 1)
    ]
    prewarm_commands = [
        build_prewarm_command(config, repetition=index)
        for index in range(1, config.repetitions + 1)
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
        "runs": [],
    }
    manifest_path = Path("artifacts/env") / f"{timestamp}-{config.profile_name}.json"
    write_json(manifest, manifest_path)
    print(f"Environment manifest: {manifest_path}")

    for index, (prewarm_command, command) in enumerate(
        zip(prewarm_commands, commands, strict=True), start=1
    ):
        print(f"Running repetition {index}/{config.repetitions}")
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
        if index < config.repetitions and config.cooldown_seconds:
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

    if skip_completed and Path("artifacts/env").is_dir():
        records = load_records_from_manifests("artifacts/env")
        pending: list[BenchmarkConfig] = []
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
                pending.append(config)
        configs = pending

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
    except (ConfigError, MetricsError, ResultError) as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
