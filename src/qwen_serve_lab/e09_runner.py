from __future__ import annotations

import json
import time
from concurrent.futures import Future, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from qwen_serve_lab.config import ConfigError
from qwen_serve_lab.e08_runner import (
    _load_active_server,
    _run_one_request,
    build_prompt_token_ids,
    load_local_tokenizer,
    request_streaming_completion,
    summarize_e08_requests,
)
from qwen_serve_lab.e09_admission import (
    E09AdmissionController,
    E09Config,
    validate_e09_trace,
)
from qwen_serve_lab.environment import collect_environment
from qwen_serve_lab.results import ResultError
from qwen_serve_lab.telemetry import NvidiaSmiSampler, summarize_telemetry


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * percentile / 100
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = rank - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def load_e09_trace(path: str | Path, config: E09Config) -> dict[str, Any]:
    trace_path = Path(path)
    try:
        trace = json.loads(trace_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ResultError(f"Cannot read E09 trace {trace_path}: {exc}") from exc
    try:
        validate_e09_trace(trace, config)
    except ConfigError as exc:
        raise ResultError(str(exc)) from exc
    return trace


def _rejected_result(
    request_data: dict[str, Any],
    arrival_lag_ms: float,
    queue_wait_ms: float,
    reason: str,
) -> dict[str, Any]:
    return {
        **request_data,
        "admitted": False,
        "status": "rejected",
        "rejection_reason": reason,
        "dispatch_lag_ms": arrival_lag_ms,
        "arrival_lag_ms": arrival_lag_ms,
        "queue_wait_ms": queue_wait_ms,
        "request_started_lag_ms": None,
        "ttft_ms": None,
        "tpot_ms": None,
        "e2e_ms": None,
        "completion_tokens": 0,
        "prompt_tokens": None,
        "stream_events": 0,
        "finish_reason": None,
        "generated_chars": 0,
        "error": None,
        "token_shape_valid": None,
        "slo_met": False,
    }


def summarize_e09_requests(
    results: list[dict[str, Any]], duration_seconds: float
) -> dict[str, Any]:
    summary = summarize_e08_requests(results, duration_seconds)
    arrival_lags = [
        float(row["arrival_lag_ms"])
        for row in results
        if isinstance(row.get("arrival_lag_ms"), (int, float))
        and not isinstance(row.get("arrival_lag_ms"), bool)
    ]
    admitted_queue_waits = [
        float(row["queue_wait_ms"])
        for row in results
        if row.get("admitted") is True
        and isinstance(row.get("queue_wait_ms"), (int, float))
        and not isinstance(row.get("queue_wait_ms"), bool)
    ]
    rejected_queue_waits = [
        float(row["queue_wait_ms"])
        for row in results
        if row.get("admitted") is False
        and isinstance(row.get("queue_wait_ms"), (int, float))
        and not isinstance(row.get("queue_wait_ms"), bool)
    ]
    summary.update(
        {
            "p95_arrival_lag_ms": _percentile(arrival_lags, 95),
            "p50_admitted_queue_wait_ms": _percentile(admitted_queue_waits, 50),
            "p95_admitted_queue_wait_ms": _percentile(admitted_queue_waits, 95),
            "max_admitted_queue_wait_ms": (
                max(admitted_queue_waits) if admitted_queue_waits else None
            ),
            "p50_rejected_queue_wait_ms": _percentile(rejected_queue_waits, 50),
            "deadline_rejections": sum(
                row.get("rejection_reason") == "queue_deadline_expired"
                for row in results
            ),
        }
    )
    return summary


def _dispatch(
    executor: ThreadPoolExecutor,
    futures: dict[Future[dict[str, Any]], dict[str, float]],
    request_data: dict[str, Any],
    scheduled_at: float,
    arrival_lag_ms: float,
    queue_wait_ms: float,
    controller: E09AdmissionController,
    config: E09Config,
    tokenizer: Any,
    requester: Callable[..., dict[str, Any]],
) -> bool:
    lease = controller.try_acquire(
        str(request_data["id"]),
        str(request_data["workload"]),
        int(request_data["token_cost"]),
    )
    if lease is None:
        return False
    future = executor.submit(
        _run_one_request,
        request_data,
        scheduled_at,
        lease,
        controller,
        config,
        tokenizer,
        requester,
    )
    futures[future] = {
        "arrival_lag_ms": arrival_lag_ms,
        "queue_wait_ms": queue_wait_ms,
    }
    return True


def _collect_done(
    futures: dict[Future[dict[str, Any]], dict[str, float]],
    results: list[dict[str, Any]],
) -> None:
    for future in [candidate for candidate in futures if candidate.done()]:
        metadata = futures.pop(future)
        row = future.result()
        row["dispatch_lag_ms"] = metadata["arrival_lag_ms"]
        row["arrival_lag_ms"] = metadata["arrival_lag_ms"]
        row["queue_wait_ms"] = metadata["queue_wait_ms"]
        row["rejection_reason"] = None
        results.append(row)


def run_e09_cell(
    config_path: str | Path,
    profile_name: str,
    policy: str,
    repetition: int,
    tokenizer_path: str | Path,
    trace_root: str | Path = "artifacts/results/e09_deadline_admission/traces",
    result_root: str | Path = "artifacts/results/e09_deadline_admission/runs",
    active_server_path: str | Path = "artifacts/server/active.json",
    requester: Callable[..., dict[str, Any]] = request_streaming_completion,
    tokenizer_loader: Callable[[str | Path], Any] = load_local_tokenizer,
) -> tuple[Path, bool]:
    config = E09Config.from_file(config_path)
    profile = config.profile(profile_name)
    if policy not in config.policies:
        raise ConfigError(f"Unknown E09 policy: {policy}")
    if repetition < 1 or repetition > config.repetitions:
        raise ConfigError(f"E09 repetition must be between 1 and {config.repetitions}")
    marker = _load_active_server(config, active_server_path)
    trace_path = Path(trace_root) / f"{profile_name}-r{repetition}.json"
    trace = load_e09_trace(trace_path, config)
    tokenizer = tokenizer_loader(tokenizer_path)

    warmup_results: list[dict[str, Any]] = []
    for index in range(config.warmup_requests):
        workload = config.workloads[index % len(config.workloads)]
        prompt_seed = int(trace["seed"]) * 1000 + index
        try:
            warmup_result = requester(
                config.base_url,
                config.served_model_name,
                build_prompt_token_ids(tokenizer, workload.input_tokens, prompt_seed),
                workload.output_tokens,
                prompt_seed,
                config.request_timeout_seconds,
                None,
            )
            warmup_result["token_shape_valid"] = bool(
                warmup_result.get("prompt_tokens") == workload.input_tokens
                and warmup_result.get("completion_tokens") == workload.output_tokens
            )
            warmup_results.append(warmup_result)
        except ResultError as exc:
            warmup_results.append({"status": "failed", "error": str(exc)})
            break

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output_dir = Path(result_root) / profile_name / policy
    output_dir.mkdir(parents=True, exist_ok=True)
    for existing_path in output_dir.glob("*-e09-*.json"):
        try:
            existing = json.loads(existing_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if (
            isinstance(existing, dict)
            and existing.get("kind") == "e09_admission_run"
            and existing.get("experiment_config_sha256") == config.source_sha256
            and existing.get("repetition") == repetition
        ):
            raise ResultError(
                "E09 cell already has preserved evidence: "
                f"{existing_path}; do not overwrite or silently rerun it"
            )

    telemetry_path = output_dir / f"{timestamp}-telemetry.csv"
    controller = E09AdmissionController(policy, config)
    results: list[dict[str, Any]] = []
    futures: dict[Future[dict[str, Any]], dict[str, float]] = {}
    pending_queue: list[dict[str, Any]] = []
    requests = list(trace["requests"])
    next_request = 0
    trace_started = time.monotonic()
    with NvidiaSmiSampler(telemetry_path):
        executor = ThreadPoolExecutor(max_workers=config.client_max_workers)
        try:
            while next_request < len(requests) or pending_queue:
                _collect_done(futures, results)
                now = time.monotonic()
                while next_request < len(requests):
                    request_data = requests[next_request]
                    scheduled_at = trace_started + float(
                        request_data["arrival_offset_seconds"]
                    )
                    if scheduled_at > now:
                        break
                    arrival_lag_ms = max(0.0, (now - scheduled_at) * 1000)
                    next_request += 1
                    if policy == "deadline_aware":
                        pending_queue.append(
                            {
                                "request": request_data,
                                "scheduled_at": scheduled_at,
                                "arrival_lag_ms": arrival_lag_ms,
                            }
                        )
                        controller.observe_queue_depth(len(pending_queue))
                        continue
                    dispatched = _dispatch(
                        executor,
                        futures,
                        request_data,
                        scheduled_at,
                        arrival_lag_ms,
                        0.0,
                        controller,
                        config,
                        tokenizer,
                        requester,
                    )
                    if not dispatched:
                        results.append(
                            _rejected_result(
                                request_data,
                                arrival_lag_ms,
                                0.0,
                                "fixed_concurrency_limit",
                            )
                        )

                if policy == "deadline_aware" and pending_queue:
                    now = time.monotonic()
                    survivors: list[dict[str, Any]] = []
                    for item in pending_queue:
                        waited_ms = (now - float(item["scheduled_at"])) * 1000
                        if waited_ms >= config.deadline_max_queue_wait_ms:
                            controller.record_deadline_expiry()
                            results.append(
                                _rejected_result(
                                    item["request"],
                                    float(item["arrival_lag_ms"]),
                                    waited_ms,
                                    "queue_deadline_expired",
                                )
                            )
                        else:
                            survivors.append(item)
                    pending_queue = survivors

                    made_progress = True
                    while pending_queue and made_progress:
                        made_progress = False
                        now = time.monotonic()
                        for position, item in enumerate(pending_queue):
                            waited_ms = (now - float(item["scheduled_at"])) * 1000
                            if _dispatch(
                                executor,
                                futures,
                                item["request"],
                                float(item["scheduled_at"]),
                                float(item["arrival_lag_ms"]),
                                waited_ms,
                                controller,
                                config,
                                tokenizer,
                                requester,
                            ):
                                if position > 0:
                                    controller.record_backfill()
                                pending_queue.pop(position)
                                made_progress = True
                                break
                    controller.observe_queue_depth(len(pending_queue))

                if next_request >= len(requests) and not pending_queue:
                    break
                sleep_seconds = config.deadline_poll_interval_ms / 1000
                if next_request < len(requests):
                    next_arrival = trace_started + float(
                        requests[next_request]["arrival_offset_seconds"]
                    )
                    sleep_seconds = min(
                        sleep_seconds if pending_queue else max(0.0, next_arrival - time.monotonic()),
                        max(0.0, next_arrival - time.monotonic()),
                    )
                if sleep_seconds > 0:
                    time.sleep(sleep_seconds)

            done, not_done = wait(
                list(futures), timeout=config.drain_timeout_seconds
            )
            del done
            _collect_done(futures, results)
            for future in not_done:
                future.cancel()
        finally:
            executor.shutdown(wait=True, cancel_futures=True)
    ended_at = time.monotonic()

    results.sort(key=lambda row: str(row["id"]))
    summary = summarize_e09_requests(results, profile.duration_seconds)
    telemetry = summarize_telemetry(telemetry_path)
    controller_state = controller.snapshot()
    warmup_valid = bool(
        len(warmup_results) == config.warmup_requests
        and all(
            row.get("status") == "completed"
            and row.get("token_shape_valid") is True
            for row in warmup_results
        )
    )
    missing_results = len(requests) - len(results)
    queue_wait_valid = bool(
        policy != "deadline_aware"
        or summary["max_admitted_queue_wait_ms"] is None
        or float(summary["max_admitted_queue_wait_ms"])
        <= config.deadline_max_queue_wait_ms
    )
    evidence_valid = bool(
        warmup_valid
        and missing_results == 0
        and summary["arrivals"] == trace["request_count"]
        and summary["admitted_error_rate"] < config.gates.maximum_admitted_error_rate
        and summary["token_shape_mismatches"] == 0
        and isinstance(summary["p95_arrival_lag_ms"], (int, float))
        and float(summary["p95_arrival_lag_ms"]) <= config.maximum_arrival_lag_ms
        and queue_wait_valid
        and controller_state["active"] == 0
        and controller_state["inflight_tokens"] == 0
        and controller_state["peak_active"] < config.client_max_workers
        and controller_state["admitted"] == summary["admitted"]
        and controller_state["rejected"] == summary["rejected"]
        and telemetry.get("sample_count", 0) > 0
        and (policy != "unbounded" or summary["rejected"] == 0)
    )
    document = {
        "schema_version": 1,
        "kind": "e09_admission_run",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "profile": profile_name,
        "formal": profile.formal,
        "policy": policy,
        "repetition": repetition,
        "seed": trace["seed"],
        "experiment_config": str(config.source_path),
        "experiment_config_sha256": config.source_sha256,
        "server_config": str(config.server_config_path),
        "server_config_sha256": config.server_config_sha256,
        "server_profile": config.server_profile,
        "active_server": marker,
        "trace": str(trace_path),
        "trace_sha256": trace["trace_sha256"],
        "offered_duration_seconds": profile.duration_seconds,
        "total_wall_seconds": ended_at - trace_started,
        "slo_ttft_ms": config.slo_ttft_ms,
        "slo_tpot_ms": config.slo_tpot_ms,
        "warmup_requests": config.warmup_requests,
        "warmup_completed": sum(
            row.get("status") == "completed" for row in warmup_results
        ),
        "warmup_results": warmup_results,
        "missing_results": missing_results,
        "summary": summary,
        "controller": controller_state,
        "telemetry_path": str(telemetry_path),
        "telemetry": telemetry,
        "environment": collect_environment(),
        "valid": evidence_valid,
        "results": results,
    }
    output_path = output_dir / (
        f"{timestamp}-e09-{profile_name}-{policy}-r{repetition}.json"
    )
    output_path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output_path, evidence_valid


def run_e09_matrix(
    config_path: str | Path,
    tokenizer_path: str | Path,
    profiles: list[str] | None = None,
    trace_root: str | Path = "artifacts/results/e09_deadline_admission/traces",
    result_root: str | Path = "artifacts/results/e09_deadline_admission/runs",
    active_server_path: str | Path = "artifacts/server/active.json",
    skip_completed: bool = False,
) -> tuple[list[Path], bool]:
    config = E09Config.from_file(config_path)
    selected = profiles or [profile.name for profile in config.profiles if profile.formal]
    if len(selected) != len(set(selected)):
        raise ConfigError("E09 matrix profiles must be unique")
    for profile_name in selected:
        if not config.profile(profile_name).formal:
            raise ConfigError("E09 matrix accepts formal profiles only")
    policy_orders = (
        ("unbounded", "fixed_concurrency", "deadline_aware"),
        ("fixed_concurrency", "deadline_aware", "unbounded"),
        ("deadline_aware", "unbounded", "fixed_concurrency"),
    )
    outputs: list[Path] = []
    all_valid = True
    first = True
    for profile_name in selected:
        for repetition in range(1, config.repetitions + 1):
            for policy in policy_orders[repetition - 1]:
                output_dir = Path(result_root) / profile_name / policy
                matching: list[Path] = []
                for path in sorted(output_dir.glob("*-e09-*.json")):
                    try:
                        data = json.loads(path.read_text(encoding="utf-8"))
                    except (OSError, json.JSONDecodeError):
                        continue
                    if (
                        isinstance(data, dict)
                        and data.get("kind") == "e09_admission_run"
                        and data.get("experiment_config_sha256") == config.source_sha256
                        and data.get("repetition") == repetition
                    ):
                        matching.append(path)
                        all_valid = all_valid and data.get("valid") is True
                if matching and skip_completed:
                    print(
                        "Skipping preserved E09 cell: "
                        f"{profile_name}/{policy}/r{repetition}"
                    )
                    outputs.extend(matching)
                    continue
                if not first:
                    time.sleep(config.cooldown_seconds)
                first = False
                print(f"Running E09 cell: {profile_name}/{policy}/r{repetition}")
                path, valid = run_e09_cell(
                    config_path=config_path,
                    profile_name=profile_name,
                    policy=policy,
                    repetition=repetition,
                    tokenizer_path=tokenizer_path,
                    trace_root=trace_root,
                    result_root=result_root,
                    active_server_path=active_server_path,
                )
                print(path)
                outputs.append(path)
                all_valid = all_valid and valid
                if not valid:
                    return outputs, False
    return outputs, all_valid
