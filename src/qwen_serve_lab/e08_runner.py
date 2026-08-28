from __future__ import annotations

import json
import os
import random
import time
from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from qwen_serve_lab.config import ConfigError
from qwen_serve_lab.e08_admission import (
    AdmissionController,
    AdmissionLease,
    E08Config,
    validate_e08_trace,
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


def _jain_index(values: list[float]) -> float:
    if not values or all(value == 0 for value in values):
        return 0.0
    numerator = sum(values) ** 2
    denominator = len(values) * sum(value * value for value in values)
    return numerator / denominator if denominator else 0.0


def _load_active_server(
    config: E08Config, active_server_path: str | Path
) -> dict[str, Any]:
    path = Path(active_server_path)
    try:
        marker = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ResultError(
            f"Cannot read controlled E08 server marker {path}: {exc}"
        ) from exc
    expected = (config.server_profile, config.server_config_sha256)
    actual = (
        marker.get("profile") if isinstance(marker, dict) else None,
        marker.get("server_config_sha256") if isinstance(marker, dict) else None,
    )
    if actual != expected:
        raise ResultError(
            "E08 requires the frozen Base server "
            f"{expected[0]} ({expected[1]}); found {actual[0]} ({actual[1]})"
        )
    pid = marker.get("pid")
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
        raise ResultError("Controlled E08 server marker has an invalid pid")
    try:
        os.kill(pid, 0)
    except ProcessLookupError as exc:
        raise ResultError(f"Controlled E08 server process {pid} is absent") from exc
    except PermissionError:
        pass
    return marker


def load_e08_trace(path: str | Path, config: E08Config) -> dict[str, Any]:
    trace_path = Path(path)
    try:
        trace = json.loads(trace_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ResultError(f"Cannot read E08 trace {trace_path}: {exc}") from exc
    try:
        validate_e08_trace(trace, config)
    except ConfigError as exc:
        raise ResultError(str(exc)) from exc
    return trace


def load_local_tokenizer(path: str | Path) -> Any:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ResultError(
            "E08 requires the existing local Transformers tokenizer"
        ) from exc
    tokenizer_path = Path(path).expanduser().resolve()
    if not tokenizer_path.is_dir():
        raise ResultError(f"E08 tokenizer directory does not exist: {tokenizer_path}")
    try:
        return AutoTokenizer.from_pretrained(
            tokenizer_path, local_files_only=True, trust_remote_code=False
        )
    except Exception as exc:
        raise ResultError(f"Cannot load local E08 tokenizer: {exc}") from exc


def build_prompt_token_ids(tokenizer: Any, length: int, seed: int) -> list[int]:
    vocab_size = getattr(tokenizer, "vocab_size", None)
    if not isinstance(vocab_size, int) or vocab_size <= 1024:
        raise ResultError("E08 tokenizer has an invalid vocab_size")
    special_ids = {
        value
        for value in getattr(tokenizer, "all_special_ids", [])
        if isinstance(value, int)
    }
    rng = random.Random(seed)
    tokens: list[int] = []
    while len(tokens) < length:
        token = rng.randrange(256, vocab_size)
        if token not in special_ids:
            tokens.append(token)
    return tokens


def request_streaming_completion(
    base_url: str,
    served_model_name: str,
    prompt_token_ids: list[int],
    output_tokens: int,
    seed: int,
    timeout_seconds: float,
    scheduled_at: float | None = None,
) -> dict[str, Any]:
    request_started = time.monotonic()
    latency_origin = scheduled_at if scheduled_at is not None else request_started
    payload = json.dumps(
        {
            "model": served_model_name,
            "prompt": prompt_token_ids,
            "max_tokens": output_tokens,
            "temperature": 0,
            "seed": seed,
            "ignore_eos": True,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
    ).encode("utf-8")
    request = Request(
        base_url.rstrip("/") + "/v1/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    first_token_at: float | None = None
    last_token_at: float | None = None
    stream_events = 0
    completion_tokens: int | None = None
    prompt_tokens: int | None = None
    finish_reason: str | None = None
    response_text: list[str] = []
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                usage = chunk.get("usage") if isinstance(chunk, dict) else None
                if isinstance(usage, dict):
                    raw_completion_tokens = usage.get("completion_tokens")
                    if isinstance(raw_completion_tokens, int):
                        completion_tokens = raw_completion_tokens
                    raw_prompt_tokens = usage.get("prompt_tokens")
                    if isinstance(raw_prompt_tokens, int):
                        prompt_tokens = raw_prompt_tokens
                choices = chunk.get("choices") if isinstance(chunk, dict) else None
                if not isinstance(choices, list) or not choices:
                    continue
                choice = choices[0]
                if not isinstance(choice, dict):
                    continue
                if isinstance(choice.get("finish_reason"), str):
                    finish_reason = choice["finish_reason"]
                text = choice.get("text")
                if not isinstance(text, str) or not text:
                    continue
                observed_at = time.monotonic()
                if first_token_at is None:
                    first_token_at = observed_at
                last_token_at = observed_at
                stream_events += 1
                response_text.append(text)
    except HTTPError as exc:
        try:
            detail = exc.read().decode("utf-8")[-1000:]
        except OSError:
            detail = str(exc)
        raise ResultError(f"E08 completion HTTP {exc.code}: {detail}") from exc
    except (URLError, OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ResultError(f"E08 streaming completion failed: {exc}") from exc
    ended_at = time.monotonic()
    if first_token_at is None or last_token_at is None:
        raise ResultError("E08 streaming response produced no output token")
    if completion_tokens is None:
        completion_tokens = stream_events
    if completion_tokens <= 0:
        raise ResultError("E08 streaming response reported no completion tokens")
    ttft_ms = (first_token_at - latency_origin) * 1000
    tpot_ms = (
        (last_token_at - first_token_at) * 1000 / (completion_tokens - 1)
        if completion_tokens > 1
        else 0.0
    )
    return {
        "status": "completed",
        "request_started_lag_ms": (request_started - latency_origin) * 1000,
        "ttft_ms": ttft_ms,
        "tpot_ms": tpot_ms,
        "e2e_ms": (ended_at - latency_origin) * 1000,
        "completion_tokens": completion_tokens,
        "prompt_tokens": prompt_tokens,
        "stream_events": stream_events,
        "finish_reason": finish_reason,
        "generated_chars": sum(len(value) for value in response_text),
        "error": None,
    }


def _run_one_request(
    request_data: dict[str, Any],
    scheduled_at: float,
    lease: AdmissionLease,
    controller: AdmissionController,
    config: E08Config,
    tokenizer: Any,
    requester: Callable[..., dict[str, Any]],
) -> dict[str, Any]:
    base = {
        "id": request_data["id"],
        "arrival_offset_seconds": request_data["arrival_offset_seconds"],
        "workload": request_data["workload"],
        "input_tokens": request_data["input_tokens"],
        "output_tokens": request_data["output_tokens"],
        "token_cost": request_data["token_cost"],
        "admitted": True,
    }
    slo_met = False
    try:
        prompt = build_prompt_token_ids(
            tokenizer, request_data["input_tokens"], request_data["prompt_seed"]
        )
        result = requester(
            config.base_url,
            config.served_model_name,
            prompt,
            request_data["output_tokens"],
            request_data["prompt_seed"],
            config.request_timeout_seconds,
            scheduled_at,
        )
        token_shape_valid = bool(
            result.get("prompt_tokens") == request_data["input_tokens"]
            and result.get("completion_tokens") == request_data["output_tokens"]
        )
        slo_met = bool(
            result.get("status") == "completed"
            and token_shape_valid
            and isinstance(result.get("ttft_ms"), (int, float))
            and result["ttft_ms"] <= config.slo_ttft_ms
            and isinstance(result.get("tpot_ms"), (int, float))
            and result["tpot_ms"] <= config.slo_tpot_ms
        )
        return {
            **base,
            **result,
            "token_shape_valid": token_shape_valid,
            "slo_met": slo_met,
        }
    except (ResultError, ValueError, RuntimeError) as exc:
        return {
            **base,
            "status": "failed",
            "request_started_lag_ms": None,
            "ttft_ms": None,
            "tpot_ms": None,
            "e2e_ms": None,
            "completion_tokens": 0,
            "prompt_tokens": None,
            "stream_events": 0,
            "finish_reason": None,
            "generated_chars": 0,
            "error": str(exc),
            "token_shape_valid": False,
            "slo_met": False,
        }
    finally:
        controller.release(lease, slo_met=slo_met)


def _class_metrics(results: list[dict[str, Any]], duration: float) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        grouped[str(result["workload"])].append(result)
    metrics: dict[str, Any] = {}
    yields: list[float] = []
    for workload in ("short", "medium", "long"):
        rows = grouped[workload]
        arrivals = len(rows)
        admitted = sum(bool(row["admitted"]) for row in rows)
        completed = sum(row["status"] == "completed" for row in rows)
        slo_met = sum(bool(row.get("slo_met")) for row in rows)
        normalized_yield = slo_met / arrivals if arrivals else 0.0
        yields.append(normalized_yield)
        metrics[workload] = {
            "arrivals": arrivals,
            "admitted": admitted,
            "rejected": arrivals - admitted,
            "completed": completed,
            "slo_met": slo_met,
            "admission_rate": admitted / arrivals if arrivals else 0.0,
            "slo_yield": normalized_yield,
            "goodput_rps": slo_met / duration,
        }
    metrics["jain_slo_yield"] = _jain_index(yields)
    return metrics


def summarize_e08_requests(
    results: list[dict[str, Any]], duration_seconds: float
) -> dict[str, Any]:
    arrivals = len(results)
    admitted = sum(bool(row["admitted"]) for row in results)
    rejected = arrivals - admitted
    completed_rows = [row for row in results if row["status"] == "completed"]
    failed = admitted - len(completed_rows)
    shape_mismatches = sum(
        row["status"] == "completed" and row.get("token_shape_valid") is not True
        for row in results
    )
    slo_met = sum(bool(row.get("slo_met")) for row in results)
    ttft = [float(row["ttft_ms"]) for row in completed_rows]
    tpot = [float(row["tpot_ms"]) for row in completed_rows]
    e2e = [float(row["e2e_ms"]) for row in completed_rows]
    dispatch_lags = [
        float(row["dispatch_lag_ms"])
        for row in results
        if isinstance(row.get("dispatch_lag_ms"), (int, float))
    ]
    return {
        "arrivals": arrivals,
        "admitted": admitted,
        "rejected": rejected,
        "completed": len(completed_rows),
        "failed": failed,
        "token_shape_mismatches": shape_mismatches,
        "admission_rate": admitted / arrivals if arrivals else 0.0,
        "rejection_rate": rejected / arrivals if arrivals else 0.0,
        "admitted_error_rate": failed / admitted if admitted else 0.0,
        "request_throughput": len(completed_rows) / duration_seconds,
        "request_goodput": slo_met / duration_seconds,
        "slo_met": slo_met,
        "slo_yield": slo_met / arrivals if arrivals else 0.0,
        "p50_ttft_ms": _percentile(ttft, 50),
        "p95_ttft_ms": _percentile(ttft, 95),
        "p99_ttft_ms": _percentile(ttft, 99),
        "p50_tpot_ms": _percentile(tpot, 50),
        "p95_tpot_ms": _percentile(tpot, 95),
        "p99_tpot_ms": _percentile(tpot, 99),
        "p95_e2e_ms": _percentile(e2e, 95),
        "p95_dispatch_lag_ms": _percentile(dispatch_lags, 95),
        "class_metrics": _class_metrics(results, duration_seconds),
    }


def run_e08_cell(
    config_path: str | Path,
    profile_name: str,
    policy: str,
    repetition: int,
    tokenizer_path: str | Path,
    trace_root: str | Path = "artifacts/results/e08_admission/traces",
    result_root: str | Path = "artifacts/results/e08_admission/runs",
    active_server_path: str | Path = "artifacts/server/active.json",
    requester: Callable[..., dict[str, Any]] = request_streaming_completion,
    tokenizer_loader: Callable[[str | Path], Any] = load_local_tokenizer,
) -> tuple[Path, bool]:
    config = E08Config.from_file(config_path)
    profile = config.profile(profile_name)
    if policy not in config.policies:
        raise ConfigError(f"Unknown E08 policy: {policy}")
    if repetition < 1 or repetition > config.repetitions:
        raise ConfigError(
            f"E08 repetition must be between 1 and {config.repetitions}"
        )
    marker = _load_active_server(config, active_server_path)
    trace_path = Path(trace_root) / f"{profile_name}-r{repetition}.json"
    trace = load_e08_trace(trace_path, config)
    tokenizer = tokenizer_loader(tokenizer_path)

    warmup_results: list[dict[str, Any]] = []
    for index in range(config.warmup_requests):
        workload = config.workloads[index % len(config.workloads)]
        prompt_seed = config.seeds[repetition - 1] * 1000 + index
        try:
            warmup_result = requester(
                config.base_url,
                config.served_model_name,
                build_prompt_token_ids(
                    tokenizer, workload.input_tokens, prompt_seed
                ),
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
    for existing_path in output_dir.glob("*-e08-*.json"):
        try:
            existing = json.loads(existing_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if (
            isinstance(existing, dict)
            and existing.get("kind") == "e08_admission_run"
            and existing.get("experiment_config_sha256") == config.source_sha256
            and existing.get("repetition") == repetition
        ):
            raise ResultError(
                "E08 cell already has preserved evidence: "
                f"{existing_path}; do not overwrite or silently rerun it"
            )
    telemetry_path = output_dir / f"{timestamp}-telemetry.csv"
    controller = AdmissionController(policy, config)
    results: list[dict[str, Any]] = []
    futures: list[Future[dict[str, Any]]] = []
    trace_started = time.monotonic()
    with NvidiaSmiSampler(telemetry_path):
        executor = ThreadPoolExecutor(max_workers=config.client_max_workers)
        try:
            for request_data in trace["requests"]:
                scheduled_at = (
                    trace_started + float(request_data["arrival_offset_seconds"])
                )
                remaining = scheduled_at - time.monotonic()
                if remaining > 0:
                    time.sleep(remaining)
                dispatched_at = time.monotonic()
                dispatch_lag_ms = max(0.0, (dispatched_at - scheduled_at) * 1000)
                lease = controller.try_acquire(
                    request_data["id"],
                    request_data["workload"],
                    request_data["token_cost"],
                )
                if lease is None:
                    results.append(
                        {
                            **request_data,
                            "admitted": False,
                            "status": "rejected",
                            "dispatch_lag_ms": dispatch_lag_ms,
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
                    )
                    continue
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
                setattr(future, "e08_dispatch_lag_ms", dispatch_lag_ms)
                futures.append(future)
            done, pending = wait(futures, timeout=config.drain_timeout_seconds)
            for future in done:
                row = future.result()
                row["dispatch_lag_ms"] = getattr(
                    future, "e08_dispatch_lag_ms", None
                )
                results.append(row)
            for future in pending:
                future.cancel()
        finally:
            executor.shutdown(wait=True, cancel_futures=True)
    ended_at = time.monotonic()
    results.sort(key=lambda row: str(row["id"]))
    summary = summarize_e08_requests(results, profile.duration_seconds)
    telemetry = summarize_telemetry(telemetry_path)
    controller_state = controller.snapshot()
    warmup_valid = (
        len(warmup_results) == config.warmup_requests
        and all(
            row.get("status") == "completed"
            and row.get("token_shape_valid") is True
            for row in warmup_results
        )
    )
    pending_count = len(trace["requests"]) - len(results)
    evidence_valid = bool(
        warmup_valid
        and pending_count == 0
        and summary["arrivals"] == trace["request_count"]
        and summary["admitted_error_rate"]
        < config.gates.maximum_admitted_error_rate
        and summary["token_shape_mismatches"] == 0
        and isinstance(summary["p95_dispatch_lag_ms"], (int, float))
        and summary["p95_dispatch_lag_ms"] <= config.maximum_dispatch_lag_ms
        and controller_state["active"] == 0
        and controller_state["inflight_tokens"] == 0
        and controller_state["peak_active"] < config.client_max_workers
        and telemetry.get("sample_count", 0) > 0
        and (policy != "unbounded" or summary["rejected"] == 0)
    )
    document = {
        "schema_version": 1,
        "kind": "e08_admission_run",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "profile": profile_name,
        "formal": profile.formal,
        "policy": policy,
        "repetition": repetition,
        "seed": config.seeds[repetition - 1],
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
        "pending_after_drain": pending_count,
        "summary": summary,
        "controller": controller_state,
        "telemetry_path": str(telemetry_path),
        "telemetry": telemetry,
        "environment": collect_environment(),
        "valid": evidence_valid,
        "results": results,
    }
    output_path = output_dir / f"{timestamp}-e08-{profile_name}-{policy}-r{repetition}.json"
    output_path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output_path, evidence_valid


def run_e08_matrix(
    config_path: str | Path,
    tokenizer_path: str | Path,
    profiles: list[str] | None = None,
    trace_root: str | Path = "artifacts/results/e08_admission/traces",
    result_root: str | Path = "artifacts/results/e08_admission/runs",
    active_server_path: str | Path = "artifacts/server/active.json",
    skip_completed: bool = False,
) -> tuple[list[Path], bool]:
    config = E08Config.from_file(config_path)
    selected = profiles or [
        profile.name for profile in config.profiles if profile.formal
    ]
    if len(selected) != len(set(selected)):
        raise ConfigError("E08 matrix profiles must be unique")
    for profile_name in selected:
        config.profile(profile_name)
    policy_orders = (
        ("unbounded", "fixed_concurrency", "token_aware"),
        ("fixed_concurrency", "token_aware", "unbounded"),
        ("token_aware", "unbounded", "fixed_concurrency"),
    )
    outputs: list[Path] = []
    all_valid = True
    first = True
    for profile_name in selected:
        for repetition in range(1, config.repetitions + 1):
            for policy in policy_orders[(repetition - 1) % len(policy_orders)]:
                output_dir = Path(result_root) / profile_name / policy
                existing = sorted(output_dir.glob("*-e08-*.json"))
                matching: list[Path] = []
                for path in existing:
                    try:
                        data = json.loads(path.read_text(encoding="utf-8"))
                    except (OSError, json.JSONDecodeError):
                        continue
                    if (
                        isinstance(data, dict)
                        and data.get("kind") == "e08_admission_run"
                        and data.get("experiment_config_sha256")
                        == config.source_sha256
                        and data.get("repetition") == repetition
                    ):
                        matching.append(path)
                        all_valid = all_valid and data.get("valid") is True
                if matching and skip_completed:
                    print(
                        "Skipping preserved E08 cell: "
                        f"{profile_name}/{policy}/r{repetition}"
                    )
                    outputs.extend(matching)
                    continue
                if not first:
                    time.sleep(config.cooldown_seconds)
                first = False
                print(
                    "Running E08 cell: "
                    f"{profile_name}/{policy}/r{repetition}"
                )
                path, valid = run_e08_cell(
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
