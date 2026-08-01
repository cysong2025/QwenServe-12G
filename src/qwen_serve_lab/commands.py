from __future__ import annotations

import math
import shlex

from qwen_serve_lab.config import BenchmarkConfig, ServeConfig


def build_serve_command(config: ServeConfig) -> list[str]:
    command = [
        "vllm",
        "serve",
        config.model,
        "--revision",
        config.revision,
        "--served-model-name",
        config.served_model_name,
        "--host",
        config.host,
        "--port",
        str(config.port),
        "--dtype",
        config.dtype,
        "--generation-config",
        config.generation_config,
        "--max-model-len",
        str(config.max_model_len),
        "--gpu-memory-utilization",
        str(config.gpu_memory_utilization),
        "--max-num-seqs",
        str(config.max_num_seqs),
        "--max-num-batched-tokens",
        str(config.max_num_batched_tokens),
        "--kv-cache-dtype",
        config.kv_cache_dtype,
    ]
    command.append(
        "--enable-prefix-caching"
        if config.enable_prefix_caching
        else "--no-enable-prefix-caching"
    )
    if config.enable_per_request_metrics:
        command.append("--enable-per-request-metrics")
    return command


def build_benchmark_command(
    config: BenchmarkConfig, repetition: int = 1
) -> list[str]:
    request_rate = "inf" if math.isinf(config.request_rate) else str(config.request_rate)
    command = [
        "vllm",
        "bench",
        "serve",
        "--backend",
        "openai",
        "--base-url",
        config.base_url,
        "--endpoint",
        config.endpoint,
        "--model",
        config.model,
        "--served-model-name",
        config.served_model_name,
        "--dataset-name",
        config.dataset_name,
        "--input-len",
        str(config.input_len),
        "--output-len",
        str(config.output_len),
        "--num-prompts",
        str(config.num_prompts),
        "--request-rate",
        request_rate,
        "--burstiness",
        f"{config.burstiness:g}",
        "--max-concurrency",
        str(config.max_concurrency),
        "--num-warmups",
        str(config.num_warmups),
        "--ready-check-timeout-sec",
        str(config.ready_check_timeout_seconds),
        "--seed",
        str(config.seed),
        "--temperature",
        f"{config.temperature:g}",
        "--percentile-metrics",
        "ttft,tpot,itl,e2el",
        "--metric-percentiles",
        ",".join(str(item) for item in config.metric_percentiles),
        "--goodput",
        f"ttft:{config.goodput_ttft_ms:g}",
        f"tpot:{config.goodput_tpot_ms:g}",
        "--metadata",
        f"profile={config.profile_name}",
        f"server_profile={config.server_profile}",
        f"repetition={repetition}",
        f"input_len={config.input_len}",
        f"output_len={config.output_len}",
        f"slo_ttft_ms={config.goodput_ttft_ms:g}",
        f"slo_tpot_ms={config.goodput_tpot_ms:g}",
        f"benchmark_config_sha256={config.source_sha256}",
        f"server_config_sha256={config.server_config_sha256}",
        "--label",
        config.profile_name,
        "--save-result",
        "--save-detailed",
        "--result-dir",
        str(config.result_dir),
    ]
    if config.ignore_eos:
        command.append("--ignore-eos")
    return command


def render_shell_command(command: list[str]) -> str:
    return shlex.join(command)
