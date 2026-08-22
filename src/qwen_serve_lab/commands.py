from __future__ import annotations

import json
import math
import shlex

from qwen_serve_lab.config import BenchmarkConfig, ServeConfig


def build_serve_environment(config: ServeConfig) -> dict[str, str]:
    return {
        "VLLM_USE_FLASHINFER_SAMPLER": (
            "1" if config.use_flashinfer_sampler else "0"
        ),
        "VLLM_WSL2_ENABLE_PIN_MEMORY": (
            "1" if config.wsl2_enable_pin_memory else "0"
        ),
    }


def build_serve_command(config: ServeConfig) -> list[str]:
    command = [
        "vllm",
        "serve",
        config.effective_model,
    ]
    if config.local_model_path is None:
        command.extend(["--revision", config.revision])
    command.extend([
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
    ])
    if config.seed is not None:
        command.extend(["--seed", str(config.seed)])
    if config.calculate_kv_scales:
        command.append("--calculate-kv-scales")
    if config.attention_backend is not None:
        command.extend(
            [
                "--attention-config",
                json.dumps(
                    {"backend": config.attention_backend},
                    separators=(",", ":"),
                ),
            ]
        )
    command.append(
        "--enable-prefix-caching"
        if config.enable_prefix_caching
        else "--no-enable-prefix-caching"
    )
    if config.enable_chunked_prefill is not None:
        command.append(
            "--enable-chunked-prefill"
            if config.enable_chunked_prefill
            else "--no-enable-chunked-prefill"
        )
    if config.enable_per_request_metrics:
        command.append("--enable-per-request-metrics")
    return command


def build_benchmark_command(
    config: BenchmarkConfig, repetition: int = 1
) -> list[str]:
    request_rate = "inf" if math.isinf(config.request_rate) else str(config.request_rate)
    effective_seed = config.seed_for_repetition(repetition)
    metadata = [
        f"profile={config.profile_name}",
        f"server_profile={config.server_profile}",
        f"repetition={repetition}",
        f"effective_seed={effective_seed}",
        f"input_len={config.input_len}",
        f"output_len={config.output_len}",
        f"slo_ttft_ms={config.goodput_ttft_ms:g}",
        f"slo_tpot_ms={config.goodput_tpot_ms:g}",
        f"benchmark_config_sha256={config.source_sha256}",
        f"server_config_sha256={config.server_config_sha256}",
    ]
    if config.dataset_name == "prefix_repetition":
        metadata.extend([
            f"prefix_len={config.prefix_len}",
            f"suffix_len={config.suffix_len}",
            f"num_prefixes={config.num_prefixes}",
            f"nominal_reuse_percent={config.nominal_reuse_percent:g}",
            "prefix_cache_enabled="
            + str(config.server_prefix_caching_enabled).lower(),
        ])
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
    ]
    if config.local_tokenizer_path is not None:
        command.extend(["--tokenizer", str(config.local_tokenizer_path)])
    command.extend([
        "--served-model-name",
        config.served_model_name,
        "--dataset-name",
        config.dataset_name,
    ])
    if config.dataset_name == "prefix_repetition":
        assert config.prefix_len is not None
        assert config.suffix_len is not None
        assert config.num_prefixes is not None
        command.extend([
            "--prefix-repetition-prefix-len",
            str(config.prefix_len),
            "--prefix-repetition-suffix-len",
            str(config.suffix_len),
            "--prefix-repetition-num-prefixes",
            str(config.num_prefixes),
            "--prefix-repetition-output-len",
            str(config.output_len),
        ])
    else:
        command.extend([
            "--input-len",
            str(config.input_len),
            "--output-len",
            str(config.output_len),
        ])
    command.extend([
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
        str(effective_seed),
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
        *metadata,
        "--label",
        config.profile_name,
        "--save-result",
        "--save-detailed",
        "--result-dir",
        str(config.result_dir),
    ])
    if config.ignore_eos:
        command.append("--ignore-eos")
    return command


def build_prewarm_command(
    config: BenchmarkConfig, repetition: int = 1
) -> list[str] | None:
    if config.prewarm_prompts == 0:
        return None
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
    ]
    if config.local_tokenizer_path is not None:
        command.extend(["--tokenizer", str(config.local_tokenizer_path)])
    command.extend([
        "--served-model-name",
        config.served_model_name,
        "--dataset-name",
        "random",
        "--input-len",
        "64",
        "--output-len",
        "16",
        "--num-prompts",
        str(config.prewarm_prompts),
        "--request-rate",
        "inf",
        "--burstiness",
        "1",
        "--max-concurrency",
        "1",
        "--num-warmups",
        "0",
        "--ready-check-timeout-sec",
        str(config.ready_check_timeout_seconds),
        "--seed",
        str(config.prewarm_seed_for_repetition(repetition)),
        "--temperature",
        "0",
        "--ignore-eos",
    ])
    return command


def render_shell_command(
    command: list[str], environment: dict[str, str] | None = None
) -> str:
    if environment:
        assignments = [f"{key}={environment[key]}" for key in sorted(environment)]
        command = ["env", *assignments, *command]
    return shlex.join(command)
