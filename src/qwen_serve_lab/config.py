from __future__ import annotations

import math
import hashlib
import json
import tomllib
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    """Raised when an experiment configuration is invalid."""


def _load_toml(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    try:
        with config_path.open("rb") as handle:
            return tomllib.load(handle)
    except FileNotFoundError as exc:
        raise ConfigError(f"Config file does not exist: {config_path}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"Invalid TOML in {config_path}: {exc}") from exc


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ConfigError(f"Cannot fingerprint config file {path}: {exc}") from exc
    return digest.hexdigest()


def _section(data: dict[str, Any], name: str) -> dict[str, Any]:
    value = data.get(name)
    if not isinstance(value, dict):
        raise ConfigError(f"Missing [{name}] section")
    return value


def _required(section: dict[str, Any], key: str, expected: type) -> Any:
    value = section.get(key)
    if not isinstance(value, expected):
        raise ConfigError(f"{key} must be {expected.__name__}")
    if expected is str and not value.strip():
        raise ConfigError(f"{key} cannot be empty")
    return value


def _positive_int(section: dict[str, Any], key: str) -> int:
    value = _required(section, key, int)
    if isinstance(value, bool) or value <= 0:
        raise ConfigError(f"{key} must be a positive integer")
    return value


@dataclass(frozen=True)
class ServeConfig:
    profile_name: str
    description: str
    source_path: Path
    source_sha256: str
    model: str
    revision: str
    served_model_name: str
    host: str
    port: int
    dtype: str
    generation_config: str
    max_model_len: int
    gpu_memory_utilization: float
    max_num_seqs: int
    max_num_batched_tokens: int
    kv_cache_dtype: str
    enable_prefix_caching: bool
    enable_per_request_metrics: bool
    wsl2_enable_pin_memory: bool
    local_model_path: Path | None = None
    local_model_manifest_sha256: str | None = None

    @classmethod
    def from_file(cls, path: str | Path) -> "ServeConfig":
        source_path = Path(path).resolve()
        data = _load_toml(source_path)
        profile = _section(data, "profile")
        server = _section(data, "server")

        gpu_memory_utilization = server.get("gpu_memory_utilization")
        if not isinstance(gpu_memory_utilization, (int, float)) or isinstance(
            gpu_memory_utilization, bool
        ):
            raise ConfigError("gpu_memory_utilization must be a number")
        gpu_memory_utilization = float(gpu_memory_utilization)
        if not 0 < gpu_memory_utilization <= 1:
            raise ConfigError("gpu_memory_utilization must be in (0, 1]")

        max_model_len = _positive_int(server, "max_model_len")
        max_num_batched_tokens = _positive_int(server, "max_num_batched_tokens")
        if max_num_batched_tokens > max_model_len * _positive_int(
            server, "max_num_seqs"
        ):
            raise ConfigError(
                "max_num_batched_tokens exceeds the profile's total sequence capacity"
            )

        return cls(
            profile_name=_required(profile, "name", str),
            description=_required(profile, "description", str),
            source_path=source_path,
            source_sha256=_sha256(source_path),
            model=_required(server, "model", str),
            revision=_required(server, "revision", str),
            served_model_name=_required(server, "served_model_name", str),
            host=_required(server, "host", str),
            port=_positive_int(server, "port"),
            dtype=_required(server, "dtype", str),
            generation_config=_required(server, "generation_config", str),
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            max_num_seqs=_positive_int(server, "max_num_seqs"),
            max_num_batched_tokens=max_num_batched_tokens,
            kv_cache_dtype=_required(server, "kv_cache_dtype", str),
            enable_prefix_caching=_required(
                server, "enable_prefix_caching", bool
            ),
            enable_per_request_metrics=_required(
                server, "enable_per_request_metrics", bool
            ),
            wsl2_enable_pin_memory=_required(
                server, "wsl2_enable_pin_memory", bool
            ),
        )

    def with_local_model(self, path: str | Path) -> "ServeConfig":
        model_path = Path(path).expanduser().resolve()
        if not model_path.is_dir():
            raise ConfigError(f"Local model directory does not exist: {model_path}")

        required_files = ("config.json", "tokenizer.json", "SHA256SUMS")
        missing = [
            name for name in required_files if not (model_path / name).is_file()
        ]
        if missing:
            raise ConfigError(
                f"Local model directory is incomplete; missing: {', '.join(missing)}"
            )
        if not any(model_path.glob("*.safetensors")):
            raise ConfigError(
                f"Local model directory contains no safetensors weights: {model_path}"
            )

        manifest_path = model_path / "SHA256SUMS"
        return replace(
            self,
            local_model_path=model_path,
            local_model_manifest_sha256=_sha256(manifest_path),
        )

    @property
    def effective_model(self) -> str:
        if self.local_model_path is not None:
            return str(self.local_model_path)
        return self.model


@dataclass(frozen=True)
class BenchmarkConfig:
    profile_name: str
    server_profile: str
    source_path: Path
    source_sha256: str
    server_config_path: Path
    server_config_sha256: str
    model: str
    served_model_name: str
    base_url: str
    endpoint: str
    dataset_name: str
    input_len: int
    output_len: int
    num_prompts: int
    request_rate: float
    burstiness: float
    max_concurrency: int
    num_warmups: int
    metric_percentiles: tuple[int, ...]
    goodput_ttft_ms: float
    goodput_tpot_ms: float
    seed: int
    temperature: float
    ignore_eos: bool
    repetitions: int
    cooldown_seconds: int
    ready_check_timeout_seconds: int
    result_dir: Path

    @classmethod
    def from_file(cls, path: str | Path) -> "BenchmarkConfig":
        source_path = Path(path).resolve()
        data = _load_toml(source_path)
        profile = _section(data, "profile")
        benchmark = _section(data, "benchmark")
        return cls.from_sections(profile, benchmark, source_path)

    @classmethod
    def from_sections(
        cls,
        profile: dict[str, Any],
        benchmark: dict[str, Any],
        source_path: Path,
    ) -> "BenchmarkConfig":
        source_path = source_path.resolve()
        server_config_raw = _required(profile, "server_config", str)
        server_config_path = (source_path.parent / server_config_raw).resolve()
        if not server_config_path.is_file():
            raise ConfigError(f"Server config does not exist: {server_config_path}")

        request_rate_raw = benchmark.get("request_rate")
        if request_rate_raw == "inf":
            request_rate = math.inf
        elif isinstance(request_rate_raw, (int, float)) and not isinstance(
            request_rate_raw, bool
        ):
            request_rate = float(request_rate_raw)
            if request_rate <= 0:
                raise ConfigError("request_rate must be positive or 'inf'")
        else:
            raise ConfigError("request_rate must be positive or 'inf'")

        burstiness_raw = benchmark.get("burstiness")
        if not isinstance(burstiness_raw, (int, float)) or isinstance(
            burstiness_raw, bool
        ):
            raise ConfigError("burstiness must be a positive number")
        burstiness = float(burstiness_raw)
        if burstiness <= 0:
            raise ConfigError("burstiness must be a positive number")

        temperature_raw = benchmark.get("temperature")
        if not isinstance(temperature_raw, (int, float)) or isinstance(
            temperature_raw, bool
        ):
            raise ConfigError("temperature must be a non-negative number")
        temperature = float(temperature_raw)
        if temperature < 0:
            raise ConfigError("temperature must be a non-negative number")

        percentiles_raw = benchmark.get("metric_percentiles")
        if not isinstance(percentiles_raw, list) or not percentiles_raw:
            raise ConfigError("metric_percentiles must be a non-empty list")
        if any(
            not isinstance(item, int) or isinstance(item, bool) or not 0 < item < 100
            for item in percentiles_raw
        ):
            raise ConfigError("metric_percentiles values must be integers in (0, 100)")

        result_dir = _required(benchmark, "result_dir", str)
        num_warmups = benchmark.get("num_warmups")
        if not isinstance(num_warmups, int) or isinstance(num_warmups, bool) or num_warmups < 0:
            raise ConfigError("num_warmups must be a non-negative integer")
        cooldown_seconds = benchmark.get("cooldown_seconds")
        if (
            not isinstance(cooldown_seconds, int)
            or isinstance(cooldown_seconds, bool)
            or cooldown_seconds < 0
        ):
            raise ConfigError("cooldown_seconds must be a non-negative integer")
        ready_check_timeout_seconds = benchmark.get("ready_check_timeout_seconds")
        if (
            not isinstance(ready_check_timeout_seconds, int)
            or isinstance(ready_check_timeout_seconds, bool)
            or ready_check_timeout_seconds < 0
        ):
            raise ConfigError(
                "ready_check_timeout_seconds must be a non-negative integer"
            )

        return cls(
            profile_name=_required(profile, "name", str),
            server_profile=_required(profile, "server_profile", str),
            source_path=source_path,
            source_sha256=_sha256(source_path),
            server_config_path=server_config_path,
            server_config_sha256=_sha256(server_config_path),
            model=_required(benchmark, "model", str),
            served_model_name=_required(benchmark, "served_model_name", str),
            base_url=_required(benchmark, "base_url", str).rstrip("/"),
            endpoint=_required(benchmark, "endpoint", str),
            dataset_name=_required(benchmark, "dataset_name", str),
            input_len=_positive_int(benchmark, "input_len"),
            output_len=_positive_int(benchmark, "output_len"),
            num_prompts=_positive_int(benchmark, "num_prompts"),
            request_rate=request_rate,
            burstiness=burstiness,
            max_concurrency=_positive_int(benchmark, "max_concurrency"),
            num_warmups=num_warmups,
            metric_percentiles=tuple(percentiles_raw),
            goodput_ttft_ms=float(_positive_int(benchmark, "goodput_ttft_ms")),
            goodput_tpot_ms=float(_positive_int(benchmark, "goodput_tpot_ms")),
            seed=_positive_int(benchmark, "seed"),
            temperature=temperature,
            ignore_eos=_required(benchmark, "ignore_eos", bool),
            repetitions=_positive_int(benchmark, "repetitions"),
            cooldown_seconds=cooldown_seconds,
            ready_check_timeout_seconds=ready_check_timeout_seconds,
            result_dir=Path(result_dir),
        )


@dataclass(frozen=True)
class BenchmarkMatrix:
    profile_name: str
    configs: tuple[BenchmarkConfig, ...]

    @classmethod
    def from_file(cls, path: str | Path) -> "BenchmarkMatrix":
        source_path = Path(path).resolve()
        data = _load_toml(source_path)
        profile = _section(data, "profile")
        matrix = _section(data, "matrix")
        common = _section(data, "benchmark")
        workloads = data.get("workloads")
        if not isinstance(workloads, list) or not workloads:
            raise ConfigError("[[workloads]] must contain at least one workload")

        concurrencies = matrix.get("concurrencies")
        if not isinstance(concurrencies, list) or not concurrencies:
            raise ConfigError("matrix.concurrencies must be a non-empty list")
        if any(
            not isinstance(item, int) or isinstance(item, bool) or item <= 0
            for item in concurrencies
        ):
            raise ConfigError("matrix.concurrencies must contain positive integers")

        result_root = _required(matrix, "result_root", str)
        profile_prefix = _required(matrix, "profile_prefix", str)
        matrix_name = _required(profile, "name", str)
        server_profile = _required(profile, "server_profile", str)
        server_config = _required(profile, "server_config", str)
        configs: list[BenchmarkConfig] = []
        seen_names: set[str] = set()

        for workload in workloads:
            if not isinstance(workload, dict):
                raise ConfigError("Each [[workloads]] entry must be a table")
            workload_name = _required(workload, "name", str)
            input_len = _positive_int(workload, "input_len")
            output_len = _positive_int(workload, "output_len")
            for concurrency in concurrencies:
                effective_name = f"{profile_prefix}_{workload_name}_c{concurrency}"
                if effective_name in seen_names:
                    raise ConfigError(f"Duplicate matrix profile: {effective_name}")
                seen_names.add(effective_name)
                effective_profile = {
                    "name": effective_name,
                    "server_profile": server_profile,
                    "server_config": server_config,
                }
                effective_benchmark = {
                    **common,
                    "input_len": input_len,
                    "output_len": output_len,
                    "max_concurrency": concurrency,
                    "result_dir": f"{result_root}/{effective_name}",
                }
                configs.append(
                    BenchmarkConfig.from_sections(
                        effective_profile, effective_benchmark, source_path
                    )
                )

        effective_payload = [
            {
                "profile": config.profile_name,
                "input_len": config.input_len,
                "output_len": config.output_len,
                "max_concurrency": config.max_concurrency,
            }
            for config in configs
        ]
        if len({json.dumps(item, sort_keys=True) for item in effective_payload}) != len(
            effective_payload
        ):
            raise ConfigError("Matrix expansion produced duplicate experiments")
        return cls(profile_name=matrix_name, configs=tuple(configs))
