from __future__ import annotations

import hashlib
import json
import math
import random
import threading
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from qwen_serve_lab.config import ConfigError, ServeConfig


POLICIES = ("unbounded", "fixed_concurrency", "token_aware")
WORKLOAD_NAMES = ("short", "medium", "long")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ConfigError(f"Cannot fingerprint E08 file {path}: {exc}") from exc
    return digest.hexdigest()


def _number(section: dict[str, Any], key: str) -> float:
    value = section.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ConfigError(f"E08 {key} must be numeric")
    return float(value)


def _positive_number(section: dict[str, Any], key: str) -> float:
    value = _number(section, key)
    if value <= 0:
        raise ConfigError(f"E08 {key} must be positive")
    return value


def _positive_int(section: dict[str, Any], key: str) -> int:
    value = section.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ConfigError(f"E08 {key} must be a positive integer")
    return value


def _string(section: dict[str, Any], key: str) -> str:
    value = section.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"E08 {key} must be a non-empty string")
    return value.strip()


def _table(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key)
    if not isinstance(value, dict):
        raise ConfigError(f"E08 config is missing [{key}]")
    return value


@dataclass(frozen=True)
class E08Workload:
    name: str
    input_tokens: int
    output_tokens: int
    weight: int

    @property
    def token_cost(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass(frozen=True)
class E08Profile:
    name: str
    formal: bool
    mode: str
    duration_seconds: float
    rate_rps: float
    burst_period_seconds: float | None = None
    burst_duration_seconds: float | None = None
    burst_rate_rps: float | None = None
    background_rate_rps: float | None = None


@dataclass(frozen=True)
class E08Gates:
    overload_profiles: tuple[str, ...]
    minimum_goodput_gain_percent_vs_best_baseline: float
    minimum_fairness_jain: float
    maximum_fairness_regression_vs_fixed: float
    maximum_admitted_error_rate: float


@dataclass(frozen=True)
class E08Config:
    source_path: Path
    source_sha256: str
    name: str
    description: str
    server_config_path: Path
    server_config_sha256: str
    server_profile: str
    base_url: str
    served_model_name: str
    policies: tuple[str, ...]
    repetitions: int
    seeds: tuple[int, ...]
    warmup_requests: int
    request_timeout_seconds: float
    drain_timeout_seconds: float
    client_max_workers: int
    maximum_dispatch_lag_ms: float
    cooldown_seconds: float
    slo_ttft_ms: float
    slo_tpot_ms: float
    fixed_max_inflight: int
    token_max_inflight: int
    initial_token_budget: int
    min_token_budget: int
    max_token_budget: int
    decrease_factor: float
    increase_step_tokens: int
    success_window: int
    workloads: tuple[E08Workload, ...]
    profiles: tuple[E08Profile, ...]
    gates: E08Gates

    @classmethod
    def from_file(cls, path: str | Path) -> "E08Config":
        source_path = Path(path).resolve()
        try:
            with source_path.open("rb") as handle:
                data = tomllib.load(handle)
        except FileNotFoundError as exc:
            raise ConfigError(f"E08 config does not exist: {source_path}") from exc
        except tomllib.TOMLDecodeError as exc:
            raise ConfigError(f"Invalid E08 TOML in {source_path}: {exc}") from exc

        experiment = _table(data, "experiment")
        policies_raw = experiment.get("policies")
        if not isinstance(policies_raw, list) or tuple(policies_raw) != POLICIES:
            raise ConfigError(
                "E08 policies must be unbounded, fixed_concurrency, token_aware"
            )
        repetitions = _positive_int(experiment, "repetitions")
        seeds_raw = experiment.get("seeds")
        if (
            not isinstance(seeds_raw, list)
            or len(seeds_raw) != repetitions
            or any(
                not isinstance(seed, int) or isinstance(seed, bool) or seed < 0
                for seed in seeds_raw
            )
            or len(set(seeds_raw)) != repetitions
        ):
            raise ConfigError("E08 seeds must be unique non-negative integers per repetition")

        root = source_path.parents[2]
        server_config_raw = Path(_string(experiment, "server_config"))
        server_config_path = (
            server_config_raw
            if server_config_raw.is_absolute()
            else root / server_config_raw
        ).resolve()
        server = ServeConfig.from_file(server_config_path)
        if (
            server.enable_lora
            or server.kv_cache_dtype != "bfloat16"
            or server.max_num_batched_tokens != 2048
            or server.enable_prefix_caching
        ):
            raise ConfigError(
                "E08 requires Base, BF16 KV, batch-token 2048, APC-off serving"
            )

        fixed = _table(data, "fixed_concurrency")
        adaptive = _table(data, "token_aware")
        min_budget = _positive_int(adaptive, "min_token_budget")
        initial_budget = _positive_int(adaptive, "initial_token_budget")
        max_budget = _positive_int(adaptive, "max_token_budget")
        if not min_budget <= initial_budget <= max_budget:
            raise ConfigError("E08 token budgets must satisfy min <= initial <= max")
        decrease_factor = _positive_number(adaptive, "decrease_factor")
        if decrease_factor >= 1:
            raise ConfigError("E08 decrease_factor must be in (0, 1)")

        workloads_table = _table(data, "workloads")
        workloads: list[E08Workload] = []
        for name in WORKLOAD_NAMES:
            section = _table(workloads_table, name)
            workloads.append(
                E08Workload(
                    name=name,
                    input_tokens=_positive_int(section, "input_tokens"),
                    output_tokens=_positive_int(section, "output_tokens"),
                    weight=_positive_int(section, "weight"),
                )
            )
        expected_shapes = {
            "short": (128, 128, 5),
            "medium": (512, 256, 3),
            "long": (2048, 256, 2),
        }
        if {
            item.name: (item.input_tokens, item.output_tokens, item.weight)
            for item in workloads
        } != expected_shapes:
            raise ConfigError("E08 workload shapes and 50/30/20 weights are frozen")
        if min_budget < max(item.token_cost for item in workloads):
            raise ConfigError("E08 minimum token budget must admit one long request")

        profiles_raw = data.get("profiles")
        if not isinstance(profiles_raw, list) or not profiles_raw:
            raise ConfigError("E08 config must contain [[profiles]]")
        profiles: list[E08Profile] = []
        names: set[str] = set()
        for section in profiles_raw:
            if not isinstance(section, dict):
                raise ConfigError("Each E08 profile must be a table")
            name = _string(section, "name")
            if name in names:
                raise ConfigError(f"Duplicate E08 profile: {name}")
            names.add(name)
            formal = section.get("formal")
            mode = _string(section, "mode")
            if not isinstance(formal, bool):
                raise ConfigError(f"E08 profile {name} formal must be boolean")
            if mode not in {"steady", "burst"}:
                raise ConfigError(f"E08 profile {name} mode must be steady or burst")
            duration = _positive_number(section, "duration_seconds")
            if formal and duration < 180:
                raise ConfigError(f"Formal E08 profile {name} must last at least 180 seconds")
            rate = _positive_number(section, "rate_rps")
            burst_fields: dict[str, float | None] = {
                "burst_period_seconds": None,
                "burst_duration_seconds": None,
                "burst_rate_rps": None,
                "background_rate_rps": None,
            }
            if mode == "burst":
                burst_fields = {
                    key: _positive_number(section, key) for key in burst_fields
                }
                period = burst_fields["burst_period_seconds"]
                burst_duration = burst_fields["burst_duration_seconds"]
                assert period is not None and burst_duration is not None
                if burst_duration >= period:
                    raise ConfigError(f"E08 profile {name} burst must be shorter than its period")
                burst_rate = burst_fields["burst_rate_rps"]
                background_rate = burst_fields["background_rate_rps"]
                assert burst_rate is not None and background_rate is not None
                mean_rate = (
                    burst_rate * burst_duration
                    + background_rate * (period - burst_duration)
                ) / period
                if not math.isclose(mean_rate, rate, rel_tol=0, abs_tol=1e-9):
                    raise ConfigError(f"E08 profile {name} burst mean must equal rate_rps")
            profiles.append(
                E08Profile(
                    name=name,
                    formal=formal,
                    mode=mode,
                    duration_seconds=duration,
                    rate_rps=rate,
                    **burst_fields,
                )
            )

        formal_names = {profile.name for profile in profiles if profile.formal}
        if formal_names != {"mixed_nominal", "mixed_overload", "mixed_burst"}:
            raise ConfigError("E08 formal profiles are frozen")
        gates_table = _table(data, "gates")
        overload_raw = gates_table.get("overload_profiles")
        if (
            not isinstance(overload_raw, list)
            or tuple(overload_raw) != ("mixed_overload", "mixed_burst")
        ):
            raise ConfigError("E08 overload profiles are frozen")
        maximum_error_rate = _number(gates_table, "maximum_admitted_error_rate")
        if not 0 <= maximum_error_rate < 1:
            raise ConfigError("E08 maximum admitted error rate must be in [0, 1)")
        gates = E08Gates(
            overload_profiles=tuple(overload_raw),
            minimum_goodput_gain_percent_vs_best_baseline=_positive_number(
                gates_table, "minimum_goodput_gain_percent_vs_best_baseline"
            ),
            minimum_fairness_jain=_positive_number(
                gates_table, "minimum_fairness_jain"
            ),
            maximum_fairness_regression_vs_fixed=_positive_number(
                gates_table, "maximum_fairness_regression_vs_fixed"
            ),
            maximum_admitted_error_rate=maximum_error_rate,
        )
        if gates.minimum_fairness_jain > 1:
            raise ConfigError("E08 minimum fairness must not exceed 1")

        return cls(
            source_path=source_path,
            source_sha256=_sha256(source_path),
            name=_string(experiment, "name"),
            description=_string(experiment, "description"),
            server_config_path=server_config_path,
            server_config_sha256=server.source_sha256,
            server_profile=server.profile_name,
            base_url=_string(experiment, "base_url"),
            served_model_name=_string(experiment, "served_model_name"),
            policies=tuple(policies_raw),
            repetitions=repetitions,
            seeds=tuple(seeds_raw),
            warmup_requests=_positive_int(experiment, "warmup_requests"),
            request_timeout_seconds=_positive_number(
                experiment, "request_timeout_seconds"
            ),
            drain_timeout_seconds=_positive_number(
                experiment, "drain_timeout_seconds"
            ),
            client_max_workers=_positive_int(experiment, "client_max_workers"),
            maximum_dispatch_lag_ms=_positive_number(
                experiment, "maximum_dispatch_lag_ms"
            ),
            cooldown_seconds=_positive_number(experiment, "cooldown_seconds"),
            slo_ttft_ms=_positive_number(experiment, "slo_ttft_ms"),
            slo_tpot_ms=_positive_number(experiment, "slo_tpot_ms"),
            fixed_max_inflight=_positive_int(fixed, "max_inflight"),
            token_max_inflight=_positive_int(adaptive, "max_inflight"),
            initial_token_budget=initial_budget,
            min_token_budget=min_budget,
            max_token_budget=max_budget,
            decrease_factor=decrease_factor,
            increase_step_tokens=_positive_int(adaptive, "increase_step_tokens"),
            success_window=_positive_int(adaptive, "success_window"),
            workloads=tuple(workloads),
            profiles=tuple(profiles),
            gates=gates,
        )

    def profile(self, name: str) -> E08Profile:
        for profile in self.profiles:
            if profile.name == name:
                return profile
        raise ConfigError(f"Unknown E08 profile: {name}")

    def workload(self, name: str) -> E08Workload:
        for workload in self.workloads:
            if workload.name == name:
                return workload
        raise ConfigError(f"Unknown E08 workload: {name}")


def _trace_payload_sha256(payload: dict[str, Any]) -> str:
    normalized = json.dumps(
        payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(normalized).hexdigest()


def build_e08_trace(
    config: E08Config, profile_name: str, repetition: int
) -> dict[str, Any]:
    profile = config.profile(profile_name)
    if repetition < 1 or repetition > config.repetitions:
        raise ConfigError(
            f"E08 repetition must be between 1 and {config.repetitions}"
        )
    seed = config.seeds[repetition - 1]
    arrival_rng = random.Random(seed)
    class_rng = random.Random(seed ^ 0xE08)
    offsets: list[float] = []
    if profile.mode == "steady":
        offset = 0.0
        while True:
            offset += arrival_rng.expovariate(profile.rate_rps)
            if offset >= profile.duration_seconds:
                break
            offsets.append(offset)
    else:
        assert profile.burst_rate_rps is not None
        assert profile.background_rate_rps is not None
        assert profile.burst_period_seconds is not None
        assert profile.burst_duration_seconds is not None
        offset = 0.0
        maximum_rate = profile.burst_rate_rps
        while True:
            offset += arrival_rng.expovariate(maximum_rate)
            if offset >= profile.duration_seconds:
                break
            in_burst = (
                offset % profile.burst_period_seconds
            ) < profile.burst_duration_seconds
            instantaneous_rate = (
                profile.burst_rate_rps
                if in_burst
                else profile.background_rate_rps
            )
            if arrival_rng.random() <= instantaneous_rate / maximum_rate:
                offsets.append(offset)

    class_cycle = [
        name
        for workload in config.workloads
        for name in [workload.name] * workload.weight
    ]
    requests: list[dict[str, Any]] = []
    for cycle_start in range(0, len(offsets), len(class_cycle)):
        cycle = list(class_cycle)
        class_rng.shuffle(cycle)
        for position, offset in enumerate(
            offsets[cycle_start : cycle_start + len(cycle)]
        ):
            index = cycle_start + position
            workload = config.workload(cycle[position])
            requests.append(
                {
                    "id": f"{profile.name}-r{repetition}-{index + 1:04d}",
                    "arrival_offset_seconds": round(offset, 9),
                    "workload": workload.name,
                    "input_tokens": workload.input_tokens,
                    "output_tokens": workload.output_tokens,
                    "token_cost": workload.token_cost,
                    "prompt_seed": seed * 100000 + index + 1,
                }
            )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "kind": "e08_request_trace",
        "experiment_config": str(config.source_path),
        "experiment_config_sha256": config.source_sha256,
        "profile": asdict(profile),
        "repetition": repetition,
        "seed": seed,
        "request_count": len(requests),
        "requests": requests,
    }
    payload["trace_sha256"] = _trace_payload_sha256(payload)
    return payload


def validate_e08_trace(trace: dict[str, Any], config: E08Config) -> None:
    if not isinstance(trace, dict) or trace.get("kind") != "e08_request_trace":
        raise ConfigError("E08 trace must be an e08_request_trace object")
    recorded_hash = trace.get("trace_sha256")
    payload = dict(trace)
    payload.pop("trace_sha256", None)
    if not isinstance(recorded_hash, str) or recorded_hash != _trace_payload_sha256(payload):
        raise ConfigError("E08 trace SHA-256 does not match its payload")
    if trace.get("experiment_config_sha256") != config.source_sha256:
        raise ConfigError("E08 trace was generated from a different config")
    profile_data = trace.get("profile")
    repetition = trace.get("repetition")
    if not isinstance(profile_data, dict) or not isinstance(repetition, int):
        raise ConfigError("E08 trace has invalid profile or repetition metadata")
    expected = build_e08_trace(config, str(profile_data.get("name")), repetition)
    if expected.get("trace_sha256") != recorded_hash:
        raise ConfigError("E08 trace does not match the deterministic frozen generator")


def prepare_e08_traces(
    config_path: str | Path,
    output_dir: str | Path = "artifacts/results/e08_admission/traces",
) -> list[Path]:
    config = E08Config.from_file(config_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for profile in config.profiles:
        for repetition in range(1, config.repetitions + 1):
            trace = build_e08_trace(config, profile.name, repetition)
            path = output / f"{profile.name}-r{repetition}.json"
            path.write_text(
                json.dumps(trace, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            paths.append(path)
    return paths


@dataclass(frozen=True)
class AdmissionLease:
    request_id: str
    workload: str
    token_cost: int


class AdmissionController:
    def __init__(self, policy: str, config: E08Config) -> None:
        if policy not in config.policies:
            raise ConfigError(f"Unknown E08 admission policy: {policy}")
        self.policy = policy
        self.config = config
        self._lock = threading.Lock()
        self._active = 0
        self._inflight_tokens = 0
        self._peak_active = 0
        self._peak_tokens = 0
        self._token_budget = config.initial_token_budget
        self._successes_since_adjustment = 0
        self._admitted = 0
        self._rejected = 0
        self._budget_decreases = 0
        self._budget_increases = 0
        self._active_ids: set[str] = set()

    def try_acquire(
        self, request_id: str, workload: str, token_cost: int
    ) -> AdmissionLease | None:
        if token_cost <= 0:
            raise ValueError("token_cost must be positive")
        with self._lock:
            if request_id in self._active_ids:
                raise RuntimeError(f"Duplicate active request id: {request_id}")
            admitted = True
            if self.policy == "fixed_concurrency":
                admitted = self._active < self.config.fixed_max_inflight
            elif self.policy == "token_aware":
                admitted = bool(
                    self._active < self.config.token_max_inflight
                    and self._inflight_tokens + token_cost <= self._token_budget
                )
            if not admitted:
                self._rejected += 1
                return None
            self._active += 1
            self._inflight_tokens += token_cost
            self._admitted += 1
            self._active_ids.add(request_id)
            self._peak_active = max(self._peak_active, self._active)
            self._peak_tokens = max(self._peak_tokens, self._inflight_tokens)
            return AdmissionLease(request_id, workload, token_cost)

    def release(self, lease: AdmissionLease, slo_met: bool) -> None:
        with self._lock:
            if lease.request_id not in self._active_ids:
                raise RuntimeError(f"Request is not active: {lease.request_id}")
            self._active_ids.remove(lease.request_id)
            self._active -= 1
            self._inflight_tokens -= lease.token_cost
            if self._active < 0 or self._inflight_tokens < 0:
                raise RuntimeError("Admission accounting underflow")
            if self.policy != "token_aware":
                return
            if not slo_met:
                decreased = max(
                    self.config.min_token_budget,
                    int(
                        self._token_budget * self.config.decrease_factor
                        // self.config.increase_step_tokens
                        * self.config.increase_step_tokens
                    ),
                )
                if decreased < self._token_budget:
                    self._token_budget = decreased
                    self._budget_decreases += 1
                self._successes_since_adjustment = 0
                return
            self._successes_since_adjustment += 1
            if self._successes_since_adjustment >= self.config.success_window:
                increased = min(
                    self.config.max_token_budget,
                    self._token_budget + self.config.increase_step_tokens,
                )
                if increased > self._token_budget:
                    self._token_budget = increased
                    self._budget_increases += 1
                self._successes_since_adjustment = 0

    def snapshot(self) -> dict[str, int | str]:
        with self._lock:
            return {
                "policy": self.policy,
                "active": self._active,
                "inflight_tokens": self._inflight_tokens,
                "peak_active": self._peak_active,
                "peak_inflight_tokens": self._peak_tokens,
                "current_token_budget": self._token_budget,
                "admitted": self._admitted,
                "rejected": self._rejected,
                "budget_decreases": self._budget_decreases,
                "budget_increases": self._budget_increases,
            }
