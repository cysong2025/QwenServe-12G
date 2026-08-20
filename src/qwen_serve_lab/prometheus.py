from __future__ import annotations

import math
import re
from datetime import datetime, timezone
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import urlopen


class MetricsError(ValueError):
    """Raised when required Prometheus evidence cannot be collected."""


SAMPLE_PATTERN = re.compile(
    r"^(?P<name>[A-Za-z_:][A-Za-z0-9_:]*)"
    r"(?:\{.*\})?\s+(?P<value>[^\s]+)(?:\s+\d+)?$"
)
QUERY_METRICS = (
    "vllm:prefix_cache_queries",
    "vllm:prefix_cache_queries_total",
    "vllm:gpu_prefix_cache_queries",
    "vllm:gpu_prefix_cache_queries_total",
)
HIT_METRICS = (
    "vllm:prefix_cache_hits",
    "vllm:prefix_cache_hits_total",
    "vllm:gpu_prefix_cache_hits",
    "vllm:gpu_prefix_cache_hits_total",
)


def fetch_metrics(base_url: str, timeout_seconds: float = 10) -> str:
    url = base_url.rstrip("/") + "/metrics"
    try:
        with urlopen(url, timeout=timeout_seconds) as response:
            return response.read().decode("utf-8")
    except (HTTPError, URLError, OSError, UnicodeDecodeError) as exc:
        raise MetricsError(f"Cannot read Prometheus metrics from {url}: {exc}") from exc


def parse_samples(text: str) -> dict[str, float]:
    samples: dict[str, float] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = SAMPLE_PATTERN.fullmatch(line)
        if match is None:
            continue
        try:
            value = float(match.group("value"))
        except ValueError:
            continue
        if not math.isfinite(value):
            continue
        name = match.group("name")
        samples[name] = samples.get(name, 0.0) + value
    return samples


def _first_metric(
    samples: dict[str, float], candidates: tuple[str, ...]
) -> tuple[str | None, float | None]:
    for name in candidates:
        if name in samples:
            return name, samples[name]
    return None, None


def prefix_snapshot(text: str) -> dict[str, Any]:
    samples = parse_samples(text)
    query_metric, query_tokens = _first_metric(samples, QUERY_METRICS)
    hit_metric, hit_tokens = _first_metric(samples, HIT_METRICS)
    return {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "query_metric": query_metric,
        "hit_metric": hit_metric,
        "query_tokens": query_tokens,
        "hit_tokens": hit_tokens,
    }


def prefix_delta(
    before: dict[str, Any], after: dict[str, Any]
) -> dict[str, Any]:
    before_queries = before.get("query_tokens")
    after_queries = after.get("query_tokens")
    before_hits = before.get("hit_tokens")
    after_hits = after.get("hit_tokens")

    if before_queries is None and after_queries is None:
        return {
            "captured": True,
            "query_metric": None,
            "hit_metric": None,
            "query_tokens": None,
            "hit_tokens": None,
            "hit_rate_percent": None,
        }
    if None in (before_queries, after_queries, before_hits, after_hits):
        raise MetricsError("Prefix cache counters appeared or disappeared during a run")

    query_tokens = float(after_queries) - float(before_queries)
    hit_tokens = float(after_hits) - float(before_hits)
    if query_tokens < 0 or hit_tokens < 0:
        raise MetricsError("Prefix cache counters decreased during a run")
    if hit_tokens > query_tokens:
        raise MetricsError("Prefix cache hit-token delta exceeds query-token delta")
    return {
        "captured": True,
        "query_metric": after.get("query_metric"),
        "hit_metric": after.get("hit_metric"),
        "query_tokens": query_tokens,
        "hit_tokens": hit_tokens,
        "hit_rate_percent": (
            hit_tokens / query_tokens * 100 if query_tokens > 0 else None
        ),
    }
