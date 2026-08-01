from __future__ import annotations

import csv
import subprocess
import threading
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TELEMETRY_FIELDS = (
    "sampled_at_utc",
    "index",
    "name",
    "gpu_utilization_percent",
    "memory_used_mib",
    "memory_total_mib",
    "temperature_c",
    "power_draw_w",
    "sm_clock_mhz",
    "error",
)


def query_nvidia_smi() -> tuple[int, str, str]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,clocks.sm",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return 127, "", str(exc)
    return completed.returncode, completed.stdout.strip(), completed.stderr.strip()


def parse_nvidia_smi_telemetry(output: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 8:
            continue
        try:
            rows.append(
                {
                    "index": int(parts[0]),
                    "name": parts[1],
                    "gpu_utilization_percent": float(parts[2]),
                    "memory_used_mib": float(parts[3]),
                    "memory_total_mib": float(parts[4]),
                    "temperature_c": float(parts[5]),
                    "power_draw_w": float(parts[6]),
                    "sm_clock_mhz": float(parts[7]),
                }
            )
        except ValueError:
            continue
    return rows


class NvidiaSmiSampler:
    def __init__(
        self,
        output_path: str | Path,
        interval_seconds: float = 1.0,
        query: Callable[[], tuple[int, str, str]] = query_nvidia_smi,
    ) -> None:
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be positive")
        self.output_path = Path(output_path)
        self.interval_seconds = interval_seconds
        self.query = query
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("Sampler has already been started")
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(5.0, self.interval_seconds * 2))

    def _run(self) -> None:
        with self.output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=TELEMETRY_FIELDS)
            writer.writeheader()
            while not self._stop.is_set():
                sampled_at = datetime.now(timezone.utc).isoformat()
                returncode, stdout, stderr = self.query()
                parsed = parse_nvidia_smi_telemetry(stdout) if returncode == 0 else []
                if parsed:
                    for row in parsed:
                        writer.writerow(
                            {
                                "sampled_at_utc": sampled_at,
                                **row,
                                "error": "",
                            }
                        )
                else:
                    writer.writerow(
                        {
                            "sampled_at_utc": sampled_at,
                            "error": stderr or f"nvidia-smi exited {returncode}",
                        }
                    )
                handle.flush()
                self._stop.wait(self.interval_seconds)

    def __enter__(self) -> "NvidiaSmiSampler":
        self.start()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.stop()


def summarize_telemetry(path: str | Path) -> dict[str, float | int | None]:
    telemetry_path = Path(path)
    values: dict[str, list[float]] = {
        "gpu_utilization_percent": [],
        "memory_used_mib": [],
        "temperature_c": [],
        "power_draw_w": [],
        "sm_clock_mhz": [],
    }
    error_samples = 0
    with telemetry_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("error"):
                error_samples += 1
                continue
            for key in values:
                raw = row.get(key)
                if raw:
                    try:
                        values[key].append(float(raw))
                    except ValueError:
                        continue

    sample_count = len(values["memory_used_mib"])
    return {
        "sample_count": sample_count,
        "error_samples": error_samples,
        "peak_memory_used_mib": max(values["memory_used_mib"], default=None),
        "mean_gpu_utilization_percent": (
            sum(values["gpu_utilization_percent"])
            / len(values["gpu_utilization_percent"])
            if values["gpu_utilization_percent"]
            else None
        ),
        "max_temperature_c": max(values["temperature_c"], default=None),
        "mean_power_draw_w": (
            sum(values["power_draw_w"]) / len(values["power_draw_w"])
            if values["power_draw_w"]
            else None
        ),
        "mean_sm_clock_mhz": (
            sum(values["sm_clock_mhz"]) / len(values["sm_clock_mhz"])
            if values["sm_clock_mhz"]
            else None
        ),
    }
