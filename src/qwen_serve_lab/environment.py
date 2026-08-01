from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _run(command: list[str], cwd: Path | None = None) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return {"ok": False, "error": str(exc)}
    return {
        "ok": completed.returncode == 0,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def _is_wsl() -> bool:
    release = platform.release().lower()
    if "microsoft" in release or "wsl" in release:
        return True
    try:
        return "microsoft" in Path("/proc/version").read_text().lower()
    except OSError:
        return False


def _torch_snapshot() -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:
        return {"installed": False, "error": str(exc)}

    snapshot: dict[str, Any] = {
        "installed": True,
        "version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        snapshot["device_count"] = torch.cuda.device_count()
        snapshot["device_name"] = torch.cuda.get_device_name(0)
        snapshot["device_capability"] = list(torch.cuda.get_device_capability(0))
    return snapshot


def _git_snapshot(project_root: Path) -> dict[str, Any]:
    revision = _run(["git", "rev-parse", "HEAD"], cwd=project_root)
    status = _run(["git", "status", "--short"], cwd=project_root)
    return {
        "available": revision.get("ok", False),
        "commit": revision.get("stdout") if revision.get("ok") else None,
        "dirty": bool(status.get("stdout")) if status.get("ok") else None,
        "status": status.get("stdout") if status.get("ok") else None,
    }


def parse_nvidia_smi_rows(output: str) -> list[dict[str, Any]]:
    gpus: list[dict[str, Any]] = []
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 5:
            continue
        try:
            memory_total_mib = float(parts[2])
            temperature_c = float(parts[3])
            power_limit_w = float(parts[4])
        except ValueError:
            continue
        gpus.append(
            {
                "name": parts[0],
                "driver_version": parts[1],
                "memory_total_mib": memory_total_mib,
                "temperature_c": temperature_c,
                "power_limit_w": power_limit_w,
            }
        )
    return gpus


def collect_environment(project_root: str | Path = ".") -> dict[str, Any]:
    root = Path(project_root).resolve()
    nvidia_query = _run(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total,temperature.gpu,power.limit",
            "--format=csv,noheader,nounits",
        ]
    )
    nvidia_query["gpus"] = parse_nvidia_smi_rows(nvidia_query.get("stdout", ""))
    return {
        "schema_version": 1,
        "collected_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(root),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "wsl": _is_wsl(),
        },
        "python": {
            "version": sys.version,
            "executable": sys.executable,
        },
        "executables": {
            name: shutil.which(name)
            for name in ("uv", "vllm", "nvidia-smi", "git")
        },
        "vllm": _run(["vllm", "--version"]),
        "nvidia_smi": nvidia_query,
        "torch": _torch_snapshot(),
        "git": _git_snapshot(root),
        "runtime": {
            "cwd": os.getcwd(),
            "pid": os.getpid(),
        },
    }


@dataclass(frozen=True)
class DoctorCheck:
    name: str
    passed: bool
    detail: str
    required: bool = True


def run_doctor() -> list[DoctorCheck]:
    snapshot = collect_environment()
    platform_info = snapshot["platform"]
    torch_info = snapshot["torch"]
    executables = snapshot["executables"]
    nvidia_info = snapshot["nvidia_smi"]
    gpus = nvidia_info.get("gpus", [])
    primary_gpu = gpus[0] if gpus else {}

    return [
        DoctorCheck(
            "linux",
            platform_info["system"] == "Linux",
            f"detected {platform_info['system']}",
        ),
        DoctorCheck(
            "wsl2",
            bool(platform_info["wsl"]),
            "WSL detected" if platform_info["wsl"] else "WSL not detected",
        ),
        DoctorCheck(
            "nvidia-smi",
            bool(executables["nvidia-smi"]) and bool(nvidia_info.get("ok")),
            nvidia_info.get("stdout") or nvidia_info.get("error", "not available"),
        ),
        DoctorCheck(
            "target-gpu",
            "RTX 5070" in str(primary_gpu.get("name", "")),
            str(primary_gpu.get("name", "target GPU not detected")),
        ),
        DoctorCheck(
            "gpu-memory",
            float(primary_gpu.get("memory_total_mib", 0)) >= 11000,
            f"{primary_gpu.get('memory_total_mib', 0)} MiB detected; >= 11000 MiB required",
        ),
        DoctorCheck(
            "vllm",
            bool(executables["vllm"]),
            snapshot["vllm"].get("stdout")
            or snapshot["vllm"].get("error", "not installed"),
        ),
        DoctorCheck(
            "torch-cuda",
            bool(torch_info.get("cuda_available")),
            json.dumps(torch_info, ensure_ascii=True, sort_keys=True),
        ),
        DoctorCheck(
            "uv",
            bool(executables["uv"]),
            executables["uv"] or "not installed",
            required=False,
        ),
    ]


def write_json(data: dict[str, Any], path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output


def checks_as_dict(checks: list[DoctorCheck]) -> list[dict[str, Any]]:
    return [asdict(check) for check in checks]
