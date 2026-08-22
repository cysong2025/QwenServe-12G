from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import tomllib
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from qwen_serve_lab.e07_data import audit_e07_dataset
from qwen_serve_lab.environment import collect_environment
from qwen_serve_lab.results import ResultError
from qwen_serve_lab.telemetry import NvidiaSmiSampler, summarize_telemetry


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _required(section: dict[str, Any], key: str, expected: type) -> Any:
    value = section.get(key)
    if not isinstance(value, expected) or (expected is str and not value.strip()):
        raise ResultError(f"E07 training field {key} must be {expected.__name__}")
    return value


def _positive_int(section: dict[str, Any], key: str) -> int:
    value = _required(section, key, int)
    if isinstance(value, bool) or value <= 0:
        raise ResultError(f"E07 training field {key} must be positive")
    return value


def _number(section: dict[str, Any], key: str) -> float:
    value = section.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ResultError(f"E07 training field {key} must be numeric")
    return float(value)


@dataclass(frozen=True)
class E07TrainingConfig:
    profile_name: str
    description: str
    source_path: Path
    source_sha256: str
    model: str
    revision: str
    train_file: Path
    validation_file: Path
    dataset_manifest: Path
    output_dir: Path
    rank: int
    alpha: int
    dropout: float
    target_modules: tuple[str, ...]
    max_seq_length: int
    micro_batch_size: int
    gradient_accumulation_steps: int
    epochs: float
    max_steps: int
    learning_rate: float
    warmup_ratio: float
    max_train_samples: int
    seed: int
    gradient_checkpointing: bool

    @classmethod
    def from_file(cls, path: str | Path) -> "E07TrainingConfig":
        source_path = Path(path).resolve()
        try:
            with source_path.open("rb") as handle:
                data = tomllib.load(handle)
        except (OSError, tomllib.TOMLDecodeError) as exc:
            raise ResultError(f"Cannot read E07 training config {source_path}: {exc}") from exc
        sections: dict[str, dict[str, Any]] = {}
        for name in ("profile", "model", "data", "lora", "training", "output"):
            value = data.get(name)
            if not isinstance(value, dict):
                raise ResultError(f"E07 training config lacks [{name}]")
            sections[name] = value
        profile = sections["profile"]
        model = sections["model"]
        dataset = sections["data"]
        lora = sections["lora"]
        training = sections["training"]
        output = sections["output"]

        target_modules = lora.get("target_modules")
        if (
            not isinstance(target_modules, list)
            or not target_modules
            or not all(isinstance(item, str) and item for item in target_modules)
            or len(set(target_modules)) != len(target_modules)
        ):
            raise ResultError("E07 target_modules must be unique non-empty strings")
        rank = _positive_int(lora, "rank")
        alpha = _positive_int(lora, "alpha")
        if rank not in {8, 16} or alpha != rank * 2:
            raise ResultError("E07 LoRA requires rank 8/16 and alpha=2*rank")
        dropout = _number(lora, "dropout")
        if not 0 <= dropout < 1:
            raise ResultError("E07 LoRA dropout must be in [0, 1)")
        epochs = _number(training, "epochs")
        learning_rate = _number(training, "learning_rate")
        warmup_ratio = _number(training, "warmup_ratio")
        if epochs <= 0 or learning_rate <= 0 or not 0 <= warmup_ratio < 1:
            raise ResultError("E07 epochs/learning_rate/warmup_ratio are invalid")
        max_steps = training.get("max_steps")
        max_train_samples = training.get("max_train_samples")
        if (
            not isinstance(max_steps, int)
            or isinstance(max_steps, bool)
            or max_steps < 0
            or not isinstance(max_train_samples, int)
            or isinstance(max_train_samples, bool)
            or max_train_samples < 0
        ):
            raise ResultError("E07 max_steps/max_train_samples must be non-negative")
        return cls(
            profile_name=_required(profile, "name", str),
            description=_required(profile, "description", str),
            source_path=source_path,
            source_sha256=_sha256(source_path),
            model=_required(model, "name", str),
            revision=_required(model, "revision", str),
            train_file=Path(_required(dataset, "train_file", str)),
            validation_file=Path(_required(dataset, "validation_file", str)),
            dataset_manifest=Path(_required(dataset, "manifest", str)),
            output_dir=Path(_required(output, "adapter_dir", str)),
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            target_modules=tuple(target_modules),
            max_seq_length=_positive_int(training, "max_seq_length"),
            micro_batch_size=_positive_int(training, "micro_batch_size"),
            gradient_accumulation_steps=_positive_int(
                training, "gradient_accumulation_steps"
            ),
            epochs=epochs,
            max_steps=max_steps,
            learning_rate=learning_rate,
            warmup_ratio=warmup_ratio,
            max_train_samples=max_train_samples,
            seed=_positive_int(training, "seed"),
            gradient_checkpointing=_required(
                training, "gradient_checkpointing", bool
            ),
        )

    @property
    def is_smoke(self) -> bool:
        return self.max_train_samples == 100 and self.max_steps > 0


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    except (OSError, json.JSONDecodeError) as exc:
        raise ResultError(f"Cannot read E07 training data {path}: {exc}") from exc
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise ResultError(f"E07 training data is empty or malformed: {path}")
    return rows


def _assert_no_active_server(path: str | Path = "artifacts/server/active.json") -> None:
    marker_path = Path(path)
    if not marker_path.is_file():
        return
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    pid = marker.get("pid") if isinstance(marker, dict) else None
    if not isinstance(pid, int) or pid <= 0:
        return
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return
    except PermissionError:
        pass
    raise ResultError(
        f"vLLM process {pid} is still active; stop terminal 1 before E07 training"
    )


def _git_state() -> dict[str, Any]:
    def command(*args: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", *args],
                check=True,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError):
            return None
        return result.stdout.strip()

    return {
        "commit": command("rev-parse", "HEAD"),
        "branch": command("branch", "--show-current"),
        "dirty": bool(command("status", "--porcelain")),
    }


def _serialize_config(config: E07TrainingConfig) -> dict[str, Any]:
    payload = asdict(config)
    for key, value in list(payload.items()):
        if isinstance(value, Path):
            payload[key] = str(value)
        elif isinstance(value, tuple):
            payload[key] = list(value)
    return payload


def _safe_telemetry_summary(path: Path) -> dict[str, float | int | None]:
    try:
        return summarize_telemetry(path)
    except OSError:
        return {
            "sample_count": 0,
            "error_samples": 0,
            "peak_memory_used_mib": None,
            "mean_gpu_utilization_percent": None,
            "max_temperature_c": None,
            "mean_power_draw_w": None,
            "mean_sm_clock_mhz": None,
        }


def _tokenize_rows(
    rows: list[dict[str, Any]], tokenizer: Any, max_seq_length: int
) -> list[dict[str, list[int]]]:
    encoded: list[dict[str, list[int]]] = []
    for row in rows:
        messages = row.get("messages")
        if not isinstance(messages, list) or len(messages) != 3:
            raise ResultError(f"E07 row {row.get('id')} must contain three messages")
        prompt_text = tokenizer.apply_chat_template(
            messages[:-1], tokenize=False, add_generation_prompt=True
        )
        full_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        prompt_ids = tokenizer(
            prompt_text,
            truncation=True,
            max_length=max_seq_length,
            add_special_tokens=False,
        )["input_ids"]
        full = tokenizer(
            full_text,
            truncation=True,
            max_length=max_seq_length,
            add_special_tokens=False,
        )
        input_ids = full["input_ids"]
        if len(prompt_ids) >= len(input_ids):
            raise ResultError(
                f"E07 row {row.get('id')} response was truncated; increase max_seq_length"
            )
        labels = [-100] * len(prompt_ids) + input_ids[len(prompt_ids) :]
        encoded.append(
            {
                "input_ids": input_ids,
                "attention_mask": full["attention_mask"],
                "labels": labels,
            }
        )
    return encoded


class _ListDataset:
    def __init__(self, rows: list[dict[str, list[int]]]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return self.rows[index]


def inspect_e07_adapter(
    adapter_dir: str | Path,
    expected_rank: int | None = None,
) -> dict[str, Any]:
    path = Path(adapter_dir)
    required = (
        "adapter_config.json",
        "adapter_model.safetensors",
        "training_manifest.json",
    )
    missing = [name for name in required if not (path / name).is_file()]
    if missing:
        raise ResultError("E07 adapter is incomplete; missing: " + ", ".join(missing))
    try:
        config = json.loads((path / "adapter_config.json").read_text(encoding="utf-8"))
        manifest = json.loads(
            (path / "training_manifest.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise ResultError(f"Cannot inspect E07 adapter: {exc}") from exc
    rank = config.get("r") if isinstance(config, dict) else None
    weights_sha256 = _sha256(path / "adapter_model.safetensors")
    manifest_config = manifest.get("config") if isinstance(manifest, dict) else None
    checks = {
        "adapter_kind": isinstance(config, dict) and config.get("peft_type") == "LORA",
        "rank": isinstance(rank, int) and rank in {8, 16},
        "expected_rank": expected_rank is None or rank == expected_rank,
        "training_manifest": (
            isinstance(manifest, dict)
            and manifest.get("kind") == "e07_training_manifest"
            and manifest.get("status") == "COMPLETE"
        ),
        "manifest_rank": (
            isinstance(manifest_config, dict)
            and manifest_config.get("rank") == rank
        ),
        "weights_hash": (
            isinstance(manifest, dict)
            and manifest.get("adapter_weights_sha256") == weights_sha256
        ),
        "weight_nonempty": (path / "adapter_model.safetensors").stat().st_size > 0,
    }
    return {
        "passed": all(checks.values()),
        "adapter_dir": str(path),
        "rank": rank,
        "checks": checks,
        "adapter_config_sha256": _sha256(path / "adapter_config.json"),
        "adapter_weights_sha256": weights_sha256,
        "training_manifest_sha256": _sha256(path / "training_manifest.json"),
        "training_manifest": manifest,
    }


def write_e07_adapter_report(
    adapter_dir: str | Path,
    output_dir: str | Path,
    expected_rank: int = 8,
) -> tuple[Path, Path, bool]:
    inspection = inspect_e07_adapter(adapter_dir, expected_rank=expected_rank)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "adapter.json"
    markdown_path = output / "adapter.md"
    json_path.write_text(
        json.dumps(inspection, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# E07 Adapter Inspection",
        "",
        f"Status: **{'PASS' if inspection['passed'] else 'FAIL'}**",
        f"Adapter directory: `{inspection['adapter_dir']}`",
        f"LoRA rank: {inspection['rank']}",
        f"Weights SHA-256: `{inspection['adapter_weights_sha256']}`",
        f"Training manifest SHA-256: `{inspection['training_manifest_sha256']}`",
        "",
        "| Check | Status |",
        "|---|---|",
    ]
    lines.extend(
        f"| {name} | {'PASS' if passed else 'FAIL'} |"
        for name, passed in inspection["checks"].items()
    )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, markdown_path, bool(inspection["passed"])


def run_e07_training(
    config_path: str | Path,
    model_path: str | Path,
) -> Path:
    config = E07TrainingConfig.from_file(config_path)
    _assert_no_active_server()
    model_dir = Path(model_path).expanduser().resolve()
    if not (model_dir / "config.json").is_file() or not any(
        model_dir.glob("*.safetensors")
    ):
        raise ResultError(f"E07 local model snapshot is incomplete: {model_dir}")
    dataset_audit = audit_e07_dataset(config.dataset_manifest.parent)
    if not dataset_audit["passed"]:
        raise ResultError("E07 dataset audit failed before training")
    manifest = dataset_audit["manifest"]
    if (
        _sha256(config.train_file) != manifest.get("train_sha256")
        or _sha256(config.validation_file) != manifest.get("validation_sha256")
    ):
        raise ResultError("E07 training config does not reference the audited dataset")
    if config.output_dir.exists() and any(config.output_dir.iterdir()):
        raise ResultError(
            f"E07 adapter output already exists and is non-empty: {config.output_dir}"
        )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    evidence_dir = Path("artifacts/results/e07_training") / config.profile_name
    evidence_dir.mkdir(parents=True, exist_ok=True)
    attempt_path = evidence_dir / f"{timestamp}-training-attempt.json"
    telemetry_path = evidence_dir / f"{timestamp}-telemetry.csv"
    attempt: dict[str, Any] = {
        "schema_version": 1,
        "kind": "e07_training_attempt",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "RUNNING",
        "profile": config.profile_name,
        "config": _serialize_config(config),
        "model_path": str(model_dir),
        "dataset_manifest_sha256": _sha256(config.dataset_manifest),
        "environment": collect_environment(),
        "git": _git_state(),
        "telemetry": str(telemetry_path),
        "error": None,
    }
    attempt_path.write_text(
        json.dumps(attempt, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )

    try:
        import accelerate
        import bitsandbytes
        import peft
        import torch
        import transformers
        from peft import (
            LoraConfig,
            get_peft_model,
            prepare_model_for_kbit_training,
        )
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            DataCollatorForSeq2Seq,
            Trainer,
            TrainingArguments,
            set_seed,
        )
    except ImportError as exc:
        attempt.update(
            {
                "status": "FAILED",
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "error": {"type": type(exc).__name__, "message": str(exc)},
            }
        )
        attempt_path.write_text(
            json.dumps(attempt, ensure_ascii=True, indent=2) + "\n",
            encoding="utf-8",
        )
        raise ResultError(
            "E07 training dependencies are missing; run make install-e07-train-deps"
        ) from exc
    if not torch.cuda.is_available():
        attempt.update(
            {
                "status": "FAILED",
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "error": {
                    "type": "CudaUnavailable",
                    "message": "torch.cuda.is_available() is false",
                },
            }
        )
        attempt_path.write_text(
            json.dumps(attempt, ensure_ascii=True, indent=2) + "\n",
            encoding="utf-8",
        )
        raise ResultError("E07 QLoRA training requires a CUDA GPU")

    try:
        set_seed(config.seed)
        tokenizer = AutoTokenizer.from_pretrained(
            model_dir, local_files_only=True, use_fast=True
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            local_files_only=True,
            quantization_config=quantization,
            device_map={"": 0},
            torch_dtype=torch.bfloat16,
        )
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=config.gradient_checkpointing,
        )
        model.config.use_cache = False
        lora_config = LoraConfig(
            r=config.rank,
            lora_alpha=config.alpha,
            lora_dropout=config.dropout,
            target_modules=list(config.target_modules),
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        trainable_parameters = sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )
        total_parameters = sum(parameter.numel() for parameter in model.parameters())
        train_rows = _load_jsonl(config.train_file)
        if config.max_train_samples:
            train_rows = train_rows[: config.max_train_samples]
        validation_rows = _load_jsonl(config.validation_file)
        train_dataset = _ListDataset(
            _tokenize_rows(train_rows, tokenizer, config.max_seq_length)
        )
        validation_dataset = _ListDataset(
            _tokenize_rows(validation_rows, tokenizer, config.max_seq_length)
        )
        config.output_dir.mkdir(parents=True, exist_ok=True)
        arguments = TrainingArguments(
            output_dir=str(config.output_dir / "checkpoints"),
            num_train_epochs=config.epochs,
            max_steps=config.max_steps if config.max_steps else -1,
            per_device_train_batch_size=config.micro_batch_size,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            learning_rate=config.learning_rate,
            lr_scheduler_type="cosine",
            warmup_ratio=config.warmup_ratio,
            logging_steps=1 if config.is_smoke else 5,
            eval_strategy="steps" if config.max_steps else "epoch",
            eval_steps=max(1, config.max_steps // 2) if config.max_steps else None,
            save_strategy="steps" if config.max_steps else "epoch",
            save_steps=config.max_steps if config.max_steps else 500,
            save_total_limit=1,
            bf16=True,
            fp16=False,
            gradient_checkpointing=config.gradient_checkpointing,
            optim="paged_adamw_8bit",
            report_to=[],
            remove_unused_columns=False,
            seed=config.seed,
            data_seed=config.seed,
        )
        collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model,
            padding=True,
            label_pad_token_id=-100,
            return_tensors="pt",
        )
        trainer = Trainer(
            model=model,
            args=arguments,
            train_dataset=train_dataset,
            eval_dataset=validation_dataset,
            data_collator=collator,
        )
        with NvidiaSmiSampler(telemetry_path):
            train_result = trainer.train()
            evaluation = trainer.evaluate()
        model.save_pretrained(config.output_dir, safe_serialization=True)
        tokenizer.save_pretrained(config.output_dir)
        weights_path = config.output_dir / "adapter_model.safetensors"
        if not weights_path.is_file():
            raise ResultError("E07 trainer did not produce adapter_model.safetensors")
    except Exception as exc:
        attempt.update(
            {
                "status": "FAILED",
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "telemetry_summary": _safe_telemetry_summary(telemetry_path),
                "error": {"type": type(exc).__name__, "message": str(exc)},
            }
        )
        attempt_path.write_text(
            json.dumps(attempt, ensure_ascii=True, indent=2) + "\n",
            encoding="utf-8",
        )
        if isinstance(exc, ResultError):
            raise
        raise ResultError(f"E07 training failed: {exc}") from exc
    training_manifest = {
        "schema_version": 1,
        "kind": "e07_training_manifest",
        "status": "COMPLETE",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "profile": config.profile_name,
        "config": _serialize_config(config),
        "model_path": str(model_dir),
        "model_config_sha256": _sha256(model_dir / "config.json"),
        "model_manifest_sha256": (
            _sha256(model_dir / "SHA256SUMS")
            if (model_dir / "SHA256SUMS").is_file()
            else None
        ),
        "dataset_manifest": str(config.dataset_manifest),
        "dataset_manifest_sha256": _sha256(config.dataset_manifest),
        "train_rows": len(train_rows),
        "validation_rows": len(validation_rows),
        "trainable_parameters": trainable_parameters,
        "total_parameters": total_parameters,
        "trainable_percent": trainable_parameters / total_parameters * 100,
        "train_metrics": {
            key: value
            for key, value in train_result.metrics.items()
            if isinstance(value, (int, float)) and math.isfinite(value)
        },
        "evaluation_metrics": {
            key: value
            for key, value in evaluation.items()
            if isinstance(value, (int, float)) and math.isfinite(value)
        },
        "adapter_weights_sha256": _sha256(weights_path),
        "training_attempt": str(attempt_path),
        "telemetry": str(telemetry_path),
        "telemetry_summary": _safe_telemetry_summary(telemetry_path),
        "log_history": trainer.state.log_history,
        "versions": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "accelerate": accelerate.__version__,
            "bitsandbytes": bitsandbytes.__version__,
            "peft": peft.__version__,
        },
        "environment": collect_environment(),
        "git": _git_state(),
    }
    manifest_path = config.output_dir / "training_manifest.json"
    manifest_path.write_text(
        json.dumps(training_manifest, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    attempt.update(
        {
            "status": "COMPLETE",
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "training_manifest": str(manifest_path),
            "adapter_weights_sha256": _sha256(weights_path),
            "telemetry_summary": _safe_telemetry_summary(telemetry_path),
        }
    )
    attempt_path.write_text(
        json.dumps(attempt, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    inspection = inspect_e07_adapter(config.output_dir, expected_rank=config.rank)
    if not inspection["passed"]:
        raise ResultError("E07 adapter inspection failed after training")
    return manifest_path
