from __future__ import annotations

import json
import hashlib
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

from qwen_serve_lab.commands import build_serve_command
from qwen_serve_lab.config import BenchmarkMatrix, ConfigError, ServeConfig
from qwen_serve_lab.e07_data import audit_e07_dataset, prepare_e07_dataset
from qwen_serve_lab.e07_readiness import write_e07_readiness_report
from qwen_serve_lab.e07_training import (
    E07TrainingConfig,
    inspect_e07_adapter,
)


ROOT = Path(__file__).resolve().parents[1]


class E07SetupTests(unittest.TestCase):
    def test_dataset_builder_is_balanced_grouped_and_test_isolated(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            train_path, validation_path, manifest_path = prepare_e07_dataset(
                ROOT / "datasets/e07_ai_infra_sft_source.json",
                ROOT / "datasets/e05_ai_infra_quality.json",
                output,
            )
            train = [json.loads(line) for line in train_path.read_text().splitlines()]
            validation = [
                json.loads(line) for line in validation_path.read_text().splitlines()
            ]
            manifest = json.loads(manifest_path.read_text())
            self.assertEqual(len(train), 250)
            self.assertEqual(len(validation), 100)
            self.assertEqual(len({row["root_cause"] for row in train}), 10)
            self.assertFalse(
                {row["source_group"] for row in train}
                & {row["source_group"] for row in validation}
            )
            self.assertEqual(manifest["train_source_groups"], 50)
            self.assertEqual(manifest["validation_source_groups"], 20)
            self.assertTrue(audit_e07_dataset(output)["passed"])

    def test_training_profiles_freeze_smoke_primary_and_rank_ablation(self) -> None:
        smoke = E07TrainingConfig.from_file(
            ROOT / "configs/train/e07_qlora_smoke_rank8.toml"
        )
        rank8 = E07TrainingConfig.from_file(
            ROOT / "configs/train/e07_qlora_rank8.toml"
        )
        rank16 = E07TrainingConfig.from_file(
            ROOT / "configs/train/e07_qlora_rank16.toml"
        )
        self.assertTrue(smoke.is_smoke)
        self.assertEqual(smoke.max_train_samples, 100)
        self.assertEqual((rank8.rank, rank16.rank), (8, 16))
        self.assertEqual((rank8.alpha, rank16.alpha), (16, 32))
        self.assertEqual(rank8.target_modules, rank16.target_modules)
        self.assertEqual(rank8.max_seq_length, 1024)
        self.assertEqual(rank8.micro_batch_size, 1)
        self.assertTrue(rank8.gradient_checkpointing)

    def test_base_lora_server_and_matrices_are_paired(self) -> None:
        base = ServeConfig.from_file(ROOT / "configs/serve/e07_base.toml")
        lora = ServeConfig.from_file(ROOT / "configs/serve/e07_lora.toml")
        ignored = {
            "profile_name",
            "description",
            "source_path",
            "source_sha256",
            "enable_lora",
            "lora_name",
            "lora_path",
            "max_loras",
            "max_lora_rank",
            "lora_manifest_sha256",
            "lora_weights_sha256",
        }
        base_data = asdict(base)
        lora_data = asdict(lora)
        self.assertTrue(
            all(
                base_data[key] == lora_data[key]
                for key in base_data
                if key not in ignored
            )
        )
        command = build_serve_command(lora)
        self.assertIn("--enable-lora", command)
        self.assertIn("ai-infra-triage-r8=artifacts/adapters/e07/rank8", command)

        base_matrix = BenchmarkMatrix.from_file(
            ROOT / "configs/matrix/e07_base.toml"
        )
        lora_matrix = BenchmarkMatrix.from_file(
            ROOT / "configs/matrix/e07_lora.toml"
        )
        self.assertEqual(len(base_matrix.configs), 6)
        self.assertEqual(len(lora_matrix.configs), 6)
        base_cells = {
            (
                config.profile_name.removeprefix("e07_base_"),
                config.input_len,
                config.output_len,
                config.max_concurrency,
                config.seed,
            )
            for config in base_matrix.configs
        }
        lora_cells = {
            (
                config.profile_name.removeprefix("e07_lora_"),
                config.input_len,
                config.output_len,
                config.max_concurrency,
                config.seed,
            )
            for config in lora_matrix.configs
        }
        self.assertEqual(base_cells, lora_cells)

    def test_adapter_inspection_checks_rank_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            adapter = Path(temporary)
            (adapter / "adapter_config.json").write_text(
                json.dumps({"peft_type": "LORA", "r": 8})
            )
            (adapter / "adapter_model.safetensors").write_bytes(b"weights")
            (adapter / "training_manifest.json").write_text(
                json.dumps(
                    {
                        "kind": "e07_training_manifest",
                        "status": "COMPLETE",
                        "config": {"rank": 8},
                        "adapter_weights_sha256": hashlib.sha256(b"weights").hexdigest(),
                    }
                )
            )
            inspection = inspect_e07_adapter(adapter, expected_rank=8)
            self.assertTrue(inspection["passed"])
            config = ServeConfig.from_file(ROOT / "configs/serve/e07_lora.toml")
            effective = config.with_lora_adapter(adapter)
            self.assertEqual(effective.lora_path, adapter.resolve())
            self.assertIsNotNone(effective.lora_manifest_sha256)
            (adapter / "adapter_config.json").write_text(
                json.dumps({"peft_type": "LORA", "r": 16})
            )
            with self.assertRaises(ConfigError):
                config.with_lora_adapter(adapter)

    def test_readiness_audit_is_explicitly_non_gpu(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            json_path, _, passed = write_e07_readiness_report(
                ROOT, Path(temporary)
            )
            report = json.loads(json_path.read_text())
            self.assertTrue(passed)
            self.assertEqual(report["status"], "READY_FOR_GPU")
            self.assertEqual(report["gpu_execution"], "DEFERRED")
            self.assertEqual(report["planned_formal_benchmark_runs"], 36)


if __name__ == "__main__":
    unittest.main()
