from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from qwen_serve_lab.commands import (
    build_benchmark_command,
    build_serve_command,
    build_serve_environment,
    render_shell_command,
)
from qwen_serve_lab.config import (
    BenchmarkConfig,
    BenchmarkMatrix,
    ConfigError,
    ServeConfig,
)


ROOT = Path(__file__).resolve().parents[1]


class ServeConfigTests(unittest.TestCase):
    def test_baseline_command_is_explicit_and_pinned(self) -> None:
        config = ServeConfig.from_file(ROOT / "configs/serve/baseline.toml")
        command = build_serve_command(config)

        self.assertEqual(command[:3], ["vllm", "serve", config.model])
        self.assertIn("--revision", command)
        self.assertIn(config.revision, command)
        self.assertIn("--no-enable-prefix-caching", command)
        self.assertIn("--enable-per-request-metrics", command)
        self.assertEqual(config.max_model_len, 8192)
        self.assertEqual(
            build_serve_environment(config),
            {"VLLM_WSL2_ENABLE_PIN_MEMORY": "1"},
        )
        self.assertTrue(
            render_shell_command(command, build_serve_environment(config)).startswith(
                "env VLLM_WSL2_ENABLE_PIN_MEMORY=1 vllm serve"
            )
        )

    def test_prefix_profile_changes_only_the_intended_switch(self) -> None:
        baseline = ServeConfig.from_file(ROOT / "configs/serve/baseline.toml")
        prefix = ServeConfig.from_file(ROOT / "configs/serve/prefix_cache.toml")

        self.assertFalse(baseline.enable_prefix_caching)
        self.assertTrue(prefix.enable_prefix_caching)
        self.assertEqual(baseline.model, prefix.model)
        self.assertEqual(baseline.revision, prefix.revision)
        self.assertEqual(baseline.max_model_len, prefix.max_model_len)
        self.assertEqual(
            baseline.wsl2_enable_pin_memory, prefix.wsl2_enable_pin_memory
        )

    def test_local_model_override_omits_remote_revision(self) -> None:
        config = ServeConfig.from_file(ROOT / "configs/serve/baseline.toml")
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir)
            for filename in ("config.json", "tokenizer.json", "model.safetensors"):
                (model_path / filename).write_text("test")
            (model_path / "SHA256SUMS").write_text(
                "fixture  config.json\n", encoding="utf-8"
            )

            local_config = config.with_local_model(model_path)
            command = build_serve_command(local_config)

        self.assertEqual(command[:3], ["vllm", "serve", str(model_path.resolve())])
        self.assertNotIn("--revision", command)
        self.assertEqual(local_config.model, config.model)
        self.assertIsNotNone(local_config.local_model_manifest_sha256)

    def test_local_model_override_rejects_incomplete_snapshot(self) -> None:
        config = ServeConfig.from_file(ROOT / "configs/serve/baseline.toml")
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(ConfigError):
                config.with_local_model(temp_dir)

    def test_invalid_memory_fraction_is_rejected(self) -> None:
        config_text = (ROOT / "configs/serve/baseline.toml").read_text()
        config_text = config_text.replace(
            "gpu_memory_utilization = 0.82", "gpu_memory_utilization = 1.2"
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "invalid.toml"
            path.write_text(config_text)
            with self.assertRaises(ConfigError):
                ServeConfig.from_file(path)


class BenchmarkConfigTests(unittest.TestCase):
    def test_smoke_command_records_slo_and_metadata(self) -> None:
        config = BenchmarkConfig.from_file(ROOT / "configs/bench/smoke.toml")
        command = build_benchmark_command(config, repetition=1)

        self.assertEqual(command[:3], ["vllm", "bench", "serve"])
        self.assertIn("ttft:1000", command)
        self.assertIn("tpot:50", command)
        self.assertIn("profile=e00_smoke", command)
        self.assertIn("repetition=1", command)
        self.assertIn("--save-detailed", command)
        self.assertIn("--ready-check-timeout-sec", command)
        self.assertIn("--temperature", command)
        self.assertIn("benchmark_config_sha256=" + config.source_sha256, command)

    def test_formal_baseline_has_three_repetitions(self) -> None:
        config = BenchmarkConfig.from_file(
            ROOT / "configs/bench/baseline_short_c1.toml"
        )
        self.assertEqual(config.repetitions, 3)
        self.assertEqual(config.num_warmups, 10)
        self.assertEqual(config.num_prompts, 100)

    def test_baseline_matrix_expands_three_workloads_and_four_concurrencies(self) -> None:
        matrix = BenchmarkMatrix.from_file(ROOT / "configs/matrix/baseline.toml")
        shapes = {
            (config.input_len, config.output_len, config.max_concurrency)
            for config in matrix.configs
        }

        self.assertEqual(len(matrix.configs), 12)
        self.assertIn((128, 128, 1), shapes)
        self.assertIn((512, 256, 8), shapes)
        self.assertIn((2048, 256, 16), shapes)


if __name__ == "__main__":
    unittest.main()
