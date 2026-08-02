from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

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
from qwen_serve_lab.cli import _run_matrix


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
            {
                "VLLM_USE_FLASHINFER_SAMPLER": "0",
                "VLLM_WSL2_ENABLE_PIN_MEMORY": "1",
            },
        )
        self.assertTrue(
            render_shell_command(command, build_serve_environment(config)).startswith(
                "env VLLM_USE_FLASHINFER_SAMPLER=0 "
                "VLLM_WSL2_ENABLE_PIN_MEMORY=1 vllm serve"
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
        self.assertEqual(
            baseline.use_flashinfer_sampler, prefix.use_flashinfer_sampler
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

    def test_local_tokenizer_is_explicit_and_fingerprinted(self) -> None:
        config = BenchmarkConfig.from_file(ROOT / "configs/bench/smoke.toml")
        with tempfile.TemporaryDirectory() as temp_dir:
            tokenizer_path = Path(temp_dir)
            for filename in ("tokenizer.json", "tokenizer_config.json"):
                (tokenizer_path / filename).write_text("test")
            (tokenizer_path / "SHA256SUMS").write_text(
                "fixture  tokenizer.json\n", encoding="utf-8"
            )

            local_config = config.with_local_tokenizer(tokenizer_path)
            command = build_benchmark_command(local_config)

        tokenizer_index = command.index("--tokenizer")
        self.assertEqual(command[tokenizer_index + 1], str(tokenizer_path.resolve()))
        self.assertIsNotNone(local_config.local_tokenizer_manifest_sha256)

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

    def test_matrix_resume_skips_three_matching_valid_repetitions(self) -> None:
        matrix_path = ROOT / "configs/matrix/baseline.toml"
        config = BenchmarkMatrix.from_file(matrix_path).configs[0]
        records = [
            SimpleNamespace(
                profile=config.profile_name,
                benchmark_config_sha256=config.source_sha256,
                server_config_sha256=config.server_config_sha256,
                input_len=config.input_len,
                output_len=config.output_len,
                max_concurrency=config.max_concurrency,
                completed=config.num_prompts,
                failed=0,
                valid=True,
                repetition=repetition,
            )
            for repetition in range(1, config.repetitions + 1)
        ]

        with (
            patch("qwen_serve_lab.cli.Path.is_dir", return_value=True),
            patch(
                "qwen_serve_lab.cli.load_records_from_manifests",
                return_value=records,
            ),
            patch("qwen_serve_lab.cli._execute_benchmark") as execute,
        ):
            returncode = _run_matrix(
                matrix_path,
                [config.profile_name],
                tokenizer_path=None,
                skip_completed=True,
            )

        self.assertEqual(returncode, 0)
        execute.assert_not_called()


if __name__ == "__main__":
    unittest.main()
