from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from qwen_serve_lab.e05 import (
    compare_e05_runs,
    parse_kv_capacity_log,
    write_e05_capacity_report,
)
from qwen_serve_lab.config import config_sha256


class E05Tests(unittest.TestCase):
    def test_paired_performance_runs_produce_valid_comparison(self) -> None:
        rows = []
        for state, throughput, ttft in (
            ("bf16", 400.0, 1200.0),
            ("fp8", 500.0, 900.0),
        ):
            for repetition in (1, 2, 3):
                rows.append(
                    {
                        "profile": f"e05_{state}_xlong_c8",
                        "state": state,
                        "workload": "xlong",
                        "repetition": repetition,
                        "effective_seed": 100 + repetition,
                        "input_len": 2048,
                        "output_len": 256,
                        "max_concurrency": 8,
                        "completed": 100,
                        "failed": 0,
                        "slo_ttft_ms": 1000.0,
                        "slo_tpot_ms": 50.0,
                        "error_rate": 0.0,
                        "request_goodput": 1.0 if state == "fp8" else 0.0,
                        "output_throughput": throughput,
                        "p95_ttft_ms": ttft,
                        "p95_tpot_ms": 15.0,
                        "peak_memory_used_mib": 11000.0,
                        "server_config_sha256": state * 32,
                        "benchmark_config_sha256": (state + "bench") * 16,
                        "generated_texts_sha256": "same" + str(repetition),
                        "valid": True,
                    }
                )

        comparisons = compare_e05_runs(rows)

        self.assertEqual(len(comparisons), 1)
        self.assertEqual(comparisons[0]["evidence"], "VALID")
        self.assertEqual(comparisons[0]["fp8_slo"], "PASS")
        self.assertEqual(comparisons[0]["performance_signal"], "BENEFIT")
        self.assertAlmostEqual(
            comparisons[0]["output_throughput_delta_percent"], 25.0
        )

    def test_capacity_parser_accepts_comma_separated_vllm_log(self) -> None:
        capacity = parse_kv_capacity_log(
            "GPU KV cache size: 63,744 tokens\n"
            "Maximum concurrency for 8,192 tokens per request: 7.78x\n"
        )

        self.assertEqual(capacity["gpu_kv_cache_tokens"], 63744)
        self.assertEqual(capacity["reference_request_tokens"], 8192)
        self.assertEqual(capacity["maximum_concurrency"], 7.78)

    def test_capacity_report_uses_latest_parseable_logs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            env_dir = root / "artifacts/env"
            server_dir = root / "artifacts/server"
            env_dir.mkdir(parents=True)
            server_dir.mkdir(parents=True)
            for index, (state, tokens, concurrency, dtype) in enumerate(
                (
                    ("bf16", 32000, 3.9, "bfloat16"),
                    ("fp8", 64000, 7.8, "fp8_e4m3"),
                )
            ):
                log_path = server_dir / f"{state}.log"
                log_path.write_text(
                    f"GPU KV cache size: {tokens:,} tokens\n"
                    f"Maximum concurrency for 8,192 tokens per request: {concurrency}x\n"
                )
                manifest = {
                    "kind": "server",
                    "profile": f"e05_kv_{state}",
                    "server_config_sha256": config_sha256(
                        Path(__file__).resolve().parents[1]
                        / f"configs/serve/e05_kv_{state}.toml"
                    ),
                    "effective_config": {"kv_cache_dtype": dtype},
                    "environment": {"project_root": str(root)},
                    "log": str(log_path.relative_to(root)),
                }
                (env_dir / f"20260821T00000{index}Z-{state}.json").write_text(
                    json.dumps(manifest)
                )

            json_path, markdown_path = write_e05_capacity_report(
                env_dir, root / "reports"
            )

            summary = json.loads(json_path.read_text())
            self.assertEqual(summary["fp8_to_bf16_token_capacity_ratio"], 2.0)
            self.assertEqual(summary["capacity_status"], "PASS")
            self.assertIn("2.000x", markdown_path.read_text())


if __name__ == "__main__":
    unittest.main()
