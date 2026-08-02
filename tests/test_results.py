from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from qwen_serve_lab.results import (
    aggregate_records,
    load_records_from_manifests,
    parse_vllm_result,
)
from qwen_serve_lab.cli import main


def sample_result(repetition: int, goodput: float) -> dict[str, object]:
    return {
        "profile": "e01_baseline_short_c1",
        "server_profile": "baseline_bf16",
        "repetition": str(repetition),
        "input_len": "128",
        "output_len": "128",
        "slo_ttft_ms": "1000",
        "slo_tpot_ms": "50",
        "benchmark_config_sha256": "a" * 64,
        "server_config_sha256": "b" * 64,
        "max_concurrency": 1,
        "completed": 100,
        "failed": 0,
        "request_throughput": 1.2,
        "request_goodput": goodput,
        "output_throughput": 153.0,
        "total_token_throughput": 306.0,
        "p50_ttft_ms": 40.0,
        "p95_ttft_ms": 60.0,
        "p99_ttft_ms": 80.0,
        "p50_tpot_ms": 8.0,
        "p95_tpot_ms": 10.0,
        "p99_tpot_ms": 12.0,
        "p95_e2el_ms": 1320.0,
    }


class ResultTests(unittest.TestCase):
    def test_manifest_evidence_is_loaded_and_aggregated(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            env_dir = root / "artifacts/env"
            result_dir = root / "artifacts/results/profile"
            env_dir.mkdir(parents=True)
            result_dir.mkdir(parents=True)
            runs = []
            for repetition, goodput in enumerate((1.0, 1.2, 1.1), start=1):
                result_path = result_dir / f"result-{repetition}.json"
                result_path.write_text(json.dumps(sample_result(repetition, goodput)))
                telemetry_path = result_dir / f"telemetry-{repetition}.csv"
                with telemetry_path.open("w", newline="") as handle:
                    writer = csv.DictWriter(
                        handle,
                        fieldnames=[
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
                        ],
                    )
                    writer.writeheader()
                    writer.writerow(
                        {
                            "gpu_utilization_percent": 80,
                            "memory_used_mib": 9000 + repetition,
                            "temperature_c": 65,
                            "power_draw_w": 200,
                            "sm_clock_mhz": 2700,
                        }
                    )
                runs.append(
                    {
                        "repetition": repetition,
                        "returncode": 0,
                        "telemetry": str(telemetry_path.relative_to(root)),
                        "result_files": [str(result_path.relative_to(root))],
                    }
                )
            manifest = {
                "profile": "e01_baseline_short_c1",
                "benchmark_config_sha256": "a" * 64,
                "environment": {"project_root": str(root)},
                "runs": runs,
            }
            (env_dir / "manifest.json").write_text(json.dumps(manifest))

            records = load_records_from_manifests(env_dir, "e01_baseline")
            filtered_records = load_records_from_manifests(
                env_dir,
                "e01_baseline",
                benchmark_config_sha256="a" * 64,
            )
            excluded_records = load_records_from_manifests(
                env_dir,
                "e01_baseline",
                benchmark_config_sha256="c" * 64,
            )
            aggregates = aggregate_records(records)
            report_dir = root / "reports"
            exit_code = main(
                [
                    "summarize",
                    "--manifest-dir",
                    str(env_dir),
                    "--output-dir",
                    str(report_dir),
                    "--profile-prefix",
                    "e01_baseline",
                ]
            )

            self.assertEqual(len(records), 3)
            self.assertEqual(len(filtered_records), 3)
            self.assertEqual(excluded_records, [])
            self.assertEqual(aggregates[0]["request_goodput"], 1.1)
            self.assertEqual(aggregates[0]["request_goodput_range"], (1.0, 1.2))
            self.assertEqual(aggregates[0]["evidence_status"], "VALID")
            self.assertEqual(aggregates[0]["slo_status"], "PASS")
            self.assertEqual(aggregates[0]["peak_memory_used_mib"], 9003)
            self.assertEqual(exit_code, 0)
            self.assertTrue((report_dir / "runs.csv").is_file())
            self.assertIn(
                "e01_baseline_short_c1",
                (report_dir / "summary.md").read_text(),
            )

    def test_slo_failure_is_separate_from_valid_evidence(self) -> None:
        records = []
        for repetition in range(1, 4):
            result = sample_result(repetition, goodput=0.1)
            result["p95_ttft_ms"] = 1200.0 if repetition == 3 else 900.0
            with tempfile.TemporaryDirectory() as temp_dir:
                result_path = Path(temp_dir) / "result.json"
                result_path.write_text(json.dumps(result))
                records.append(
                    parse_vllm_result(
                        result_path,
                        Path(temp_dir) / "manifest.json",
                        telemetry_summary={"peak_memory_used_mib": 9000},
                    )
                )

        aggregate = aggregate_records(records)[0]

        self.assertEqual(aggregate["evidence_status"], "VALID")
        self.assertEqual(aggregate["slo_status"], "FAIL")


if __name__ == "__main__":
    unittest.main()
