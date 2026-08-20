from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from qwen_serve_lab.comparison import (
    build_e02_comparison,
    load_e02_runs,
    write_e02_comparison,
)


class E02ComparisonTests(unittest.TestCase):
    def _write_runs(self, path: Path) -> None:
        rows = []
        for budget, throughput, ttft, temperature, power, clock in (
            (2048, 110.0, 80.0, 70.0, 150.0, 2500.0),
            (8192, 100.0, 100.0, 65.0, 155.0, 2600.0),
        ):
            for repetition in range(1, 4):
                rows.append(
                    {
                        "profile": f"e02_bt{budget}_short_c4",
                        "repetition": repetition,
                        "benchmark_config_sha256": str(budget) * 16,
                        "server_config_sha256": str(budget + 1) * 16,
                        "input_len": 128,
                        "output_len": 128,
                        "slo_ttft_ms": 1000,
                        "slo_tpot_ms": 50,
                        "max_concurrency": 4,
                        "completed": 100,
                        "failed": 0,
                        "error_rate": 0,
                        "request_goodput": 2.0,
                        "output_throughput": throughput,
                        "p95_ttft_ms": ttft,
                        "p95_tpot_ms": 10.0,
                        "peak_memory_used_mib": 10000,
                        "mean_gpu_utilization_percent": 90,
                        "max_temperature_c": temperature,
                        "mean_power_draw_w": power,
                        "mean_sm_clock_mhz": clock,
                        "valid": True,
                    }
                )
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def test_comparison_calculates_deltas_and_run_state_warning(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            runs_path = root / "runs.csv"
            self._write_runs(runs_path)

            rows = load_e02_runs(runs_path)
            compared, telemetry = build_e02_comparison(rows)
            csv_path, markdown_path = write_e02_comparison(runs_path, root / "out")

            candidate = next(row for row in compared if row["budget"] == 2048)
            self.assertEqual(candidate["output_throughput_delta_percent"], 10.0)
            self.assertEqual(candidate["p95_ttft_delta_percent"], -20.0)
            self.assertEqual(candidate["evidence_status"], "VALID")
            self.assertEqual(candidate["slo_status"], "PASS")
            self.assertEqual(len(telemetry), 2)
            self.assertTrue(csv_path.is_file())
            report = markdown_path.read_text(encoding="utf-8")
            self.assertIn("differs from 8192 in run-state telemetry", report)
            self.assertIn("SM clock -100 MHz", report)


if __name__ == "__main__":
    unittest.main()
