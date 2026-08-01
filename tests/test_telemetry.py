from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path

from qwen_serve_lab.telemetry import (
    NvidiaSmiSampler,
    parse_nvidia_smi_telemetry,
    summarize_telemetry,
)


class TelemetryTests(unittest.TestCase):
    def test_parser_extracts_numeric_gpu_fields(self) -> None:
        rows = parse_nvidia_smi_telemetry(
            "0, NVIDIA GeForce RTX 5070, 87, 9412, 12227, 68, 212.5, 2812"
        )

        self.assertEqual(rows[0]["index"], 0)
        self.assertEqual(rows[0]["memory_used_mib"], 9412)
        self.assertEqual(rows[0]["gpu_utilization_percent"], 87)

    def test_sampler_writes_csv_and_summary(self) -> None:
        def fake_query() -> tuple[int, str, str]:
            return 0, "0, RTX 5070, 75, 9000, 12227, 65, 200, 2700", ""

        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "telemetry.csv"
            with NvidiaSmiSampler(output, interval_seconds=0.01, query=fake_query):
                time.sleep(0.03)
            summary = summarize_telemetry(output)

        self.assertGreaterEqual(summary["sample_count"], 1)
        self.assertEqual(summary["peak_memory_used_mib"], 9000)
        self.assertEqual(summary["mean_gpu_utilization_percent"], 75)


if __name__ == "__main__":
    unittest.main()
