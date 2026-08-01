from __future__ import annotations

import unittest
from pathlib import Path

from qwen_serve_lab.environment import collect_environment, parse_nvidia_smi_rows


class EnvironmentTests(unittest.TestCase):
    def test_collection_is_non_fatal_without_gpu(self) -> None:
        snapshot = collect_environment(Path(__file__).resolve().parents[1])

        self.assertEqual(snapshot["schema_version"], 1)
        self.assertIn("platform", snapshot)
        self.assertIn("torch", snapshot)
        self.assertIn("vllm_bench", snapshot)
        self.assertIn("nvidia_smi", snapshot)
        self.assertIn("git", snapshot)

    def test_nvidia_smi_rows_are_parsed_for_doctor_checks(self) -> None:
        rows = parse_nvidia_smi_rows(
            "NVIDIA GeForce RTX 5070, 999.1, 12227, 41, 250.00"
        )

        self.assertEqual(rows[0]["name"], "NVIDIA GeForce RTX 5070")
        self.assertEqual(rows[0]["memory_total_mib"], 12227)


if __name__ == "__main__":
    unittest.main()
