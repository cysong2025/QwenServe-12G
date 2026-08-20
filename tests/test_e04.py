from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from qwen_serve_lab.e04 import compare_e04_runs, write_e04_comparison


def sample_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for repetition in range(1, 4):
        seed = 20260821 + (repetition - 1) * 100
        for state in ("off", "on"):
            rows.append(
                {
                    "condition": "reuse90_p1024",
                    "state": state,
                    "max_concurrency": 4,
                    "input_len": 2048,
                    "output_len": 256,
                    "prefix_len": 1024,
                    "suffix_len": 1024,
                    "num_prefixes": 10,
                    "nominal_reuse_percent": 90.0,
                    "repetition": repetition,
                    "effective_seed": seed,
                    "profile": f"e04_{state}_reuse90_p1024_c4",
                    "generated_texts_sha256": f"hash-{repetition}",
                    "prefix_cache_enabled": state == "on",
                    "prefix_cache_hit_rate_percent": 45.0 if state == "on" else None,
                    "valid": True,
                    "completed": 100,
                    "failed": 0,
                    "error_rate": 0.0,
                    "prefix_cache_query_tokens": 204800.0 if state == "on" else None,
                    "prefix_cache_hit_tokens": 92160.0 if state == "on" else None,
                    "p95_ttft_ms": 90.0 if state == "on" else 100.0,
                    "p95_tpot_ms": 12.0,
                    "slo_ttft_ms": 1000.0,
                    "slo_tpot_ms": 50.0,
                    "output_throughput": 505.0 if state == "on" else 500.0,
                    "request_goodput": 1.1 if state == "on" else 1.0,
                    "peak_memory_used_mib": 11000.0,
                }
            )
    return rows


class E04ComparisonTests(unittest.TestCase):
    def test_csv_report_round_trip(self) -> None:
        rows = sample_rows()
        with tempfile.TemporaryDirectory() as temp_dir:
            runs_csv = Path(temp_dir) / "runs.csv"
            with runs_csv.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)

            csv_path, markdown_path = write_e04_comparison(runs_csv, temp_dir)

            self.assertTrue(csv_path.is_file())
            self.assertIn("BENEFIT", markdown_path.read_text(encoding="utf-8"))

    def test_paired_valid_runs_can_establish_benefit(self) -> None:
        comparison = compare_e04_runs(sample_rows())[0]

        self.assertEqual(comparison["evidence"], "VALID")
        self.assertTrue(comparison["output_match"])
        self.assertEqual(comparison["p95_ttft_delta_percent"], -10)
        self.assertEqual(comparison["actual_hit_rate_percent"], 45)
        self.assertEqual(comparison["decision"], "BENEFIT")

    def test_output_mismatch_invalidates_comparison(self) -> None:
        rows = sample_rows()
        rows[-1]["generated_texts_sha256"] = "different"

        comparison = compare_e04_runs(rows)[0]

        self.assertEqual(comparison["evidence"], "INCOMPLETE")
        self.assertFalse(comparison["output_match"])
        self.assertEqual(comparison["decision"], "UNKNOWN")


if __name__ == "__main__":
    unittest.main()
