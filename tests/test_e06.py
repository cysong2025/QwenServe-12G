from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from qwen_serve_lab.e06 import compare_e06_runs, write_e06_comparison


CELLS = {
    "bt8192_off": (100.0, 12.0, 500.0, 1.0, False),
    "bt2048_off": (90.0, 11.8, 505.0, 1.05, False),
    "bt8192_on": (80.0, 11.7, 520.0, 1.2, True),
    "bt2048_on": (70.0, 11.6, 540.0, 1.4, True),
}


def sample_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for repetition in (1, 2, 3):
        seed = 20260823 + (repetition - 1) * 100
        for cell, (ttft, tpot, throughput, goodput, apc) in CELLS.items():
            budget = 2048 if "2048" in cell else 8192
            state = "on" if apc else "off"
            rows.append(
                {
                    "profile": f"e06_bt{budget}_{state}_reuse90_p1024_c4",
                    "cell": cell,
                    "condition": "reuse90_p1024",
                    "budget": budget,
                    "apc": state,
                    "server_profile": f"e06_bt{budget}_apc_{state}",
                    "repetition": repetition,
                    "effective_seed": seed,
                    "input_len": 2048,
                    "output_len": 256,
                    "max_concurrency": 4,
                    "prefix_len": 1024,
                    "suffix_len": 1024,
                    "num_prefixes": 10,
                    "nominal_reuse_percent": 90.0,
                    "completed": 100,
                    "failed": 0,
                    "slo_ttft_ms": 1000.0,
                    "slo_tpot_ms": 50.0,
                    "error_rate": 0.0,
                    "request_goodput": goodput,
                    "output_throughput": throughput,
                    "p95_ttft_ms": ttft,
                    "p95_tpot_ms": tpot,
                    "peak_memory_used_mib": 11000.0,
                    "prefix_cache_enabled": apc,
                    "prefix_cache_query_tokens": 204800.0 if apc else None,
                    "prefix_cache_hit_tokens": 102400.0 if apc else None,
                    "prefix_cache_hit_rate_percent": 50.0 if apc else None,
                    "benchmark_config_sha256": f"benchmark-{cell}",
                    "server_config_sha256": f"server-{cell}",
                    "generated_texts_sha256": f"outputs-{repetition}",
                    "valid": True,
                }
            )
    return rows


class E06ComparisonTests(unittest.TestCase):
    def test_complete_factorial_establishes_stacked_benefit(self) -> None:
        comparison = compare_e06_runs(sample_rows())[0]

        self.assertEqual(comparison["evidence"], "VALID")
        self.assertEqual(comparison["combined_slo"], "PASS")
        self.assertEqual(comparison["decision"], "STACKED_BENEFIT")
        self.assertAlmostEqual(
            comparison["combined_vs_best_single_ttft_percent"], -12.5
        )
        self.assertAlmostEqual(
            comparison["combined_vs_best_single_throughput_percent"],
            (540.0 - 520.0) / 520.0 * 100,
        )
        self.assertAlmostEqual(
            comparison["ttft_interaction_percentage_points"],
            (-22.2222222222) - (-20.0),
            places=5,
        )

    def test_missing_cell_invalidates_factorial_evidence(self) -> None:
        rows = [row for row in sample_rows() if row["cell"] != "bt2048_on"]

        comparison = compare_e06_runs(rows)[0]

        self.assertEqual(comparison["evidence"], "INCOMPLETE")
        self.assertEqual(comparison["decision"], "UNKNOWN")

    def test_random_output_mismatch_is_diagnostic_only(self) -> None:
        rows = sample_rows()
        rows[-1]["generated_texts_sha256"] = "different"

        comparison = compare_e06_runs(rows)[0]

        self.assertEqual(comparison["evidence"], "VALID")
        self.assertEqual(
            comparison["bt2048_on_output_matches_control"], "2/3"
        )

    def test_comparison_writes_csv_and_markdown(self) -> None:
        rows = sample_rows()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            runs_csv = root / "runs.csv"
            with runs_csv.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)

            csv_path, markdown_path = write_e06_comparison(runs_csv, root)

            self.assertTrue(csv_path.is_file())
            report = markdown_path.read_text(encoding="utf-8")
            self.assertIn("STACKED_BENEFIT", report)
            self.assertIn("Factorial interaction", report)


if __name__ == "__main__":
    unittest.main()
