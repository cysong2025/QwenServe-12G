from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from qwen_serve_lab.e07 import (
    compare_e07_runs,
    write_e07_comparison,
    write_e07_final_report,
)


def _rows(lora_throughput: float = 90.0) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for workload, input_len, output_len, seed in (
        ("short", 128, 128, 20260824),
        ("medium", 512, 256, 20270824),
    ):
        for concurrency in (1, 4, 8):
            for state in ("base", "lora"):
                for repetition in (1, 2, 3):
                    treatment = state == "lora"
                    rows.append(
                        {
                            "profile": f"e07_{state}_{workload}_c{concurrency}",
                            "server_profile": f"e07_{state}",
                            "repetition": str(repetition),
                            "benchmark_config_sha256": f"bench-{state}",
                            "server_config_sha256": f"server-{state}",
                            "input_len": str(input_len),
                            "output_len": str(output_len),
                            "slo_ttft_ms": "1000",
                            "slo_tpot_ms": "50",
                            "max_concurrency": str(concurrency),
                            "completed": "100",
                            "failed": "0",
                            "error_rate": "0",
                            "request_goodput": "1.8" if treatment else "2.0",
                            "output_throughput": (
                                str(lora_throughput) if treatment else "100"
                            ),
                            "p95_ttft_ms": "110" if treatment else "100",
                            "p95_tpot_ms": "11" if treatment else "10",
                            "peak_memory_used_mib": (
                                "11200" if treatment else "11000"
                            ),
                            "valid": "True",
                            "effective_seed": str(seed + (repetition - 1) * 100),
                        }
                    )
    return rows


class E07ComparisonTests(unittest.TestCase):
    def test_complete_paired_matrix_passes_online_cost_gate(self) -> None:
        comparison = compare_e07_runs(_rows())
        self.assertEqual(len(comparison), 6)
        self.assertTrue(all(row["evidence"] == "VALID" for row in comparison))
        self.assertTrue(all(row["online_cost"] == "PASS" for row in comparison))
        self.assertAlmostEqual(comparison[0]["output_throughput_delta_percent"], -10)

    def test_cost_regression_is_preserved_as_failure(self) -> None:
        comparison = compare_e07_runs(_rows(lora_throughput=70))
        self.assertTrue(all(row["online_cost"] == "FAIL" for row in comparison))

    def test_missing_treatment_is_marked_incomplete(self) -> None:
        rows = [row for row in _rows() if row["profile"] != "e07_lora_medium_c8"]
        comparison = compare_e07_runs(rows)
        target = next(
            row
            for row in comparison
            if row["workload"] == "medium" and row["max_concurrency"] == 8
        )
        self.assertEqual(target["evidence"], "INCOMPLETE")
        self.assertEqual(target["online_cost"], "UNKNOWN")

    def test_report_writer_emits_six_cells(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runs_path = root / "runs.csv"
            rows = _rows()
            with runs_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            csv_path, markdown_path, passed = write_e07_comparison(
                runs_path, root / "reports"
            )
            self.assertTrue(passed)
            with csv_path.open("r", encoding="utf-8", newline="") as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 6)
            self.assertIn("Overall online-cost gate: **PASS**", markdown_path.read_text())

    def test_final_report_requires_all_four_gates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            (output / "adapter.json").write_text('{"passed": true}')
            (output / "quality.json").write_text(
                '{"automated_status": "PASS"}'
            )
            (output / "human_review_summary.json").write_text(
                '{"status": "PASS"}'
            )
            with (output / "comparison.csv").open(
                "w", encoding="utf-8", newline=""
            ) as handle:
                writer = csv.DictWriter(handle, fieldnames=["online_cost"])
                writer.writeheader()
                writer.writerows([{"online_cost": "PASS"}] * 6)
            _, markdown_path, passed = write_e07_final_report(output)
            self.assertTrue(passed)
            self.assertIn("Overall status: **PASS**", markdown_path.read_text())


if __name__ == "__main__":
    unittest.main()
